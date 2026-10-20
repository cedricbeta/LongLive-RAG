"""Transactional, shot-buffered agent runtime for causal video generation.

The module deliberately contains no model-specific tensor synthesis.  A controller may
choose only the bounded actions defined below; a backend remains responsible for mapping
those actions onto existing prompt/KV/V intervention paths.  The session owns the trust
boundary: provisional observations are candidates until a verifier commits them, and a
quarantined candidate is never returned by a trusted-memory read.

Everything is opt-in.  :class:`AgentRuntimeConfig` defaults to ``enabled=False`` and the
pipeline does not instantiate a session on the legacy path.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


ALLOWED_STEER_LAYERS = frozenset({0, 7, 14, 21, 29})
ALLOWED_STEER_ALPHA = frozenset({0.25, 0.5, 1.0})
ALLOWED_LOGIT_BIAS = frozenset({0.0, 1.0, 2.0})
# Masked-renoise entry sigmas measured in the SDEdit repair pilot
# (docs/editing-repair-report.zh.md); sigma=1.0 (full erase inside the mask)
# was the strength champion and is the default.
ALLOWED_EDIT_SIGMA = frozenset({0.63, 0.83, 0.94, 1.0})
# Upper bound on masked-renoise repair rounds per shot (each round costs one
# shot replay); the pre-registered multi-round experiment uses 3.
MAX_EDIT_ROUNDS = 4
# Default repair-prompt template for edit_shot: authored shot caption joined
# with the judge's drift description.  Overridable via
# AgentRuntimeConfig.edit_prompt_template.
DEFAULT_EDIT_PROMPT_TEMPLATE = (
    "{caption} Repair the subject so it matches its first established appearance: {drift}"
)
ACTION_NAMES = frozenset(
    {
        "patch_prompt",
        "inject_entity_kv",
        "steer_v",
        "inject_scene_anchor_kv",
        "rope_remap",
        "logit_bias",
        "quarantine",
        "restore_and_retry",
        "edit_shot",
        "abstain",
    }
)

# A later strict-streaming implementation may filter on this field without changing the
# action schema.  Rollback itself is the only intrinsically non-forward-only operation.
FORWARD_ONLY = MappingProxyType(
    {
        "patch_prompt": True,
        "inject_entity_kv": True,
        "steer_v": True,
        "inject_scene_anchor_kv": True,
        "rope_remap": True,
        "logit_bias": True,
        "quarantine": True,
        "restore_and_retry": False,
        # edit_shot rewrites the current provisional shot by replaying it from
        # the boundary snapshot with a masked-renoise plan installed; strict
        # streaming cannot rewrite already-streamed frames.
        "edit_shot": False,
        "abstain": True,
    }
)


def _plain(value: Any) -> Any:
    """Convert config/dataclass containers into stable JSON-compatible values."""
    if dataclasses.is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "shape"):
        tensor = value.detach().cpu().contiguous()
        try:
            raw = tensor.numpy().tobytes()
        except TypeError:
            # NumPy has no native bfloat16 scalar.  Hash the exact storage bytes.
            raw = tensor.view(__import__("torch").uint8).numpy().tobytes()
        return {
            "tensor_sha256": hashlib.sha256(raw).hexdigest(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    raise TypeError(f"expected a mapping-like config, got {type(value).__name__}")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _freeze(value: Any) -> Any:
    """Recursively make JSON-like normative/memory payloads immutable."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class AgentRuntimeConfig:
    enabled: bool = False
    mode: str = "shot_buffered"
    max_retries_per_shot: int = 1
    ledger_dir: str = "videos/agentic_runtime_v1/ledgers"
    decision_cache_dir: str = "videos/agentic_runtime_v1/vlm_cache"
    lock_path: str = "/tmp/longlive_agent_runtime.lock"
    lock_nonblocking: bool = True
    verify_every_boundary: bool = True
    require_streaming_vae_off: bool = True
    archive_failures: bool = True
    # edit_shot (masked-renoise in-place repair) budget and defaults.  The edit
    # budget is per shot, in addition to the single retry slot: up to
    # ``max_edits_per_shot`` repair rounds run back to back, each one restoring
    # the boundary snapshot and replaying the shot with the verifier's latest
    # directed plan (shrinking mask, positive feedback, fresh noise seed).
    # When the budget is spent the existing restore+retry path runs, and the
    # best-scoring rejected candidate (verdict metric ``repair_score``) is
    # delivered under quarantine.
    max_edits_per_shot: int = 1
    edit_entry_sigma: float = 1.0
    edit_prompt_template: str = DEFAULT_EDIT_PROMPT_TEMPLATE
    # Hold a repaired candidate locally while comparing at most this many
    # affected future shots against the nearest stable parent branch. Preview
    # inputs are audited at their own future shot indices and never enter the
    # current controller context. A verifier may supply the affected indices in
    # ``repair_dependencies.future_shots``; contiguous shots are the legacy
    # fallback.
    # Both suffixes are judged from the verifier state that preceded the
    # current candidate's own judgment, so the guard asks whether the future
    # still meets the script, not whether it matches the reference the repair
    # has just frozen.
    # Zero preserves the original immediate-commit behavior.
    # Offline/buffered commit guard.  Values >0 intentionally inspect a
    # tentative suffix and therefore must not be reported as online/shot-causal.
    commit_lookahead_shots: int = 0

    @classmethod
    def from_config(cls, value: Any) -> "AgentRuntimeConfig":
        cfg = _mapping(value)
        result = cls(
            enabled=_as_bool(cfg.get("enabled", False), False),
            mode=str(cfg.get("mode", "shot_buffered")).lower().replace("-", "_"),
            max_retries_per_shot=int(cfg.get("max_retries_per_shot", 1)),
            ledger_dir=str(cfg.get("ledger_dir", cls.ledger_dir)),
            decision_cache_dir=str(cfg.get("decision_cache_dir", cls.decision_cache_dir)),
            lock_path=str(cfg.get("lock_path", cls.lock_path)),
            lock_nonblocking=_as_bool(cfg.get("lock_nonblocking", True), True),
            verify_every_boundary=_as_bool(cfg.get("verify_every_boundary", True), True),
            require_streaming_vae_off=_as_bool(cfg.get("require_streaming_vae_off", True), True),
            archive_failures=_as_bool(cfg.get("archive_failures", True), True),
            max_edits_per_shot=int(cfg.get("max_edits_per_shot", 1)),
            edit_entry_sigma=float(cfg.get("edit_entry_sigma", 1.0)),
            edit_prompt_template=str(cfg.get("edit_prompt_template", cls.edit_prompt_template)),
            commit_lookahead_shots=int(cfg.get("commit_lookahead_shots", 0)),
        )
        if result.mode != "shot_buffered":
            raise ValueError("agent runtime v1 supports only mode='shot_buffered'")
        if result.max_retries_per_shot != 1:
            raise ValueError("agent runtime v1 requires exactly one retry slot per shot")
        if not 0 <= result.max_edits_per_shot <= MAX_EDIT_ROUNDS:
            raise ValueError(
                f"max_edits_per_shot must be within 0..{MAX_EDIT_ROUNDS} edit_shot rounds per shot"
            )
        if result.edit_entry_sigma not in ALLOWED_EDIT_SIGMA:
            raise ValueError(
                f"edit_entry_sigma must be one of {sorted(ALLOWED_EDIT_SIGMA)}"
            )
        if result.commit_lookahead_shots < 0:
            raise ValueError("commit_lookahead_shots must be non-negative")
        return result


@dataclass(frozen=True)
class TemporalInput:
    """One auditable controller/verifier input and its causal availability."""

    kind: str
    identifier: str
    sha256: str
    available_after_shot: int
    authored: bool = False

    @classmethod
    def from_value(
        cls,
        *,
        kind: str,
        identifier: str,
        value: Any,
        available_after_shot: int,
        authored: bool = False,
    ) -> "TemporalInput":
        return cls(kind, identifier, fingerprint(value), int(available_after_shot), bool(authored))


def audit_temporal_inputs(inputs: Iterable[TemporalInput], *, as_of_shot: int) -> list[str]:
    violations: list[str] = []
    for item in inputs:
        if int(item.available_after_shot) > int(as_of_shot):
            violations.append(
                f"{item.kind}:{item.identifier} available_after_shot="
                f"{item.available_after_shot} > as_of_shot={as_of_shot}"
            )
    return violations


@dataclass(frozen=True)
class NormativeView:
    scene: str
    as_of_shot: int
    global_state: Mapping[str, Any]
    shot_state: tuple[Mapping[str, Any], ...]
    source_fingerprints: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        # Round-trip through the canonical encoder so nested MappingProxyType /
        # tuples from the immutable normative partition become plain JSON values.
        return json.loads(canonical_json({
            "scene": self.scene,
            "as_of_shot": self.as_of_shot,
            "global_state": dict(self.global_state),
            "shot_state": [dict(x) for x in self.shot_state],
            "source_fingerprints": dict(self.source_fingerprints),
        }))


class NormativeGraph:
    """Authored, immutable scene state compiled before generation.

    The full authored graph is retained privately, while ``view(k)`` exposes only global
    material plus per-shot documents through k.  This makes future-shot text unavailable
    to both controller and verifier APIs even though compilation occurs up front.
    """

    def __init__(
        self,
        scene: str,
        global_state: Mapping[str, Any],
        shots: Sequence[Mapping[str, Any]],
        source_fingerprints: Mapping[str, str],
    ) -> None:
        self.scene = str(scene)
        self._global_state = _freeze(global_state)
        self._shots = tuple(_freeze(shot) for shot in shots)
        self._source_fingerprints = MappingProxyType(dict(source_fingerprints))
        self._frozen_fingerprint = fingerprint(
            {"scene": self.scene, "global": global_state, "shots": shots}
        )

    @property
    def frozen_fingerprint(self) -> str:
        return self._frozen_fingerprint

    @property
    def num_shots(self) -> int:
        return len(self._shots)

    def view(self, shot_index: int) -> NormativeView:
        k = int(shot_index)
        if not 0 <= k < len(self._shots):
            raise IndexError(f"shot_index {k} outside normative graph of {len(self._shots)} shots")
        allowed_sources = {
            key: digest
            for key, digest in self._source_fingerprints.items()
            if key == "global.json" or (key.endswith(".json") and key[:-5].isdigit() and int(key[:-5]) <= k)
        }
        return NormativeView(
            scene=self.scene,
            as_of_shot=k,
            global_state=self._global_state,
            shot_state=self._shots[: k + 1],
            source_fingerprints=MappingProxyType(allowed_sources),
        )

    def assert_unchanged(self) -> None:
        current = fingerprint(
            {
                "scene": self.scene,
                "global": dict(self._global_state),
                "shots": [dict(x) for x in self._shots],
            }
        )
        if current != self._frozen_fingerprint:
            raise RuntimeError("normative memory changed after compilation")

    @classmethod
    def compile_scene(
        cls, scene_dir: str | Path, *, allow_missing_global: bool = False
    ) -> "NormativeGraph":
        root = Path(scene_dir)
        global_path = root / "global.json"
        if not global_path.exists():
            # Self-contained prompt suites (e.g. NarraStream-Bench) carry no
            # authored global.json by design; callers must opt in explicitly so
            # the authored-benchmark fail-fast contract stays intact.
            if not allow_missing_global:
                raise FileNotFoundError(global_path)
            global_raw: dict[str, Any] = {}
        else:
            global_raw = json.loads(global_path.read_text(encoding="utf-8"))
        shot_paths = sorted(
            (path for path in root.glob("*.json") if path.stem.isdigit()),
            key=lambda path: int(path.stem),
        )
        if not shot_paths:
            raise ValueError(f"no per-shot JSON files found in {root}")

        global_state = {
            "global_caption": global_raw.get("global_caption", global_raw.get("caption", "")),
            "invariant_caption": global_raw.get("invariant_caption", ""),
            "contrast_caption": global_raw.get("contrast_caption", ""),
            "entities": global_raw.get("entities", []),
            "invariants": global_raw.get("invariants", []),
            "allowed_state_changes": global_raw.get("allowed_state_changes", []),
            "source": "authored_global" if global_raw else "no_global_json",
        }
        shots: list[dict[str, Any]] = []
        for expected, path in enumerate(shot_paths):
            index = int(path.stem)
            if index != expected:
                raise ValueError(f"shot JSON indices must be contiguous; expected {expected}, got {index}")
            raw = json.loads(path.read_text(encoding="utf-8"))
            shots.append(
                {
                    "shot_index": index,
                    "caption": raw.get("caption", raw.get("prompt", "")),
                    "invariants": raw.get("invariants", []),
                    "allowed_state_changes": raw.get("allowed_state_changes", []),
                    "entities": raw.get("entities", []),
                    "source": "authored_shot",
                }
            )
        source_hashes = (
            {"global.json": file_sha256(global_path)} if global_path.exists() else {}
        )
        source_hashes.update({path.name: file_sha256(path) for path in shot_paths})
        return cls(root.name, global_state, shots, source_hashes)


@dataclass(frozen=True)
class MemoryPayload:
    shot_index: int
    attempt: int
    visual: Any
    model_native: Any
    provenance: tuple[TemporalInput, ...] = ()
    payload_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "visual", _freeze(self.visual))
        object.__setattr__(self, "model_native", _freeze(self.model_native))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        if not self.payload_fingerprint:
            object.__setattr__(
                self,
                "payload_fingerprint",
                fingerprint(
                    {
                        "shot_index": self.shot_index,
                        "attempt": self.attempt,
                        "visual": self.visual,
                        "model_native": self.model_native,
                    }
                ),
            )


class TrustPartitionedMemory:
    """Candidate -> persistent/quarantine memory with fail-closed trusted reads."""

    def __init__(self) -> None:
        self._candidate: dict[tuple[int, int], MemoryPayload] = {}
        self._persistent: dict[int, MemoryPayload] = {}
        self._quarantine: dict[tuple[int, int], MemoryPayload] = {}

    def stage(self, payload: MemoryPayload) -> None:
        key = (int(payload.shot_index), int(payload.attempt))
        if key in self._persistent or key in self._quarantine:
            raise RuntimeError(f"memory payload {key} already finalized")
        self._candidate[key] = payload

    def admit(self, shot_index: int, attempt: int) -> MemoryPayload:
        key = (int(shot_index), int(attempt))
        payload = self._candidate.pop(key)
        prior = self._persistent.get(int(shot_index))
        if prior is not None and prior.payload_fingerprint != payload.payload_fingerprint:
            raise RuntimeError(f"shot {shot_index} already admitted with another payload")
        self._persistent[int(shot_index)] = payload
        return payload

    def quarantine(self, shot_index: int, attempt: int) -> MemoryPayload:
        key = (int(shot_index), int(attempt))
        payload = self._candidate.pop(key)
        self._quarantine[key] = payload
        return payload

    def trusted(self, *, as_of_shot: int) -> tuple[MemoryPayload, ...]:
        return tuple(
            self._persistent[index]
            for index in sorted(self._persistent)
            if index < int(as_of_shot)
        )

    def candidate(self, shot_index: int, attempt: int) -> MemoryPayload | None:
        return self._candidate.get((int(shot_index), int(attempt)))

    def quarantined_payload(self, shot_index: int, attempt: int) -> MemoryPayload:
        return self._quarantine[(int(shot_index), int(attempt))]

    def partition_counts(self) -> dict[str, int]:
        return {
            "candidate": len(self._candidate),
            "persistent": len(self._persistent),
            "quarantine": len(self._quarantine),
        }

    def audit_invisible(self) -> list[str]:
        trusted = {payload.payload_fingerprint for payload in self._persistent.values()}
        quarantined = {payload.payload_fingerprint for payload in self._quarantine.values()}
        overlap = trusted.intersection(quarantined)
        return [f"quarantined payload visible in persistent memory: {value}" for value in sorted(overlap)]

    def snapshot_state(self) -> dict[str, Any]:
        # Payloads are immutable; shallow dict copies preserve their identity safely.
        return {
            "candidate": dict(self._candidate),
            "persistent": dict(self._persistent),
            "quarantine": dict(self._quarantine),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self._candidate = dict(state["candidate"])
        self._persistent = dict(state["persistent"])
        self._quarantine = dict(state["quarantine"])


@dataclass(frozen=True)
class Action:
    tool: str
    params: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    forward_only: bool | None = None

    def normalized(self, *, shot_index: int) -> "Action":
        tool = str(self.tool)
        if tool not in ACTION_NAMES:
            raise ValueError(f"unsupported agent tool {tool!r}")
        params = dict(self.params or {})
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError(f"{tool} requires a non-empty reason")

        if tool == "patch_prompt":
            invariant = str(params.get("invariant", "")).strip()
            if not invariant:
                raise ValueError("patch_prompt requires a non-empty invariant")
            if not bool(params.get("review_approved", False)):
                raise ValueError("patch_prompt requires review_approved=true")
            if params.get("operation", "append") != "append":
                raise ValueError("patch_prompt is append-only")
            params = {"invariant": invariant, "review_approved": True, "operation": "append"}
        elif tool == "inject_entity_kv":
            entity = str(params.get("entity", "")).strip()
            anchors = [int(x) for x in params.get("anchors", [])]
            if not entity or not anchors or len(anchors) > 4 or len(set(anchors)) != len(anchors):
                raise ValueError("inject_entity_kv requires an entity and 1..4 unique anchors")
            if params.get("region_mask") in (None, "", []):
                raise ValueError("inject_entity_kv requires a region_mask")
            boundary = int(params.get("boundary", shot_index))
            if boundary != int(shot_index):
                raise ValueError("inject_entity_kv boundary must equal the current shot")
            params = {
                "entity": entity,
                "boundary": boundary,
                "anchors": anchors,
                "region_mask": params["region_mask"],
            }
        elif tool == "steer_v":
            entity = str(params.get("entity", "")).strip()
            layers = sorted({int(x) for x in params.get("layers", [])})
            alpha = float(params.get("alpha", -1))
            if not entity or not layers or not set(layers).issubset(ALLOWED_STEER_LAYERS):
                raise ValueError(f"steer_v layers must be a non-empty subset of {sorted(ALLOWED_STEER_LAYERS)}")
            if alpha not in ALLOWED_STEER_ALPHA:
                raise ValueError(f"steer_v alpha must be one of {sorted(ALLOWED_STEER_ALPHA)}")
            params = {"entity": entity, "layers": layers, "alpha": alpha}
        elif tool == "inject_scene_anchor_kv":
            boundary = int(params.get("boundary", shot_index))
            if boundary != int(shot_index):
                raise ValueError("inject_scene_anchor_kv boundary must equal the current shot")
            params = {"boundary": boundary}
        elif tool == "rope_remap":
            entries = [int(x) for x in params.get("entries", [])]
            if not entries or len(entries) > 4 or len(set(entries)) != len(entries):
                raise ValueError("rope_remap requires 1..4 unique entry indices")
            params = {"entries": entries}
        elif tool == "logit_bias":
            value = float(params.get("lambda", -1))
            if value not in ALLOWED_LOGIT_BIAS:
                raise ValueError(f"logit_bias lambda must be one of {sorted(ALLOWED_LOGIT_BIAS)}")
            params = {"lambda": value}
        elif tool == "quarantine":
            target = int(params.get("shot", shot_index))
            if target != int(shot_index):
                raise ValueError("quarantine can target only the current shot")
            params = {"shot": target}
        elif tool == "edit_shot":
            target = int(params.get("shot", shot_index))
            if target != int(shot_index):
                raise ValueError("edit_shot can target only the current shot")
            prompt = str(params.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("edit_shot requires a non-empty repair prompt")
            # The repaint region is either a CLIPSeg phrase (legacy judge) or a
            # precomputed mask payload on disk (entity judge: SAM2 tracks).
            mask_phrase = str(params.get("mask_phrase", "")).strip()
            mask_path = str(params.get("mask_path", "")).strip()
            if not mask_phrase and not mask_path:
                raise ValueError("edit_shot requires a non-empty mask_phrase or mask_path")
            entry_sigma = float(params.get("entry_sigma", 1.0))
            if entry_sigma not in ALLOWED_EDIT_SIGMA:
                raise ValueError(
                    f"edit_shot entry_sigma must be one of {sorted(ALLOWED_EDIT_SIGMA)}"
                )
            params = {"shot": target, "prompt": prompt, "entry_sigma": entry_sigma}
            if mask_phrase:
                params["mask_phrase"] = mask_phrase
            if mask_path:
                params["mask_path"] = mask_path
        elif tool == "restore_and_retry":
            raw_program = params.get("action_program")
            if not isinstance(raw_program, Sequence) or isinstance(raw_program, (str, bytes)):
                raise ValueError("restore_and_retry requires action_program")
            nested = [
                Action(
                    tool=str(item["tool"]),
                    params=item.get("params", {}),
                    reason=str(item.get("reason", reason)),
                ).normalized(shot_index=shot_index)
                if isinstance(item, Mapping)
                else item.normalized(shot_index=shot_index)
                for item in raw_program
            ]
            if any(action.tool in ("restore_and_retry", "edit_shot") for action in nested):
                raise ValueError("restore_and_retry cannot nest restore_and_retry or edit_shot")
            params = {"action_program": [action.to_dict() for action in nested]}
        elif tool == "abstain":
            params = {}

        expected_forward = bool(FORWARD_ONLY[tool])
        if self.forward_only is not None and bool(self.forward_only) != expected_forward:
            raise ValueError(f"{tool} forward_only must be {expected_forward}")
        return Action(tool=tool, params=MappingProxyType(params), reason=reason, forward_only=expected_forward)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "params": dict(self.params),
            "reason": self.reason,
            "forward_only": self.forward_only,
        }


@dataclass(frozen=True)
class ActionProgram:
    actions: tuple[Action, ...]
    diagnosis: str
    input_fingerprint: str
    controller_call_id: str = ""

    @classmethod
    def build(
        cls,
        actions: Iterable[Action | Mapping[str, Any]],
        *,
        diagnosis: str,
        input_fingerprint: str,
        shot_index: int,
        controller_call_id: str = "",
    ) -> "ActionProgram":
        normalized: list[Action] = []
        for raw in actions:
            action = raw if isinstance(raw, Action) else Action(
                tool=str(raw.get("tool", "")),
                params=raw.get("params", {}),
                reason=str(raw.get("reason", diagnosis)),
                forward_only=raw.get("forward_only"),
            )
            normalized.append(action.normalized(shot_index=shot_index))
        if not normalized:
            raise ValueError("an action program must contain at least abstain()")
        if len(normalized) > 8:
            raise ValueError("an action program may contain at most 8 bounded actions")
        if any(action.tool == "abstain" for action in normalized) and len(normalized) != 1:
            raise ValueError("abstain must be the only action in its program")
        if not str(diagnosis).strip():
            raise ValueError("action program requires a diagnosis")
        return cls(
            tuple(normalized),
            str(diagnosis).strip(),
            str(input_fingerprint),
            str(controller_call_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "diagnosis": self.diagnosis,
            "input_fingerprint": self.input_fingerprint,
            "controller_call_id": self.controller_call_id,
        }


class SessionLock:
    """Shared process lock.  Acquisition is non-blocking by default."""

    def __init__(self, path: str | Path, *, nonblocking: bool = True) -> None:
        self.path = Path(path)
        self.nonblocking = bool(nonblocking)
        self._handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if self.nonblocking else 0)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(f"agent runtime lock is already held: {self.path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "acquired_at": time.time()}))
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "SessionLock":
        self.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


class JsonlLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        if self.path.exists():
            self._sequence = sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())

    def append(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "schema_version": 1,
            "sequence": self._sequence,
            "event": str(event),
            "wall_time_unix": time.time(),
            **_plain(payload),
        }
        record["record_sha256"] = fingerprint(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._sequence += 1
        return record

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class ShotContext:
    shot_index: int
    normative: NormativeView
    trusted_memory: tuple[MemoryPayload, ...]
    telemetry: Mapping[str, Any]
    inputs: tuple[TemporalInput, ...]
    input_fingerprint: str


@dataclass(frozen=True)
class GeneratedShot:
    shot_index: int
    attempt: int
    output: Any
    visual_candidate: Any
    model_native_candidate: Any
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    input_refs: tuple[TemporalInput, ...] = ()


@dataclass(frozen=True)
class ShotObservation:
    shot_index: int
    attempt: int
    frames: Any
    crops: Any
    telemetry: Mapping[str, Any]
    inputs: tuple[TemporalInput, ...]
    input_fingerprint: str


@dataclass(frozen=True)
class Verdict:
    accept: bool
    reason: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    directed_actions: tuple[Action | Mapping[str, Any], ...] = ()
    call_id: str = ""
    input_fingerprint: str = ""


class Controller(Protocol):
    def decide(self, context: ShotContext) -> ActionProgram: ...

    def repair(self, context: ShotContext, verdict: Verdict) -> ActionProgram: ...


class Verifier(Protocol):
    def judge(self, observation: ShotObservation, normative: NormativeView) -> Verdict: ...


class ShotBackend(Protocol):
    def snapshot(self, shot_index: int) -> Any: ...

    def restore(self, snapshot: Any) -> None: ...

    def apply(self, shot_index: int, program: ActionProgram, *, retry: bool) -> list[Mapping[str, Any]]: ...

    def generate(self, shot_index: int, attempt: int) -> GeneratedShot: ...

    def observe(self, generated: GeneratedShot) -> ShotObservation: ...

    def commit(self, generated: GeneratedShot, *, quarantined: bool) -> Any: ...

    def telemetry(self, shot_index: int) -> Mapping[str, Any]: ...

    # Optional capability used only when commit_lookahead_shots > 0. It must
    # expose a future shot from the branch identified by ``generated`` without
    # mutating backend state.  The runtime calls it for both the tentative
    # candidate and its nearest stable parent to perform a paired comparison.
    def preview(self, generated: GeneratedShot, shot_index: int) -> GeneratedShot: ...

    # Optional capability: an in-place masked-renoise repair of the current
    # provisional shot.  Called after the session has restored the boundary
    # snapshot; the backend replays the shot with the repair plan installed
    # (see CausalDiffusionInferencePipeline.set_runtime_edit_plan) and returns
    # the repaired candidate plus a JSON-compatible result mapping for the
    # ledger.  Backends without this method never receive edit_shot.
    def edit(
        self,
        shot_index: int,
        attempt: int,
        action: "Action",
        verdict: "Verdict",
    ) -> tuple[GeneratedShot, Mapping[str, Any]]: ...


@dataclass(frozen=True)
class ShotResult:
    shot_index: int
    attempts: int
    accepted: bool
    quarantined: bool
    delivered: Any
    final_verdict: Verdict


class CausalShotSession:
    """Execute the v1 shot transaction with one and only one retry slot."""

    def __init__(
        self,
        config: AgentRuntimeConfig,
        scene: NormativeGraph,
        backend: ShotBackend,
        controller: Controller,
        verifier: Verifier,
        *,
        ledger_path: str | Path,
        acquire_lock: bool = True,
    ) -> None:
        if not config.enabled:
            raise ValueError("CausalShotSession requires agent_runtime.enabled=true")
        self.config = config
        self.scene = scene
        self.backend = backend
        self.controller = controller
        self.verifier = verifier
        self.memory = TrustPartitionedMemory()
        self.ledger = JsonlLedger(ledger_path)
        self.retry_counts: dict[int, int] = {}
        self.edit_counts: dict[int, int] = {}
        self.committed_shots: set[int] = set()
        self._verifier_candidate_states: dict[tuple[int, int], Any] = {}
        self._lock = SessionLock(config.lock_path, nonblocking=config.lock_nonblocking)
        self._owns_lock = bool(acquire_lock)
        if self._owns_lock:
            self._lock.acquire()
        self.ledger.append(
            "session_start",
            scene=scene.scene,
            normative_fingerprint=scene.frozen_fingerprint,
            config=config,
        )

    def close(self) -> None:
        if self._owns_lock:
            self._lock.release()
            self._owns_lock = False

    def __enter__(self) -> "CausalShotSession":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _inputs_or_fail(
        self,
        inputs: Iterable[TemporalInput],
        *,
        shot_index: int,
        stage: str,
    ) -> tuple[TemporalInput, ...]:
        resolved = tuple(inputs)
        violations = audit_temporal_inputs(resolved, as_of_shot=shot_index)
        self.ledger.append(
            "temporal_audit",
            stage=stage,
            shot_index=shot_index,
            as_of_shot=shot_index,
            inputs=resolved,
            violations=violations,
            passed=not violations,
        )
        if violations:
            raise RuntimeError(f"future information leak at {stage}: {'; '.join(violations)}")
        return resolved

    def pre_shot(self, shot_index: int) -> ShotContext:
        k = int(shot_index)
        if k > 0 and k - 1 not in self.committed_shots:
            raise RuntimeError(f"shot {k} cannot start before shot {k - 1} is delivered")
        normative = self.scene.view(k)
        trusted = self.memory.trusted(as_of_shot=k)
        telemetry = dict(self.backend.telemetry(k))
        inputs: list[TemporalInput] = [
            TemporalInput.from_value(
                kind="normative_global",
                identifier=f"{self.scene.scene}:global",
                value=normative.global_state,
                available_after_shot=-1,
                authored=True,
            ),
            TemporalInput.from_value(
                kind="telemetry",
                identifier=f"pre_shot:{k}",
                value=telemetry,
                available_after_shot=k - 1,
            ),
        ]
        inputs.extend(
            TemporalInput.from_value(
                kind="normative_shot",
                identifier=f"{self.scene.scene}:shot{index}",
                value=shot_state,
                available_after_shot=index,
                authored=True,
            )
            for index, shot_state in enumerate(normative.shot_state)
        )
        for payload in trusted:
            inputs.extend(payload.provenance)
        audited = self._inputs_or_fail(inputs, shot_index=k, stage="pre_shot")
        digest = fingerprint(
            {
                "shot_index": k,
                "normative": normative.to_dict(),
                "trusted_memory": [payload.payload_fingerprint for payload in trusted],
                "telemetry": telemetry,
                "inputs": audited,
            }
        )
        context = ShotContext(k, normative, trusted, MappingProxyType(telemetry), audited, digest)
        self.ledger.append(
            "pre_shot",
            shot_index=k,
            input_fingerprint=digest,
            trusted_memory_fingerprints=[payload.payload_fingerprint for payload in trusted],
            normative_source_fingerprints=normative.source_fingerprints,
            telemetry=telemetry,
        )
        return context

    def apply(self, shot_index: int, program: ActionProgram, *, retry: bool) -> list[Mapping[str, Any]]:
        expected = int(self.retry_counts.get(int(shot_index), 0)) > 0
        if bool(retry) != expected:
            raise RuntimeError(f"retry flag {retry} disagrees with retry counter {self.retry_counts.get(shot_index, 0)}")
        # Rebuild so an untrusted controller cannot bypass validation by directly
        # constructing a frozen ActionProgram dataclass.
        checked = ActionProgram.build(
            program.actions,
            diagnosis=program.diagnosis,
            input_fingerprint=program.input_fingerprint,
            shot_index=shot_index,
            controller_call_id=program.controller_call_id,
        )
        if any(action.tool == "edit_shot" for action in checked.actions):
            raise ValueError(
                "edit_shot executes through the session edit stage after a verifier "
                "rejection, never through apply()"
            )
        results = list(self.backend.apply(int(shot_index), checked, retry=bool(retry)))
        for index, action in enumerate(checked.actions):
            result = results[index] if index < len(results) else {}
            self.ledger.append(
                "action",
                shot_index=shot_index,
                retry=bool(retry),
                diagnostic_input_fingerprint=checked.input_fingerprint,
                controller_call_id=checked.controller_call_id,
                diagnosis=checked.diagnosis,
                tool=action.tool,
                params=action.params,
                reason=action.reason,
                forward_only=action.forward_only,
                attention_mass_check=result.get("attention_mass_check", "not_applicable"),
                result_metrics=result.get("metrics", {}),
                result=result,
            )
        return results

    def _observe_and_judge(
        self,
        generated: GeneratedShot,
        normative: NormativeView,
        *,
        ledger_event: str = "verifier",
        ledger_extra: Mapping[str, Any] | None = None,
    ) -> tuple[ShotObservation, Verdict]:
        observation = self.backend.observe(generated)
        if observation.shot_index != generated.shot_index or observation.attempt != generated.attempt:
            raise RuntimeError("backend observation does not identify its generated candidate")
        normative_inputs = [
            TemporalInput.from_value(
                kind="normative_global",
                identifier=f"{self.scene.scene}:global",
                value=normative.global_state,
                available_after_shot=-1,
                authored=True,
            )
        ]
        normative_inputs.extend(
            TemporalInput.from_value(
                kind="normative_shot",
                identifier=f"{self.scene.scene}:shot{index}",
                value=shot_state,
                available_after_shot=index,
                authored=True,
            )
            for index, shot_state in enumerate(normative.shot_state)
        )
        inputs = self._inputs_or_fail(
            (*observation.inputs, *normative_inputs),
            shot_index=generated.shot_index,
            stage="verifier",
        )
        digest = fingerprint(
            {
                "shot_index": observation.shot_index,
                "attempt": observation.attempt,
                "frames": observation.frames,
                "crops": observation.crops,
                "telemetry": observation.telemetry,
                "normative": normative.to_dict(),
                "inputs": inputs,
            }
        )
        if observation.input_fingerprint and observation.input_fingerprint != digest:
            raise RuntimeError("backend observation fingerprint mismatch")
        observation = dataclasses.replace(observation, inputs=inputs, input_fingerprint=digest)
        verdict = self.verifier.judge(observation, normative)
        if verdict.input_fingerprint and verdict.input_fingerprint != digest:
            raise RuntimeError("verifier verdict fingerprint does not match its audited input")
        verdict = dataclasses.replace(verdict, input_fingerprint=digest)
        self.ledger.append(
            ledger_event,
            shot_index=generated.shot_index,
            attempt=generated.attempt,
            input_fingerprint=digest,
            call_id=verdict.call_id,
            accept=verdict.accept,
            reason=verdict.reason,
            metrics=verdict.metrics,
            directed_actions=[
                item.to_dict() if isinstance(item, Action) else dict(item)
                for item in verdict.directed_actions
            ],
            **dict(ledger_extra or {}),
        )
        return observation, verdict

    def _stage_and_judge(
        self,
        context: ShotContext,
        generated: GeneratedShot,
        *,
        expected_attempt: int,
    ) -> Verdict:
        k = context.shot_index
        if generated.shot_index != k or generated.attempt != int(expected_attempt):
            raise RuntimeError("backend generated candidate has an invalid shot/attempt id")
        candidate_inputs = self._inputs_or_fail(
            generated.input_refs,
            shot_index=k,
            stage="generate_candidate",
        )
        payload = MemoryPayload(
            shot_index=k,
            attempt=generated.attempt,
            visual=generated.visual_candidate,
            model_native=generated.model_native_candidate,
            provenance=candidate_inputs,
        )
        self.memory.stage(payload)
        self.ledger.append(
            "candidate",
            shot_index=k,
            attempt=generated.attempt,
            payload_fingerprint=payload.payload_fingerprint,
            partition_counts=self.memory.partition_counts(),
        )
        snapshot_state = getattr(self.verifier, "snapshot_state", None)
        # Snapshot every stateful verifier before judgment.  A rejected ordinary
        # candidate or retry is just as untrusted as a rejected edit and must not
        # be able to teach the next candidate.  Accepted candidates pop the
        # snapshot in _commit_accepted and keep their deliberate state changes.
        if callable(snapshot_state):
            self._verifier_candidate_states[(k, generated.attempt)] = snapshot_state()
        _, verdict = self._observe_and_judge(generated, context.normative)
        return verdict

    @staticmethod
    def _verdict_summary(verdict: Verdict) -> dict[str, Any]:
        """Return the stable, human-readable fields used by the paired guard."""
        raw_score = verdict.metrics.get("repair_score")
        try:
            repair_score = None if raw_score is None else float(raw_score)
        except (TypeError, ValueError):
            repair_score = None
        return {
            "accept": bool(verdict.accept),
            "reason": str(verdict.reason),
            "repair_score": repair_score,
            "metrics": dict(verdict.metrics),
        }

    @staticmethod
    def _compare_lookahead_verdicts(
        parent: Verdict,
        candidate: Verdict,
    ) -> tuple[bool, dict[str, Any]]:
        """Return whether a candidate suffix is no worse than its parent.

        Constraint-level assessments take priority when both verdicts expose
        them.  Older verifiers remain compatible through a lexicographic
        ``accept`` then ``repair_score`` comparison.  Equal failures are not a
        regression: an existing future defect must not erase a successful
        repair of the current shot.
        """
        # Local import keeps the generic runtime's legacy/no-lookahead import
        # path light.  The comparator itself owns both the generic constraint
        # policy and the accept/repair_score migration fallback.
        from utils.agent_constraints import compare_constraint_assessments

        decision = compare_constraint_assessments(parent, candidate)
        details = dict(decision.to_dict())
        details["source"] = (
            "constraint_assessments"
            if decision.comparison_mode == "constraints"
            else "legacy_verdict_metrics"
        )
        return bool(decision.allow_commit), details

    def _lookahead_indices(
        self,
        generated: GeneratedShot,
        requested: int,
    ) -> tuple[list[int], dict[str, Any]]:
        """Select up to ``requested`` affected future shots.

        A repair-aware verifier may identify future appearances of the edited
        entity/object in ``repair_dependencies.future_shots``.  This avoids a
        person-specific or fixed-distance heuristic.  Legacy verifiers fall
        back to the next contiguous authored shots.
        """
        current = int(generated.shot_index)
        dependencies = generated.telemetry.get("repair_dependencies")
        if isinstance(dependencies, Mapping) and "future_shots" in dependencies:
            raw = dependencies.get("future_shots")
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError("repair_dependencies.future_shots must be a sequence")
            eligible: set[int] = (
                set(range(current + 1, self.scene.num_shots))
                if _as_bool(dependencies.get("all_future", False))
                else set()
            )
            for value in raw:
                try:
                    index = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "repair_dependencies.future_shots must contain integers"
                    ) from exc
                if current < index < self.scene.num_shots:
                    eligible.add(index)
            ordered = sorted(eligible)
            return ordered[:requested], {
                "source": "repair_dependencies",
                "eligible": ordered,
                "limited": len(ordered) > requested,
            }

        stop = min(self.scene.num_shots, current + requested + 1)
        contiguous = list(range(current + 1, stop))
        return contiguous, {
            "source": "contiguous_fallback",
            "eligible": contiguous,
            "limited": stop < current + requested + 1,
        }

    def _lookahead_guard(
        self,
        generated: GeneratedShot,
        parent_generated: GeneratedShot | None = None,
        *,
        basis_state: Any = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Compare a repaired suffix with the nearest stable parent suffix.

        The current repaired shot has already passed the verifier absolutely.
        Future shots need not be perfect: they veto the repair only on a clear
        regression or an invalid/incomplete constraint comparison.  Parent and
        candidate judgments each start from the same frozen verifier state:
        ``basis_state`` when given (the state that preceded the current
        candidate's own judgment), otherwise the post-current state.  Judging
        from the pre-judgment state matters for identity verifiers: the entity
        just repaired has no frozen reference yet, so both suffixes are compared
        with the script description instead of with the repair itself, which
        would otherwise make the candidate win every identity comparison by
        construction.  Preview observations never enter trusted memory,
        consume attempts, call ``commit()``, or mutate backend state.
        """
        requested = int(self.config.commit_lookahead_shots)
        if requested <= 0 or not bool(generated.telemetry.get("edited", False)):
            return True, {
                "enabled": False,
                "policy": "paired_no_regression",
                "requested": requested,
                "checked": [],
                "decision": "commit_guard_disabled",
            }
        if parent_generated is None:
            raise RuntimeError(
                "paired commit lookahead requires the nearest stable parent candidate"
            )
        if int(parent_generated.shot_index) != int(generated.shot_index):
            raise RuntimeError("lookahead parent candidate belongs to another shot")
        preview = getattr(self.backend, "preview", None)
        if not callable(preview):
            raise RuntimeError(
                "commit lookahead requires backend.preview(generated, shot_index)"
            )
        snapshot_state = getattr(self.verifier, "snapshot_state", None)
        restore_state = getattr(self.verifier, "restore_state", None)
        if not callable(snapshot_state) or not callable(restore_state):
            raise RuntimeError(
                "commit lookahead requires verifier snapshot_state/restore_state"
            )
        current = int(generated.shot_index)
        future_indices, selection = self._lookahead_indices(generated, requested)
        checked: list[dict[str, Any]] = []
        preview_error = ""
        post_current_state = snapshot_state()
        judging_state = post_current_state if basis_state is None else basis_state
        backend_snapshot = self.backend.snapshot(current)
        backend_snapshot_digest = fingerprint(backend_snapshot)
        try:
            for future_index in future_indices:
                # Each side is judged from exactly the same verifier state.
                # This prevents the parent judgment from teaching or otherwise
                # contaminating the candidate judgment (and vice versa).
                restore_state(judging_state)
                parent_future = preview(parent_generated, future_index)
                _, parent_verdict = self._observe_and_judge(
                    parent_future,
                    self.scene.view(future_index),
                    ledger_event="lookahead_verifier",
                    ledger_extra={
                        "tentative_shot_index": current,
                        "decision_as_of_shot": future_index,
                        "branch_role": "parent",
                    },
                )

                restore_state(judging_state)
                candidate_future = preview(generated, future_index)
                _, candidate_verdict = self._observe_and_judge(
                    candidate_future,
                    self.scene.view(future_index),
                    ledger_event="lookahead_verifier",
                    ledger_extra={
                        "tentative_shot_index": current,
                        "decision_as_of_shot": future_index,
                        "branch_role": "candidate",
                    },
                )
                allow_commit, comparison = self._compare_lookahead_verdicts(
                    parent_verdict, candidate_verdict
                )
                checked.append(
                    {
                        "shot_index": future_index,
                        "parent": self._verdict_summary(parent_verdict),
                        "candidate": self._verdict_summary(candidate_verdict),
                        "comparison": comparison,
                        "decision": (
                            "keep_candidate" if allow_commit else "rollback_candidate"
                        ),
                    }
                )
                if not allow_commit:
                    break
        except Exception as exc:
            # A missing/failed paired judgment cannot establish no-regression.
            # Keep the failure in the ledger-facing result and roll back.
            preview_error = f"{type(exc).__name__}: {exc}"
        finally:
            restore_state(post_current_state)
        restored_backend_digest = fingerprint(self.backend.snapshot(current))
        if restored_backend_digest != backend_snapshot_digest:
            preview_error = preview_error or (
                "RuntimeError: backend.preview mutated backend state "
                f"({backend_snapshot_digest} != {restored_backend_digest})"
            )
        passed = (
            not preview_error
            and len(checked) == len(future_indices)
            and all(row["decision"] == "keep_candidate" for row in checked)
        )
        return passed, {
            "enabled": True,
            "policy": "paired_no_regression",
            "requested": requested,
            "verifier_basis": (
                "post_current_judgment" if basis_state is None else "pre_current_judgment"
            ),
            "selection": selection,
            "checked": checked,
            "error": preview_error,
            "truncated": bool(selection["limited"]),
            "decision_as_of_shot": checked[-1]["shot_index"] if checked else current,
            "decision": (
                "commit_no_regression"
                if passed
                else (
                    "rollback_guard_error"
                    if preview_error
                    else "rollback_relative_regression"
                )
            ),
        }

    def _commit_accepted(
        self,
        generated: GeneratedShot,
        verdict: Verdict,
        *,
        boundary_snapshot: Any,
        boundary_snapshot_digest: str,
        fallback: tuple[GeneratedShot, Any, str] | None = None,
        defer_rollback_delivery: bool = False,
    ) -> ShotResult | Verdict:
        k = generated.shot_index
        if not verdict.accept:
            raise RuntimeError("_commit_accepted requires an accepted current-shot verdict")
        parent_generated = fallback[0] if fallback is not None else None
        # The state snapshotted before this candidate was judged (see
        # _stage_and_judge).  Missing only for verifiers without snapshot_state,
        # in which case the guard refuses to run anyway.
        basis_state = self._verifier_candidate_states.get((k, generated.attempt))
        lookahead_passed, lookahead = self._lookahead_guard(
            generated, parent_generated, basis_state=basis_state
        )
        if lookahead["enabled"]:
            self.ledger.append(
                "commit_guard",
                shot_index=k,
                attempt=generated.attempt,
                passed=lookahead_passed,
                lookahead=lookahead,
            )
        if not lookahead_passed:
            if fallback is None:
                # Lookahead is only valid when the caller retained both the
                # parent-branch output and its post-generation state.  Falling
                # back to ``generated`` here would deliver the very repair that
                # failed the guard, while restoring some other backend state.
                self._reject_candidate(generated, verdict)
                self._restore_checked(
                    k,
                    boundary_snapshot,
                    boundary_snapshot_digest,
                    purpose="lookahead_rollback_missing_fallback",
                    decision_as_of_shot=lookahead.get("decision_as_of_shot", k),
                )
                raise RuntimeError(
                    "commit lookahead rejected an edited candidate without a "
                    "saved parent-branch fallback"
                )
            fallback_generated, fallback_snapshot, fallback_snapshot_digest = fallback
            if fallback_generated.shot_index != k:
                raise RuntimeError("lookahead fallback belongs to another shot")
            guard_verdict = Verdict(
                accept=False,
                reason="reject_commit_lookahead",
                metrics={"lookahead": lookahead},
                input_fingerprint=verdict.input_fingerprint,
            )
            self._reject_candidate(generated, guard_verdict)
            self._restore_checked(
                k,
                fallback_snapshot,
                fallback_snapshot_digest,
                purpose="lookahead_rollback",
                decision_as_of_shot=lookahead.get("decision_as_of_shot", k),
            )
            if defer_rollback_delivery:
                # An explicitly offline repair session may use this rejection
                # in another budgeted attempt. Nothing has been delivered yet.
                return guard_verdict
            # Deliver the current shot from the nearest stable parent branch,
            # not the edited candidate whose suffix just failed.  Its payload
            # was already quarantined when the repair was requested, so the
            # ledger can point at the actual delivered visual/model-native pair.
            delivered = self.backend.commit(fallback_generated, quarantined=True)
            self.committed_shots.add(k)
            fallback_payload = self.memory.quarantined_payload(
                k, fallback_generated.attempt
            )
            self.ledger.append(
                "commit",
                shot_index=k,
                attempt=fallback_generated.attempt,
                rejected_candidate_attempt=generated.attempt,
                payload_fingerprint=fallback_payload.payload_fingerprint,
                quarantined=True,
                reason="lookahead_rollback",
                lookahead=lookahead,
                partition_counts=self.memory.partition_counts(),
            )
            self.scene.assert_unchanged()
            return ShotResult(
                k,
                generated.attempt + 1,
                False,
                True,
                delivered,
                guard_verdict,
            )

        self._verifier_candidate_states.pop((k, generated.attempt), None)
        admitted = self.memory.admit(k, generated.attempt)
        delivered = self.backend.commit(generated, quarantined=False)
        self.committed_shots.add(k)
        self.ledger.append(
            "commit",
            shot_index=k,
            attempt=generated.attempt,
            payload_fingerprint=admitted.payload_fingerprint,
            quarantined=False,
            partition_counts=self.memory.partition_counts(),
        )
        self.scene.assert_unchanged()
        return ShotResult(k, generated.attempt + 1, True, False, delivered, verdict)

    def _reject_candidate(self, generated: GeneratedShot, verdict: Verdict) -> MemoryPayload:
        # Rejected candidates are archived in quarantine immediately.  Only a
        # final rejected candidate is delivered; none can enter trusted reads.
        verifier_state = self._verifier_candidate_states.pop(
            (int(generated.shot_index), int(generated.attempt)), None
        )
        restore_state = getattr(self.verifier, "restore_state", None)
        if verifier_state is not None and callable(restore_state):
            restore_state(verifier_state)
        quarantined = self.memory.quarantine(generated.shot_index, generated.attempt)
        self.ledger.append(
            "reject_candidate",
            shot_index=generated.shot_index,
            attempt=generated.attempt,
            payload_fingerprint=quarantined.payload_fingerprint,
            reason=verdict.reason,
            partition_counts=self.memory.partition_counts(),
        )
        return quarantined

    def _restore_checked(
        self,
        shot_index: int,
        snapshot: Any,
        snapshot_digest: str,
        **extra: Any,
    ) -> None:
        self.backend.restore(snapshot)
        restored_digest = fingerprint(self.backend.snapshot(int(shot_index)))
        self.ledger.append(
            "restore",
            shot_index=int(shot_index),
            expected_snapshot_fingerprint=snapshot_digest,
            restored_snapshot_fingerprint=restored_digest,
            equal=restored_digest == snapshot_digest,
            **extra,
        )
        if restored_digest != snapshot_digest:
            raise RuntimeError("snapshot restore was not state-identical")

    def _select_edit_action(self, shot_index: int, verdict: Verdict) -> Action | None:
        """Return the verifier-directed edit_shot action when it may run.

        The edit budget (at most ``max_edits_per_shot`` per shot) and backend
        support are both required; otherwise the rejection proceeds directly to
        the restore+retry / accept-with-quarantine degradation path.
        """
        k = int(shot_index)
        if int(self.config.max_edits_per_shot) <= int(self.edit_counts.get(k, 0)):
            return None
        if not callable(getattr(self.backend, "edit", None)):
            return None
        for raw in verdict.directed_actions:
            action = raw if isinstance(raw, Action) else Action(
                tool=str(raw.get("tool", "")),
                params=raw.get("params", {}),
                reason=str(raw.get("reason", verdict.reason)),
            )
            if str(action.tool) == "edit_shot":
                return action.normalized(shot_index=k)
        return None

    def _edit_and_review(
        self,
        context: ShotContext,
        verdict: Verdict,
        action: Action,
        snapshot: Any,
        snapshot_digest: str,
        attempt: int,
    ) -> tuple[GeneratedShot, Verdict]:
        """Run one masked-renoise repair of the current shot and re-judge it.

        The boundary snapshot is restored first, then the backend replays the
        shot with the repair plan installed (pass-1 denoise, masked renoise to
        the entry sigma with the repair prompt, background pinned per step, the
        edited chunks entering the clean recache).  The repaired candidate goes
        through the same observe -> verifier stage as any other candidate.
        """
        k = context.shot_index
        self.edit_counts[k] = self.edit_counts.get(k, 0) + 1
        if self.edit_counts[k] > self.config.max_edits_per_shot:
            raise RuntimeError("edit budget exceeded before restore")
        self._restore_checked(
            k, snapshot, snapshot_digest, purpose="edit_shot", edit_count=self.edit_counts[k]
        )
        edited, raw_result = self.backend.edit(k, int(attempt), action, verdict)
        # The runtime, not a backend convention, knows this candidate came from
        # edit_shot.  Stamp it here so commit guards and fallback policy cannot
        # be accidentally bypassed by an omitted telemetry field.
        edited = dataclasses.replace(
            edited,
            telemetry=MappingProxyType(
                {
                    **dict(edited.telemetry),
                    "edited": True,
                    **(
                        {
                            "repair_dependencies": _plain(
                                verdict.metrics["repair_dependencies"]
                            )
                        }
                        if "repair_dependencies" in verdict.metrics
                        else {}
                    ),
                }
            ),
        )
        result = dict(raw_result or {})
        self.ledger.append(
            "action",
            shot_index=k,
            retry=False,
            edit=True,
            edit_count=self.edit_counts[k],
            diagnostic_input_fingerprint=verdict.input_fingerprint,
            controller_call_id=verdict.call_id,
            diagnosis="identity_drift_masked_renoise_repair",
            tool=action.tool,
            params=action.params,
            reason=action.reason,
            forward_only=action.forward_only,
            attention_mass_check=result.get("attention_mass_check", "not_applicable_latent_rewrite"),
            result_metrics=result.get("metrics", {}),
            result=result,
        )
        edit_verdict = self._stage_and_judge(context, edited, expected_attempt=attempt)
        return edited, edit_verdict

    def run_shot(self, shot_index: int) -> ShotResult:
        k = int(shot_index)
        context = self.pre_shot(k)
        program = self.controller.decide(context)
        if program.input_fingerprint != context.input_fingerprint:
            raise RuntimeError("controller decision fingerprint does not match pre_shot input")
        self.apply(k, program, retry=False)

        # Per protocol, the boundary snapshot is captured after the initial action
        # program.  A repair is therefore applied on top of that restored baseline.
        snapshot = self.backend.snapshot(k)
        snapshot_digest = fingerprint(snapshot)
        self.ledger.append("snapshot", shot_index=k, inventory_fingerprint=snapshot_digest)

        attempt = 0
        generated = self.backend.generate(k, attempt)
        # A guarded edit may later fail because of its future suffix.  Keep the
        # parent branch's current-shot output *and post-generation state* so
        # rollback can deliver and continue from one coherent version.
        fallback_snapshot = self.backend.snapshot(k)
        fallback_snapshot_digest = fingerprint(fallback_snapshot)
        verdict = self._stage_and_judge(context, generated, expected_attempt=attempt)
        stable_parent_generated = generated
        stable_parent_verdict = verdict
        if verdict.accept:
            return self._commit_accepted(
                generated,
                verdict,
                boundary_snapshot=snapshot,
                boundary_snapshot_digest=snapshot_digest,
            )
        self._reject_candidate(generated, verdict)
        rejected: list[tuple[GeneratedShot, Verdict]] = [(generated, verdict)]

        # edit_shot stage: verifier-directed in-place repair rounds, budgeted
        # by max_edits_per_shot.  Every round restores the boundary snapshot
        # and replays the shot with the latest directed plan; a passing review
        # commits the repaired shot, a failing review feeds the next round.
        # When the verifier stops directing an edit (nothing left to mask) or
        # the budget is spent, control falls through to the degradation path.
        while True:
            edit_action = self._select_edit_action(k, verdict)
            if edit_action is None:
                break
            attempt += 1
            edited, edit_verdict = self._edit_and_review(
                context, verdict, edit_action, snapshot, snapshot_digest, attempt
            )
            if edit_verdict.accept:
                return self._commit_accepted(
                    edited,
                    edit_verdict,
                    boundary_snapshot=snapshot,
                    boundary_snapshot_digest=snapshot_digest,
                    fallback=(generated, fallback_snapshot, fallback_snapshot_digest),
                )
            self._reject_candidate(edited, edit_verdict)
            rejected.append((edited, edit_verdict))
            verdict = edit_verdict

        # Restore + directed retry: exactly one slot per shot.
        self.retry_counts[k] = self.retry_counts.get(k, 0) + 1
        if self.retry_counts[k] > self.config.max_retries_per_shot:
            raise RuntimeError("retry budget exceeded before restore")
        self._restore_checked(
            k, snapshot, snapshot_digest, retry_count=self.retry_counts[k]
        )
        repair = self.controller.repair(context, verdict)
        if repair.input_fingerprint != verdict.input_fingerprint:
            raise RuntimeError("repair fingerprint does not match verifier input")
        self.apply(k, repair, retry=True)

        attempt += 1
        generated = self.backend.generate(k, attempt)
        verdict = self._stage_and_judge(context, generated, expected_attempt=attempt)
        if verdict.accept:
            return self._commit_accepted(
                generated,
                verdict,
                boundary_snapshot=snapshot,
                boundary_snapshot_digest=snapshot_digest,
            )

        self._reject_candidate(generated, verdict)
        rejected.append((generated, verdict))

        # Every candidate was rejected.  Once an edit branch was attempted, a
        # failed repair must fall back to the current shot from the nearest
        # stable parent branch.  With no edit branch, the legacy best-of-N
        # policy still chooses among ordinary retry candidates.
        # Quarantined payloads never enter trusted reads regardless of choice.
        scores = [float(v.metrics.get("repair_score", 0.0)) for _, v in rejected]
        edit_failed = any(
            bool(g.telemetry.get("edited", False)) for g, _ in rejected
        )
        if edit_failed:
            self._restore_checked(
                k,
                fallback_snapshot,
                fallback_snapshot_digest,
                purpose="stable_parent_fallback",
            )
            best_generated, best_verdict = (
                stable_parent_generated,
                stable_parent_verdict,
            )
            selection = "nearest_stable_parent"
        else:
            best_index = max(range(len(rejected)), key=lambda i: (scores[i], i))
            best_generated, best_verdict = rejected[best_index]
            selection = "best_repair_score"
        delivered = self.backend.commit(best_generated, quarantined=True)
        self.committed_shots.add(k)
        self.ledger.append(
            "commit",
            shot_index=k,
            attempt=best_generated.attempt,
            payload_fingerprint=self.memory.quarantined_payload(k, best_generated.attempt).payload_fingerprint,
            quarantined=True,
            retry_count=self.retry_counts.get(k, 0),
            edit_count=self.edit_counts.get(k, 0),
            selection=selection,
            best_of={"attempts": [g.attempt for g, _ in rejected], "repair_scores": scores},
            partition_counts=self.memory.partition_counts(),
        )
        self.scene.assert_unchanged()
        invisible_errors = self.memory.audit_invisible()
        if invisible_errors:
            raise RuntimeError("; ".join(invisible_errors))
        return ShotResult(k, attempt + 1, False, True, delivered, best_verdict)

    def run(self, shot_indices: Iterable[int] | None = None) -> list[ShotResult]:
        indices = list(range(self.scene.num_shots)) if shot_indices is None else [int(x) for x in shot_indices]
        results: list[ShotResult] = []
        try:
            for shot_index in indices:
                results.append(self.run_shot(shot_index))
            self.ledger.append(
                "session_complete",
                shots=len(results),
                quarantined=sum(1 for result in results if result.quarantined),
                partition_counts=self.memory.partition_counts(),
            )
            return results
        except Exception as exc:
            self.ledger.append("session_failure", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.close()


class AbstainingController:
    """Deterministic controller used for equivalence and no-action checks."""

    def decide(self, context: ShotContext) -> ActionProgram:
        return ActionProgram.build(
            [Action("abstain", reason="no directed action requested")],
            diagnosis="no action",
            input_fingerprint=context.input_fingerprint,
            shot_index=context.shot_index,
        )

    def repair(self, context: ShotContext, verdict: Verdict) -> ActionProgram:
        # edit_shot runs only through the session edit stage; a directed edit
        # that already failed its review must not re-enter as a retry program.
        directed = tuple(
            item
            for item in verdict.directed_actions
            if str(item.tool if isinstance(item, Action) else item.get("tool", "")) != "edit_shot"
        )
        actions = directed or (
            Action("abstain", reason="verifier supplied no directed repair"),
        )
        return ActionProgram.build(
            actions,
            diagnosis=f"repair after verifier rejection: {verdict.reason}",
            input_fingerprint=verdict.input_fingerprint,
            shot_index=context.shot_index,
            controller_call_id=verdict.call_id,
        )
