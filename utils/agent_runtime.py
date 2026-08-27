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
    # budget is per shot, in addition to the single retry slot: a shot may be
    # repaired at most once, and a failed repair review falls back to the
    # existing restore+retry / accept-with-quarantine degradation path.
    max_edits_per_shot: int = 1
    edit_entry_sigma: float = 1.0
    edit_prompt_template: str = DEFAULT_EDIT_PROMPT_TEMPLATE

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
        )
        if result.mode != "shot_buffered":
            raise ValueError("agent runtime v1 supports only mode='shot_buffered'")
        if result.max_retries_per_shot != 1:
            raise ValueError("agent runtime v1 requires exactly one retry slot per shot")
        if result.max_edits_per_shot not in (0, 1):
            raise ValueError("agent runtime v1 allows at most one edit_shot repair per shot")
        if result.edit_entry_sigma not in ALLOWED_EDIT_SIGMA:
            raise ValueError(
                f"edit_entry_sigma must be one of {sorted(ALLOWED_EDIT_SIGMA)}"
            )
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
    def compile_scene(cls, scene_dir: str | Path) -> "NormativeGraph":
        root = Path(scene_dir)
        global_path = root / "global.json"
        if not global_path.exists():
            raise FileNotFoundError(global_path)
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
            "source": "authored_global",
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
        source_hashes = {"global.json": file_sha256(global_path)}
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
            mask_phrase = str(params.get("mask_phrase", "")).strip()
            if not mask_phrase:
                raise ValueError("edit_shot requires a non-empty mask_phrase")
            entry_sigma = float(params.get("entry_sigma", 1.0))
            if entry_sigma not in ALLOWED_EDIT_SIGMA:
                raise ValueError(
                    f"edit_shot entry_sigma must be one of {sorted(ALLOWED_EDIT_SIGMA)}"
                )
            params = {
                "shot": target,
                "prompt": prompt,
                "mask_phrase": mask_phrase,
                "entry_sigma": entry_sigma,
            }
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
            "verifier",
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
        _, verdict = self._observe_and_judge(generated, context.normative)
        return verdict

    def _commit_accepted(self, generated: GeneratedShot, verdict: Verdict) -> ShotResult:
        k = generated.shot_index
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
        verdict = self._stage_and_judge(context, generated, expected_attempt=attempt)
        if verdict.accept:
            return self._commit_accepted(generated, verdict)
        self._reject_candidate(generated, verdict)

        # edit_shot stage: a verifier-directed in-place repair, budgeted to one
        # per shot.  A passing review commits the repaired shot; a failing
        # review falls through to the pre-existing degradation path below.
        edit_action = self._select_edit_action(k, verdict)
        if edit_action is not None:
            attempt += 1
            edited, edit_verdict = self._edit_and_review(
                context, verdict, edit_action, snapshot, snapshot_digest, attempt
            )
            if edit_verdict.accept:
                return self._commit_accepted(edited, edit_verdict)
            self._reject_candidate(edited, edit_verdict)
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
            return self._commit_accepted(generated, verdict)

        quarantined = self._reject_candidate(generated, verdict)
        delivered = self.backend.commit(generated, quarantined=True)
        self.committed_shots.add(k)
        self.ledger.append(
            "commit",
            shot_index=k,
            attempt=generated.attempt,
            payload_fingerprint=quarantined.payload_fingerprint,
            quarantined=True,
            retry_count=self.retry_counts.get(k, 0),
            partition_counts=self.memory.partition_counts(),
        )
        self.scene.assert_unchanged()
        invisible_errors = self.memory.audit_invisible()
        if invisible_errors:
            raise RuntimeError("; ".join(invisible_errors))
        return ShotResult(k, generated.attempt + 1, False, True, delivered, verdict)

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
