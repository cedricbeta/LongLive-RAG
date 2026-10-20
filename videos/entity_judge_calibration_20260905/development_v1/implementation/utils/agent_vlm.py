"""Schema-constrained, fingerprint-cached VLM controller and verifier.

This is a small reusable extraction of the Qwen call discipline used by the Round 17/18
optimizer: ordered image hashes are part of the cache key, JSON Schema is requested from
the server, local validation is mandatory, and an old transcript without an exact input
fingerprint is never reused.
"""

from __future__ import annotations

import base64
import copy
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.agent_runtime import (
    Action,
    ActionProgram,
    Controller,
    NormativeView,
    ShotContext,
    ShotObservation,
    TemporalInput,
    Verdict,
    Verifier,
    audit_temporal_inputs,
    file_sha256,
    fingerprint,
)
from utils.agent_edit import build_repair_prompt


def _action_branch(
    tool: str,
    *,
    params_properties: Mapping[str, Any],
    params_required: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tool": {"type": "string", "enum": [tool]},
            "params": {
                "type": "object",
                "additionalProperties": False,
                "properties": dict(params_properties),
                "required": list(params_required),
            },
            "reason": {"type": "string", "enum": [reason]},
        },
        "required": ["tool", "params", "reason"],
    }


ACTION_ITEM_SCHEMA: dict[str, Any] = {
    "oneOf": [
        _action_branch(
            "patch_prompt",
            params_properties={
                "invariant": {"type": "string", "minLength": 1, "maxLength": 256},
                "review_approved": {"type": "boolean", "enum": [True]},
                "operation": {"type": "string", "enum": ["append"]},
            },
            params_required=["invariant", "review_approved", "operation"],
            reason="authored_invariant_requires_append",
        ),
        _action_branch(
            "inject_entity_kv",
            params_properties={
                "entity": {"type": "string", "minLength": 1, "maxLength": 80},
                "boundary": {"type": "integer"},
                "anchors": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "uniqueItems": True,
                    "items": {"type": "integer"},
                },
                "region_mask": {"type": "string", "minLength": 1, "maxLength": 256},
            },
            params_required=["entity", "boundary", "anchors", "region_mask"],
            reason="entity_anchor_required",
        ),
        _action_branch(
            "steer_v",
            params_properties={
                "entity": {"type": "string", "minLength": 1, "maxLength": 80},
                "layers": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "uniqueItems": True,
                    "items": {"type": "integer", "enum": [0, 7, 14, 21, 29]},
                },
                "alpha": {"type": "number", "enum": [0.25, 0.5, 1.0]},
            },
            params_required=["entity", "layers", "alpha"],
            reason="bounded_v_steer_scheduled",
        ),
        _action_branch(
            "inject_scene_anchor_kv",
            params_properties={"boundary": {"type": "integer"}},
            params_required=["boundary"],
            reason="scene_anchor_required",
        ),
        _action_branch(
            "rope_remap",
            params_properties={
                "entries": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "uniqueItems": True,
                    "items": {"type": "integer"},
                }
            },
            params_required=["entries"],
            reason="rope_remap_required",
        ),
        _action_branch(
            "logit_bias",
            params_properties={"lambda": {"type": "number", "enum": [0, 1, 2]}},
            params_required=["lambda"],
            reason="bounded_logit_bias_required",
        ),
        _action_branch(
            "quarantine",
            params_properties={"shot": {"type": "integer"}},
            params_required=["shot"],
            reason="current_shot_must_be_quarantined",
        ),
        _action_branch(
            "abstain",
            params_properties={},
            params_required=[],
            reason="insufficient_evidence_abstain",
        ),
    ]
}


ACTION_PROGRAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "diagnosis": {
            "type": "string",
            "enum": [
                "no_action",
                "text_invariant_failure",
                "subject_missing",
                "identity_drift",
                "scene_layout_drift",
                "historical_tokens_downweighted",
                "current_shot_untrusted",
                "insufficient_confidence",
                "audited_action_replay",
            ],
        },
        "actions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": ACTION_ITEM_SCHEMA,
        },
    },
    "required": ["diagnosis", "actions"],
}


_CONTRACT_REASON = {
    "patch_prompt": "authored_invariant_requires_append",
    "inject_entity_kv": "entity_anchor_required",
    "steer_v": "bounded_v_steer_scheduled",
    "inject_scene_anchor_kv": "scene_anchor_required",
    "rope_remap": "rope_remap_required",
    "logit_bias": "bounded_logit_bias_required",
    "quarantine": "current_shot_must_be_quarantined",
    "edit_shot": "masked_renoise_repair_scheduled",
    "abstain": "insufficient_evidence_abstain",
}


# edit_shot is deliberately NOT part of ACTION_PROGRAM_SCHEMA: controllers may
# only select bounded pre-generation tools.  The masked-renoise repair is a
# verifier-directed action that the session runs through its dedicated edit
# stage after a rejection (utils/agent_runtime.py run_shot).
EDIT_SHOT_ACTION_SCHEMA: dict[str, Any] = _action_branch(
    "edit_shot",
    params_properties={
        "shot": {"type": "integer"},
        "prompt": {"type": "string", "minLength": 1, "maxLength": 512},
        "mask_phrase": {"type": "string", "minLength": 1, "maxLength": 80},
        "entry_sigma": {"type": "number", "enum": [0.63, 0.83, 0.94, 1.0]},
    },
    params_required=["shot", "prompt", "mask_phrase", "entry_sigma"],
    reason="masked_renoise_repair_scheduled",
)

DIRECTED_ACTION_ITEM_SCHEMA: dict[str, Any] = {
    "oneOf": [*ACTION_ITEM_SCHEMA["oneOf"], EDIT_SHOT_ACTION_SCHEMA]
}


VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "accept": {"type": "boolean"},
        "reason": {
            "type": "string",
            "enum": [
                "accept_numeric_and_visual_guards",
                "reject_invariant_mismatch",
                "reject_subject_missing",
                "reject_identity_drift",
                "reject_scene_layout_drift",
                "insufficient_confidence",
            ],
        },
        "metrics": {
            "type": "object",
            "additionalProperties": False,
            "maxProperties": 5,
            "properties": {
                "subject_visible": {"type": "boolean"},
                "identity_consistent": {"type": "boolean"},
                "scene_layout_consistent": {"type": "boolean"},
                "confidence": {"type": "number", "enum": [0, 0.25, 0.5, 0.75, 1]},
                "evidence_frames": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
            },
        },
        "directed_actions": {
            **ACTION_PROGRAM_SCHEMA["properties"]["actions"],
            "items": DIRECTED_ACTION_ITEM_SCHEMA,
            "minItems": 0,
            "maxItems": 1,
        },
    },
    "required": ["accept", "reason", "metrics", "directed_actions"],
}


DRIFT_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "subject_drifted": {"type": "boolean"},
        "drift_description": {"type": "string", "minLength": 0, "maxLength": 240},
        "reference_description": {"type": "string", "minLength": 0, "maxLength": 240},
        "repaint_region": {"type": "string", "minLength": 0, "maxLength": 120},
        "confidence": {"type": "number", "enum": [0, 0.25, 0.5, 0.75, 1]},
    },
    "required": [
        "subject_drifted",
        "drift_description",
        "reference_description",
        "repaint_region",
        "confidence",
    ],
}


STATE_COMPILER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {"type": "array", "items": {"type": "string"}},
        "invariants": {"type": "array", "items": {"type": "string"}},
        "allowed_state_changes": {"type": "array", "items": {"type": "string"}},
        "caption": {"type": "string"},
    },
    "required": ["entities", "invariants", "allowed_state_changes", "caption"],
}


def _validate_json(instance: Any, schema: Mapping[str, Any]) -> None:
    """Validate the deliberately small JSON-Schema subset used by this module.

    Keeping this local avoids making the inference environment depend on the
    optional ``jsonschema`` package.  The server still receives the full schema;
    this is the mandatory, independent fail-closed check of its response.
    """

    def walk(value: Any, rule: Mapping[str, Any], path: str) -> None:
        if "oneOf" in rule:
            matches = 0
            errors = []
            for branch in rule["oneOf"]:
                try:
                    walk(value, branch, path)
                    matches += 1
                except ValueError as exc:
                    errors.append(str(exc))
            if matches != 1:
                raise ValueError(
                    f"{path} must match exactly one bounded schema branch (matched {matches})"
                )
            return
        expected = rule.get("type")
        if expected == "object":
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be an object")
            required = set(rule.get("required", []))
            missing = required.difference(value)
            if missing:
                raise ValueError(f"{path} missing required keys {sorted(missing)}")
            properties = dict(rule.get("properties", {}))
            if rule.get("additionalProperties") is False:
                extra = set(value).difference(properties)
                if extra:
                    raise ValueError(f"{path} has unsupported keys {sorted(extra)}")
            if "maxProperties" in rule and len(value) > int(rule["maxProperties"]):
                raise ValueError(f"{path} has too many properties")
            for key, item in value.items():
                if key in properties:
                    walk(item, properties[key], f"{path}.{key}")
        elif expected == "array":
            if not isinstance(value, list):
                raise ValueError(f"{path} must be an array")
            if len(value) < int(rule.get("minItems", 0)):
                raise ValueError(f"{path} has too few items")
            if "maxItems" in rule and len(value) > int(rule["maxItems"]):
                raise ValueError(f"{path} has too many items")
            if rule.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
                raise ValueError(f"{path} items must be unique")
            item_rule = rule.get("items")
            if item_rule:
                for index, item in enumerate(value):
                    walk(item, item_rule, f"{path}[{index}]")
        elif expected == "string":
            if not isinstance(value, str):
                raise ValueError(f"{path} must be a string")
            if len(value) < int(rule.get("minLength", 0)):
                raise ValueError(f"{path} is too short")
            if "maxLength" in rule and len(value) > int(rule["maxLength"]):
                raise ValueError(f"{path} is too long")
        elif expected == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{path} must be boolean")
        elif expected == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{path} must be an integer")
        elif expected == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{path} must be numeric")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"{path} must be one of {rule['enum']}")

    walk(instance, schema, "$")


def _image_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("VLM reply must be one JSON object")
    return parsed


class FingerprintedVLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        cache_dir: str | Path,
        timeout: int = 240,
        max_parse_retries: int = 1,
        max_tokens: int = 4096,
        seed: int = 0,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.cache_dir = Path(cache_dir)
        self.timeout = int(timeout)
        self.max_parse_retries = int(max_parse_retries)
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)

    @property
    def sampling_config(self) -> dict[str, Any]:
        return {"temperature": 0, "seed": self.seed, "max_tokens": self.max_tokens}

    def call_fingerprint(
        self,
        *,
        prompt: str,
        image_paths: Iterable[str | Path],
        schema: Mapping[str, Any],
    ) -> str:
        paths = [Path(path) for path in image_paths]
        return fingerprint(
            {
                "model": self.model,
                "prompt": prompt,
                "ordered_images": [file_sha256(path) for path in paths],
                "schema": schema,
                "request_version": 2,
                "sampling": self.sampling_config,
            }
        )

    def _post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        if not isinstance(parsed, dict):
            raise RuntimeError("VLM endpoint returned a non-object response")
        return parsed

    def call(
        self,
        *,
        prompt: str,
        image_paths: Iterable[str | Path],
        schema: Mapping[str, Any],
        call_kind: str,
        shot_index: int,
        temporal_inputs: Iterable[TemporalInput],
    ) -> dict[str, Any]:
        inputs = tuple(temporal_inputs)
        violations = audit_temporal_inputs(inputs, as_of_shot=int(shot_index))
        if violations:
            raise RuntimeError(f"future information leak before VLM call: {'; '.join(violations)}")
        paths = [Path(path) for path in image_paths]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        digest = self.call_fingerprint(prompt=prompt, image_paths=paths, schema=schema)
        transcript_path = self.cache_dir / str(call_kind) / f"{digest}.json"
        if transcript_path.exists():
            cached = json.loads(transcript_path.read_text(encoding="utf-8"))
            parsed = cached.get("parsed")
            if cached.get("cache_fingerprint") == digest:
                _validate_json(parsed, schema)
                return {
                    "parsed": parsed,
                    "call_id": digest,
                    "transcript": str(transcript_path),
                    "cached": True,
                    "wall_seconds": 0.0,
                }

        content: list[dict[str, Any]] = [{"type": "text", "text": str(prompt)}]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_uri(path)}} for path in paths
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            **self.sampling_config,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": str(call_kind), "schema": dict(schema), "strict": True},
            },
        }
        started = time.time()
        attempts: list[dict[str, Any]] = []
        raw: dict[str, Any] = {}
        parsed: dict[str, Any] | None = None
        schema_enforced = True
        for attempt in range(self.max_parse_retries + 1):
            try:
                raw = self._post(payload)
            except urllib.error.HTTPError as exc:
                if schema_enforced and exc.code in {400, 422}:
                    attempts.append({"attempt": attempt, "error": f"HTTP {exc.code}: schema rejected"})
                    payload.pop("response_format", None)
                    schema_enforced = False
                    raw = self._post(payload)
                else:
                    raise
            reply = raw["choices"][0]["message"]["content"]
            try:
                parsed = _extract_json(reply)
                _validate_json(parsed, schema)
                break
            except Exception as exc:
                attempts.append({"attempt": attempt, "error": str(exc), "raw": str(reply)[:2000]})
                if attempt >= self.max_parse_retries:
                    transcript_path.parent.mkdir(parents=True, exist_ok=True)
                    failure_path = transcript_path.with_suffix(".failure.json")
                    failure_path.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "cache_fingerprint": digest,
                                "call_kind": call_kind,
                                "shot_index": int(shot_index),
                                "model": self.model,
                                "ordered_image_paths": [str(path) for path in paths],
                                "ordered_image_hashes": [file_sha256(path) for path in paths],
                                "temporal_inputs": [vars(item) for item in inputs],
                                "schema": schema,
                                "parse_attempts": attempts,
                                "last_response": raw,
                                "fatal": True,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    raise
                payload["messages"].append({"role": "assistant", "content": str(reply)})
                payload["messages"].append(
                    {
                        "role": "user",
                        "content": "The reply failed the required JSON Schema. Return only one corrected JSON object.",
                    }
                )
        assert parsed is not None
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        safe_payload = copy.deepcopy(payload)
        safe_payload["messages"][0]["content"] = [
            item
            if item.get("type") == "text"
            else {"type": "image_url", "image_url": {"url": "<base64 omitted>"}}
            for item in safe_payload["messages"][0]["content"]
        ]
        transcript = {
            "schema_version": 1,
            "cache_fingerprint": digest,
            "call_kind": call_kind,
            "shot_index": int(shot_index),
            "model": self.model,
            "request": safe_payload,
            "ordered_image_paths": [str(path) for path in paths],
            "ordered_image_hashes": [file_sha256(path) for path in paths],
            "temporal_inputs": [vars(item) for item in inputs],
            "schema": schema,
            "schema_enforced_by_server": schema_enforced,
            "parse_attempts": attempts,
            "response": raw,
            "parsed": parsed,
            "wall_seconds": time.time() - started,
        }
        transcript_path.write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "parsed": parsed,
            "call_id": digest,
            "transcript": str(transcript_path),
            "cached": False,
            "wall_seconds": transcript["wall_seconds"],
        }


def _paths_from_visual(value: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and path.exists():
            paths.append(path)
    elif isinstance(value, Mapping):
        for item in value.values():
            paths.extend(_paths_from_visual(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            paths.extend(_paths_from_visual(item))
    return paths


class BoundedVLMController(Controller):
    def __init__(self, client: FingerprintedVLMClient) -> None:
        self.client = client

    @staticmethod
    def _prompt(context: ShotContext, *, repair_reason: str | None = None) -> str:
        prompt = {
            "role": "causal boundary controller",
            "rule": "Select bounded tools only; never emit tensors. Prefer abstain when evidence is insufficient.",
            "diagnosis_and_reason": "Use only the short enum codes in the supplied schema; do not emit prose in those fields.",
            "shot_index": context.shot_index,
            "normative_as_of": context.normative.to_dict(),
            "trusted_memory": [payload.payload_fingerprint for payload in context.trusted_memory],
            "telemetry": dict(context.telemetry),
            "repair_reason": repair_reason,
            "tool_domains": {
                "patch_prompt": "append authored invariant only; params invariant, review_approved=true, operation=append",
                "inject_entity_kv": "1..4 earlier anchor indices plus region_mask",
                "steer_v": "layers subset [0,7,14,21,29], alpha in [0.25,0.5,1.0]",
                "inject_scene_anchor_kv": "current boundary",
                "rope_remap": "1..4 entry indices",
                "logit_bias": "lambda in [0,1,2]",
                "quarantine": "current shot",
                "abstain": "empty params",
            },
            "output": "strict JSON matching the supplied schema",
        }
        return json.dumps(prompt, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _schema(context: ShotContext) -> Mapping[str, Any]:
        """Return a one-action schema when telemetry names an audited contract.

        Normal online decisions retain the complete bounded tool schema.  Replay-backed
        engineering demonstrations additionally expose the action that was actually
        executed by the source render.  Constraining that case to the exact normalized
        contract prevents an otherwise valid VLM response from inventing extra actions
        and makes graph-reach evidence unambiguous.
        """

        raw_contract = context.telemetry.get("executed_action_contract")
        if raw_contract is None:
            return ACTION_PROGRAM_SCHEMA
        if not isinstance(raw_contract, Mapping):
            raise ValueError("executed_action_contract must be a mapping")
        tool = str(raw_contract.get("tool", ""))
        if tool not in _CONTRACT_REASON:
            raise ValueError(f"unsupported executed action contract tool {tool!r}")
        raw_params = raw_contract.get("params", {})
        if not isinstance(raw_params, Mapping):
            raise ValueError("executed_action_contract.params must be a mapping")
        reason = _CONTRACT_REASON[tool]
        normalized = Action(tool=tool, params=dict(raw_params), reason=reason).normalized(
            shot_index=context.shot_index
        )
        params = dict(normalized.params)
        action_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tool": {"type": "string", "enum": [tool]},
                "params": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        key: {"enum": [value]} for key, value in sorted(params.items())
                    },
                    "required": sorted(params),
                },
                "reason": {"type": "string", "enum": [reason]},
            },
            "required": ["tool", "params", "reason"],
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "enum": ["no_action" if tool == "abstain" else "audited_action_replay"],
                },
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": action_schema,
                },
            },
            "required": ["diagnosis", "actions"],
        }

    def _call(self, context: ShotContext, *, repair_reason: str | None = None, inputs: tuple[TemporalInput, ...] | None = None) -> ActionProgram:
        images: list[Path] = []
        for payload in context.trusted_memory:
            images.extend(_paths_from_visual(payload.visual))
        used_inputs = context.inputs if inputs is None else inputs
        result = self.client.call(
            prompt=self._prompt(context, repair_reason=repair_reason),
            image_paths=images,
            schema=self._schema(context),
            call_kind="controller_repair" if repair_reason else "controller_decide",
            shot_index=context.shot_index,
            temporal_inputs=used_inputs,
        )
        parsed = result["parsed"]
        input_digest = context.input_fingerprint if repair_reason is None else fingerprint(
            {"context": context.input_fingerprint, "repair_reason": repair_reason}
        )
        return ActionProgram.build(
            parsed["actions"],
            diagnosis=parsed["diagnosis"],
            input_fingerprint=input_digest,
            shot_index=context.shot_index,
            controller_call_id=result["call_id"],
        )

    def decide(self, context: ShotContext) -> ActionProgram:
        return self._call(context)

    def repair(self, context: ShotContext, verdict: Verdict) -> ActionProgram:
        # The session contract keys repair to the verifier's exact observation
        # fingerprint.  Include that fingerprint as an explicit temporal input.
        repair_input = TemporalInput(
            kind="verifier_verdict",
            identifier=verdict.call_id or f"shot{context.shot_index}:verdict",
            sha256=verdict.input_fingerprint,
            available_after_shot=context.shot_index,
        )
        result = self.client.call(
            prompt=self._prompt(context, repair_reason=verdict.reason),
            image_paths=[],
            schema=self._schema(context),
            call_kind="controller_repair",
            shot_index=context.shot_index,
            temporal_inputs=(*context.inputs, repair_input),
        )
        parsed = result["parsed"]
        return ActionProgram.build(
            parsed["actions"],
            diagnosis=parsed["diagnosis"],
            input_fingerprint=verdict.input_fingerprint,
            shot_index=context.shot_index,
            controller_call_id=result["call_id"],
        )


class BoundedVLMVerifier(Verifier):
    def __init__(self, client: FingerprintedVLMClient) -> None:
        self.client = client

    def judge(self, observation: ShotObservation, normative: NormativeView) -> Verdict:
        prompt = json.dumps(
            {
                "role": "causal current-shot verifier",
                "rule": "Judge only supplied current/past evidence against authored state available as of this shot. Return no quality comparison.",
                "reason": "Use one reason enum code from the supplied schema, never free-form prose.",
                "shot_index": observation.shot_index,
                "attempt": observation.attempt,
                "normative_as_of": normative.to_dict(),
                "telemetry": dict(observation.telemetry),
                "output": "strict JSON matching the supplied schema",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        images = _paths_from_visual(observation.frames) + _paths_from_visual(observation.crops)
        result = self.client.call(
            prompt=prompt,
            image_paths=images,
            schema=VERIFIER_SCHEMA,
            call_kind="verifier",
            shot_index=observation.shot_index,
            temporal_inputs=observation.inputs,
        )
        parsed = result["parsed"]
        actions = tuple(
            Action(
                tool=item["tool"],
                params=item.get("params", {}),
                reason=item.get("reason", parsed["reason"]),
            ).normalized(shot_index=observation.shot_index)
            for item in parsed["directed_actions"]
        )
        return Verdict(
            accept=bool(parsed["accept"]),
            reason=str(parsed["reason"]),
            metrics=dict(parsed.get("metrics", {})),
            directed_actions=actions,
            call_id=result["call_id"],
            input_fingerprint=observation.input_fingerprint,
        )


class ComparativeDriftVerifier(Verifier):
    """Comparative subject-drift judge driving the edit_shot repair loop.

    Instead of judging the current shot in isolation, the VLM sees the
    REFERENCE frames (the subject as first established, supplied by the backend
    through ``observation.telemetry['reference_frames']`` and audited through
    ``observation.inputs``) next to the CURRENT provisional frames and answers
    one comparative question: has the subject changed relative to the
    reference?  On drift the verdict is a rejection carrying one directed
    ``edit_shot`` action whose repair prompt is the shot's authored caption
    joined with the judged drift description (template hook:
    ``AgentRuntimeConfig.edit_prompt_template``).
    """

    def __init__(
        self,
        client: FingerprintedVLMClient,
        *,
        entity_phrase: str,
        edit_entry_sigma: float = 1.0,
        prompt_template: str | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        self.client = client
        self.entity_phrase = str(entity_phrase)
        self.edit_entry_sigma = float(edit_entry_sigma)
        self.prompt_template = prompt_template
        self.min_confidence = float(min_confidence)

    def judge(self, observation: ShotObservation, normative: NormativeView) -> Verdict:
        reference = [Path(item) for item in observation.telemetry.get("reference_frames", [])]
        current = _paths_from_visual(observation.frames) + _paths_from_visual(observation.crops)
        if not reference:
            # Nothing committed to compare against yet (opening shot).
            return Verdict(
                accept=True,
                reason="accept_numeric_and_visual_guards",
                metrics={"drift_reference_available": False},
                input_fingerprint=observation.input_fingerprint,
            )
        prompt = json.dumps(
            {
                "role": "comparative subject-drift judge",
                "rule": (
                    "The first images are REFERENCE frames showing the subject as first "
                    "established in this scene; the remaining images are the CURRENT "
                    "provisional shot. Judge ONLY whether an ESTABLISHED character's "
                    "identity or appearance has changed relative to the reference "
                    "(species, color, face, clothing, distinguishing marks), or a "
                    "character has been replaced or duplicated. The current shot's "
                    "authored caption (current_caption below) may legitimately "
                    "introduce new characters or remove existing ones — narrative "
                    "entrances and exits that the caption calls for are NOT drift. "
                    "Ignore pose, camera angle, framing, lighting, and allowed state "
                    "changes."
                ),
                "current_caption": (
                    str(normative.shot_state[-1].get("caption", ""))
                    if normative.shot_state
                    else ""
                ),
                "subject": self.entity_phrase,
                "reference_image_count": len(reference),
                "current_image_count": len(current),
                "shot_index": observation.shot_index,
                "attempt": observation.attempt,
                "normative_as_of": normative.to_dict(),
                "drift_description": (
                    "When drifted, describe concretely what changed versus the reference "
                    "in one short sentence; empty string otherwise."
                ),
                "reference_description": (
                    "Describe the subject's appearance AS SEEN IN THE REFERENCE frames in "
                    "one positive phrase (colors, clothing, distinguishing marks). This "
                    "text repaints the subject, so name ONLY what should appear — never "
                    "mention the wrong or drifted appearance."
                ),
                "repaint_region": (
                    "Short segmentation phrase naming everything in the CURRENT frames "
                    "that must be erased and repainted: the drifted subject PLUS any "
                    "person/object that the current caption does NOT call for. Never "
                    "include characters the caption legitimately places in the shot. "
                    "Use a broad noun like 'person' only when every visible person is "
                    "wrong. Empty string if not drifted."
                ),
                "output": "strict JSON matching the supplied schema",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        result = self.client.call(
            prompt=prompt,
            image_paths=[*reference, *current],
            schema=DRIFT_JUDGE_SCHEMA,
            call_kind="drift_judge",
            shot_index=observation.shot_index,
            temporal_inputs=observation.inputs,
        )
        parsed = result["parsed"]
        confidence = float(parsed["confidence"])
        drifted = bool(parsed["subject_drifted"]) and confidence >= self.min_confidence
        metrics = {
            "subject_drifted": bool(parsed["subject_drifted"]),
            "confidence": confidence,
            "drift_description": str(parsed["drift_description"]),
            "reference_description": str(parsed.get("reference_description", "")),
            "repaint_region": str(parsed.get("repaint_region", "")),
        }
        if not drifted:
            return Verdict(
                accept=True,
                reason="accept_numeric_and_visual_guards",
                metrics=metrics,
                call_id=result["call_id"],
                input_fingerprint=observation.input_fingerprint,
            )
        caption = str(normative.shot_state[-1].get("caption", "")) if normative.shot_state else ""
        # The repaint prompt must carry only the TARGET appearance. Diffusion text
        # conditioning is bag-of-concepts: feeding the drift narration ("a woman
        # with red hair appeared") paints the drift back in. The drift text stays
        # in metrics/ledger for audit; the prompt gets the positive reference
        # description, falling back to the authored caption alone.
        repair_prompt = build_repair_prompt(
            caption,
            str(parsed.get("reference_description", "")),
            template=self.prompt_template,
        )
        # Mask what the judge says must be repainted — the drifted subject plus
        # any extra person/object that appeared. A static entity phrase leaves
        # appeared-content outside the mask, where the background pin preserves
        # it verbatim (pilot ns100: repaired man came back, spurious woman
        # survived pinned beside him).
        mask_phrase = str(parsed.get("repaint_region", "")).strip() or self.entity_phrase
        action = Action(
            "edit_shot",
            {
                "shot": observation.shot_index,
                "prompt": repair_prompt,
                "mask_phrase": mask_phrase,
                "entry_sigma": self.edit_entry_sigma,
            },
            reason="masked_renoise_repair_scheduled",
        ).normalized(shot_index=observation.shot_index)
        return Verdict(
            accept=False,
            reason="reject_identity_drift",
            metrics=metrics,
            directed_actions=(action,),
            call_id=result["call_id"],
            input_fingerprint=observation.input_fingerprint,
        )


def compile_normative_scene_with_vlm(
    client: FingerprintedVLMClient,
    scene_dir: str | Path,
) -> tuple["NormativeGraph", list[dict[str, Any]]]:
    """Compile authored global/shot JSON independently with the real VLM.

    Each shot document is compiled in isolation and later exposed only by
    ``NormativeGraph.view(k)``.  The authored source remains embedded verbatim;
    generated observations never enter this function and cannot rewrite the
    normative partition.
    """
    from utils.agent_runtime import NormativeGraph

    root = Path(scene_dir)
    global_path = root / "global.json"
    shot_paths = sorted(
        (path for path in root.glob("*.json") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not global_path.exists() or not shot_paths:
        raise FileNotFoundError(f"incomplete authored scene directory: {root}")

    records: list[dict[str, Any]] = []

    def compile_one(path: Path, *, shot_index: int, kind: str) -> dict[str, Any]:
        authored = json.loads(path.read_text(encoding="utf-8"))
        source_ref = TemporalInput(
            kind=f"authored_{kind}",
            identifier=str(path),
            sha256=file_sha256(path),
            available_after_shot=-1 if kind == "global" else shot_index,
            authored=True,
        )
        prompt = json.dumps(
            {
                "role": "normative state compiler",
                "rule": (
                    "Extract entities, invariants, and explicitly allowed state changes only. "
                    "Do not invent facts, weaken authored invariants, or use generated imagery."
                ),
                "scene": root.name,
                "document_kind": kind,
                "shot_index": None if kind == "global" else shot_index,
                "authored_json": authored,
                "output": "strict JSON matching the supplied schema",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        result = client.call(
            prompt=prompt,
            image_paths=[],
            schema=STATE_COMPILER_SCHEMA,
            call_kind=f"state_compiler_{kind}",
            shot_index=max(0, shot_index),
            temporal_inputs=(source_ref,),
        )
        parsed = dict(result["parsed"])
        parsed["authored_source"] = authored
        parsed["source_sha256"] = source_ref.sha256
        parsed["source"] = f"vlm_compiled_authored_{kind}"
        records.append(
            {
                "kind": kind,
                "shot_index": None if kind == "global" else shot_index,
                "source": str(path),
                "source_sha256": source_ref.sha256,
                "call_id": result["call_id"],
                "transcript": result["transcript"],
                "cached": result["cached"],
            }
        )
        return parsed

    global_state = compile_one(global_path, shot_index=-1, kind="global")
    shots = [
        {"shot_index": int(path.stem), **compile_one(path, shot_index=int(path.stem), kind="shot")}
        for path in shot_paths
    ]
    sources = {"global.json": file_sha256(global_path)}
    sources.update({path.name: file_sha256(path) for path in shot_paths})
    return NormativeGraph(root.name, global_state, shots, sources), records
