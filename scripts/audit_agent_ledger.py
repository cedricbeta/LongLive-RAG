#!/usr/bin/env python3
"""Audit causal scope, action completeness, and cached-decision replay in ledgers."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.agent_runtime import TemporalInput, audit_temporal_inputs, file_sha256, fingerprint
from utils.agent_vlm import FingerprintedVLMClient


ACTION_REQUIRED = {
    "diagnostic_input_fingerprint",
    "tool",
    "params",
    "reason",
    "attention_mass_check",
    "result_metrics",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_record_hash(record: dict[str, Any]) -> bool:
    payload = dict(record)
    actual = payload.pop("record_sha256", "")
    return actual == fingerprint(payload)


def cache_transcript(cache_root: Path, call_id: str) -> Path | None:
    matches = sorted(cache_root.rglob(f"{call_id}.json"))
    return matches[0] if len(matches) == 1 else None


def audit_cache(path: Path, cache_root: Path) -> dict[str, Any]:
    transcript = json.loads(path.read_text(encoding="utf-8"))
    shot = int(transcript["shot_index"])
    inputs = tuple(TemporalInput(**item) for item in transcript.get("temporal_inputs", []))
    temporal_violations = audit_temporal_inputs(inputs, as_of_shot=shot)
    image_errors = []
    paths = [Path(item) for item in transcript.get("ordered_image_paths", [])]
    hashes = list(transcript.get("ordered_image_hashes", []))
    if len(paths) != len(hashes):
        image_errors.append("ordered image path/hash length mismatch")
    for image, expected in zip(paths, hashes):
        if not image.exists():
            image_errors.append(f"missing image {image}")
        elif file_sha256(image) != expected:
            image_errors.append(f"image hash changed {image}")
    prompt = transcript["request"]["messages"][0]["content"][0]["text"]
    recomputed = fingerprint(
        {
            "model": transcript["model"],
            "prompt": prompt,
            "ordered_images": hashes,
            "schema": transcript["schema"],
        }
    )
    replay_error = None
    replayed = None
    try:
        # Port 9 is deliberately unusable.  A successful result therefore proves
        # this exact request was reconstructed from the fingerprint cache without
        # contacting a model server.
        replay_client = FingerprintedVLMClient(
            base_url="http://127.0.0.1:9/v1",
            model=transcript["model"],
            cache_dir=cache_root,
            timeout=1,
        )
        replayed = replay_client.call(
            prompt=prompt,
            image_paths=paths,
            schema=transcript["schema"],
            call_kind=transcript["call_kind"],
            shot_index=shot,
            temporal_inputs=inputs,
        )
    except Exception as exc:
        replay_error = f"{type(exc).__name__}: {exc}"
    replay_equal = bool(
        replayed
        and replayed.get("cached") is True
        and replayed.get("call_id") == transcript.get("cache_fingerprint")
        and replayed.get("parsed") == transcript.get("parsed")
    )
    schema_enforced = transcript.get("schema_enforced_by_server") is True
    return {
        "path": str(path.relative_to(ROOT)),
        "temporal_violations": temporal_violations,
        "image_errors": image_errors,
        "expected_fingerprint": transcript.get("cache_fingerprint"),
        "recomputed_fingerprint": recomputed,
        "fingerprint_equal": recomputed == transcript.get("cache_fingerprint"),
        "parsed_fingerprint": fingerprint(transcript.get("parsed")),
        "schema_enforced_by_server": schema_enforced,
        "offline_cache_replay_equal": replay_equal,
        "offline_cache_replay_error": replay_error,
        "passed": (
            not temporal_violations
            and not image_errors
            and recomputed == transcript.get("cache_fingerprint")
            and replay_equal
            and schema_enforced
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledgers", nargs="+", help="JSONL ledger paths")
    parser.add_argument("--cache_root", default="videos/agentic_runtime_v1/vlm_cache")
    parser.add_argument("--output", default="videos/agentic_runtime_v1/ledger_audit.json")
    parser.add_argument(
        "--demo_summary",
        default="videos/agentic_runtime_v1/demos/summary.json",
        help="Demo summary containing the referenced normative-compiler calls",
    )
    parser.add_argument("--replay_actions", type=int, default=3)
    args = parser.parse_args()

    cache_root = ROOT / args.cache_root
    ledger_results = []
    replay_candidates: list[dict[str, Any]] = []
    vlm_references: dict[str, dict[str, Any]] = {}
    for raw_path in args.ledgers:
        path = (ROOT / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path)
        records = read_jsonl(path)
        hash_errors = [row["sequence"] for row in records if not verify_record_hash(row)]
        sequence_errors = [
            index for index, row in enumerate(records) if int(row.get("sequence", -1)) != index
        ]
        temporal_failures = [
            row for row in records if row.get("event") == "temporal_audit" and not row.get("passed")
        ]
        actions = [row for row in records if row.get("event") == "action"]
        incomplete = [
            row.get("sequence") for row in actions if ACTION_REQUIRED.difference(row)
        ]
        for row in actions:
            if row.get("controller_call_id"):
                replay_candidates.append({"ledger": path, "action": row})
                call_id = str(row["controller_call_id"])
                vlm_references[call_id] = {
                    "call_id": call_id,
                    "kind": "controller",
                    "scene": path.parent.name,
                }
        for row in records:
            if row.get("event") == "verifier" and row.get("call_id"):
                call_id = str(row["call_id"])
                vlm_references[call_id] = {
                    "call_id": call_id,
                    "kind": "verifier",
                    "scene": path.parent.name,
                }
        ledger_results.append(
            {
                "path": str(path.relative_to(ROOT)),
                "records": len(records),
                "actions": len(actions),
                "hash_errors": hash_errors,
                "sequence_errors": sequence_errors,
                "future_leak_events": [row.get("sequence") for row in temporal_failures],
                "incomplete_action_events": incomplete,
                "passed": not hash_errors and not sequence_errors and not temporal_failures and not incomplete,
            }
        )

    summary_path = ROOT / args.demo_summary
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for scene, record in summary.get("scenes", {}).items():
        for compiler in record.get("vlm_state_compiler_calls", []):
            call_id = str(compiler["call_id"])
            vlm_references[call_id] = {
                "call_id": call_id,
                "kind": "state_compiler",
                "scene": scene,
            }

    vlm_call_audits = []
    for call_id, reference in sorted(vlm_references.items()):
        transcript_path = cache_transcript(cache_root, call_id)
        if transcript_path is None:
            vlm_call_audits.append(
                {**reference, "passed": False, "reason": "exact cache transcript not found"}
            )
            continue
        vlm_call_audits.append({**reference, "cache": audit_cache(transcript_path, cache_root)})
        vlm_call_audits[-1]["passed"] = bool(vlm_call_audits[-1]["cache"]["passed"])

    rng = random.Random(20260716)
    rng.shuffle(replay_candidates)
    chosen = replay_candidates[: int(args.replay_actions)]
    replays = []
    for candidate in chosen:
        action = candidate["action"]
        call_id = str(action["controller_call_id"])
        transcript_path = cache_transcript(cache_root, call_id)
        if transcript_path is None:
            replays.append(
                {
                    "call_id": call_id,
                    "action_sequence": action["sequence"],
                    "passed": False,
                    "reason": "exact cache transcript not found",
                }
            )
            continue
        cache_audit = audit_cache(transcript_path, cache_root)
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        parsed_actions = transcript.get("parsed", {}).get("actions", [])
        action_match = any(
            item.get("tool") == action.get("tool")
            and item.get("params", {}) == action.get("params", {})
            and item.get("reason") == action.get("reason")
            for item in parsed_actions
        )
        replays.append(
            {
                "call_id": call_id,
                "action_sequence": action["sequence"],
                "ledger": str(candidate["ledger"].relative_to(ROOT)),
                "cache": cache_audit,
                "same_decision": action_match,
                "passed": bool(cache_audit["passed"] and action_match),
            }
        )

    enough_replays = len(replays) == int(args.replay_actions)
    all_vlm_calls_pass = bool(vlm_call_audits) and all(
        item["passed"] for item in vlm_call_audits
    )
    result = {
        "schema_version": 1,
        "gate": "ledger_and_no_future_leak",
        "ledgers": ledger_results,
        "replay_sample_seed": 20260716,
        "requested_replay_actions": int(args.replay_actions),
        "replays": replays,
        "referenced_vlm_calls": vlm_call_audits,
        "referenced_vlm_call_count": len(vlm_call_audits),
        "all_referenced_vlm_calls_pass": all_vlm_calls_pass,
        "no_future_leak_pass": all(item["passed"] for item in ledger_results),
        "ledger_complete_pass": all(item["passed"] for item in ledger_results),
        "replay_pass": enough_replays and all(item["passed"] for item in replays),
        "passed": (
            all(item["passed"] for item in ledger_results)
            and enough_replays
            and all(item["passed"] for item in replays)
            and all_vlm_calls_pass
        ),
        "quality_claim": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "no_future_leak_pass",
                    "ledger_complete_pass",
                    "replay_pass",
                    "referenced_vlm_call_count",
                    "all_referenced_vlm_calls_pass",
                    "passed",
                )
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
