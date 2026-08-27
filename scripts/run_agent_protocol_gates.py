#!/usr/bin/env python3
"""Run deterministic engineering gates for snapshot, quarantine, and retry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.agent_runtime import (
    Action,
    ActionProgram,
    AgentRuntimeConfig,
    CausalShotSession,
    GeneratedShot,
    NormativeGraph,
    ShotObservation,
    TemporalInput,
    Verdict,
    fingerprint,
)
from utils.agent_snapshot import PipelineBoundaryAdapter


SCENES = (
    "bearded_watchmaker_workshop",
    "tattooed_noodle_chef",
    "brown_bear_river",
    "sandy_beach_driftwood",
)


class Controller:
    def decide(self, context):
        return ActionProgram.build(
            [Action("abstain", reason="deterministic protocol gate")],
            diagnosis="no initial intervention",
            input_fingerprint=context.input_fingerprint,
            shot_index=context.shot_index,
        )

    def repair(self, context, verdict):
        return ActionProgram.build(
            [Action("logit_bias", {"lambda": 1}, "directed forced-fail repair")],
            diagnosis="one directed retry",
            input_fingerprint=verdict.input_fingerprint,
            shot_index=context.shot_index,
        )


class Verifier:
    def __init__(self):
        self.calls = {}

    def judge(self, observation, normative):
        key = observation.shot_index
        attempt = self.calls.get(key, 0)
        self.calls[key] = attempt + 1
        # Shot 1 is the injected mandatory-failure stub.  Every other shot passes.
        accept = key != 1
        return Verdict(
            accept=accept,
            reason="numeric guard pass" if accept else "injected mandatory verifier failure",
            metrics={"forced_fail": key == 1, "attempt": observation.attempt},
            input_fingerprint=observation.input_fingerprint,
        )


class Backend:
    def __init__(self, scene: str):
        self.scene = scene
        self.state = 0
        self.restores = 0
        self.generations = []
        self.commits = []

    def telemetry(self, shot_index):
        return {"state": self.state, "available_after_shot": shot_index - 1}

    def apply(self, shot_index, program, *, retry):
        return [
            {
                "metrics": {"reached_backend": True, "retry": retry},
                "attention_mass_check": (
                    {"required": True, "finite": True, "changed": action.tool == "logit_bias"}
                    if action.tool in {"logit_bias", "inject_scene_anchor_kv", "inject_entity_kv"}
                    else "not_applicable"
                ),
            }
            for action in program.actions
        ]

    def snapshot(self, shot_index):
        return {"scene": self.scene, "shot_index": shot_index, "state": self.state}

    def restore(self, snapshot):
        self.state = int(snapshot["state"])
        self.restores += 1

    def generate(self, shot_index, attempt):
        self.state += 1
        self.generations.append({"shot": shot_index, "attempt": attempt, "state": self.state})
        ref = TemporalInput.from_value(
            kind="current_shot_generated",
            identifier=f"{self.scene}:shot{shot_index}:attempt{attempt}",
            value=self.state,
            available_after_shot=shot_index,
        )
        return GeneratedShot(
            shot_index,
            attempt,
            {"state": self.state},
            {"crop": f"{self.scene}:{shot_index}:{attempt}"},
            {"kv": f"{self.scene}:{shot_index}:{attempt}"},
            {"state": self.state},
            (ref,),
        )

    def observe(self, generated):
        ref = TemporalInput.from_value(
            kind="current_shot_observation",
            identifier=f"{self.scene}:shot{generated.shot_index}:attempt{generated.attempt}",
            value=generated.output,
            available_after_shot=generated.shot_index,
        )
        return ShotObservation(
            generated.shot_index,
            generated.attempt,
            generated.output,
            generated.visual_candidate,
            generated.telemetry,
            (ref,),
            "",
        )

    def commit(self, generated, *, quarantined):
        record = {"shot": generated.shot_index, "attempt": generated.attempt, "quarantined": quarantined}
        self.commits.append(record)
        return record


def snapshot_gate(scene: str) -> list[dict]:
    records = []
    scene_seed = int(fingerprint(scene)[:8], 16)
    for boundary in (1, 4):
        torch.manual_seed(scene_seed + boundary)
        pipeline = SimpleNamespace(
            kv_cache_pos=[{"k": torch.randn(8)}],
            kv_cache_neg=[{"k": torch.randn(8)}],
            crossattn_cache_pos=[{"is_init": True, "v": torch.randn(4)}],
            crossattn_cache_neg=[{"is_init": True, "v": torch.randn(4)}],
            kv_rag_pos=None,
            kv_rag_neg=None,
            _dit_model=SimpleNamespace(rope_temporal_offset=float(boundary)),
        )
        output = torch.zeros(1, 8, 1, 2, 2)
        output[:, :2] = torch.randn(1, 2, 1, 2, 2)
        noise = torch.randn_like(output)
        adapter = PipelineBoundaryAdapter(pipeline, output=output, noise=noise)
        snap = adapter.capture(
            shot_index=boundary,
            committed_end_frame=2,
            shot_noise_start=2,
            shot_noise_end=4,
            current_start_frame=2,
            cache_start_frame=2,
            vae_pending_state={"pending": []},
            vae_output_cursor=2,
            extra_indices={"chunk_index": boundary},
        )

        def generate_once():
            candidate = noise[:, 2:4] + torch.rand_like(noise[:, 2:4])
            candidate = candidate + pipeline.kv_cache_pos[0]["k"].sum()
            output[:, 2:4] = candidate
            pipeline.kv_cache_pos[0]["k"].add_(1)
            return candidate.clone(), fingerprint(pipeline.kv_cache_pos)

        continuous, continuous_cache = generate_once()
        adapter.restore(snap, shot_noise_start=2, shot_noise_end=4)
        replay, replay_cache = generate_once()
        equal = torch.equal(continuous, replay) and continuous_cache == replay_cache
        records.append(
            {
                "scene": scene,
                "boundary_shot_index": boundary,
                "same_noise_decoded_latent_bit_equal": torch.equal(continuous, replay),
                "post_generation_cache_equal": continuous_cache == replay_cache,
                "passed": equal,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="videos/agentic_runtime_v1/protocol_gates/results.json")
    args = parser.parse_args()
    root = ROOT / "videos/agentic_runtime_v1/protocol_gates"
    root.mkdir(parents=True, exist_ok=True)
    snapshot_records = []
    session_records = []
    all_session_pass = True
    for scene in SCENES:
        snapshot_records.extend(snapshot_gate(scene))
        graph = NormativeGraph.compile_scene(ROOT / "example/benchmark_hard_multishot" / scene)
        backend = Backend(scene)
        verifier = Verifier()
        config = AgentRuntimeConfig.from_config(
            {"enabled": True, "lock_path": str(root / f"{scene}.lock")}
        )
        ledger = root / f"{scene}.jsonl"
        if ledger.exists():
            ledger.unlink()
        session = CausalShotSession(
            config,
            graph,
            backend,
            Controller(),
            verifier,
            ledger_path=ledger,
            acquire_lock=False,
        )
        results = [session.run_shot(index) for index in range(3)]
        trusted_at_3 = session.memory.trusted(as_of_shot=3)
        passed = bool(
            results[0].accepted
            and results[1].quarantined
            and results[1].attempts == 2
            and results[2].accepted
            and backend.restores == 1
            and [payload.shot_index for payload in trusted_at_3] == [0, 2]
            and all(payload.shot_index != 1 for payload in trusted_at_3)
            and session.memory.audit_invisible() == []
        )
        all_session_pass &= passed
        session_records.append(
            {
                "scene": scene,
                "ledger": str(ledger.relative_to(ROOT)),
                "shot_results": [
                    {"shot": item.shot_index, "attempts": item.attempts, "accepted": item.accepted, "quarantined": item.quarantined}
                    for item in results
                ],
                "restore_count": backend.restores,
                "trusted_shots_after_quarantine": [payload.shot_index for payload in trusted_at_3],
                "partition_counts": session.memory.partition_counts(),
                "passed": passed,
            }
        )
    result = {
        "schema_version": 1,
        "snapshot_restore": {
            "boundaries_per_scene": 2,
            "records": snapshot_records,
            "passed": all(row["passed"] for row in snapshot_records),
        },
        "quarantine_invisible_e2e": {
            "records": session_records,
            "passed": all_session_pass,
        },
        "retry_budget_forced_fail": {
            "expected": "exactly one restore/retry, then accept-with-quarantine",
            "passed": all_session_pass,
        },
        "quality_claim": False,
        "passed": all(row["passed"] for row in snapshot_records) and all_session_pass,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"snapshot_restore": result["snapshot_restore"]["passed"], "quarantine": result["quarantine_invisible_e2e"]["passed"], "retry": result["retry_budget_forced_fail"]["passed"], "passed": result["passed"]}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
