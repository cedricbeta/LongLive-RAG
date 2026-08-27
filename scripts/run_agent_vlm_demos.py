#!/usr/bin/env python3
"""Run two replay-backed full agent sessions with a real local Qwen3-VL.

The decoded action streams are the already-audited full-vector intervention renders from
the capsule channel test.  This reuses those expensive true model actions while exercising
the new online transaction/VLM/ledger boundary shot by shot.  It is an engineering demo,
not a new render and not evidence that the action improved quality.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.closed_judge import find_free_port, probe_qwen3_optimizer, qwen3_sglang_launch_command
from utils.agent_runtime import (
    AgentRuntimeConfig,
    CausalShotSession,
    GeneratedShot,
    NormativeGraph,
    ShotObservation,
    TemporalInput,
    file_sha256,
)
from utils.agent_vlm import (
    BoundedVLMController,
    BoundedVLMVerifier,
    FingerprintedVLMClient,
    compile_normative_scene_with_vlm,
)


SCENE_ARMS = {
    "bearded_watchmaker_workshop": "bearded_watchmaker_workshop__all5",
    "sandy_beach_driftwood": "sandy_beach_driftwood__all5",
}
ENTITIES = {
    "bearded_watchmaker_workshop": "elderly watchmaker",
    "sandy_beach_driftwood": "weathered driftwood log",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_free_mib(index: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            str(index),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return int(result.stdout.strip())


def start_server(args, output_root: Path) -> tuple[subprocess.Popen[Any], str, dict[str, Any]]:
    free = gpu_free_mib(args.gpu)
    if free < 61440:
        raise RuntimeError(f"GPU {args.gpu} has {free} MiB free; demo requires >= 61440 MiB")
    probe = probe_qwen3_optimizer(sglang_python=args.sglang_python, model=args.model)
    if probe.get("blocked_reason"):
        raise RuntimeError(str(probe["blocked_reason"]))
    port = find_free_port()
    launch = qwen3_sglang_launch_command(
        port=port,
        gpu=args.gpu,
        sglang_python=args.sglang_python,
        model=args.model,
        mem_fraction_static=args.mem_fraction_static,
    )
    # XGrammar otherwise permits an unbounded run of JSON whitespace between
    # required fields.  The native compact-JSON mode removes that language path.
    launch["cmd"].append("--constrained-json-disable-any-whitespace")
    server_dir = output_root / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    log_path = server_dir / "sglang.log"
    env = os.environ.copy()
    env.update(launch["env"])
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            launch["cmd"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"SGLang exited {proc.returncode}: {tail}")
        try:
            with urllib.request.urlopen(f"{launch['base_url']}/models", timeout=5) as response:
                models = json.loads(response.read().decode("utf-8"))
            return proc, launch["base_url"], {
                "gpu": args.gpu,
                "free_mib_before_start": free,
                "model": args.model,
                "probe": probe,
                "launch": {"cmd": launch["cmd"], "base_url": launch["base_url"], "mem_fraction_static": launch["mem_fraction_static"]},
                "models": models,
                "log": str(log_path.relative_to(ROOT)),
            }
        except Exception:
            time.sleep(5)
    proc.terminate()
    raise RuntimeError("SGLang startup timed out")


def shot_ranges(scene: str) -> list[tuple[int, int]]:
    durations = [
        int(value)
        for value in (ROOT / "example/benchmark_hard_multishot" / scene / "shot_durations.txt")
        .read_text(encoding="utf-8")
        .split()
    ]
    ranges, start = [], 0
    for chunks in durations:
        end = start + chunks * 8
        ranges.append((start, end))
        start = end
    return ranges


def decode_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != 480:
        raise RuntimeError(f"expected 480 frames from {path}, got {len(frames)}")
    return np.stack(frames)


def prepare_observations(scene: str, video: Path, output_root: Path) -> dict[int, dict[str, Any]]:
    frames = decode_video(video)
    mask_payload = torch.load(
        ROOT / f"videos/capsule_pilot_20260715/clipseg_masks/seed0/{scene}.pt",
        map_location="cpu",
        weights_only=False,
    )
    masks = mask_payload["binary"].bool()
    scene_dir = output_root / scene / "observations"
    scene_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for shot, (start, end) in enumerate(shot_ranges(scene)):
        indices = np.linspace(start, end - 1, num=3).round().astype(int).tolist()
        paths = []
        for local, index in enumerate(indices):
            path = scene_dir / f"shot{shot:02d}_frame{local}_idx{index:04d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR))
            paths.append(path)
        middle = indices[1]
        mask = cv2.resize(
            masks[middle].numpy().astype(np.uint8),
            (frames.shape[2], frames.shape[1]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        crop_path = None
        if mask.any():
            ys, xs = np.where(mask)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            crop = frames[middle, y0:y1, x0:x1]
            if crop.size:
                crop_path = scene_dir / f"shot{shot:02d}_entity_crop.png"
                cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        records[shot] = {
            "frame_indices": indices,
            "frames": [str(path) for path in paths],
            "crop": None if crop_path is None else str(crop_path),
        }
    return records


class ReplayActionBackend:
    def __init__(
        self,
        *,
        scene: str,
        observations: dict[int, dict[str, Any]],
        action_manifest: dict[str, Any],
        action_boundaries: list[int],
    ) -> None:
        self.scene = scene
        self.observations = observations
        self.manifest = action_manifest
        self.action_boundaries = set(action_boundaries)
        self.cursor = 0
        self.applied_reach: list[dict[str, Any]] = []
        self.delivered: list[dict[str, Any]] = []

    def telemetry(self, shot_index):
        contract = None
        if shot_index in self.action_boundaries:
            contract = {
                "tool": "steer_v",
                "params": {
                    "entity": ENTITIES[self.scene],
                    "layers": [0, 7, 14, 21, 29],
                    "alpha": 1.0,
                },
                "status": "this exact bounded action is scheduled in the audited source render",
                "controller_instruction": "select exactly this action to audit the real model-native intervention",
            }
        else:
            contract = {
                "tool": "abstain",
                "params": {},
                "status": "no intervention was scheduled at this boundary",
                "controller_instruction": "select abstain",
            }
        return {
            "fixed_boundary_full_verification": True,
            "executed_action_contract": contract,
            "committed_shots": self.cursor,
        }

    def apply(self, shot_index, program, *, retry):
        contract = self.telemetry(shot_index)["executed_action_contract"]
        results = []
        for action in program.actions:
            reached = action.tool == contract["tool"] and dict(action.params) == contract["params"]
            record = {
                "shot_index": shot_index,
                "retry": retry,
                "tool": action.tool,
                "params": dict(action.params),
                "matches_audited_render_action": reached,
            }
            self.applied_reach.append(record)
            results.append(
                {
                    "scheduled_graph_hook": reached,
                    "source_render_manifest": self.manifest.get("sample_name"),
                    "metrics": {"matches_audited_render_action": reached},
                    "attention_mass_check": "not_applicable_v_edit" if action.tool == "steer_v" else "not_applicable",
                }
            )
        return results

    def snapshot(self, shot_index):
        return {"scene": self.scene, "shot_index": shot_index, "cursor": self.cursor}

    def restore(self, snapshot):
        self.cursor = int(snapshot["cursor"])

    def generate(self, shot_index, attempt):
        observation = self.observations[shot_index]
        refs = tuple(
            TemporalInput(
                kind="current_shot_frame",
                identifier=str(path),
                sha256=file_sha256(path),
                available_after_shot=shot_index,
            )
            for path in observation["frames"]
        )
        crop_path = observation["crop"] or observation["frames"][1]
        refs = (*refs, TemporalInput(
            kind="current_shot_crop",
            identifier=str(crop_path),
            sha256=file_sha256(crop_path),
            available_after_shot=shot_index,
        ))
        actual_edits = [
            edit
            for edit in self.manifest.get("stats", {}).get("edits", [])
            if int(edit.get("boundary_shot", -1)) == shot_index
        ]
        return GeneratedShot(
            shot_index=shot_index,
            attempt=attempt,
            output=observation["frames"],
            visual_candidate={"crop": observation["crop"] or observation["frames"][1]},
            model_native_candidate={
                "source": "audited_true_capsule_v_edit_render",
                "runtime_edits": actual_edits,
                "manifest_fingerprint": file_sha256(
                    ROOT / self.manifest["_manifest_relative_path"]
                ),
            },
            telemetry={
                "actual_edit_calls": len(actual_edits),
                "actual_edited_tokens": sum(int(item.get("edited_tokens", 0)) for item in actual_edits),
            },
            input_refs=refs,
        )

    def observe(self, generated):
        crop = generated.visual_candidate["crop"]
        inputs = list(generated.input_refs)
        return ShotObservation(
            shot_index=generated.shot_index,
            attempt=generated.attempt,
            frames=generated.output,
            crops=crop,
            telemetry=generated.telemetry,
            inputs=tuple(inputs),
            input_fingerprint="",
        )

    def commit(self, generated, *, quarantined):
        self.cursor = max(self.cursor, generated.shot_index + 1)
        result = {
            "shot": generated.shot_index,
            "attempt": generated.attempt,
            "quarantined": quarantined,
        }
        self.delivered.append(result)
        return result


def make_two_up(scene: str, baseline: Path, action: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        "[0:v]scale=640:352,drawtext=text='incumbent':x=18:y=18:fontsize=26:"
        "fontcolor=white:box=1:boxcolor=black@0.6[l];"
        "[1:v]scale=640:352,drawtext=text='true-action stream (engineering demo)':"
        "x=18:y=18:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6[r];"
        "[l][r]hstack=inputs=2[v]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(baseline), "-i", str(action),
            "-filter_complex", filter_graph, "-map", "[v]", "-an", "-c:v", "libx264",
            "-crf", "20", "-preset", "veryfast", str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--sglang_python", default="/home/chendong/miniconda3/envs/sglang/bin/python")
    parser.add_argument("--mem_fraction_static", type=float, default=0.45)
    parser.add_argument("--startup_timeout", type=int, default=300)
    parser.add_argument("--output_root", default="videos/agentic_runtime_v1/demos")
    args = parser.parse_args()

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    channel = load_json(ROOT / "videos/capsule_pilot_20260715/channel_test/channel_results.json")
    proc = None
    summary: dict[str, Any] = {"schema_version": 1, "quality_claim": False, "replay_backed": True, "scenes": {}}
    try:
        proc, base_url, server = start_server(args, output_root)
        summary["vlm_server"] = server
        compiler_client = FingerprintedVLMClient(
            base_url=base_url,
            model=args.model,
            cache_dir=ROOT / "videos/agentic_runtime_v1/vlm_cache",
        )
        # Agent decisions are small bounded objects.  A separate generation cap
        # also prevents a constrained decoder from spending the remaining budget
        # on legal trailing whitespace after the JSON object is complete.
        agent_client = FingerprintedVLMClient(
            base_url=base_url,
            model=args.model,
            cache_dir=ROOT / "videos/agentic_runtime_v1/vlm_cache",
            max_tokens=512,
        )
        controller = BoundedVLMController(agent_client)
        verifier = BoundedVLMVerifier(agent_client)
        for scene, arm_name in SCENE_ARMS.items():
            arm = channel["arms"][arm_name]
            baseline = ROOT / arm["baseline"]
            action_video = ROOT / arm["edited"]
            manifest_path = ROOT / arm["runtime_integrity"]["manifest"]
            manifest = load_json(manifest_path)
            manifest["_manifest_relative_path"] = str(manifest_path.relative_to(ROOT))
            observations = prepare_observations(scene, action_video, output_root)
            backend = ReplayActionBackend(
                scene=scene,
                observations=observations,
                action_manifest=manifest,
                action_boundaries=[int(value) for value in arm["boundaries"]],
            )
            ledger = output_root / scene / "ledger.jsonl"
            if ledger.exists():
                ledger.unlink()
            config = AgentRuntimeConfig.from_config(
                {
                    "enabled": True,
                    "lock_path": str(output_root / f"{scene}.lock"),
                    "decision_cache_dir": str(ROOT / "videos/agentic_runtime_v1/vlm_cache"),
                }
            )
            graph, compiler_records = compile_normative_scene_with_vlm(
                compiler_client, ROOT / "example/benchmark_hard_multishot" / scene
            )
            with CausalShotSession(
                config,
                graph,
                backend,
                controller,
                verifier,
                ledger_path=ledger,
                acquire_lock=True,
            ) as session:
                results = session.run()
                counts = session.memory.partition_counts()
            two_up = output_root / scene / "two_up.mp4"
            make_two_up(scene, baseline, action_video, two_up)
            true_action_matches = [row for row in backend.applied_reach if row["matches_audited_render_action"] and row["tool"] == "steer_v"]
            summary["scenes"][scene] = {
                "source_baseline": str(baseline.relative_to(ROOT)),
                "source_true_action_render": str(action_video.relative_to(ROOT)),
                "source_runtime_manifest": str(manifest_path.relative_to(ROOT)),
                "source_runtime_manifest_sha256": file_sha256(manifest_path),
                "ledger": str(ledger.relative_to(ROOT)),
                "two_up": str(two_up.relative_to(ROOT)),
                "vlm_state_compiler_calls": compiler_records,
                "shots": len(results),
                "quarantined_shots": [result.shot_index for result in results if result.quarantined],
                "memory_partition_counts": counts,
                "vlm_selected_true_action_matches": true_action_matches,
                "true_action_match_count": len(true_action_matches),
                "passed": len(results) == 8 and len(true_action_matches) >= 1,
            }
        summary["passed"] = all(record["passed"] for record in summary["scenes"].values())
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": summary.get("passed"), "scenes": {key: value.get("passed") for key, value in summary.get("scenes", {}).items()}}, indent=2))
    if not summary.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
