#!/usr/bin/env python3
"""Closed-loop agent edit_shot demo: judge drift -> masked-renoise repair -> re-review.

For each demo scene (bearded_watchmaker_workshop, sandy_beach_driftwood) this driver:

1. renders the scene once as the deterministic baseline (seeded replay);
2. runs a CausalShotSession over its shots, where the comparative VLM drift judge
   (utils.agent_vlm.ComparativeDriftVerifier) compares each provisional shot against the
   delivered opening-shot reference frames;
3. on a drift rejection, executes the directed edit_shot action: CLIPSeg subject masks for
   the shot's chunks (utils.agent_edit), a repair prompt from the authored caption plus the
   judged drift description, and a re-render of the scene with the masked-renoise
   edit_renoise block active for exactly those chunks (sigma default 1.0) -- deterministic
   replay keeps every pre-edit chunk bit-identical, so this is precisely "restore the
   boundary and replay the shot with the repair installed";
4. re-reviews the repaired shot with the same judge: pass -> the repaired render becomes
   the delivery stream for this and subsequent shots (butterfly effect: later shots are
   re-judged against the repaired history); fail -> the pre-existing restore+retry /
   accept-with-quarantine degradation path (retry of a deterministic replay reproduces the
   same content, so a failed repair normally ends accept-with-quarantine);
5. writes the full ledger, a summary.json, and a 2-up baseline-vs-final video.

Engineering demo only: it shows the loop turning, it does not claim quality improvements.
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.closed_judge import find_free_port, probe_qwen3_optimizer, qwen3_sglang_launch_command
from utils.agent_edit import clipseg_chunk_masks, mask_coverage, save_mask_payload
from utils.agent_runtime import (
    AbstainingController,
    AgentRuntimeConfig,
    CausalShotSession,
    GeneratedShot,
    NormativeGraph,
    ShotObservation,
    TemporalInput,
    file_sha256,
)
from utils.agent_vlm import ComparativeDriftVerifier, FingerprintedVLMClient

SCENES = ("bearded_watchmaker_workshop", "sandy_beach_driftwood")
ENTITIES = {
    "bearded_watchmaker_workshop": "elderly watchmaker",
    "sandy_beach_driftwood": "weathered driftwood log",
}
FRAMES_PER_CHUNK = 8
PYTHON_BIN = "/home/chendong/miniconda3/bin/python"


def gpu_free_mib(index: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(index)],
        check=True,
        text=True,
        capture_output=True,
    )
    return int(result.stdout.strip())


def start_vlm_server(args, output_root: Path):
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
            with urllib.request.urlopen(f"{launch['base_url']}/models", timeout=5):
                return proc, launch["base_url"]
        except Exception:
            time.sleep(5)
    proc.terminate()
    raise RuntimeError("SGLang startup timed out")


def shot_chunk_ranges(scene_dir: Path) -> list[tuple[int, int]]:
    """Inclusive [chunk_start, chunk_end] per shot from shot_durations.txt."""
    durations = [int(value) for value in (scene_dir / "shot_durations.txt").read_text().split()]
    ranges, start = [], 0
    for chunks in durations:
        ranges.append((start, start + chunks - 1))
        start += chunks
    return ranges


def build_render_config(
    base_config: Path,
    *,
    scene: str,
    data_dir: Path,
    output_folder: Path,
    prefix: str,
    edit: dict[str, Any] | None,
    target: Path,
) -> Path:
    data = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    data["output_folder"] = str(output_folder)
    data.setdefault("data", {})["data_path"] = str(data_dir)
    inference = data.setdefault("inference", {})
    inference["output_folder"] = str(output_folder)
    inference["filename_prefix"] = prefix
    inference["filename_from_sample_name"] = True
    data["filename_from_sample_name"] = True
    data.setdefault("logging", {})["seed"] = int(data.get("logging", {}).get("seed", 0))
    if edit is not None:
        inference["edit_renoise"] = {"enabled": True, **edit}
    else:
        inference.pop("edit_renoise", None)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def render(config_path: Path, *, gpu: int, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
        }
    )
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [PYTHON_BIN, "inference.py", "--config_path", str(config_path)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def find_render(output_folder: Path, prefix: str, scene: str) -> Path:
    matches = sorted(output_folder.glob(f"{prefix}-rank*-*{scene}*_regular.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {prefix} render for {scene}, found {matches}")
    return matches[0]


def extract_frames(video: Path, frame_indices: list[int], out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    paths = []
    for local, index in enumerate(frame_indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not read frame {index} from {video}")
        path = out_dir / f"{stem}_f{local}_idx{index:04d}.png"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    capture.release()
    return paths


class EditRenderBackend:
    """Shot backend over deterministic full-scene renders.

    generate() slices the current active render; edit() re-renders the scene with the
    masked-renoise repair active for exactly the shot's chunks (deterministic replay ==
    restore-to-boundary + replay-with-plan); commit() of an accepted repaired shot swaps
    the active render so later shots inherit the repaired history.
    """

    def __init__(
        self,
        *,
        scene: str,
        scene_dir: Path,
        out_dir: Path,
        base_config: Path,
        baseline_video: Path,
        render_gpu: int,
        clipseg_device: str,
        noise_seed: int = 4242,
    ) -> None:
        self.scene = scene
        self.scene_dir = scene_dir
        self.out_dir = out_dir
        self.base_config = base_config
        self.render_gpu = int(render_gpu)
        self.clipseg_device = clipseg_device
        self.noise_seed = int(noise_seed)
        self.chunk_ranges = shot_chunk_ranges(scene_dir)
        self.active_video = Path(baseline_video)
        self.cursor = 0
        self.reference_frames: list[str] = []
        self._pending_edit_video: dict[tuple[int, int], Path] = {}
        self.delivered: list[dict[str, Any]] = []
        self.applied: list[dict[str, Any]] = []

    # -- helpers ---------------------------------------------------------
    def _shot_frame_indices(self, shot_index: int, count: int = 3) -> list[int]:
        c0, c1 = self.chunk_ranges[shot_index]
        start, end = c0 * FRAMES_PER_CHUNK, (c1 + 1) * FRAMES_PER_CHUNK
        step = max(1, (end - start - 1) // max(1, count - 1))
        return [min(end - 1, start + i * step) for i in range(count)]

    def _shot_candidate(self, shot_index: int, attempt: int, video: Path, *, edited: bool) -> GeneratedShot:
        stem = f"shot{shot_index:02d}_attempt{attempt}" + ("_edited" if edited else "")
        frames = extract_frames(
            video, self._shot_frame_indices(shot_index), self.out_dir / "attempts", stem
        )
        refs = tuple(
            TemporalInput(
                kind="current_shot_frame",
                identifier=str(path),
                sha256=file_sha256(path),
                available_after_shot=shot_index,
            )
            for path in frames
        ) + tuple(
            TemporalInput(
                kind="reference_frame",
                identifier=str(path),
                sha256=file_sha256(path),
                available_after_shot=0,
            )
            for path in self.reference_frames
        )
        return GeneratedShot(
            shot_index=shot_index,
            attempt=attempt,
            output={"video": str(video), "frames": [str(p) for p in frames]},
            visual_candidate={"frames": [str(p) for p in frames]},
            model_native_candidate={
                "source_video": str(video),
                "source_video_sha256": file_sha256(video),
                "edited": edited,
            },
            telemetry={"edited": edited},
            input_refs=refs,
        )

    # -- ShotBackend protocol -------------------------------------------
    def telemetry(self, shot_index):
        return {
            "active_video": str(self.active_video),
            "committed_shots": self.cursor,
        }

    def apply(self, shot_index, program, *, retry):
        results = []
        for action in program.actions:
            record = {
                "shot_index": shot_index,
                "retry": retry,
                "tool": action.tool,
                "note": "deterministic-replay demo: pre-generation tools are recorded, not executed",
            }
            self.applied.append(record)
            results.append({"metrics": {"noop": True}, "attention_mass_check": "not_applicable"})
        return results

    def snapshot(self, shot_index):
        return {
            "scene": self.scene,
            "shot_index": int(shot_index),
            "cursor": int(self.cursor),
            "active_video": str(self.active_video),
        }

    def restore(self, snapshot):
        self.cursor = int(snapshot["cursor"])
        self.active_video = Path(snapshot["active_video"])

    def generate(self, shot_index, attempt):
        return self._shot_candidate(shot_index, attempt, self.active_video, edited=False)

    def observe(self, generated):
        return ShotObservation(
            shot_index=generated.shot_index,
            attempt=generated.attempt,
            frames=generated.output["frames"],
            crops=[],
            telemetry={
                "reference_frames": list(self.reference_frames),
                "edited": bool(generated.telemetry.get("edited", False)),
            },
            inputs=generated.input_refs,
            input_fingerprint="",
        )

    def edit(self, shot_index, attempt, action, verdict):
        params = dict(action.params)
        c0, c1 = self.chunk_ranges[shot_index]
        masks = clipseg_chunk_masks(
            self.active_video,
            range(c0, c1 + 1),
            params["mask_phrase"],
            device=self.clipseg_device,
        )
        mask_path = save_mask_payload(
            masks,
            self.out_dir / "edit" / f"shot{shot_index:02d}_masks.pt",
            phrase=params["mask_phrase"],
            video=self.active_video,
        )
        edit_cfg = {
            "chunk_start": c0,
            "chunk_end": c1,
            "entry_sigma": float(params["entry_sigma"]),
            "mask_path": str(mask_path),
            "prompt": params["prompt"],
            "noise_seed": self.noise_seed,
        }
        prefix = f"edit_shot{shot_index:02d}"
        render_dir = self.out_dir / "renders"
        config_path = build_render_config(
            self.base_config,
            scene=self.scene,
            data_dir=self.out_dir / "data",
            output_folder=render_dir,
            prefix=prefix,
            edit=edit_cfg,
            target=self.out_dir / "configs" / f"{prefix}.yaml",
        )
        render(config_path, gpu=self.render_gpu, log_path=self.out_dir / "logs" / f"{prefix}.log")
        edited_video = find_render(render_dir, prefix, self.scene)
        self._pending_edit_video[(int(shot_index), int(attempt))] = edited_video
        generated = self._shot_candidate(shot_index, attempt, edited_video, edited=True)
        result = {
            "metrics": {
                "chunk_start": c0,
                "chunk_end": c1,
                "entry_sigma": float(params["entry_sigma"]),
                "mask_coverage": mask_coverage(masks),
            },
            "repair_prompt": params["prompt"],
            "mask_path": str(mask_path),
            "config": str(config_path),
            "edited_video": str(edited_video),
            "edited_video_sha256": file_sha256(edited_video),
            "attention_mass_check": "not_applicable_latent_rewrite",
        }
        return generated, result

    def commit(self, generated, *, quarantined):
        key = (int(generated.shot_index), int(generated.attempt))
        edited_video = self._pending_edit_video.get(key)
        if edited_video is not None and not quarantined:
            # Butterfly effect: the repaired render becomes the delivery stream,
            # so every later shot is generated from (and judged against) the
            # repaired history.
            self.active_video = edited_video
        self.cursor = max(self.cursor, generated.shot_index + 1)
        if generated.shot_index == 0 and not quarantined:
            self.reference_frames = list(generated.output["frames"])
        record = {
            "shot": generated.shot_index,
            "attempt": generated.attempt,
            "quarantined": quarantined,
            "video": str(self.active_video),
            "edited": bool(generated.telemetry.get("edited", False)),
        }
        self.delivered.append(record)
        return record


def make_two_up(baseline: Path, final: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        "[0:v]scale=640:352,drawtext=text='baseline':x=18:y=18:fontsize=26:"
        "fontcolor=white:box=1:boxcolor=black@0.6[l];"
        "[1:v]scale=640:352,drawtext=text='agent edit_shot delivery (engineering demo)':"
        "x=18:y=18:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[r];"
        "[l][r]hstack=inputs=2[v]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(baseline), "-i", str(final),
            "-filter_complex", filter_graph, "-map", "[v]", "-an", "-c:v", "libx264",
            "-crf", "20", "-preset", "veryfast", str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def resolve_entity_phrase(args, scene: str) -> str:
    if getattr(args, "entities_json", None):
        table = json.loads((ROOT / args.entities_json).read_text())
        if scene in table:
            return table[scene]
    if scene in ENTITIES:
        return ENTITIES[scene]
    raise KeyError(
        f"no entity phrase for scene {scene}; pass --entities_json with an entry for it"
    )


def run_scene(args, scene: str, base_url: str, output_root: Path) -> dict[str, Any]:
    scene_src = ROOT / args.data_root / scene
    out_dir = output_root / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scene-only dataset directory (MultiTextConcatDataset renders one sample per
    # scene subfolder).
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)
    link = data_dir / scene
    if not link.exists():
        link.symlink_to(scene_src)

    render_dir = out_dir / "renders"
    baseline_config = build_render_config(
        ROOT / args.base_config,
        scene=scene,
        data_dir=data_dir,
        output_folder=render_dir,
        prefix="baseline",
        edit=None,
        target=out_dir / "configs" / "baseline.yaml",
    )
    try:
        baseline_video = find_render(render_dir, "baseline", scene)
        print(f"[{scene}] reusing baseline render {baseline_video}")
    except RuntimeError:
        print(f"[{scene}] rendering baseline on GPU {args.render_gpu}")
        render(baseline_config, gpu=args.render_gpu, log_path=out_dir / "logs" / "baseline.log")
        baseline_video = find_render(render_dir, "baseline", scene)

    backend = EditRenderBackend(
        scene=scene,
        scene_dir=scene_src,
        out_dir=out_dir,
        base_config=baseline_config,
        baseline_video=baseline_video,
        render_gpu=args.render_gpu,
        clipseg_device=args.clipseg_device,
    )
    client = FingerprintedVLMClient(
        base_url=base_url,
        model=args.model,
        cache_dir=output_root / "vlm_cache",
        max_tokens=512,
    )
    runtime_config = AgentRuntimeConfig.from_config(
        {
            "enabled": True,
            "lock_path": str(output_root / f"{scene}.lock"),
            "decision_cache_dir": str(output_root / "vlm_cache"),
            "edit_entry_sigma": args.entry_sigma,
        }
    )
    verifier = ComparativeDriftVerifier(
        client,
        entity_phrase=resolve_entity_phrase(args, scene),
        edit_entry_sigma=runtime_config.edit_entry_sigma,
        prompt_template=runtime_config.edit_prompt_template,
    )
    graph = NormativeGraph.compile_scene(scene_src)
    ledger = out_dir / "ledger.jsonl"
    if ledger.exists():
        ledger.unlink()
    with CausalShotSession(
        runtime_config,
        graph,
        backend,
        AbstainingController(),
        verifier,
        ledger_path=ledger,
        acquire_lock=True,
    ) as session:
        results = session.run()
        counts = session.memory.partition_counts()
        edit_counts = dict(session.edit_counts)

    two_up = out_dir / "two_up.mp4"
    make_two_up(baseline_video, backend.active_video, two_up)
    edited_commits = [row for row in backend.delivered if row["edited"] and not row["quarantined"]]
    return {
        "baseline_video": str(baseline_video),
        "final_delivery_video": str(backend.active_video),
        "ledger": str(ledger),
        "two_up": str(two_up),
        "shots": len(results),
        "quarantined_shots": [r.shot_index for r in results if r.quarantined],
        "edited_shots": dict(edit_counts),
        "edited_commits": edited_commits,
        "memory_partition_counts": counts,
        "delivered": backend.delivered,
        "passed": len(results) == graph.num_shots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=3, help="GPU for the Qwen3-VL judge server")
    parser.add_argument("--render_gpu", type=int, default=2, help="GPU for pipeline renders")
    parser.add_argument("--clipseg_device", default="cuda:0", help="Device for CLIPSeg masks")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--sglang_python", default="/home/chendong/miniconda3/envs/sglang/bin/python")
    parser.add_argument("--mem_fraction_static", type=float, default=0.45)
    parser.add_argument("--startup_timeout", type=int, default=300)
    parser.add_argument("--base_config", default="configs/inference_kv_rag_long_multishot.yaml")
    parser.add_argument("--data_root", default="example/benchmark_hard_multishot")
    parser.add_argument("--output_root", default="videos/agent_edit_demo")
    parser.add_argument("--entry_sigma", type=float, default=1.0)
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    parser.add_argument(
        "--entities_json",
        default=None,
        help="JSON file mapping scene name -> CLIPSeg entity phrase (overrides built-ins)",
    )
    args = parser.parse_args()

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "quality_claim": False,
        "loop": "drift_judge -> edit_shot(masked renoise) -> re-review -> commit/quarantine",
        "entry_sigma": args.entry_sigma,
        "scenes": {},
    }
    proc = None
    try:
        proc, base_url = start_vlm_server(args, output_root)
        for scene in args.scenes:
            summary["scenes"][scene] = run_scene(args, scene, base_url, output_root)
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
    print(
        json.dumps(
            {
                "passed": summary.get("passed"),
                "scenes": {
                    key: {
                        "passed": value.get("passed"),
                        "edited_shots": value.get("edited_shots"),
                        "quarantined_shots": value.get("quarantined_shots"),
                    }
                    for key, value in summary.get("scenes", {}).items()
                },
            },
            indent=2,
        )
    )
    if not summary.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
