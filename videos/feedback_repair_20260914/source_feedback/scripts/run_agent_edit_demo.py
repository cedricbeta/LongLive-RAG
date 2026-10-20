#!/usr/bin/env python3
"""Closed-loop agent edit_shot demo: judge drift -> masked-renoise repair -> re-review.

For each demo scene (bearded_watchmaker_workshop, sandy_beach_driftwood) this driver:

1. renders the scene once as the deterministic baseline (seeded replay);
2. runs a CausalShotSession over its shots with the selected comparative or entity verifier;
3. on a repairable rejection, executes the verifier-directed edit_shot action and a
   deterministic full-scene replay carrying every accepted parent edit plus the new plan;
4. re-reviews the repaired shot with the same verifier. With the entity judge, a local pass is
   tentative: affected future appearances are sliced from both the repaired render and its
   stable parent, then compared constraint by constraint without entering trusted memory.
   The child is promoted when it causes no clear regression; an existing future defect is
   left for that future shot instead of erasing the successful current repair;
5. writes the full ledger, a summary.json, and a 2-up baseline-vs-final video.

Every child branch is checked against its parent before it is judged: the latent
chunks before the new edit must be bit-identical (decoded frames within codec noise
when a latent dump is missing).  A divergent prefix means the deterministic-replay
assumption behind the branch model is broken, and the run fails closed.

Engineering/offline-buffered demo only: it shows the transaction and cumulative replay
working; because the full suffix already exists, it does not claim real-time streaming or
quality improvements.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.closed_judge import find_free_port, probe_qwen3_optimizer, qwen3_sglang_launch_command
from utils.agent_edit import clipseg_chunk_masks, mask_coverage, save_mask_payload
from utils.agent_entity_judge import (
    EntityDriftVerifier,
    Sam2Tracker,
    VLMComparer,
    VLMLocalizer,
    load_entity_specs,
)
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
    edit_plans: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> Path:
    if edit is not None and edit_plans:
        raise ValueError("pass either edit or edit_plans, not both")

    plans = [deepcopy(plan) for plan in (edit_plans or ())]
    for index, plan in enumerate(plans):
        try:
            start = int(plan["chunk_start"])
            end = int(plan["chunk_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"edit plan {index} requires integer chunk_start/chunk_end") from exc
        if start < 0 or end < start:
            raise ValueError(f"edit plan {index} has invalid chunk range [{start}, {end}]")
        plan["chunk_start"], plan["chunk_end"] = start, end
    plans.sort(key=lambda plan: (plan["chunk_start"], plan["chunk_end"]))
    for previous, current in zip(plans, plans[1:]):
        if current["chunk_start"] <= previous["chunk_end"]:
            raise ValueError(
                "edit plans overlap: "
                f"[{previous['chunk_start']}, {previous['chunk_end']}] and "
                f"[{current['chunk_start']}, {current['chunk_end']}]"
            )

    data = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    data["output_folder"] = str(output_folder)
    data.setdefault("data", {})["data_path"] = str(data_dir)
    inference = data.setdefault("inference", {})
    inference["output_folder"] = str(output_folder)
    inference["filename_prefix"] = prefix
    inference["filename_from_sample_name"] = True
    data["filename_from_sample_name"] = True
    data.setdefault("logging", {})["seed"] = int(data.get("logging", {}).get("seed", 0))
    # Remove both grouped and legacy-flat forms.  normalize_config preserves
    # flat compatibility, so leaving a stale top-level key could otherwise
    # make the generated cumulative-plan config appear doubly configured.
    data.pop("edit_renoise", None)
    data.pop("edit_renoise_plans", None)
    inference.pop("edit_renoise", None)
    inference.pop("edit_renoise_plans", None)
    if plans:
        inference["edit_renoise_plans"] = [
            {"enabled": True, **plan} for plan in plans
        ]
    elif edit is not None:
        inference["edit_renoise"] = {"enabled": True, **edit}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def render(
    config_path: Path, *, gpu: int, log_path: Path, latent_dir: Path | None = None
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if latent_dir is not None:
        # The pipeline dumps the full output latent [B, T, C, H, W] here; the
        # branch prefix check compares it chunk by chunk with the parent.
        env["LLV2_DUMP_LATENT_DIR"] = str(latent_dir)
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


def latent_dir_for(output_folder: Path, prefix: str) -> Path:
    return output_folder / f"{prefix}_latents"


def find_latents(output_folder: Path, prefix: str) -> Path | None:
    """Return the render's latent evidence: per-chunk digests, else the raw dump."""
    folder = latent_dir_for(output_folder, prefix)
    for name in ("chunk_digests.json", "latent_0000.pt"):
        path = folder / name
        if path.exists():
            return path
    return None


def latent_chunk_digests(latent: torch.Tensor, num_chunks: int) -> list[str]:
    """SHA-256 of every chunk's raw latent bytes, layout [B, T, C, H, W]."""
    if latent.shape[1] % int(num_chunks):
        raise RuntimeError(
            f"latent has {latent.shape[1]} frames, not divisible by {num_chunks} chunks"
        )
    per_chunk = latent.shape[1] // int(num_chunks)
    digests = []
    for chunk in range(int(num_chunks)):
        block = latent[:, chunk * per_chunk : (chunk + 1) * per_chunk].contiguous()
        if block.dtype == torch.bfloat16:
            block = block.view(torch.int16)  # numpy has no bfloat16; bytes are what matter
        digests.append(hashlib.sha256(block.numpy().tobytes()).hexdigest())
    return digests


def condense_latent_dump(output_folder: Path, prefix: str, num_chunks: int) -> Path | None:
    """Replace a ~130 MB latent dump by its per-chunk digests (exact equality is all we need)."""
    folder = latent_dir_for(output_folder, prefix)
    raw = folder / "latent_0000.pt"
    digests_path = folder / "chunk_digests.json"
    if digests_path.exists():
        return digests_path
    if not raw.exists():
        return None
    latent = torch.load(raw, map_location="cpu", weights_only=True)
    payload = {
        "shape": list(latent.shape),
        "dtype": str(latent.dtype),
        "num_chunks": int(num_chunks),
        "chunk_digests": latent_chunk_digests(latent, num_chunks),
    }
    digests_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    raw.unlink()
    return digests_path


def prefix_identity_report(
    parent_video: Path,
    child_video: Path,
    chunk_end: int,
    *,
    num_chunks: int,
    parent_latents: Path | None = None,
    child_latents: Path | None = None,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
    frame_tolerance: float = 8.0,
) -> dict[str, Any]:
    """Compare chunks ``[0, chunk_end)`` of a child replay with its parent.

    With latent evidence on both sides (per-chunk digests, or raw dumps) the
    prefix must be bit-identical.  Without it the decoded first and middle frame of every prefix chunk are compared
    and must stay within ``frame_tolerance`` mean absolute error: x264's
    rate-control lookahead makes frames shortly before a content change encode
    differently.  Measured on bit-identical latents: 2.5 to 2.6 max per-frame
    MAE in the 2026-09-03 smoke renders, and 4.02 to 4.57 in three scenes of
    the 2026-09-09 33-scene run (ns060/ns110/ns200, later proven identical by
    latent replay), against 16 to 40 for genuinely different chunks.  The
    tolerance therefore sits at 8.0; prefer latent digests whenever a parent
    has them, this fallback cannot certify bit-exactness.
    """
    chunk_end = int(chunk_end)
    if chunk_end <= 0:
        return {"basis": "none", "chunks": 0, "identical": True}
    if (
        parent_latents is not None
        and child_latents is not None
        and Path(parent_latents).suffix == ".json"
        and Path(child_latents).suffix == ".json"
    ):
        parent = json.loads(Path(parent_latents).read_text(encoding="utf-8"))
        child = json.loads(Path(child_latents).read_text(encoding="utf-8"))
        if parent["shape"] != child["shape"] or parent["num_chunks"] != child["num_chunks"]:
            raise RuntimeError(f"latent layouts differ: {parent['shape']} vs {child['shape']}")
        if int(parent["num_chunks"]) != int(num_chunks):
            raise RuntimeError(f"digests cover {parent['num_chunks']} chunks, expected {num_chunks}")
        mismatched = [
            chunk
            for chunk in range(chunk_end)
            if parent["chunk_digests"][chunk] != child["chunk_digests"][chunk]
        ]
        return {
            "basis": "latent_digests",
            "chunks": chunk_end,
            "mismatched_chunks": mismatched,
            "identical": not mismatched,
            "parent_latents": str(parent_latents),
            "child_latents": str(child_latents),
        }
    if parent_latents is not None and child_latents is not None:
        parent = torch.load(parent_latents, map_location="cpu", weights_only=True)
        child = torch.load(child_latents, map_location="cpu", weights_only=True)
        if tuple(parent.shape) != tuple(child.shape):
            raise RuntimeError(
                f"latent shapes differ: {tuple(parent.shape)} vs {tuple(child.shape)}"
            )
        if parent.shape[1] % int(num_chunks):
            raise RuntimeError(
                f"latent has {parent.shape[1]} frames, not divisible by {num_chunks} chunks"
            )
        per_chunk = parent.shape[1] // int(num_chunks)
        frames = chunk_end * per_chunk
        diff = (parent[:, :frames].float() - child[:, :frames].float()).abs()
        max_diff = float(diff.max()) if diff.numel() else 0.0
        return {
            "basis": "latent",
            "chunks": chunk_end,
            "latent_frames": frames,
            "max_abs_diff": max_diff,
            "identical": max_diff == 0.0,
            "parent_latents": str(parent_latents),
            "child_latents": str(child_latents),
        }

    indices: list[int] = []
    for chunk in range(chunk_end):
        indices.extend([chunk * frames_per_chunk, chunk * frames_per_chunk + frames_per_chunk // 2])

    def read(video: Path) -> list[Any]:
        capture = cv2.VideoCapture(str(video))
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not read frame {index} from {video}")
            frames.append(frame.astype("float32"))
        capture.release()
        return frames

    parent_frames = read(parent_video)
    child_frames = read(child_video)
    maes = [float(abs(a - b).mean()) for a, b in zip(parent_frames, child_frames)]
    max_mae = max(maes) if maes else 0.0
    return {
        "basis": "decoded_frames",
        "chunks": chunk_end,
        "frames": len(maes),
        "max_frame_mae": max_mae,
        "tolerance": float(frame_tolerance),
        "identical": max_mae <= float(frame_tolerance),
    }


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


@dataclass(frozen=True)
class RenderBranchState:
    """One deterministic replay branch and its repair-plan lineage."""

    branch_id: str
    parent: str | None
    video: Path
    edit_plans: tuple[dict[str, Any], ...]
    safe: bool
    tentative: bool
    verified: bool
    # Highest shot accepted on this branch.  Not "every shot up to here is
    # verified": shots delivered under quarantine are listed separately.
    last_accepted_shot: int = -1
    delivered_through_shot: int = -1
    quarantined_shots: tuple[int, ...] = ()
    reference_frames: tuple[str, ...] = ()
    quarantined: bool = False
    rollback_reason: str = ""
    latents: str | None = None
    prefix_check: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "parent": self.parent,
            "video": str(self.video),
            "edit_plans": deepcopy(list(self.edit_plans)),
            "safe": self.safe,
            "tentative": self.tentative,
            "verified": self.verified,
            "last_accepted_shot": self.last_accepted_shot,
            "delivered_through_shot": self.delivered_through_shot,
            "quarantined_shots": list(self.quarantined_shots),
            "reference_frames": list(self.reference_frames),
            "quarantined": self.quarantined,
            "rollback_reason": self.rollback_reason,
            "latents": self.latents,
            "prefix_check": deepcopy(self.prefix_check),
        }


class EditRenderBackend:
    """Shot backend over deterministic full-scene renders.

    generate() slices the current safe render. edit() creates a tentative child by
    replaying the parent's accepted plans plus the current repair. commit() promotes
    that child; restore/rejection quarantines it without changing the safe parent.
    """

    PREFIX_RENDER_ATTEMPTS = 2

    def __init__(
        self,
        *,
        scene: str,
        scene_dir: Path,
        out_dir: Path,
        base_config: Path,
        baseline_video: Path,
        baseline_latents: Path | None = None,
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
        self.cursor = 0
        self._branch_sequence = 0
        root = RenderBranchState(
            branch_id=self._next_branch_id(),
            parent=None,
            video=Path(baseline_video),
            edit_plans=(),
            safe=True,
            tentative=False,
            verified=True,
            latents=str(baseline_latents) if baseline_latents else None,
        )
        self.branches: dict[str, RenderBranchState] = {root.branch_id: root}
        self.active_branch_id = root.branch_id
        self.safe_branch_id = root.branch_id
        self._pending_edit_branch: dict[tuple[int, int], str] = {}
        self.delivered: list[dict[str, Any]] = []
        self.applied: list[dict[str, Any]] = []

    # -- helpers ---------------------------------------------------------
    def _next_branch_id(self) -> str:
        branch_id = f"branch-{self._branch_sequence:04d}"
        self._branch_sequence += 1
        return branch_id

    @property
    def active_branch(self) -> RenderBranchState:
        return self.branches[self.active_branch_id]

    @property
    def active_video(self) -> Path:
        return self.active_branch.video

    @property
    def reference_frames(self) -> list[str]:
        return list(self.active_branch.reference_frames)

    def branch_manifest(self) -> list[dict[str, Any]]:
        return [branch.to_dict() for branch in self.branches.values()]

    def _is_descendant(self, branch_id: str, ancestor_id: str) -> bool:
        current = self.branches.get(branch_id)
        seen: set[str] = set()
        while current is not None and current.parent is not None:
            if current.parent == ancestor_id:
                return True
            if current.branch_id in seen:
                raise RuntimeError("render branch graph contains a cycle")
            seen.add(current.branch_id)
            current = self.branches.get(current.parent)
        return False

    def _quarantine_descendants(self, branch_id: str, *, reason: str) -> list[str]:
        invalidated = []
        for candidate_id, candidate in list(self.branches.items()):
            if not self._is_descendant(candidate_id, branch_id):
                continue
            if candidate.quarantined and not candidate.safe and not candidate.tentative:
                continue
            self.branches[candidate_id] = replace(
                candidate,
                safe=False,
                tentative=False,
                quarantined=True,
                rollback_reason=reason,
            )
            invalidated.append(candidate_id)
        return invalidated

    def _drop_pending_branches(self, branch_ids: list[str]) -> None:
        invalidated = set(branch_ids)
        if not invalidated:
            return
        self._pending_edit_branch = {
            key: value
            for key, value in self._pending_edit_branch.items()
            if value not in invalidated
        }

    def rollback_to(self, branch_id: str, *, reason: str = "explicit_rollback") -> dict[str, Any]:
        """Renderer half of a multi-shot rollback.

        Not wired into the session loop: the runtime currently returns to the
        nearest stable parent only within the current shot (lookahead failure
        or exhausted edit budget).  A session-level rollback across shots would
        also have to restore verifier references and trusted memory; until that
        policy exists this helper is exercised by unit tests only.
        """
        if branch_id not in self.branches:
            raise KeyError(f"unknown render branch {branch_id}")
        target = self.branches[branch_id]
        if not target.safe or target.tentative or not target.verified or target.quarantined:
            raise ValueError(f"render branch {branch_id} is not a verified safe rollback target")
        invalidated = self._quarantine_descendants(branch_id, reason=reason)
        self._drop_pending_branches(invalidated)
        self.active_branch_id = branch_id
        self.safe_branch_id = branch_id
        self.cursor = max(0, target.delivered_through_shot + 1)
        return {
            "active_branch_id": branch_id,
            "safe_branch_id": branch_id,
            "invalidated_branches": invalidated,
            "reason": reason,
        }

    def _shot_frame_indices(self, shot_index: int, count: int = 3) -> list[int]:
        c0, c1 = self.chunk_ranges[shot_index]
        start, end = c0 * FRAMES_PER_CHUNK, (c1 + 1) * FRAMES_PER_CHUNK
        step = max(1, (end - start - 1) // max(1, count - 1))
        return [min(end - 1, start + i * step) for i in range(count)]

    def _shot_candidate(
        self,
        shot_index: int,
        attempt: int,
        video: Path,
        *,
        edited: bool,
        branch_id: str,
    ) -> GeneratedShot:
        branch = self.branches[branch_id]
        # Include the immutable branch id so rolling back and replaying the
        # same shot/attempt never overwrites evidence from an older branch.
        stem = (
            f"{branch_id}_shot{shot_index:02d}_attempt{attempt}"
            + ("_edited" if edited else "")
        )
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
            for path in branch.reference_frames
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
                "branch_id": branch_id,
                "parent": branch.parent,
                "edit_plan_count": len(branch.edit_plans),
            },
            telemetry={
                "edited": edited,
                "branch_id": branch_id,
                "parent": branch.parent,
                "safe": branch.safe,
                "tentative": branch.tentative,
                "verified": branch.verified,
            },
            input_refs=refs,
        )

    # -- ShotBackend protocol -------------------------------------------
    def telemetry(self, shot_index):
        return {
            "active_video": str(self.active_video),
            "active_branch_id": self.active_branch_id,
            "safe_branch_id": self.safe_branch_id,
            "edit_plan_count": len(self.active_branch.edit_plans),
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
            "active_branch_id": self.active_branch_id,
            "safe_branch_id": self.safe_branch_id,
            "branch": self.active_branch.to_dict(),
        }

    def restore(self, snapshot):
        branch_id = str(snapshot.get("active_branch_id", ""))
        if not branch_id:
            matches = [
                candidate.branch_id
                for candidate in self.branches.values()
                if candidate.video == Path(snapshot["active_video"])
            ]
            if len(matches) != 1:
                raise RuntimeError("snapshot active video does not identify one render branch")
            branch_id = matches[0]
        if branch_id not in self.branches:
            raise RuntimeError(f"snapshot references unknown render branch {branch_id}")
        expected = snapshot.get("branch")
        if expected is not None and self.branches[branch_id].to_dict() != expected:
            raise RuntimeError(f"safe render branch {branch_id} changed after snapshot")
        invalidated = self._quarantine_descendants(
            branch_id, reason="snapshot_restore"
        )
        self._drop_pending_branches(invalidated)
        self.cursor = int(snapshot["cursor"])
        self.active_branch_id = branch_id
        self.safe_branch_id = str(snapshot.get("safe_branch_id", branch_id))

    def generate(self, shot_index, attempt):
        return self._shot_candidate(
            shot_index,
            attempt,
            self.active_video,
            edited=False,
            branch_id=self.active_branch_id,
        )

    def preview(self, generated, shot_index):
        """Slice a future shot from the already-rendered tentative branch, without mutation."""
        branch_id = str(generated.model_native_candidate.get("branch_id", ""))
        if branch_id not in self.branches:
            raise RuntimeError(f"preview references unknown render branch {branch_id}")
        branch = self.branches[branch_id]
        return self._shot_candidate(
            int(shot_index),
            int(generated.attempt),
            branch.video,
            edited=bool(generated.telemetry.get("edited", False)),
            branch_id=branch_id,
        )

    def observe(self, generated):
        c0, c1 = self.chunk_ranges[generated.shot_index]
        branch_id = str(generated.model_native_candidate.get("branch_id", self.active_branch_id))
        branch = self.branches[branch_id]
        return ShotObservation(
            shot_index=generated.shot_index,
            attempt=generated.attempt,
            frames=generated.output["frames"],
            crops=[],
            telemetry={
                "reference_frames": list(branch.reference_frames),
                "edited": bool(generated.telemetry.get("edited", False)),
                # entity judge inputs: the candidate render and the shot's chunk range
                "shot_video": str(generated.output["video"]),
                "chunk_start": c0,
                "chunk_end": c1,
            },
            inputs=generated.input_refs,
            input_fingerprint="",
        )

    def edit(self, shot_index, attempt, action, verdict):
        params = dict(action.params)
        c0, c1 = self.chunk_ranges[shot_index]
        parent = self.active_branch
        for existing in parent.edit_plans:
            p0, p1 = int(existing["chunk_start"]), int(existing["chunk_end"])
            if c0 <= p1 and p0 <= c1:
                raise ValueError(
                    f"new edit range [{c0}, {c1}] overlaps accepted parent plan "
                    f"[{p0}, {p1}] on {parent.branch_id}"
                )
        if params.get("mask_path"):
            # Precomputed repaint region (entity judge: SAM2 tracks of drifted
            # entities and unassigned subjects), already in payload format.
            mask_path = Path(params["mask_path"])
            masks = torch.load(mask_path, map_location="cpu", weights_only=False)["masks"]
        else:
            masks = clipseg_chunk_masks(
                self.active_video,
                range(c0, c1 + 1),
                params["mask_phrase"],
                device=self.clipseg_device,
            )
            mask_path = save_mask_payload(
                masks,
                self.out_dir / "edit" / f"shot{shot_index:02d}_attempt{attempt}_masks.pt",
                phrase=params["mask_phrase"],
                video=self.active_video,
            )
        edit_cfg = {
            "chunk_start": c0,
            "chunk_end": c1,
            "entry_sigma": float(params["entry_sigma"]),
            "mask_path": str(mask_path),
            "prompt": params["prompt"],
            # every repair round draws fresh noise inside the mask
            "noise_seed": self.noise_seed + int(attempt),
        }
        candidate_plans = tuple(deepcopy(plan) for plan in parent.edit_plans) + (
            deepcopy(edit_cfg),
        )
        prefix = f"edit_shot{shot_index:02d}_attempt{attempt}"
        render_dir = self.out_dir / "renders"
        config_path = build_render_config(
            self.base_config,
            scene=self.scene,
            data_dir=self.out_dir / "data",
            output_folder=render_dir,
            prefix=prefix,
            edit=None,
            edit_plans=candidate_plans,
            target=self.out_dir / "configs" / f"{prefix}.yaml",
        )
        # A prefix divergence is usually a one-off numeric event (ns200 on
        # 2026-09-11 diverged from chunk 0 between two otherwise identical
        # renders 15 minutes apart), so the render is retried once from scratch
        # before the branch model gives up; a second divergence is fatal.
        prefix_checks = []
        for render_try in range(1, self.PREFIX_RENDER_ATTEMPTS + 1):
            if render_try > 1:
                for stale in render_dir.glob(f"{prefix}-rank*"):
                    stale.unlink()
                shutil.rmtree(latent_dir_for(render_dir, prefix), ignore_errors=True)
            render(
                config_path,
                gpu=self.render_gpu,
                log_path=self.out_dir / "logs" / f"{prefix}{'' if render_try == 1 else f'_retry{render_try}'}.log",
                latent_dir=latent_dir_for(render_dir, prefix),
            )
            edited_video = find_render(render_dir, prefix, self.scene)
            condense_latent_dump(render_dir, prefix, self.chunk_ranges[-1][1] + 1)
            edited_latents = find_latents(render_dir, prefix)
            prefix_check = prefix_identity_report(
                parent.video,
                edited_video,
                c0,
                num_chunks=self.chunk_ranges[-1][1] + 1,
                parent_latents=Path(parent.latents) if parent.latents else None,
                child_latents=edited_latents,
            )
            prefix_check["render_attempt"] = render_try
            prefix_checks.append(prefix_check)
            if prefix_check["identical"]:
                break
            print(f"[{self.scene}] {prefix}: replay prefix diverged on render attempt {render_try}: {prefix_check}", flush=True)
        if not prefix_check["identical"]:
            raise RuntimeError(
                f"{prefix}: replay prefix diverged from {parent.branch_id} before chunk {c0} on "
                f"{len(prefix_checks)} render attempts: {prefix_checks}. The branch model assumes deterministic "
                "replay (same seed, same inductor cache); this render cannot inherit the parent's verification."
            )
        prefix_check["earlier_attempts"] = prefix_checks[:-1]
        branch_id = self._next_branch_id()
        candidate = RenderBranchState(
            branch_id=branch_id,
            parent=parent.branch_id,
            video=edited_video,
            edit_plans=candidate_plans,
            safe=False,
            tentative=True,
            verified=False,
            last_accepted_shot=parent.last_accepted_shot,
            delivered_through_shot=parent.delivered_through_shot,
            quarantined_shots=parent.quarantined_shots,
            reference_frames=parent.reference_frames,
            latents=str(edited_latents) if edited_latents else None,
            prefix_check=prefix_check,
        )
        self.branches[branch_id] = candidate
        self._pending_edit_branch[(int(shot_index), int(attempt))] = branch_id
        generated = self._shot_candidate(
            shot_index,
            attempt,
            edited_video,
            edited=True,
            branch_id=branch_id,
        )
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
            "branch": candidate.to_dict(),
            "parent_branch_id": parent.branch_id,
            "edit_plan_count": len(candidate_plans),
            "prefix_check": prefix_check,
            "attention_mass_check": "not_applicable_latent_rewrite",
        }
        return generated, result

    def commit(self, generated, *, quarantined):
        key = (int(generated.shot_index), int(generated.attempt))
        candidate_id = self._pending_edit_branch.pop(key, None)
        generated_branch_id = str(
            (generated.model_native_candidate or {}).get("branch_id", self.active_branch_id)
        )
        branch_id = candidate_id or generated_branch_id
        if branch_id not in self.branches:
            raise RuntimeError(f"generated candidate references unknown branch {branch_id}")

        if candidate_id is not None and not quarantined:
            candidate = self.branches[candidate_id]
            if candidate.parent != self.active_branch_id:
                raise RuntimeError(
                    f"tentative branch {candidate_id} parent {candidate.parent} is not active"
                )
            refs = candidate.reference_frames
            if generated.shot_index == 0:
                refs = tuple(str(path) for path in generated.output["frames"])
            promoted = replace(
                candidate,
                safe=True,
                tentative=False,
                verified=True,
                last_accepted_shot=max(
                    candidate.last_accepted_shot, int(generated.shot_index)
                ),
                delivered_through_shot=max(
                    candidate.delivered_through_shot, int(generated.shot_index)
                ),
                reference_frames=refs,
            )
            self.branches[candidate_id] = promoted
            self.active_branch_id = candidate_id
            self.safe_branch_id = candidate_id
        elif candidate_id is not None:
            candidate = self.branches[candidate_id]
            self.branches[candidate_id] = replace(
                candidate,
                safe=False,
                tentative=False,
                quarantined=True,
                rollback_reason="candidate_rejected",
            )
            current = self.active_branch
            self.branches[current.branch_id] = replace(
                current,
                delivered_through_shot=max(
                    current.delivered_through_shot, int(generated.shot_index)
                ),
                quarantined_shots=tuple(
                    sorted(set(current.quarantined_shots) | {int(generated.shot_index)})
                ),
            )
        elif not quarantined:
            current = self.active_branch
            refs = current.reference_frames
            if generated.shot_index == 0:
                refs = tuple(str(path) for path in generated.output["frames"])
            self.branches[current.branch_id] = replace(
                current,
                verified=True,
                last_accepted_shot=max(
                    current.last_accepted_shot, int(generated.shot_index)
                ),
                delivered_through_shot=max(
                    current.delivered_through_shot, int(generated.shot_index)
                ),
                reference_frames=refs,
            )
        else:
            # A quarantined current shot may still be delivered from the active
            # stable branch.  Advance delivery progress without calling it a
            # verified identity observation.
            current = self.active_branch
            self.branches[current.branch_id] = replace(
                current,
                delivered_through_shot=max(
                    current.delivered_through_shot, int(generated.shot_index)
                ),
                quarantined_shots=tuple(
                    sorted(set(current.quarantined_shots) | {int(generated.shot_index)})
                ),
            )
        self.cursor = max(self.cursor, generated.shot_index + 1)
        delivered_branch = self.active_branch
        candidate_branch = self.branches[branch_id]
        record = {
            "shot": generated.shot_index,
            "attempt": generated.attempt,
            "quarantined": quarantined,
            "video": str(self.active_video),
            "edited": (
                delivered_branch.branch_id == branch_id
                and bool(generated.telemetry.get("edited", False))
            ),
            "candidate_edited": bool(generated.telemetry.get("edited", False)),
            "branch_id": delivered_branch.branch_id,
            "candidate_branch_id": candidate_branch.branch_id,
            "parent": candidate_branch.parent,
            "safe": delivered_branch.safe,
            "tentative": delivered_branch.tentative,
            "verified": delivered_branch.verified,
            "edit_plan_count": len(delivered_branch.edit_plans),
            "edit_plan_chunks": [
                [plan["chunk_start"], plan["chunk_end"]]
                for plan in delivered_branch.edit_plans
            ],
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


_SAM2_TRACKER = None


def _sam2_tracker(args):
    """One SAM2 predictor per process (weights load once, shared across scenes)."""
    global _SAM2_TRACKER
    if _SAM2_TRACKER is None:
        _SAM2_TRACKER = Sam2Tracker(device=args.clipseg_device)
    return _SAM2_TRACKER


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
        render(
            baseline_config,
            gpu=args.render_gpu,
            log_path=out_dir / "logs" / "baseline.log",
            latent_dir=latent_dir_for(render_dir, "baseline"),
        )
        baseline_video = find_render(render_dir, "baseline", scene)
    condense_latent_dump(render_dir, "baseline", shot_chunk_ranges(scene_src)[-1][1] + 1)

    backend = EditRenderBackend(
        scene=scene,
        scene_dir=scene_src,
        out_dir=out_dir,
        base_config=baseline_config,
        baseline_video=baseline_video,
        baseline_latents=find_latents(render_dir, "baseline"),
        render_gpu=args.render_gpu,
        clipseg_device=args.clipseg_device,
    )
    client = FingerprintedVLMClient(
        base_url=base_url,
        model=args.model,
        cache_dir=getattr(args, "judge_cache_dir", None) or output_root / "vlm_cache",
        max_tokens=1024 if args.judge == "entity" else 512,
        seed=getattr(args, "judge_seed", 0),
    )
    runtime_config = AgentRuntimeConfig.from_config(
        {
            "enabled": True,
            "lock_path": str(output_root / f"{scene}.lock"),
            "decision_cache_dir": str(client.cache_dir),
            "edit_entry_sigma": args.entry_sigma,
            "max_edits_per_shot": int(args.max_edits_per_shot),
            "commit_lookahead_shots": (
                int(args.commit_lookahead_shots) if args.judge == "entity" else 0
            ),
        }
    )
    entity_specs_source = None
    if args.judge == "entity":
        # Entity-level judge: enumerate / assign roles / compare, SAM2 tracks as the
        # first generic constraint adapter and masked-renoise repair tool.
        specs_path = (ROOT / args.entity_specs).resolve()
        specs = load_entity_specs(specs_path, scene)
        entity_specs_source = {
            "path": str(specs_path),
            "sha256": file_sha256(specs_path),
        }
        if args.comparer == "color":
            from utils.agent_appearance_color import ColorCheckedAppearanceComparer, set_part_device
            set_part_device(args.clipseg_device)
            comparer = ColorCheckedAppearanceComparer(client)
        elif args.comparer == "evidence":
            from utils.agent_appearance_evidence import EvidenceAppearanceComparer
            comparer = EvidenceAppearanceComparer(client)
        else:
            comparer = VLMComparer(client)
        if args.role_localizer == "indexed":
            from utils.agent_role_assignment import ContinuityRoleLocalizer
            localizer = ContinuityRoleLocalizer(client)
        else:
            localizer = VLMLocalizer(client)
        rewriter = None
        if args.absent_action == "regenerate" and getattr(args, "repair_policy", "legacy") == "legacy":
            from utils.agent_prompt_rewrite import AbsentEntityPromptRewriter

            rewriter = AbsentEntityPromptRewriter(client)
        verifier = EntityDriftVerifier(
            specs,
            localizer=localizer,
            comparer=comparer,
            tracker=_sam2_tracker(args),
            work_dir=out_dir / "entity_judge",
            edit_entry_sigma=runtime_config.edit_entry_sigma,
            reference_registration_min_confidence=(
                args.reference_registration_min_confidence
            ),
            place_absent=args.place_absent,
            placement_prompt=args.placement_prompt,
            uncertain_policy=args.uncertain_policy,
            attribute_masks=args.attribute_masks,
            part_device=args.clipseg_device,
            judge_chunks=args.judge_chunks,
            absent_action=args.absent_action,
            prompt_rewriter=rewriter,
            defer_repair_planning=getattr(args, "repair_policy", "legacy") == "feedback",
        )
    else:
        verifier = ComparativeDriftVerifier(
            client,
            entity_phrase=resolve_entity_phrase(args, scene),
            edit_entry_sigma=runtime_config.edit_entry_sigma,
            prompt_template=runtime_config.edit_prompt_template,
        )
    graph = NormativeGraph.compile_scene(
        scene_src, allow_missing_global=bool(getattr(args, "allow_missing_global", False))
    )
    ledger = out_dir / "ledger.jsonl"
    if ledger.exists():
        ledger.unlink()
    session_class = CausalShotSession
    session_extra = {}
    if getattr(args, "repair_policy", "legacy") == "feedback":
        from utils.agent_repair_planner import OfflineFeedbackShotSession, VLMRepairPlanner
        session_class = OfflineFeedbackShotSession
        session_extra = {
            "planner": VLMRepairPlanner(client),
            "max_repair_seconds": getattr(args, "max_repair_seconds", 1800),
        }
    with session_class(
        runtime_config,
        graph,
        backend,
        AbstainingController(),
        verifier,
        ledger_path=ledger,
        acquire_lock=True,
        **session_extra,
    ) as session:
        results = session.run()
        counts = session.memory.partition_counts()
        edit_counts = dict(session.edit_counts)
        ledger_records = session.ledger.records()

    two_up = out_dir / "two_up.mp4"
    make_two_up(baseline_video, backend.active_video, two_up)
    edited_commits = [row for row in backend.delivered if row["edited"] and not row["quarantined"]]
    # Which kind of edit produced each accepted repair: the edit rounds of a
    # shot are numbered like their attempts, and the ledger action carries the
    # verifier's reason (masked repaint, placement, or prompt regeneration).
    edit_reasons = {
        (int(row.get("shot_index")), int(row.get("edit_count"))): str(row.get("reason", ""))
        for row in ledger_records
        if row.get("event") == "action" and row.get("edit")
    }
    for row in edited_commits:
        row["repair_kind"] = edit_reasons.get((int(row["shot"]), int(row["attempt"])), "")
    run_completed = len(results) == graph.num_shots
    unresolved_shots = [result.shot_index for result in results if not result.accepted]
    all_evaluated_constraints_valid = run_completed and not unresolved_shots
    rolled_back_repairs = [
        {
            "shot_index": row.get("shot_index"),
            "attempt": row.get("attempt"),
            "decision": row.get("lookahead", {}).get("decision"),
            "decision_as_of_shot": row.get("lookahead", {}).get(
                "decision_as_of_shot"
            ),
            "error": row.get("lookahead", {}).get("error", ""),
        }
        for row in ledger_records
        if row.get("event") == "commit_guard" and not row.get("passed", False)
    ]
    branch_manifest = backend.branch_manifest()
    record = {
        "baseline_video": str(baseline_video),
        "repair_policy": getattr(args, "repair_policy", "legacy"),
        "repair_planning_mode": "offline_with_explicit_observation_horizon" if session_extra else "legacy",
        "repair_plan_calls": sum(len(r.get("planning", {}).get("calls", [])) for r in ledger_records if r.get("event") == "repair_plan"),
        "repair_planning_seconds": sum(r.get("wall_seconds", 0) for r in ledger_records if r.get("event") == "repair_plan"),
        "final_delivery_video": str(backend.active_video),
        "ledger": str(ledger),
        "two_up": str(two_up),
        "shots": len(results),
        "quarantined_shots": [r.shot_index for r in results if r.quarantined],
        "edit_attempts": dict(edit_counts),
        "edited_shots": dict(edit_counts),
        "edited_shots_semantics": "deprecated alias of edit_attempts",
        "edited_commits": edited_commits,
        "accepted_repairs": edited_commits,
        "rolled_back_repairs": rolled_back_repairs,
        "unresolved_shots": unresolved_shots,
        "memory_partition_counts": counts,
        "delivered": backend.delivered,
        "active_branch_id": backend.active_branch_id,
        "safe_branch_id": backend.safe_branch_id,
        "final_edit_plans": deepcopy(list(backend.active_branch.edit_plans)),
        "branches": branch_manifest,
        "commit_policy": (
            "offline_dependency_paired_no_clear_regression"
            if runtime_config.commit_lookahead_shots
            else "immediate"
        ),
        "commit_lookahead_shots": runtime_config.commit_lookahead_shots,
        "commit_lookahead_semantics": "maximum affected future appearances",
        "constraint_adapter": (
            "entity_appearance_v1" if args.judge == "entity" else None
        ),
        "evaluated_constraint_kinds": (
            [
                "presence",
                "identity",
                "attribute",
                "conditional_unassigned_count",
                "spatial_relation_left_right_beside_non_gating",
            ]
            if args.judge == "entity"
            else []
        ),
        "unevaluated_constraint_kinds": (
            ["action", "relation_depth_behind_front", "state"] if args.judge == "entity" else []
        ),
        "place_absent": args.place_absent if args.judge == "entity" else None,
        "placement_prompt": args.placement_prompt if args.judge == "entity" else None,
        "comparer": args.comparer if args.judge == "entity" else None,
        "role_localizer": args.role_localizer if args.judge == "entity" else None,
        "uncertain_policy": args.uncertain_policy if args.judge == "entity" else None,
        "attribute_masks": args.attribute_masks if args.judge == "entity" else None,
        "judge_chunks": args.judge_chunks if args.judge == "entity" else None,
        "absent_action": args.absent_action if args.judge == "entity" else None,
        "entity_specs_source": entity_specs_source,
        "judge_sampling": client.sampling_config,
        "judge_cache_dir": str(client.cache_dir),
        "entity_judge_version": "enumerate_joint_roles_crop_votes_v2" if args.judge == "entity" else None,
        "excluded_ambiguous_entity_shots": (
            [{"shot": shot, "entity": spec.name}
             for shot in range(graph.num_shots) for spec in specs if spec.presence_in(shot) == "ambiguous"]
            if args.judge == "entity" else []
        ),
        "presence_exclusion_scope": "entity x shot; other required entities in that shot remain evaluated",
        "reference_registration_policy": "three_separate_positive_crop_votes" if args.judge == "entity" else None,
        "reference_registration_min_confidence": (
            args.reference_registration_min_confidence if args.judge == "entity" else None
        ),
        "run_completed": run_completed,
        "all_evaluated_constraints_valid": all_evaluated_constraints_valid,
        "validation_scope": "active verifier only; not a claim that every script predicate was checked",
    }
    # Per-scene summary so a crashed multi-scene run keeps its finished scenes
    # (narrastream_prepare_eval.py reads g*/<scene>/summary.json).
    (out_dir / "summary.json").write_text(
        json.dumps({"schema_version": 1, "scenes": {scene: record}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


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
    parser.add_argument("--judge_seed", type=int, default=0, help="fixed VLM sampling seed (included in cache fingerprints)")
    parser.add_argument("--judge_cache_dir", help="shared content-fingerprinted VLM cache; default: output_root/vlm_cache")
    parser.add_argument(
        "--commit_lookahead_shots",
        type=int,
        default=2,
        help="maximum affected future appearances paired before an entity repair branch is promoted",
    )
    parser.add_argument(
        "--reference_registration_min_confidence",
        type=float,
        default=0.75,
        help="legacy comparer threshold; the VLM entity comparer additionally requires three positive crop votes",
    )
    parser.add_argument(
        "--comparer",
        choices=("vote", "evidence", "color"),
        default="vote",
        help=(
            "entity judge appearance comparer: vote = three yes/no crop votes (v1); evidence = blind description "
            "+ visible-evidence lists (09-05 candidate); color = evidence plus a pixel colour veto on CLIPSeg "
            "hair/clothing parts against the script and the reference"
        ),
    )
    parser.add_argument(
        "--role_localizer",
        choices=("baseline", "indexed"),
        default="baseline",
        help="role assignment: baseline status/index protocol, or the single-index protocol with same-shot continuity",
    )
    parser.add_argument(
        "--uncertain_policy",
        choices=("reject", "accept_unknown"),
        default="reject",
        help="what an unresolved role assignment does to an otherwise clean shot: reject (v1) or accept with UNKNOWN constraints",
    )
    parser.add_argument(
        "--judge_chunks",
        choices=("all", "sparse"),
        default="all",
        help="enumerate/assign on every chunk-first frame (all) or on the first, middle and last only (sparse)",
    )
    parser.add_argument(
        "--attribute_masks",
        action="store_true",
        help="repaint only person ∩ CLIPSeg parts the colour comparer vetoed (hair / clothing) instead of the whole person",
    )
    parser.add_argument(
        "--place_absent",
        choices=("never", "relation_or_history", "always"),
        default="relation_or_history",
        help=(
            "entity judge: when a scripted entity is absent with no impostor to repaint, synthesise a "
            "placement box from a caption relation (\"to Nora's left\") or the entity's last accepted "
            "track and repaint there; never = old nothing-to-mask behaviour"
        ),
    )
    parser.add_argument(
        "--placement_prompt",
        choices=("caption_and_spec", "spec_only"),
        default="caption_and_spec",
        help="prompt for a placement repaint: shot caption + entity description, or the description alone",
    )
    parser.add_argument(
        "--absent_action",
        choices=("none", "regenerate"),
        default="none",
        help=(
            "entity judge: what an absent scripted entity triggers. regenerate = resample the whole shot "
            "(full-frame mask, sigma 1.0) with a caption the judge's language model rewrote to put the "
            "absent entity first; takes precedence over placement / impostor repaints. none = leave it "
            "to --place_absent"
        ),
    )
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    parser.add_argument(
        "--entities_json",
        default=None,
        help="JSON file mapping scene name -> CLIPSeg entity phrase (overrides built-ins)",
    )
    parser.add_argument(
        "--allow_missing_global",
        action="store_true",
        help="permit scenes without global.json (self-contained suites like NarraStream)",
    )
    parser.add_argument(
        "--judge",
        choices=("comparative", "entity"),
        default="comparative",
        help="comparative = single free-form drift call (pilot); entity = locate/enumerate/compare + SAM2 tracks",
    )
    parser.add_argument(
        "--entity_specs",
        default="example/narrastream_bench/entity_specs_pilot3.json",
        help="JSON {scene: [{name, description, shots}]} for --judge entity",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        help="use an already running OpenAI-compatible judge server (e.g. http://127.0.0.1:53456/v1) instead of launching one",
    )
    parser.add_argument(
        "--max_edits_per_shot",
        type=int,
        default=1,
        help="masked-renoise repair rounds per shot (each round: fresh seed, shrinking mask, positive feedback)",
    )
    parser.add_argument("--repair_policy", choices=("legacy", "feedback"), default="legacy",
                        help="feedback = bounded offline visual-feedback planner; legacy keeps existing action selection")
    parser.add_argument("--max_repair_seconds", type=float, default=1800,
                        help="feedback mode: stop starting new repair attempts after this per-shot wall budget")
    args = parser.parse_args()
    if args.repair_policy == "feedback" and args.judge != "entity":
        parser.error("feedback repair currently requires --judge entity")
    if not args.scenes:
        parser.error("--scenes must contain at least one scene")

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if args.judge == "entity":
        loop_description = (
            "entity constraint adapter -> cumulative repair -> current-shot re-review -> "
            "offline paired no-clear-regression guard -> promote/nearest-stable rollback"
        )
    else:
        loop_description = (
            "comparative drift judge -> single repair -> current-shot re-review -> immediate commit"
        )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "quality_claim": False,
        "loop": loop_description,
        "entry_sigma": args.entry_sigma,
        "commit_lookahead_shots": (
            args.commit_lookahead_shots if args.judge == "entity" else 0
        ),
        "scenes": {},
    }
    proc = None
    try:
        if args.base_url:
            # Shared external judge server (one per machine; several scene
            # drivers may run in parallel against it).
            base_url = str(args.base_url).rstrip("/")
            with urllib.request.urlopen(f"{base_url}/models", timeout=10):
                pass
        else:
            proc, base_url = start_vlm_server(args, output_root)
        for scene in args.scenes:
            summary["scenes"][scene] = run_scene(args, scene, base_url, output_root)
        summary["run_completed"] = all(
            record["run_completed"] for record in summary["scenes"].values()
        )
        summary["all_evaluated_constraints_valid"] = all(
            record["all_evaluated_constraints_valid"]
            for record in summary["scenes"].values()
        )
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
                "run_completed": summary.get("run_completed"),
                "all_evaluated_constraints_valid": summary.get(
                    "all_evaluated_constraints_valid"
                ),
                "scenes": {
                    key: {
                        "run_completed": value.get("run_completed"),
                        "all_evaluated_constraints_valid": value.get(
                            "all_evaluated_constraints_valid"
                        ),
                        "unresolved_shots": value.get("unresolved_shots"),
                        "edited_shots": value.get("edited_shots"),
                        "quarantined_shots": value.get("quarantined_shots"),
                    }
                    for key, value in summary.get("scenes", {}).items()
                },
            },
            indent=2,
        )
    )
    if not summary.get("run_completed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
