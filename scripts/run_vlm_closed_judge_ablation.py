#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Run the OAuth-judged Qwen3-in-the-loop four-arm consistency ablation."""

from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.closed_judge import (  # noqa: E402
    ClaudeOAuthMessagesClient,
    CodexExecJudgeClient,
    JUDGE_SCHEMA,
    JudgeFixture,
    OpenAIResponsesError,
    file_sha256,
    find_free_port,
    fixture_video_hash,
    prompt_hash,
    probe_qwen3_optimizer,
    qwen3_sglang_launch_command,
    run_oauth_judge_validation,
    select_free_gpu,
    utc_timestamp,
    validate_verdict_shape,
    write_json,
)
from evaluation.multiview_prompts import build_spec_resolver, load_shot_specs  # noqa: E402
from evaluation.video_consistency import (  # noqa: E402
    chunk_durations_to_boundaries,
    compare_cross_perspective_dirs,
    evaluate_cross_perspective_gate,
    read_video,
    shot_ranges,
)
try:
    from scripts.run_kv_rag_ablation import (  # noqa: E402
        DEFAULT_KV_RAG,
        _apply_noise_floor_gate,
        _base_kv_rag_block,
        _build_long_scorers,
        _copy_result_with_records,
        _finalist_kv_rag,
        _frame_contract_blocked_reason,
        _frame_contract_stats,
        _load_attention_diagnostics,
        _num_blocks_from_cfg,
        _preflight_config,
        _set_nested,
    )
except ModuleNotFoundError as exc:  # Miniconda ships a site-packages `scripts` package.
    if exc.name not in {"scripts", "scripts.run_kv_rag_ablation"}:
        raise
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_kv_rag_ablation import (  # type: ignore  # noqa: E402
        DEFAULT_KV_RAG,
        _apply_noise_floor_gate,
        _base_kv_rag_block,
        _build_long_scorers,
        _copy_result_with_records,
        _finalist_kv_rag,
        _frame_contract_blocked_reason,
        _frame_contract_stats,
        _load_attention_diagnostics,
        _num_blocks_from_cfg,
        _preflight_config,
        _set_nested,
    )


ARMS = ("baseline", "prompt_only", "kv_only", "both")


_TQDM_ELAPSED_RE = re.compile(r"\[(?P<elapsed>(?:\d+:)?\d{1,2}:\d{2}),")


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_elapsed_seconds(text: str) -> float | None:
    matches = list(_TQDM_ELAPSED_RE.finditer(text))
    if not matches:
        return None
    parts = [int(p) for p in matches[-1].group("elapsed").split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    hours, minutes, seconds = parts
    return float(hours * 3600 + minutes * 60 + seconds)


def recovered_log_seconds(log_path: str | Path | None) -> float | None:
    if not log_path:
        return None
    path = Path(log_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    try:
        return parse_elapsed_seconds(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None


def mtime_span_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    mtimes = [p.stat().st_mtime for p in path.rglob("*") if p.is_file()]
    if len(mtimes) < 2:
        return None
    return float(max(mtimes) - min(mtimes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    stamp = dt.datetime.now().strftime("%Y%m%d")
    parser.add_argument("--output_root", default=f"docs/vlm_closed_judge_ablation_{stamp}")
    parser.add_argument("--video_root", default=f"videos/vlm_closed_judge_ablation_{stamp}")
    parser.add_argument("--config_path", default="configs/inference_kv_rag_long_multishot.yaml")
    parser.add_argument("--admission_json", default="docs/multiview_gate_results/frame_strategy_A_drift_audit.json")
    parser.add_argument("--numeric_reference_json", default="docs/multiview_gate_results/round14_scene_memory_mechanism_sweep.json")
    parser.add_argument("--judge_validation_json", default="docs/closed_judge_loop_oauth_rerun/judge_validation.json")
    parser.add_argument("--judge_validation_output", default=None)
    parser.add_argument("--codex_model", default="gpt-5.5")
    parser.add_argument("--claude_model", default="claude-opus-4-8")
    parser.add_argument("--judge_timeout", type=int, default=300)
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--sglang_python", default=None)
    parser.add_argument("--qwen_startup_timeout", type=int, default=300)
    parser.add_argument("--qwen_mem_fraction_static", type=float, default=0.45)
    parser.add_argument("--optimizer_gpu", type=int, default=None)
    parser.add_argument("--generator_gpu", type=int, default=None)
    parser.add_argument("--clip_model", default="ViT-B-32")
    parser.add_argument("--clip_pretrained", default="openai")
    parser.add_argument("--clip_device", default="cuda:0")
    parser.add_argument("--motion_backend", default="raft", choices=("raft", "farneback"))
    parser.add_argument("--motion_tolerance", type=float, default=0.2)
    parser.add_argument("--diversity_tolerance", type=float, default=0.0)
    parser.add_argument("--adherence_tolerance", type=float, default=0.02)
    parser.add_argument("--prompt_similarity_floor", type=float, default=0.12)
    parser.add_argument("--single_seed_consistency_delta_threshold", type=float, default=0.02)
    parser.add_argument("--baseline_seed", type=int, default=0)
    parser.add_argument("--kv_anchor_cap", type=int, default=2)
    parser.add_argument("--optimizer_frames_per_shot", type=int, default=4)
    parser.add_argument("--judge_frames_per_shot", type=int, default=1)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse existing optimizer decisions/prompts/videos when present.")
    parser.add_argument("--max_scenes", type=int, default=0, help="Explicit logged cap for debugging; 0 means all admitted scenes.")
    parser.add_argument("--min_admitted_main_scenes", type=int, default=8,
                        help="Fail closed unless the admission JSON provides at least this many main scenes.")
    parser.add_argument("--primary_logit_bias_lambda", type=float, default=1.0,
                        help="Persistent logit-bias lambda for primary kv_only/both arms.")
    parser.add_argument("--kv_selection_mode", default="per_boundary",
                        choices=("per_boundary", "legacy_single_call"),
                        help="per_boundary: one retrieval-style VLM call per shot boundary with "
                             "candidate thumbnails (per-scene plans). legacy_single_call: whole "
                             "plan in the diagnosis call (Round 17 behaviour).")
    parser.add_argument("--kv_candidate_frames_per_shot", type=int, default=8,
                        help="Denser per-shot sampling used only as candidate thumbnails for "
                             "per-boundary KV selection.")
    parser.add_argument("--kv_max_candidates_per_boundary", type=int, default=24,
                        help="Uniformly subsample a boundary's eligible candidates down to this "
                             "many thumbnails per selection call (late boundaries accumulate "
                             "candidates from every prior shot).")
    parser.add_argument("--prompt_review", default="on", choices=("on", "off"),
                        help="Independent VLM review of proposed prompt additions against the "
                             "authored invariants (drop/rewrite contradictions).")
    parser.add_argument("--optimizer_video_dir", default=None,
                        help="Directory of renders the OPTIMIZER watches (multi-pass feedback: "
                             "point at a prior round's arm dir). Judges/metrics still compare "
                             "against the baseline. Default: the baseline dir.")
    parser.add_argument("--optimizer_video_prefix", default=None,
                        help="Filename prefix inside --optimizer_video_dir (e.g. 'both').")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_token(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:120] or "sample"


def _gpu_process_users() -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    try:
        pmon = subprocess.run(["nvidia-smi", "pmon", "-c", "1"], text=True, capture_output=True, timeout=20)
        rows: list[tuple[int, str]] = []
        pids: set[str] = set()
        for line in pmon.stdout.splitlines():
            if line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                rows.append((int(parts[0]), parts[1]))
                pids.add(parts[1])
        pid_to_user: dict[str, str] = {}
        if pids:
            ps = subprocess.run(
                ["ps", "-o", "user=,pid=", "-p", ",".join(sorted(pids))],
                text=True,
                capture_output=True,
                timeout=20,
            )
            for line in ps.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    pid_to_user[parts[1]] = parts[0]
        for gpu, pid in rows:
            out.setdefault(gpu, []).append(pid_to_user.get(pid, "unknown"))
    except Exception:
        pass
    return out


def gpu_state() -> list[dict[str, Any]]:
    q = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    users = _gpu_process_users()
    rows = []
    for line in q.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        rows.append({
            "index": int(parts[0]),
            "name": parts[1],
            "memory_used_mib": int(parts[2]),
            "memory_free_mib": int(parts[3]),
            "memory_total_mib": int(parts[4]),
            "utilization_gpu_percent": int(parts[5]),
            "utilization_memory_percent": int(parts[6]),
            "process_users": sorted(set(users.get(int(parts[0]), []))),
        })
    return rows


def choose_gpus(args: argparse.Namespace) -> dict[str, Any]:
    selection = select_free_gpu(max_utilization=15, min_free_memory_mib=45000)
    candidates = selection.get("candidates") or []
    if args.optimizer_gpu is not None:
        optimizer = args.optimizer_gpu
    else:
        optimizer = int(selection.get("selected_gpu")) if selection.get("selected_gpu") is not None else None
    if optimizer is None:
        raise RuntimeError(selection.get("blocked_reason") or "no idle GPU for optimizer")
    if args.generator_gpu is not None:
        generator = args.generator_gpu
    else:
        rest = [int(c["index"]) for c in candidates if int(c["index"]) != int(optimizer)]
        generator = rest[0] if rest else int(optimizer)
    return {
        "optimizer_gpu": int(optimizer),
        "generator_gpu": int(generator),
        "selection": selection,
        "pre_run_state": gpu_state(),
        "policy": "use idle headroom: util <= 15% and enough free memory; idle shared allocations are allowed",
    }


def ensure_gpu_not_maxed(gpu: int, *, max_util: int = 85) -> dict[str, Any]:
    rows = gpu_state()
    row = next((r for r in rows if int(r["index"]) == int(gpu)), None)
    if row is None:
        raise RuntimeError(f"GPU {gpu} not visible")
    if int(row["utilization_gpu_percent"]) >= max_util:
        idle = [
            r for r in rows
            if int(r["utilization_gpu_percent"]) < 15 and int(r["memory_free_mib"]) >= 45000
        ]
        if not idle:
            raise RuntimeError(f"GPU {gpu} became active and no idle replacement is available: {rows}")
        replacement = sorted(idle, key=lambda r: (int(r["utilization_gpu_percent"]), -int(r["memory_free_mib"])))[0]
        return {"gpu": int(replacement["index"]), "switched": True, "previous": row, "replacement": replacement}
    return {"gpu": int(gpu), "switched": False, "state": row}


def ensure_judge_valid(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    path = Path(args.judge_validation_json)
    if path.exists():
        cached = load_json(path)
        cached_fields = set(cached.get("schema_required_fields") or [])
        current_fields = set(JUDGE_SCHEMA["required"])
        if (
            cached.get("closed_judge_validated")
            and cached.get("selected_provider") == "codex"
            and current_fields.issubset(cached_fields)
        ):
            return {
                "reused": True,
                "path": str(path),
                "selected_provider": "codex",
                "validation": cached,
            }
    validation_dir = Path(args.judge_validation_output or output_root / "judge_revalidation")
    validation = run_oauth_judge_validation(
        output_dir=validation_dir,
        provider="codex",
        codex_model=args.codex_model,
        primary_model=args.claude_model,
    )
    validation["closed_judge_validated"] = bool(validation.get("passed"))
    if not validation.get("passed"):
        raise RuntimeError("closed judge revalidation failed: " + str(validation.get("blocked_reason")))
    return {
        "reused": False,
        "path": str(validation_dir / "judge_validation.json"),
        "selected_provider": validation.get("selected_provider"),
        "validation": validation,
    }


def start_qwen3_server(args: argparse.Namespace, gpu: int, output_root: Path) -> tuple[subprocess.Popen, dict[str, Any]]:
    probe = probe_qwen3_optimizer(sglang_python=args.sglang_python, model=args.qwen_model)
    if probe.get("blocked_reason"):
        raise RuntimeError("Qwen3 optimizer failed validation: " + str(probe["blocked_reason"]))
    port = find_free_port()
    launch = qwen3_sglang_launch_command(
        port=port,
        gpu=int(gpu),
        sglang_python=args.sglang_python,
        model=args.qwen_model,
        mem_fraction_static=args.qwen_mem_fraction_static,
    )
    out_dir = output_root / "qwen3_optimizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sglang_server.log"
    env = os.environ.copy()
    env.update(launch["env"])
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            launch["cmd"],
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    deadline = time.time() + int(args.qwen_startup_timeout)
    models_url = f"{launch['base_url']}/models"
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:] if log_path.exists() else ""
            raise RuntimeError(f"SGLang server exited before readiness with code {proc.returncode}: {tail}")
        try:
            with urllib.request.urlopen(models_url, timeout=5) as response:
                models = json.loads(response.read().decode("utf-8"))
            meta = {
                "probe": probe,
                "launch": {
                    "cmd": launch["cmd"],
                    "env": launch["env"],
                    "base_url": launch["base_url"],
                    "mem_fraction_static": launch["mem_fraction_static"],
                },
                "models_response": models,
                "log_path": str(log_path),
                "gpu": int(gpu),
                "startup_seconds": time.time() - started,
                "server_started": True,
            }
            return proc, meta
        except Exception as exc:
            last_error = str(exc)
            time.sleep(5)
    raise RuntimeError(f"SGLang Qwen3 server did not become ready within {args.qwen_startup_timeout}s: {last_error}")


def _image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _post_chat(base_url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def qwen_call_fingerprint(
    *,
    model: str,
    prompt: str,
    image_paths: list[Path],
    json_schema: dict[str, Any] | None,
) -> str:
    """Content hash of everything that determines a Qwen call's output.

    Image ORDER matters (labels like Q0/C3 refer to positions), so hash the
    ordered list of per-image digests, not a set."""
    import hashlib

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [file_sha256(p) for p in image_paths],
        "schema": json_schema,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def qwen_chat_json(
    *,
    base_url: str,
    model: str,
    prompt: str,
    image_paths: list[Path],
    transcript_path: Path,
    timeout: int = 240,
    json_schema: dict[str, Any] | None = None,
    max_parse_retries: int = 1,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_data_uri(path)}})
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 1800,
    }
    schema_active = bool(json_schema)
    if schema_active:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "decision", "schema": json_schema, "strict": True},
        }
    started = time.time()
    attempts: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    raw: dict[str, Any] = {}
    for attempt in range(int(max_parse_retries) + 1):
        try:
            raw = _post_chat(base_url, payload, timeout)
        except urllib.error.HTTPError as exc:
            # Older SGLang builds reject response_format for VLMs; drop the
            # constraint once and rely on prompt + parse-retry instead.
            if schema_active and exc.code in (400, 422):
                attempts.append({"attempt": attempt, "error": f"HTTP {exc.code}: response_format rejected"})
                payload.pop("response_format", None)
                schema_active = False
                raw = _post_chat(base_url, payload, timeout)
            else:
                raise
        message = raw["choices"][0]["message"]["content"]
        try:
            parsed = _extract_json(message)
            break
        except Exception as exc:
            attempts.append({"attempt": attempt, "error": f"json parse failed: {exc}", "raw_text": str(message)[:2000]})
            if attempt >= int(max_parse_retries):
                raise
            payload["messages"].append({"role": "assistant", "content": str(message)})
            payload["messages"].append({
                "role": "user",
                "content": (
                    f"Your previous reply was not valid JSON ({exc}). "
                    "Reply again with ONLY the JSON object, no prose, no code fences."
                ),
            })
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = copy.deepcopy(payload)
    safe_payload["messages"][0]["content"] = [
        item if item.get("type") == "text" else {"type": "image_url", "image_url": {"url": "<base64 image omitted>"}}
        for item in safe_payload["messages"][0]["content"]
    ]
    write_json(transcript_path, {
        "request": safe_payload,
        "image_paths": [str(p) for p in image_paths],
        "image_hashes": {str(p): file_sha256(p) for p in image_paths},
        "cache_fingerprint": qwen_call_fingerprint(
            model=model, prompt=prompt, image_paths=image_paths, json_schema=json_schema
        ),
        "response": raw,
        "parsed": parsed,
        "schema_enforced": schema_active,
        "parse_attempts": attempts,
        "wall_clock_seconds": time.time() - started,
    })
    return {"parsed": parsed, "raw": raw, "transcript": str(transcript_path), "wall_clock_seconds": time.time() - started}


def scene_video(directory: Path, prefix: str, scene: str, seed: int) -> Path:
    token = _safe_token(scene)
    matches = sorted(directory.glob(f"{prefix}-rank*-{token}-seed{seed}_regular.mp4"))
    usable = [p for p in matches if p.exists() and p.stat().st_size > 0]
    if not usable:
        raise FileNotFoundError(f"missing video for scene={scene}, prefix={prefix}, dir={directory}")
    return usable[0]


def link_scene_video_subset(
    *,
    source_dir: Path,
    prefix: str,
    scenes: list[str],
    seed: int,
    subset_dir: Path,
) -> dict[str, Any]:
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)
    links: list[dict[str, str]] = []
    for scene in scenes:
        src = scene_video(source_dir, prefix, scene, seed).resolve()
        dst = subset_dir / src.name
        try:
            dst.symlink_to(src)
            link_type = "symlink"
        except OSError:
            shutil.copy2(src, dst)
            link_type = "copy"
        links.append({
            "scene": scene,
            "source": str(src),
            "path": str(dst),
            "link_type": link_type,
        })
    return {"dir": str(subset_dir), "videos": links}


def arm_complete(directory: Path, prefix: str, scenes: list[str], seed: int) -> bool:
    try:
        for scene in scenes:
            scene_video(directory, prefix, scene, seed)
        return True
    except FileNotFoundError:
        return False


def missing_arm_scenes(directory: Path, prefix: str, scenes: list[str], seed: int) -> list[str]:
    missing: list[str] = []
    for scene in scenes:
        try:
            scene_video(directory, prefix, scene, seed)
        except FileNotFoundError:
            missing.append(scene)
    return missing


def copy_prompt_subset(source_prompt_dir: Path, scenes: list[str], dest_prompt_dir: Path) -> Path:
    if dest_prompt_dir.exists():
        shutil.rmtree(dest_prompt_dir)
    dest_prompt_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        shutil.copytree(source_prompt_dir / scene, dest_prompt_dir / scene)
    return dest_prompt_dir


def save_rgb(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def sample_scene_frames(
    *,
    video_path: Path,
    spec: dict[str, Any],
    out_dir: Path,
    frames_per_shot: int,
) -> dict[str, Any]:
    frames = read_video(video_path)
    boundaries = chunk_durations_to_boundaries(frames.shape[0], list(spec["chunk_durations"]))
    ranges = shot_ranges(frames.shape[0], boundaries=boundaries)
    shots: list[list[Path]] = []
    shot_records: list[dict[str, Any]] = []
    for shot_idx, (start, end) in enumerate(ranges):
        count = max(1, min(frames_per_shot, end - start))
        idxs = np.linspace(start, end - 1, num=count).round().astype(int).tolist()
        paths: list[Path] = []
        for local_idx, frame_idx in enumerate(idxs):
            path = out_dir / f"shot{shot_idx:02d}_frame{local_idx:02d}_idx{frame_idx:04d}.png"
            save_rgb(path, frames[frame_idx])
            paths.append(path)
        shots.append(paths)
        shot_records.append({
            "shot_index": shot_idx,
            "start_frame": int(start),
            "end_frame": int(end),
            "sampled_frame_indices": [int(i) for i in idxs],
            "paths": [str(p) for p in paths],
        })
    return {
        "shots": shots,
        "records": shot_records,
        "num_frames": int(frames.shape[0]),
        "video_path": str(video_path),
    }


def candidate_frames_by_boundary(shot_records: list[dict[str, Any]], *, local_attn_size: int) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for rec in shot_records[1:]:
        boundary = int(rec["shot_index"])
        boundary_start = int(rec["start_frame"])
        eligible: list[int] = []
        for prev in shot_records[:boundary]:
            for frame in prev.get("sampled_frame_indices", []):
                if int(frame) < boundary_start - int(local_attn_size):
                    eligible.append(int(frame))
        out[boundary] = sorted(set(eligible))
    return out


OPTIMIZER_DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "which_cut_broke": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cut_index": {"type": ["integer", "null"]},
                "how": {"type": "string"},
            },
            "required": ["cut_index", "how"],
        },
        "global_invariant_additions": {"type": "array", "items": {"type": "string"}},
        "per_shot_additions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "shot_index": {"type": "integer"},
                    "text": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["shot_index", "text", "rationale"],
            },
        },
        "kv_anchor_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "boundary_shot_index": {"type": "integer"},
                    "decoded_frame_indices": {"type": "array", "items": {"type": "integer"}},
                    "rationale": {"type": "string"},
                },
                "required": ["boundary_shot_index", "decoded_frame_indices", "rationale"],
            },
        },
        "stop_rule": {"type": "string"},
    },
    "required": ["which_cut_broke", "global_invariant_additions", "per_shot_additions", "kv_anchor_plan", "stop_rule"],
}

BOUNDARY_SELECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_frame_indices": {"type": "array", "items": {"type": "integer"}},
        "skip": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["selected_frame_indices", "skip", "rationale"],
}

PROMPT_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "index": {"type": "integer"},
                    "target": {"type": "string", "enum": ["global", "per_shot"]},
                    "verdict": {"type": "string", "enum": ["keep", "drop", "rewrite"]},
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "target", "verdict", "text", "reason"],
            },
        },
    },
    "required": ["verdicts"],
}


def build_optimizer_prompt(
    scene: str,
    spec: dict[str, Any],
    shot_records: list[dict[str, Any]],
    candidates: dict[int, list[int]],
    *,
    anchor_cap: int,
    include_kv_plan: bool = True,
) -> str:
    lines = [
        "You are the LOCAL Qwen3-VL optimizer, not the held-out judge.",
        "Diagnose which shot cut breaks cross-shot consistency and propose constrained fixes.",
        "Return strict JSON only with this shape:",
        "{",
        '  "which_cut_broke": {"cut_index": integer_or_null, "how": "short reason"},',
        '  "global_invariant_additions": ["additive invariant text", ...],',
        '  "per_shot_additions": [{"shot_index": 0, "text": "additive invariant text", "rationale": "why"}, ...],',
    ]
    if include_kv_plan:
        lines.append('  "kv_anchor_plan": [{"boundary_shot_index": 1, "decoded_frame_indices": [0], "rationale": "why"}, ...],')
    else:
        lines.append('  "kv_anchor_plan": [],')
    lines += [
        '  "stop_rule": "one bounded refinement round"',
        "}",
        "Rules: ADD subject/scene invariants only. Do not delete or weaken per-shot camera/action.",
        "Invariant additions must RESTATE or SHARPEN the authored global invariant below.",
        "If the rendered video contradicts that invariant (wrong clothing/color/object), that is a",
        "defect to FIX: re-assert the authored invariant. NEVER codify the rendered appearance when",
        "it conflicts with the authored invariant.",
        "Do not collapse shots into near-identical text.",
    ]
    if include_kv_plan:
        lines += [
            "Choose only listed decoded frame indices.",
            f"Cap KV anchors to {anchor_cap} frames per boundary. Use only source shots outside the live local window.",
        ]
    else:
        lines.append("KV anchors are chosen in a separate per-boundary pass; return an empty kv_anchor_plan.")
    lines += [
        f"Scene: {scene}",
        f"Global invariant: {spec.get('invariant_caption') or spec.get('global_caption') or ''}",
    ]
    for idx, caption in enumerate(spec.get("captions", [])):
        rec = shot_records[idx] if idx < len(shot_records) else {}
        lines.append(
            f"SHOT {idx}: frames={rec.get('sampled_frame_indices', [])}; caption={caption}"
        )
    if include_kv_plan:
        for boundary, frames in sorted(candidates.items()):
            lines.append(f"KV candidates for boundary entering shot {boundary}: {frames}")
    return "\n".join(lines)


def sanitize_optimizer_decision(
    raw: dict[str, Any],
    *,
    candidates: dict[int, list[int]],
    anchor_cap: int,
    num_shots: int,
) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    decision = {
        "which_cut_broke": raw.get("which_cut_broke") if isinstance(raw.get("which_cut_broke"), dict) else {
            "cut_index": None,
            "how": "optimizer returned no parseable cut diagnosis",
        },
        "global_invariant_additions": [],
        "per_shot_additions": [],
        "kv_anchor_plan": [],
        "stop_rule": raw.get("stop_rule", "one bounded refinement round"),
    }
    for text in raw.get("global_invariant_additions", []) if isinstance(raw.get("global_invariant_additions"), list) else []:
        if isinstance(text, str) and text.strip():
            decision["global_invariant_additions"].append(text.strip())
    for item in raw.get("per_shot_additions", []) if isinstance(raw.get("per_shot_additions"), list) else []:
        if not isinstance(item, dict):
            continue
        try:
            shot = int(item.get("shot_index"))
        except Exception:
            continue
        text = str(item.get("text", "")).strip()
        if 0 <= shot < num_shots and text:
            decision["per_shot_additions"].append({
                "shot_index": shot,
                "text": text,
                "rationale": str(item.get("rationale", "")).strip(),
            })
    for item in raw.get("kv_anchor_plan", []) if isinstance(raw.get("kv_anchor_plan"), list) else []:
        if not isinstance(item, dict):
            continue
        try:
            boundary = int(item.get("boundary_shot_index"))
        except Exception:
            continue
        allowed = candidates.get(boundary, [])
        if not allowed:
            continue
        chosen: list[int] = []
        for frame in item.get("decoded_frame_indices", []):
            try:
                f = int(frame)
            except Exception:
                continue
            if f not in allowed:
                nearest = min(allowed, key=lambda x: abs(int(x) - f))
                notes.append(f"boundary {boundary}: snapped invalid frame {f} to {nearest}")
                f = nearest
            if f not in chosen:
                chosen.append(f)
            if len(chosen) >= int(anchor_cap):
                break
        if chosen:
            decision["kv_anchor_plan"].append({
                "boundary_shot_index": boundary,
                "decoded_frame_indices": chosen,
                "rationale": str(item.get("rationale", "")).strip(),
            })
    missing_boundaries = [b for b, vals in candidates.items() if vals and not any(p["boundary_shot_index"] == b for p in decision["kv_anchor_plan"])]
    if missing_boundaries:
        notes.append(f"optimizer omitted boundaries with eligible candidates: {missing_boundaries}; left unforced")
    return decision, notes


def qwen_chat_json_cached(
    *,
    get_base_url,
    model: str,
    prompt: str,
    image_paths: list[Path],
    transcript_path: Path,
    timeout: int = 240,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serve a prior transcript only when its content fingerprint (model + prompt
    text + ordered image hashes + schema) matches this call exactly; otherwise
    make the live call (starting the server lazily via get_base_url). Transcripts
    written before fingerprints existed never match, so they are recomputed
    instead of silently reused across prompt or mode changes."""
    if transcript_path.exists():
        try:
            cached = load_json(transcript_path)
            parsed = cached.get("parsed")
            fingerprint = qwen_call_fingerprint(
                model=model, prompt=prompt, image_paths=image_paths, json_schema=json_schema
            )
            if isinstance(parsed, dict) and cached.get("cache_fingerprint") == fingerprint:
                return {
                    "parsed": parsed,
                    "raw": cached.get("response"),
                    "transcript": str(transcript_path),
                    "wall_clock_seconds": 0.0,
                    "cached": True,
                }
        except Exception:
            pass
    return qwen_chat_json(
        base_url=get_base_url(),
        model=model,
        prompt=prompt,
        image_paths=image_paths,
        transcript_path=transcript_path,
        timeout=timeout,
        json_schema=json_schema,
    )


def sanitize_boundary_selection(
    raw: dict[str, Any],
    *,
    boundary: int,
    allowed: list[int],
    anchor_cap: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate one per-boundary selection: snap to eligible frames, dedup, cap."""
    notes: list[str] = []
    if not allowed:
        return None, notes
    if bool(raw.get("skip")):
        return None, [f"boundary {boundary}: optimizer skipped ({str(raw.get('rationale', ''))[:160]})"]
    chosen: list[int] = []
    for frame in raw.get("selected_frame_indices", []) if isinstance(raw.get("selected_frame_indices"), list) else []:
        try:
            f = int(frame)
        except Exception:
            continue
        if f not in allowed:
            nearest = min(allowed, key=lambda x: abs(int(x) - f))
            notes.append(f"boundary {boundary}: snapped invalid frame {f} to {nearest}")
            f = nearest
        if f not in chosen:
            chosen.append(f)
        if len(chosen) >= int(anchor_cap):
            break
    if not chosen:
        return None, notes + [f"boundary {boundary}: selection empty after sanitization"]
    return {
        "boundary_shot_index": int(boundary),
        "decoded_frame_indices": chosen,
        "rationale": str(raw.get("rationale", "")).strip(),
    }, notes


def build_boundary_select_prompt(
    *,
    scene: str,
    spec: dict[str, Any],
    boundary: int,
    allowed: list[int],
    candidate_labels: list[tuple[int, str]],
    query_labels: list[tuple[int, str]],
    anchor_cap: int,
) -> str:
    captions = spec.get("captions", [])
    entering_caption = captions[boundary] if boundary < len(captions) else ""
    lines = [
        "You are the LOCAL Qwen3-VL KV-anchor selector.",
        f"Task: pick which PAST decoded frames get re-injected into the KV cache at the cut into shot {boundary},",
        "so the subject and scene identity stay consistent across the cut.",
        f"Scene: {scene}",
        f"Authored invariant: {spec.get('invariant_caption') or spec.get('global_caption') or ''}",
        f"Entering shot {boundary} caption: {entering_caption}",
        "",
        f"QUERY images (start of shot {boundary}, what the transition must stay consistent with):",
    ]
    for frame_idx, tag in query_labels:
        lines.append(f"- image {tag}: decoded frame {frame_idx}")
    lines.append("")
    lines.append("CANDIDATE anchor images (past frames outside the live attention window):")
    for frame_idx, tag in candidate_labels:
        lines.append(f"- image {tag}: decoded frame {frame_idx}")
    lines += [
        "",
        f"Choose up to {anchor_cap} candidate frames whose CONTENT best pins the subject identity",
        "(face/clothing/markings) and scene layout for this specific cut.",
        "Judge by what the images show, not by position in the video: an early frame is NOT",
        "automatically better. Prefer frames with a clear, unoccluded view of the subject as it",
        "should appear entering this shot. Skip only if no candidate shows the relevant subject.",
        "Return strict JSON: {\"selected_frame_indices\": [..], \"skip\": false, \"rationale\": \"content-based reason naming what the chosen frames show\"}",
        f"Valid frame indices: {allowed}",
    ]
    return "\n".join(lines)


def subsample_candidates(allowed: list[int], max_candidates: int) -> tuple[list[int], list[int]]:
    """Uniformly thin an eligible-frame list to max_candidates, keeping temporal
    spread (first and last always survive). Returns (kept, dropped)."""
    if max_candidates <= 0 or len(allowed) <= max_candidates:
        return list(allowed), []
    ordered = sorted(allowed)
    idxs = np.linspace(0, len(ordered) - 1, num=int(max_candidates)).round().astype(int)
    kept = sorted({ordered[i] for i in idxs})
    dropped = [f for f in ordered if f not in kept]
    return kept, dropped


def select_kv_anchors_per_boundary(
    *,
    get_base_url,
    model: str,
    scene: str,
    spec: dict[str, Any],
    diagnosis_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    candidates: dict[int, list[int]],
    anchor_cap: int,
    transcript_dir: Path,
    timeout: int,
    max_candidates_per_boundary: int = 24,
) -> tuple[list[dict[str, Any]], list[str]]:
    """One retrieval-style VLM call per shot boundary.

    Every boundary with eligible candidates gets an explicit decision (select or
    skip) -- the single-call mode silently dropped late boundaries, exactly where
    drift accumulates.
    """
    frame_to_path: dict[int, Path] = {}
    for rec in candidate_records:
        for frame_idx, path in zip(rec.get("sampled_frame_indices", []), rec.get("paths", [])):
            frame_to_path[int(frame_idx)] = Path(path)
    plan: list[dict[str, Any]] = []
    notes: list[str] = []
    for boundary in sorted(candidates):
        allowed = [f for f in candidates[boundary] if f in frame_to_path]
        dropped = sorted(set(candidates[boundary]) - set(allowed))
        if dropped:
            notes.append(f"boundary {boundary}: {len(dropped)} candidates lacked thumbnails; dropped {dropped[:8]}")
        if not allowed:
            notes.append(f"boundary {boundary}: no eligible candidates with thumbnails")
            continue
        allowed, thinned = subsample_candidates(allowed, max_candidates_per_boundary)
        if thinned:
            notes.append(
                f"boundary {boundary}: thinned {len(thinned)} candidates to cap "
                f"{max_candidates_per_boundary} (uniform spread kept)"
            )
        query_rec = diagnosis_records[boundary] if boundary < len(diagnosis_records) else {}
        query_paths = [Path(p) for p in (query_rec.get("paths") or [])[:2]]
        query_frames = [int(f) for f in (query_rec.get("sampled_frame_indices") or [])[:2]]
        images: list[Path] = []
        query_labels: list[tuple[int, str]] = []
        for i, (f, p) in enumerate(zip(query_frames, query_paths)):
            images.append(p)
            query_labels.append((f, f"Q{i}"))
        candidate_labels: list[tuple[int, str]] = []
        for i, f in enumerate(allowed):
            images.append(frame_to_path[f])
            candidate_labels.append((f, f"C{i}"))
        prompt = build_boundary_select_prompt(
            scene=scene,
            spec=spec,
            boundary=boundary,
            allowed=allowed,
            candidate_labels=candidate_labels,
            query_labels=query_labels,
            anchor_cap=anchor_cap,
        )
        qres = qwen_chat_json_cached(
            get_base_url=get_base_url,
            model=model,
            prompt=prompt,
            image_paths=images,
            transcript_path=transcript_dir / f"{scene}__boundary{boundary:02d}.json",
            timeout=timeout,
            json_schema=BOUNDARY_SELECT_SCHEMA,
        )
        item, item_notes = sanitize_boundary_selection(
            qres["parsed"] or {},
            boundary=boundary,
            allowed=allowed,
            anchor_cap=anchor_cap,
        )
        notes.extend(item_notes)
        if item:
            plan.append(item)
    return plan, notes


def build_prompt_review_prompt(
    *,
    scene: str,
    spec: dict[str, Any],
    global_additions: list[str],
    per_shot_additions: list[dict[str, Any]],
) -> str:
    lines = [
        "You are an independent REVIEWER of proposed prompt edits (not their author).",
        "The authored invariants below are the ground truth of what the video MUST show.",
        f"Scene: {scene}",
        f"Authored global caption: {spec.get('global_caption') or ''}",
        f"Authored invariant: {spec.get('invariant_caption') or ''}",
        "",
        "Proposed additions:",
    ]
    for i, text in enumerate(global_additions):
        lines.append(f"- [global #{i}] {text}")
    for i, item in enumerate(per_shot_additions):
        lines.append(f"- [per_shot #{i} shot={item.get('shot_index')}] {item.get('text')}")
    lines += [
        "",
        "For EACH addition return a verdict:",
        "- keep: consistent with the authored invariants (may sharpen them).",
        "- rewrite: contradicts an authored invariant (e.g. different clothing/color/object than",
        "  authored) or drifts toward what a flawed render showed; provide `text` restating the",
        "  AUTHORED invariant for that aspect instead.",
        "- drop: unrelated to invariants, collapses shot diversity, or is unsalvageable.",
        "For keep/drop verdicts set `text` to the original addition text unchanged.",
        "Return strict JSON: {\"verdicts\": [{\"index\": 0, \"target\": \"global\"|\"per_shot\",",
        "\"verdict\": \"keep\"|\"drop\"|\"rewrite\", \"text\": \"...\", \"reason\": \"...\"}]}",
        "Cover every addition exactly once.",
    ]
    return "\n".join(lines)


def apply_review_verdicts(
    decision: dict[str, Any],
    verdicts: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply reviewer verdicts to a sanitized decision; unreviewed additions keep."""
    global_adds = list(decision.get("global_invariant_additions", []))
    per_shot = [dict(item) for item in decision.get("per_shot_additions", [])]
    record: dict[str, Any] = {"applied": [], "invalid": []}
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        target = v.get("target")
        verdict = str(v.get("verdict", "")).lower()
        try:
            idx = int(v.get("index"))
        except Exception:
            record["invalid"].append(v)
            continue
        rewrite_text = str(v.get("text", "")).strip()
        if verdict not in {"keep", "drop", "rewrite"} or (verdict == "rewrite" and not rewrite_text):
            record["invalid"].append(v)
            continue
        if target == "global" and 0 <= idx < len(global_adds):
            if verdict == "drop":
                global_adds[idx] = None
            elif verdict == "rewrite":
                global_adds[idx] = rewrite_text
        elif target == "per_shot" and 0 <= idx < len(per_shot):
            if verdict == "drop":
                per_shot[idx] = None
            elif verdict == "rewrite":
                per_shot[idx]["text"] = rewrite_text
        else:
            record["invalid"].append(v)
            continue
        record["applied"].append({k: v.get(k) for k in ("index", "target", "verdict", "reason")})
    reviewed = dict(decision)
    reviewed["global_invariant_additions"] = [t for t in global_adds if t]
    reviewed["per_shot_additions"] = [p for p in per_shot if p]
    return reviewed, record


def per_scene_anchor_plan(decisions: dict[str, dict[str, Any]], *, anchor_cap: int) -> dict[str, Any]:
    """v2 manual plan: each scene keeps ITS OWN boundary->frames choices.

    The legacy `boundaries` union stays as a fallback for samples whose name
    matches no scene key (utils/kv_rag.py resolves scenes first, then falls
    back to it)."""
    scenes: dict[str, dict[str, list[dict[str, int]]]] = {}
    union: dict[int, list[int]] = {}
    for scene, decision in decisions.items():
        per_boundary: dict[str, list[dict[str, int]]] = {}
        for item in decision.get("kv_anchor_plan", []):
            boundary = int(item["boundary_shot_index"])
            frames = [int(f) for f in item.get("decoded_frame_indices", [])][: int(anchor_cap)]
            if frames:
                per_boundary[str(boundary)] = [{"frame_index": f} for f in frames]
                union.setdefault(boundary, []).extend(frames)
        if per_boundary:
            scenes[scene] = per_boundary
    boundaries: dict[str, list[dict[str, int]]] = {}
    for boundary, frames in sorted(union.items()):
        unique = sorted(set(frames))[: int(anchor_cap)]
        boundaries[str(boundary)] = [{"frame_index": f} for f in unique]
    return {
        "version": 2,
        "source": "qwen3_vl_per_boundary_retrieval",
        "aggregation": "per-scene boundary plans; legacy `boundaries` union is fallback only",
        "anchor_cap": int(anchor_cap),
        "scenes": scenes,
        "boundaries": boundaries,
    }


def load_optimizer_decisions_from_transcripts(
    *,
    transcript_dir: Path,
    baseline_dir: Path,
    baseline_prefix: str,
    prompt_subset: Path,
    scenes: list[str],
    seed: int,
    num_blocks: int,
    local_attn_size: int,
    optimizer_frames_per_shot: int,
    kv_anchor_cap: int,
    output_root: Path,
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    resolver = build_spec_resolver(prompt_subset, with_captions=True, max_chunks=num_blocks)
    for scene in scenes:
        transcript_path = transcript_dir / f"{scene}.json"
        if not transcript_path.exists():
            raise FileNotFoundError(f"missing Qwen transcript for resume: {transcript_path}")
        video = scene_video(baseline_dir, baseline_prefix, scene, seed)
        spec = resolver(video.stem)
        if spec is None:
            raise RuntimeError(f"could not resolve prompt spec for {video}")
        sampled = sample_scene_frames(
            video_path=video,
            spec=spec,
            out_dir=output_root / "optimizer_frames" / scene,
            frames_per_shot=optimizer_frames_per_shot,
        )
        candidates = candidate_frames_by_boundary(
            sampled["records"],
            local_attn_size=local_attn_size,
        )
        transcript = load_json(transcript_path)
        decision, notes = sanitize_optimizer_decision(
            transcript.get("parsed") or {},
            candidates=candidates,
            anchor_cap=kv_anchor_cap,
            num_shots=len(spec["captions"]),
        )
        decisions[scene] = {
            "decision": decision,
            "sanitization_notes": notes,
            "transcript": str(transcript_path),
            "frames_shown": sampled["records"],
            "kv_candidates": candidates,
            "resumed_from_transcript": True,
        }
    return decisions


def apply_prompt_refinements(
    *,
    source_prompt_dir: Path,
    dest_prompt_dir: Path,
    decisions: dict[str, dict[str, Any]],
    scenes: list[str],
) -> list[dict[str, Any]]:
    if dest_prompt_dir.exists():
        shutil.rmtree(dest_prompt_dir)
    dest_prompt_dir.mkdir(parents=True, exist_ok=True)
    diffs: list[dict[str, Any]] = []
    for scene in scenes:
        shutil.copytree(source_prompt_dir / scene, dest_prompt_dir / scene)
        decision = decisions.get(scene, {})
        global_add = decision.get("global_invariant_additions", [])
        if global_add:
            path = dest_prompt_dir / scene / "global.json"
            data = load_json(path) if path.exists() else {}
            before = copy.deepcopy(data)
            addition = " ".join(global_add)
            for key in ("caption", "invariant_caption"):
                current = str(data.get(key, "") or "")
                data[key] = (current + " Added invariant: " + addition).strip()
            path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            diffs.append({"scene": scene, "path": str(path), "before": before, "after": data, "rationale": "Qwen3 global invariant additions"})
        for item in decision.get("per_shot_additions", []):
            shot = int(item["shot_index"])
            path = dest_prompt_dir / scene / f"{shot}.json"
            if not path.exists():
                continue
            data = load_json(path)
            before = copy.deepcopy(data)
            current = str(data.get("caption", "") or "")
            data["caption"] = (current + " Maintain invariant: " + str(item["text"]).strip()).strip()
            path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            diffs.append({"scene": scene, "path": str(path), "before": before, "after": data, "rationale": item.get("rationale", "")})
    return diffs


def universal_anchor_plan(decisions: dict[str, dict[str, Any]], *, anchor_cap: int) -> dict[str, Any]:
    by_boundary: dict[int, list[int]] = {}
    for decision in decisions.values():
        for item in decision.get("kv_anchor_plan", []):
            boundary = int(item["boundary_shot_index"])
            by_boundary.setdefault(boundary, []).extend(int(f) for f in item.get("decoded_frame_indices", []))
    boundaries: dict[str, list[dict[str, int]]] = {}
    for boundary, frames in sorted(by_boundary.items()):
        unique = sorted(set(frames))
        if len(unique) > int(anchor_cap):
            arr = np.asarray(unique, dtype=np.float64)
            quantiles = np.linspace(0.0, 1.0, num=int(anchor_cap))
            picked = [int(unique[int(np.argmin(np.abs(arr - np.quantile(arr, q))))]) for q in quantiles]
            unique = sorted(set(picked))[: int(anchor_cap)]
        boundaries[str(boundary)] = [{"frame_index": int(f)} for f in unique[: int(anchor_cap)]]
    return {
        "source": "qwen3_vl_decoded_frame_plan",
        "aggregation": "per-boundary union/quantile from scene-level Qwen3 choices; KVRAG snaps to available whole-frame entries",
        "anchor_cap": int(anchor_cap),
        "boundaries": boundaries,
    }


def write_arm_config(
    *,
    base_cfg,
    output_root: Path,
    arm: str,
    prompt_dir: Path,
    kv_settings: dict[str, Any],
    seed: int,
    config_name: str | None = None,
) -> tuple[Path, Path]:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    out_dir = output_root / arm
    _set_nested(cfg, "data", "data_path", str(prompt_dir))
    _set_nested(cfg, "logging", "seed", int(seed))
    _set_nested(cfg, "inference", "sparse_long_multishot", True)
    cfg.output_folder = str(out_dir)
    cfg.inference.output_folder = str(out_dir)
    cfg.inference.filename_prefix = arm
    cfg.inference.filename_from_sample_name = True
    cfg.filename_from_sample_name = True
    cfg.inference.kv_rag = kv_settings
    cfg_path = output_root / "configs" / f"{config_name or arm}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, cfg_path)
    return cfg_path, out_dir


def run_inference_config(config_path: Path, *, gpu: int, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "inference.py"), "--config_path", str(config_path)],
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "config": str(config_path),
        "gpu": int(gpu),
        "returncode": proc.returncode,
        "wall_clock_seconds": time.time() - started,
        "log_path": str(log_path),
    }


def build_video_fixture(
    *,
    scene: str,
    arm: str,
    video_path: Path,
    prompt_dir: Path,
    output_dir: Path,
    frames_per_shot: int,
    num_blocks: int,
) -> tuple[JudgeFixture, dict[str, Any]]:
    resolver = build_spec_resolver(prompt_dir, with_captions=True, max_chunks=num_blocks)
    spec = resolver(video_path.stem)
    if spec is None:
        raise RuntimeError(f"could not resolve prompt spec for {video_path}")
    sampled = sample_scene_frames(
        video_path=video_path,
        spec=spec,
        out_dir=output_dir / scene / arm,
        frames_per_shot=frames_per_shot,
    )
    fixture = JudgeFixture(
        name=f"{scene}_{arm}",
        description=f"Production long-regime multishot scene {scene}, arm {arm}.",
        shots=sampled["shots"],
        captions=list(spec["captions"]),
        expected="production_consistency",
    )
    return fixture, sampled


def judge_fixture_cached(
    *,
    fixture: JudgeFixture,
    output_dir: Path,
    codex: CodexExecJudgeClient,
    claude: ClaudeOAuthMessagesClient,
    claude_model: str,
    timeout: int,
) -> dict[str, Any]:
    cache_dir = output_dir / "judge_cache"
    transcript_dir = output_dir / "judge_transcripts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    vhash = fixture_video_hash(fixture)
    phash = prompt_hash(fixture.captions)
    cache_path = cache_dir / f"{vhash}_{phash}.json"
    if cache_path.exists():
        cached = load_json(cache_path)
        return {**cached, "from_cache": True}
    started = time.time()
    provider = "codex"
    try:
        response = codex.create({}, fixture=fixture, output_dir=transcript_dir, timeout=timeout)
    except Exception as codex_exc:
        provider = "claude"
        last_exc: Exception = codex_exc
        for delay in (5, 15, 45):
            try:
                payload = claude.payload_for_fixture(fixture, model=claude_model)
                response = claude.create(payload, timeout=timeout)
                break
            except Exception as exc:
                last_exc = exc
                if "429" not in str(exc):
                    break
                time.sleep(delay)
        else:
            raise OpenAIResponsesError(f"both judge providers failed; codex={codex_exc}; claude={last_exc}") from last_exc
        if "response" not in locals():
            raise OpenAIResponsesError(f"both judge providers failed; codex={codex_exc}; claude={last_exc}") from last_exc
    verdict = json.loads(response["output_text"])
    validate_verdict_shape(verdict)
    out = {
        "fixture": fixture.name,
        "provider": provider,
        "video_hash": vhash,
        "prompt_hash": phash,
        "verdict": verdict,
        "usage": response.get("usage", {}),
        "client": response.get("client", {}),
        "frames_shown": [[str(p) for p in shot] for shot in fixture.shots],
        "captions": fixture.captions,
        "wall_clock_seconds": time.time() - started,
        "from_cache": False,
    }
    write_json(cache_path, out)
    return out


def score_from_verdict(verdict: dict[str, Any]) -> float:
    return float(verdict.get("overall_consistency_score", 0.0))


def evaluate_numeric_arm(
    *,
    baseline_dir: Path,
    arm_dir: Path,
    prompt_dir: Path,
    scenes: list[str],
    negative_controls: list[str],
    admission: dict[str, Any],
    args: argparse.Namespace,
    num_blocks: int,
    output_dir: Path,
) -> dict[str, Any]:
    scorers, scorer_coverage = _build_long_scorers(args, require_adherence=True)
    missing = sorted(
        name for name, info in scorer_coverage.items()
        if isinstance(info, dict) and info.get("requested") and not info.get("loaded")
    )
    if missing:
        return {"passed": False, "blocked_reason": f"numeric scorer(s) unavailable: {missing}", "scorer_coverage": scorer_coverage}
    shots_for = build_spec_resolver(prompt_dir, with_captions=True, max_chunks=num_blocks)
    eval_scenes = list(dict.fromkeys(list(scenes) + list(negative_controls)))
    subset_root = output_dir / "pair_subsets" / arm_dir.name
    baseline_subset = link_scene_video_subset(
        source_dir=baseline_dir,
        prefix=f"baseline_seed{args.baseline_seed}",
        scenes=eval_scenes,
        seed=args.baseline_seed,
        subset_dir=subset_root / "baseline",
    )
    modified_subset = link_scene_video_subset(
        source_dir=arm_dir,
        prefix=arm_dir.name,
        scenes=eval_scenes,
        seed=args.baseline_seed,
        subset_dir=subset_root / "modified",
    )
    result_all = compare_cross_perspective_dirs(
        baseline_subset["dir"],
        modified_subset["dir"],
        shots_for=shots_for,
        adherence_scorer=scorers.get("adherence_scorer"),
        invariant_scorer=scorers.get("invariant_scorer"),
        subject_encoder=scorers.get("subject_encoder"),
        background_encoder=scorers.get("background_encoder"),
        dynamic_scorer=scorers.get("dynamic_scorer"),
        max_frames=None,
        stride=1,
    )
    allowed = set(scenes) | set(negative_controls)
    records = []
    for row in result_all["records"]:
        if row.get("theme") not in allowed:
            continue
        rec = copy.deepcopy(row)
        rec["negative_control"] = rec.get("theme") in set(negative_controls)
        records.append(rec)
    result = _copy_result_with_records(result_all, records)
    min_wins = math.ceil(len(scenes) / 2)
    gate = evaluate_cross_perspective_gate(
        result,
        metric="anchor_drift_aggregate_consistency",
        min_consistency_wins=min_wins,
        adherence_tolerance=args.adherence_tolerance,
        diversity_tolerance=args.diversity_tolerance,
        motion_tolerance=args.motion_tolerance,
        require_adherence=True,
        require_invariant=True,
    )
    gate = _apply_noise_floor_gate(
        gate,
        metric="anchor_drift_aggregate_consistency",
        noise_floor_by_scene=admission.get("noise_floor", {}),
        sigma_multiplier=float(admission.get("guards", {}).get("noise_sigma_multiplier", 2.0)),
    )
    attention = _load_attention_diagnostics(arm_dir, shots_for)
    frame_reason = _frame_contract_blocked_reason({
        scene: attention.get(scene, {"diagnostic": {}})
        for scene in allowed
    })
    frame_contract = {
        scene: _frame_contract_stats(rec.get("diagnostic", {}))
        for scene, rec in attention.items()
    }
    logit_bias_calls = sum(
        int(stats.get("persistent_logit_bias_calls", 0) or 0)
        for stats in frame_contract.values()
    )
    logit_bias_lambda = 0.0
    for rec in attention.values():
        diag = rec.get("diagnostic", {}) if isinstance(rec, dict) else {}
        for bank in ("pos", "neg"):
            cfg = (diag.get(bank) or {}).get("config", {})
            if isinstance(cfg, dict):
                logit_bias_lambda = max(
                    logit_bias_lambda,
                    float(cfg.get("persistent_logit_bias_lambda", 0.0) or 0.0),
                )
    arm_cfg_path = arm_dir.parent / "configs" / f"{arm_dir.name}.yaml"
    if arm_cfg_path.exists():
        try:
            arm_cfg = OmegaConf.load(arm_cfg_path)
            kv_cfg = arm_cfg.get("inference", {}).get("kv_rag", {})
            logit_bias_lambda = max(
                logit_bias_lambda,
                float(kv_cfg.get("persistent_logit_bias_lambda", 0.0) or 0.0),
            )
        except Exception:
            pass
    if logit_bias_lambda > 0.0 and arm_dir.name in {"kv_only", "both"} and logit_bias_calls <= 0:
        extra = "persistent logit-bias configured but persistent_logit_bias_calls=0"
        frame_reason = "; ".join(([frame_reason] if frame_reason else []) + [extra])
    if frame_reason and arm_dir.name in {"kv_only", "both"}:
        prev = gate.get("blocked_reason")
        gate["blocked_reason"] = "; ".join(([prev] if prev else []) + [frame_reason])
        gate["passed"] = False
        gate["frame_contract_ok"] = False
    else:
        gate["frame_contract_ok"] = True
    out = {
        "passed": bool(gate.get("passed")),
        "gate": gate,
        "comparison": result,
        "attention_diagnostics": attention,
        "frame_contract": frame_contract,
        "scorer_coverage": scorer_coverage,
        "scene_role_source": "admission_negative_controls_v2",
        "pair_subset": {
            "baseline": baseline_subset,
            "modified": modified_subset,
            "scenes": eval_scenes,
            "source_baseline_dir": str(baseline_dir),
            "source_modified_dir": str(arm_dir),
        },
    }
    write_json(output_dir / f"numeric_{arm_dir.name}.json", out)
    return out


def write_degraded_negative_fixture(
    *,
    baseline_video: Path,
    prompt_dir: Path,
    scene: str,
    output_dir: Path,
    num_blocks: int,
) -> JudgeFixture:
    resolver = build_spec_resolver(prompt_dir, with_captions=True, max_chunks=num_blocks)
    spec = resolver(baseline_video.stem)
    if spec is None:
        raise RuntimeError(f"could not resolve negative control spec for {baseline_video}")
    frames = read_video(baseline_video)
    first = frames[0]
    shots: list[list[Path]] = []
    for idx, _ in enumerate(spec["captions"]):
        path = output_dir / f"degraded_copy_shot{idx:02d}.png"
        save_rgb(path, first)
        shots.append([path])
    return JudgeFixture(
        name=f"{scene}_degraded_copy_negative_control",
        description="Sane negative control: copied/frozen first frame reused for every shot caption.",
        shots=shots,
        captions=list(spec["captions"]),
        expected="copy_cheat_flagged",
    )


def numeric_failure_reason(numeric_result: dict[str, Any]) -> str | None:
    reason = numeric_result.get("blocked_reason") or (numeric_result.get("gate") or {}).get("blocked_reason")
    if reason:
        return str(reason)
    gate = numeric_result.get("gate") or {}
    if numeric_result.get("passed"):
        return None
    false_keys = [
        key for key in (
            "consistency_ok",
            "adherence_ok",
            "diversity_ok",
            "motion_ok",
            "invariant_ok",
            "scorable_ok",
            "negative_control_ok",
            "frame_contract_ok",
        )
        if gate.get(key) is False
    ]
    if not false_keys:
        return None
    details = []
    if "consistency_ok" in false_keys:
        details.append(
            "consistency_wins={wins}/{need}".format(
                wins=gate.get("consistency_wins"),
                need=gate.get("min_consistency_wins"),
            )
        )
    return "numeric guard failed: " + ", ".join(false_keys + details)


def build_ledger(
    *,
    args: argparse.Namespace,
    output_root: Path,
    video_root: Path,
    judge_validation: dict[str, Any],
    gpu: dict[str, Any],
    qwen_meta: dict[str, Any] | None,
    optimizer_decisions: dict[str, Any],
    prompt_diffs: list[dict[str, Any]],
    anchor_plan: dict[str, Any],
    generation: dict[str, Any],
    judge_results: dict[str, Any],
    numeric_results: dict[str, Any],
    negative_control: dict[str, Any],
    admission: dict[str, Any],
    started: float,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    scenes = list(admission["admitted_main_scenes"])
    min_wins = math.ceil(len(scenes) / 2)
    arm_rows = []
    winner = None
    for arm in ARMS:
        if arm == "baseline":
            continue
        j = judge_results.get("arms", {}).get(arm, {})
        n = numeric_results.get(arm, {})
        attention_vals = [
            float(rec.get("mean_persistent_attention_mass", 0.0) or 0.0)
            for rec in (n.get("attention_diagnostics") or {}).values()
            if isinstance(rec, dict)
        ]
        frame_attention_vals = [
            float(rec.get("mean_persistent_frame_attention_mass", 0.0) or 0.0)
            for rec in (n.get("attention_diagnostics") or {}).values()
            if isinstance(rec, dict)
        ]
        contract_vals = list((n.get("frame_contract") or {}).values())
        persistent_logit_bias_calls = sum(
            int(rec.get("persistent_logit_bias_calls", 0) or 0)
            for rec in contract_vals
            if isinstance(rec, dict)
        )
        persistent_logit_bias_frames = sum(
            int(rec.get("persistent_logit_bias_frames", 0) or 0)
            for rec in contract_vals
            if isinstance(rec, dict)
        )
        judge_wins = int(j.get("judge_wins", 0))
        passed = bool(judge_wins >= min_wins and n.get("passed") and negative_control.get("passed"))
        row = {
            "arm": arm,
            "judge_wins": judge_wins,
            "min_wins": min_wins,
            "judge_pass": judge_wins >= min_wins,
            "numeric_pass": bool(n.get("passed")),
            "negative_control_pass": bool(negative_control.get("passed")),
            "passed": passed,
            "mean_judge_delta": j.get("mean_delta"),
            "numeric_blocked_reason": numeric_failure_reason(n),
            "persistent_rag_attention_mass_mean": (
                float(np.mean(attention_vals)) if attention_vals else 0.0
            ),
            "persistent_rag_frame_attention_mass_mean": (
                float(np.mean(frame_attention_vals)) if frame_attention_vals else 0.0
            ),
            "persistent_logit_bias_calls": int(persistent_logit_bias_calls),
            "persistent_logit_bias_frames": int(persistent_logit_bias_frames),
        }
        arm_rows.append(row)
        if passed and (winner is None or (row.get("mean_judge_delta") or -999) > (winner.get("mean_judge_delta") or -999)):
            winner = row
    if blocked_reason is None and winner is None:
        blocked_reason = None
    generation_stage_seconds: dict[str, float] = {}
    generation_stage_sources: dict[str, str] = {}
    for arm, rec in (generation.get("runs") or {}).items():
        seconds = float(rec.get("wall_clock_seconds") or 0.0)
        source = "run_timer"
        if seconds <= 0.0:
            recovered = recovered_log_seconds(rec.get("log_path"))
            if recovered is not None:
                seconds = float(recovered)
                source = "tqdm_elapsed_from_generation_log"
        generation_stage_seconds[arm] = seconds
        generation_stage_sources[arm] = source
    closed_judge_seconds = mtime_span_seconds(output_root / "closed_judge" / "judge_cache")
    numeric_seconds = mtime_span_seconds(output_root / "numeric")
    ledger = {
        "timestamp_utc": utc_timestamp(),
        "objective": "VLM-in-the-loop four-arm cross-shot consistency ablation with independent OAuth closed judge",
        "pass_criterion": (
            "judge_wins >= ceil(N/2) plus numeric guard non-regression "
            "(motion/diversity/adherence/invariant), sane negative control, and frame-level KV contract"
        ),
        "consistency_delta_gate": {
            "baseline_seeds": admission.get("baseline_seeds"),
            "policy": (admission.get("guards") or {}).get("consistency_delta_gate"),
            "threshold": (admission.get("guards") or {}).get("single_seed_consistency_delta_threshold"),
            "noise_sigma_multiplier": (admission.get("guards") or {}).get("noise_sigma_multiplier"),
            "reduced_power_single_seed": bool(
                (admission.get("guards") or {}).get("reduced_power_single_seed", False)
            ),
            "note": (admission.get("guards") or {}).get("gate_power_note"),
        },
        "pass": bool(winner),
        "winner": winner,
        "is_null_result": winner is None,
        "blocked_reason": blocked_reason,
        "next_hypothesis": (
            "If null: Qwen-selected decoded-frame anchors may be too sparse/late for Wan 5B; "
            "test latent re-anchoring or a stronger prompt-invariant schedule while keeping the same held-out judge."
        ),
        "judge_validation": {
            "reused": judge_validation.get("reused"),
            "path": judge_validation.get("path"),
            "selected_provider": judge_validation.get("selected_provider"),
            "closed_judge_validated": True,
            "oauth_only": True,
            "paid_api_key_used": False,
        },
        "separation": {
            "optimizer": "local Qwen3-VL-8B via SGLang OpenAI-compatible localhost server",
            "judge": "codex exec OAuth closed model primary; Claude OAuth alternate",
            "shared_transcript": False,
            "optimizer_model": args.qwen_model,
            "judge_model": args.codex_model,
        },
        "gpu": gpu,
        "qwen3_optimizer": qwen_meta,
        "admission": admission,
        "optimizer_decisions": optimizer_decisions,
        "prompt_refine_diffs": prompt_diffs,
        "manual_anchor_plan": anchor_plan,
        "generation": generation,
        "closed_judge": judge_results,
        "numeric_guards": numeric_results,
        "negative_control": negative_control,
        "ablation_table": arm_rows,
        "resources": {
            "final_run_wall_clock_seconds": time.time() - started,
            "stage_wall_clock_seconds": {
                "qwen3_startup": (qwen_meta or {}).get("startup_seconds", 0),
                "generation_by_arm": generation_stage_seconds,
                "generation_total": sum(generation_stage_seconds.values()),
                "closed_judge_cache_mtime_span": closed_judge_seconds,
                "numeric_mtime_span": numeric_seconds,
            },
            "stage_timing_sources": {
                "generation_by_arm": generation_stage_sources,
                "closed_judge_cache_mtime_span": "mtime span of cached verdict JSON files",
                "numeric_mtime_span": "mtime span of numeric guard JSON files",
                "note": "This ledger may be produced by a resumed run; stage timings preserve recovered prior-stage durations where possible.",
            },
            "api_tokens": judge_results.get("api_usage", {}),
            "api_cost_usd": None,
            "api_cost_note": "OAuth paths only; codex-exec does not expose billable token/cost accounting here.",
            "gpu_time_seconds": {
                "qwen3_startup": (qwen_meta or {}).get("startup_seconds", 0),
                "generation": sum(generation_stage_seconds.values()),
            },
        },
        "artifacts": {
            "output_root": str(output_root),
            "video_root": str(video_root),
        },
    }
    return ledger


def main() -> int:
    args = parse_args()
    started = time.time()
    output_root = Path(args.output_root)
    video_root = Path(args.video_root)
    output_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)

    gpu = choose_gpus(args)
    judge_validation = ensure_judge_valid(args, output_root)

    admission_src = load_json(args.admission_json)
    numeric_ref = load_json(args.numeric_reference_json) if Path(args.numeric_reference_json).exists() else {}
    admitted = list(admission_src.get("admitted_main_scenes") or numeric_ref.get("admitted_main_scenes") or [])
    negatives = list(admission_src.get("negative_controls") or numeric_ref.get("negative_controls") or [])
    synthetic_negative = (
        admission_src.get("synthetic_negative_control")
        or numeric_ref.get("synthetic_negative_control")
        or {}
    )
    if args.max_scenes and args.max_scenes > 0:
        admitted = admitted[: int(args.max_scenes)]
    if not admitted:
        raise RuntimeError("no admitted scenes available")
    if len(admitted) < int(args.min_admitted_main_scenes):
        raise RuntimeError(
            f"admission JSON has {len(admitted)} admitted main scenes; "
            f"need >= {int(args.min_admitted_main_scenes)} for this run"
        )
    synthetic_negative_source = str(synthetic_negative.get("source_scene") or "").strip()
    if not negatives and not synthetic_negative_source:
        raise RuntimeError("negative control scene missing and no synthetic_negative_control.source_scene provided")
    prompt_subset = Path(admission_src.get("prompt_subset_dir") or "videos/frame_strategy_A/prompt_subset")
    baseline_dirs = admission_src.get("baseline_seed_dirs") or numeric_ref.get("baseline_seed_dirs") or {}
    baseline_dir = Path(baseline_dirs.get(str(args.baseline_seed), "videos/frame_strategy_A/baseline_seed0"))
    baseline_prefix = f"baseline_seed{args.baseline_seed}"

    base_cfg = OmegaConf.load(args.config_path)
    num_blocks = _num_blocks_from_cfg(base_cfg)
    if num_blocks is None:
        raise RuntimeError("could not derive num_blocks from config")
    local_attn_size = int(base_cfg.get("model_kwargs", {}).get("local_attn_size", 32))
    admission = {
        "source_json": args.admission_json,
        "numeric_reference_json": args.numeric_reference_json,
        "prompt_subset_dir": str(prompt_subset),
        "baseline_dir": str(baseline_dir),
        "baseline_prefix": baseline_prefix,
        "baseline_seeds": admission_src.get("baseline_seeds") or numeric_ref.get("baseline_seeds") or [args.baseline_seed],
        "admitted_main_scenes": admitted,
        "negative_controls": negatives,
        "admission": admission_src.get("admission") or numeric_ref.get("admission") or {},
        "synthetic_negative_control": synthetic_negative,
        "deferred_main_scenes": admission_src.get("deferred_main_scenes") or [],
        "exploratory_main_scenes": admission_src.get("exploratory_main_scenes") or [],
        "noise_floor": admission_src.get("noise_floor") or numeric_ref.get("noise_floor") or {},
        "guards": admission_src.get("guards") or numeric_ref.get("guards") or {"noise_sigma_multiplier": 2.0},
        "explicit_scene_cap": int(args.max_scenes),
    }

    # Declared before try so `finally` can terminate a lazily-started server even
    # when the optimizer loop raises mid-scene.
    qwen_state: dict[str, Any] = {"proc": None, "meta": None, "base_url": None}
    qwen_meta: dict[str, Any] | None = None
    generation: dict[str, Any] = {"runs": {}, "skipped_generation": bool(args.skip_generation)}
    optimizer_decisions: dict[str, Any] = {}
    prompt_diffs: list[dict[str, Any]] = []
    anchor_plan: dict[str, Any] = {}
    judge_results: dict[str, Any] = {"arms": {}, "api_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
    numeric_results: dict[str, Any] = {}
    negative_control_result: dict[str, Any] = {"passed": False}
    blocked_reason: str | None = None

    try:
        refined_prompt_dir = video_root / "prompt_refined"
        original_prompt_dir = video_root / "prompt_original"
        anchor_plan_path = output_root / "manual_anchor_plan.json"
        prior_ledger_path = output_root / "ledger.json"
        eval_scenes = admitted + negatives
        # Everything that changes what the optimizer would produce; resume is
        # only valid when a prior run recorded the SAME configuration.
        optimizer_marker_path = output_root / "optimizer_config.json"
        optimizer_config_expected = {
            "kv_selection_mode": args.kv_selection_mode,
            "prompt_review": args.prompt_review,
            "kv_anchor_cap": int(args.kv_anchor_cap),
            "optimizer_frames_per_shot": int(args.optimizer_frames_per_shot),
            "kv_candidate_frames_per_shot": int(args.kv_candidate_frames_per_shot),
            "kv_max_candidates_per_boundary": int(args.kv_max_candidates_per_boundary),
            "qwen_model": args.qwen_model,
            "optimizer_video_dir": str(args.optimizer_video_dir) if args.optimizer_video_dir else None,
            "optimizer_video_prefix": args.optimizer_video_prefix or None,
            "baseline_seed": int(args.baseline_seed),
        }
        prior_optimizer_config = (
            load_json(optimizer_marker_path) if optimizer_marker_path.exists() else None
        )
        can_resume_optimizer = (
            args.resume
            and prior_optimizer_config == optimizer_config_expected
            and (output_root / "optimizer_decisions.json").exists()
            and anchor_plan_path.exists()
            and refined_prompt_dir.exists()
            and original_prompt_dir.exists()
        )
        if args.resume and prior_optimizer_config is not None and prior_optimizer_config != optimizer_config_expected:
            print(
                "[vlm-ablation] optimizer config changed since prior run; "
                "recomputing optimizer artifacts instead of resuming",
                flush=True,
            )
        can_resume_transcripts = (
            args.resume
            and args.kv_selection_mode == "legacy_single_call"
            and args.prompt_review == "off"
            and all(
                (output_root / "qwen3_optimizer" / "transcripts" / f"{scene}.json").exists()
                for scene in eval_scenes
            )
        )
        if can_resume_optimizer:
            prior_ledger = load_json(prior_ledger_path) if prior_ledger_path.exists() else {}
            optimizer_decisions = load_json(output_root / "optimizer_decisions.json")
            anchor_plan = load_json(anchor_plan_path)
            prompt_diffs = list(prior_ledger.get("prompt_refine_diffs") or [])
            qwen_meta = prior_ledger.get("qwen3_optimizer")
            generation["resumed_optimizer_artifacts"] = True
        elif can_resume_transcripts:
            optimizer_decisions = load_optimizer_decisions_from_transcripts(
                transcript_dir=output_root / "qwen3_optimizer" / "transcripts",
                baseline_dir=baseline_dir,
                baseline_prefix=baseline_prefix,
                prompt_subset=prompt_subset,
                scenes=eval_scenes,
                seed=args.baseline_seed,
                num_blocks=num_blocks,
                local_attn_size=local_attn_size,
                optimizer_frames_per_shot=args.optimizer_frames_per_shot,
                kv_anchor_cap=args.kv_anchor_cap,
                output_root=output_root,
            )
            prompt_diffs = apply_prompt_refinements(
                source_prompt_dir=prompt_subset,
                dest_prompt_dir=refined_prompt_dir,
                decisions={k: v["decision"] for k, v in optimizer_decisions.items()},
                scenes=eval_scenes,
            )
            if original_prompt_dir.exists():
                shutil.rmtree(original_prompt_dir)
            copy_prompt_subset(prompt_subset, eval_scenes, original_prompt_dir)
            anchor_plan = universal_anchor_plan(
                {k: v["decision"] for k, v in optimizer_decisions.items()},
                anchor_cap=args.kv_anchor_cap,
            )
            write_json(anchor_plan_path, anchor_plan)
            write_json(optimizer_marker_path, optimizer_config_expected)
            qwen_meta = {
                "reused_transcripts": True,
                "transcript_dir": str(output_root / "qwen3_optimizer" / "transcripts"),
                "model": args.qwen_model,
            }
            generation["resumed_optimizer_transcripts"] = True
        else:
            def get_base_url() -> str:
                if qwen_state["base_url"] is None:
                    proc, meta = start_qwen3_server(args, gpu["optimizer_gpu"], output_root)
                    qwen_state.update(proc=proc, meta=meta, base_url=meta["launch"]["base_url"])
                return qwen_state["base_url"]

            # Multi-pass feedback: the optimizer can watch a prior round's arm
            # render instead of the baseline; judge/metrics still pair against
            # the baseline.
            watch_dir = Path(args.optimizer_video_dir) if args.optimizer_video_dir else baseline_dir
            watch_prefix = args.optimizer_video_prefix or baseline_prefix
            per_boundary = args.kv_selection_mode == "per_boundary"

            for scene in eval_scenes:
                video = scene_video(watch_dir, watch_prefix, scene, args.baseline_seed)
                spec = build_spec_resolver(prompt_subset, with_captions=True, max_chunks=num_blocks)(video.stem)
                sampled = sample_scene_frames(
                    video_path=video,
                    spec=spec,
                    out_dir=output_root / "optimizer_frames" / scene,
                    frames_per_shot=args.optimizer_frames_per_shot,
                )
                if per_boundary:
                    candidate_sampled = sample_scene_frames(
                        video_path=video,
                        spec=spec,
                        out_dir=output_root / "kv_candidate_frames" / scene,
                        frames_per_shot=args.kv_candidate_frames_per_shot,
                    )
                    candidates = candidate_frames_by_boundary(
                        candidate_sampled["records"], local_attn_size=local_attn_size
                    )
                else:
                    candidate_sampled = None
                    candidates = candidate_frames_by_boundary(sampled["records"], local_attn_size=local_attn_size)
                prompt = build_optimizer_prompt(
                    scene,
                    spec,
                    sampled["records"],
                    candidates,
                    anchor_cap=args.kv_anchor_cap,
                    include_kv_plan=not per_boundary,
                )
                images = [Path(p) for rec in sampled["records"] for p in rec["paths"]]
                qres = qwen_chat_json_cached(
                    get_base_url=get_base_url,
                    model=args.qwen_model,
                    prompt=prompt,
                    image_paths=images,
                    transcript_path=output_root / "qwen3_optimizer" / "transcripts" / f"{scene}.json",
                    json_schema=OPTIMIZER_DIAGNOSIS_SCHEMA,
                )
                decision, notes = sanitize_optimizer_decision(
                    qres["parsed"],
                    # In per-boundary mode the diagnosis call owns no KV plan, so
                    # give it no candidates: stray plan entries drop and the
                    # "omitted boundaries" note (owned by the selection pass) is
                    # not emitted here.
                    candidates={} if per_boundary else candidates,
                    anchor_cap=args.kv_anchor_cap,
                    num_shots=len(spec["captions"]),
                )
                review_record: dict[str, Any] | None = None
                if args.prompt_review == "on" and (
                    decision["global_invariant_additions"] or decision["per_shot_additions"]
                ):
                    review_prompt = build_prompt_review_prompt(
                        scene=scene,
                        spec=spec,
                        global_additions=decision["global_invariant_additions"],
                        per_shot_additions=decision["per_shot_additions"],
                    )
                    rres = qwen_chat_json_cached(
                        get_base_url=get_base_url,
                        model=args.qwen_model,
                        prompt=review_prompt,
                        image_paths=[],
                        transcript_path=output_root / "qwen3_optimizer" / "prompt_review" / f"{scene}.json",
                        json_schema=PROMPT_REVIEW_SCHEMA,
                    )
                    verdicts = (rres["parsed"] or {}).get("verdicts") or []
                    decision, review_record = apply_review_verdicts(decision, verdicts)
                    review_record["transcript"] = rres["transcript"]
                if per_boundary:
                    kv_plan, kv_notes = select_kv_anchors_per_boundary(
                        get_base_url=get_base_url,
                        model=args.qwen_model,
                        scene=scene,
                        spec=spec,
                        diagnosis_records=sampled["records"],
                        candidate_records=candidate_sampled["records"],
                        candidates=candidates,
                        anchor_cap=args.kv_anchor_cap,
                        transcript_dir=output_root / "qwen3_optimizer" / "kv_selection",
                        timeout=240,
                        max_candidates_per_boundary=args.kv_max_candidates_per_boundary,
                    )
                    decision["kv_anchor_plan"] = kv_plan
                    notes.extend(kv_notes)
                optimizer_decisions[scene] = {
                    "decision": decision,
                    "sanitization_notes": notes,
                    "transcript": qres["transcript"],
                    "frames_shown": sampled["records"],
                    "kv_candidates": candidates,
                    "kv_selection_mode": args.kv_selection_mode,
                    "prompt_review": review_record,
                    "optimizer_watched_video": str(video),
                }

            qwen_meta = qwen_state["meta"] or {
                "reused_transcripts": True,
                "transcript_dir": str(output_root / "qwen3_optimizer" / "transcripts"),
                "model": args.qwen_model,
            }

            prompt_diffs = apply_prompt_refinements(
                source_prompt_dir=prompt_subset,
                dest_prompt_dir=refined_prompt_dir,
                decisions={k: v["decision"] for k, v in optimizer_decisions.items()},
                scenes=eval_scenes,
            )
            if original_prompt_dir.exists():
                shutil.rmtree(original_prompt_dir)
            copy_prompt_subset(prompt_subset, eval_scenes, original_prompt_dir)

            plan_decisions = {k: v["decision"] for k, v in optimizer_decisions.items()}
            if per_boundary:
                anchor_plan = per_scene_anchor_plan(plan_decisions, anchor_cap=args.kv_anchor_cap)
            else:
                anchor_plan = universal_anchor_plan(plan_decisions, anchor_cap=args.kv_anchor_cap)
            write_json(anchor_plan_path, anchor_plan)
            write_json(optimizer_marker_path, optimizer_config_expected)

        base_kv = _base_kv_rag_block(base_cfg)
        manual_kv = dict(DEFAULT_KV_RAG)
        manual_kv.update(base_kv)
        manual_kv.update({
            "enabled": True,
            "top_k": 0,
            "scene_memory_enabled": True,
            "scene_memory_rolling": True,
            "scene_memory_injection_schedule": "boundary",
            "scene_memory_max_entries": 64,
            "boundary_inject_anchors": int(args.kv_anchor_cap),
            "manual_anchor_plan_path": str(anchor_plan_path),
            "retrieval_key_mode": "pooled",
            "retrieval_value_mode": "raw",
            "max_frames_per_entry": 1,
            "frame_aligned_store": True,
            "require_frame_aligned": True,
            "reinject_rope": True,
            "attention_diagnostic": True,
            "persistent_logit_bias_lambda": float(args.primary_logit_bias_lambda),
        })
        arm_configs = {
            "prompt_only": {"prompt_dir": refined_prompt_dir, "kv": {"enabled": False}},
            "kv_only": {"prompt_dir": original_prompt_dir, "kv": manual_kv},
            "both": {"prompt_dir": refined_prompt_dir, "kv": manual_kv},
        }
        _preflight_config(base_cfg, args.config_path)
        arm_dirs = {"baseline": baseline_dir}
        if not args.skip_generation:
            for arm, info in arm_configs.items():
                cfg_path, arm_dir = write_arm_config(
                    base_cfg=base_cfg,
                    output_root=video_root,
                    arm=arm,
                    prompt_dir=info["prompt_dir"],
                    kv_settings=info["kv"],
                    seed=args.baseline_seed,
                )
                missing = missing_arm_scenes(arm_dir, arm, eval_scenes, args.baseline_seed)
                if args.resume and not missing:
                    generation["runs"][arm] = {
                        "config": str(cfg_path),
                        "gpu": None,
                        "returncode": 0,
                        "wall_clock_seconds": 0.0,
                        "log_path": str(output_root / "generation_logs" / f"{arm}.log"),
                        "reused": True,
                    }
                    arm_dirs[arm] = arm_dir
                    continue
                render_cfg_path = cfg_path
                if args.resume and missing and len(missing) < len(eval_scenes):
                    subset_prompt_dir = copy_prompt_subset(
                        Path(info["prompt_dir"]),
                        missing,
                        video_root / "render_subsets" / arm,
                    )
                    render_cfg_path, arm_dir = write_arm_config(
                        base_cfg=base_cfg,
                        output_root=video_root,
                        arm=arm,
                        prompt_dir=subset_prompt_dir,
                        kv_settings=info["kv"],
                        seed=args.baseline_seed,
                        config_name=f"{arm}_missing",
                    )
                checked = ensure_gpu_not_maxed(gpu["generator_gpu"])
                if checked.get("switched"):
                    gpu["generator_gpu"] = checked["gpu"]
                    gpu.setdefault("stage_switches", []).append(checked)
                run = run_inference_config(
                    render_cfg_path,
                    gpu=gpu["generator_gpu"],
                    log_path=output_root / "generation_logs" / f"{arm}.log",
                )
                run["missing_scenes_rendered"] = missing
                run["full_config"] = str(cfg_path)
                generation["runs"][arm] = run
                if run["returncode"] != 0:
                    raise RuntimeError(f"generation failed for {arm}; see {run['log_path']}")
                still_missing = missing_arm_scenes(arm_dir, arm, eval_scenes, args.baseline_seed)
                if still_missing:
                    raise RuntimeError(f"generation incomplete for {arm}; missing scenes={still_missing}")
                arm_dirs[arm] = arm_dir
        else:
            for arm in ("prompt_only", "kv_only", "both"):
                arm_dirs[arm] = video_root / arm

        codex = CodexExecJudgeClient(model=args.codex_model)
        claude = ClaudeOAuthMessagesClient()
        judge_frame_dir = output_root / "judge_frames"
        for arm in ARMS:
            prompt_dir = original_prompt_dir if arm in {"baseline", "kv_only"} else refined_prompt_dir
            arm_dir = arm_dirs[arm]
            scene_scores = {}
            scene_results = {}
            for scene in eval_scenes:
                prefix = baseline_prefix if arm == "baseline" else arm
                video = scene_video(arm_dir, prefix, scene, args.baseline_seed)
                fixture, sampled = build_video_fixture(
                    scene=scene,
                    arm=arm,
                    video_path=video,
                    prompt_dir=prompt_dir,
                    output_dir=judge_frame_dir,
                    frames_per_shot=args.judge_frames_per_shot,
                    num_blocks=num_blocks,
                )
                verdict = judge_fixture_cached(
                    fixture=fixture,
                    output_dir=output_root / "closed_judge",
                    codex=codex,
                    claude=claude,
                    claude_model=args.claude_model,
                    timeout=args.judge_timeout,
                )
                scene_scores[scene] = score_from_verdict(verdict["verdict"])
                scene_results[scene] = {
                    **verdict,
                    "video_path": str(video),
                    "sampled": sampled,
                    "negative_control_scene": scene in negatives,
                }
                usage = verdict.get("usage") or {}
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    judge_results["api_usage"][key] += int(usage.get(key, 0) or 0)
            judge_results["arms"][arm] = {"scores": scene_scores, "results": scene_results}

        baseline_scores = judge_results["arms"]["baseline"]["scores"]
        for arm in ("prompt_only", "kv_only", "both"):
            scores = judge_results["arms"][arm]["scores"]
            deltas = {
                scene: float(scores[scene]) - float(baseline_scores[scene])
                for scene in admitted
            }
            wins = [scene for scene, delta in deltas.items() if delta > 0.0]
            judge_results["arms"][arm].update({
                "deltas_vs_baseline": deltas,
                "judge_wins": len(wins),
                "winning_scenes": wins,
                "mean_delta": float(np.mean(list(deltas.values()))) if deltas else float("nan"),
            })

        degraded_source_scene = synthetic_negative_source or negatives[0]
        degraded_fixture = write_degraded_negative_fixture(
            baseline_video=scene_video(baseline_dir, baseline_prefix, degraded_source_scene, args.baseline_seed),
            prompt_dir=prompt_subset,
            scene=degraded_source_scene,
            output_dir=output_root / "negative_control_degraded",
            num_blocks=num_blocks,
        )
        neg_judge = judge_fixture_cached(
            fixture=degraded_fixture,
            output_dir=output_root / "closed_judge",
            codex=codex,
            claude=claude,
            claude_model=args.claude_model,
            timeout=args.judge_timeout,
        )
        flags = neg_judge["verdict"]["cheat_flags"]
        negative_control_result = {
            "passed": bool(flags.get("copy_cheat") or flags.get("freeze_cheat") or flags.get("prompt_collapse")),
            "judge": neg_judge,
            "required": "degraded copied/frozen input must be flagged and must not pass as consistent",
            "synthetic_only": not bool(negatives),
            "source_scene": degraded_source_scene,
            "fixture_kind": synthetic_negative.get("fixture_kind") or "degraded_copy_cheat",
        }

        for arm in ("prompt_only", "kv_only", "both"):
            prompt_dir = original_prompt_dir if arm == "kv_only" else refined_prompt_dir
            numeric_path = output_root / "numeric" / f"numeric_{arm_dirs[arm].name}.json"
            if args.resume and numeric_path.exists():
                cached_numeric = load_json(numeric_path)
                if cached_numeric.get("scene_role_source") == "admission_negative_controls_v2":
                    numeric_results[arm] = cached_numeric
                    continue
                print(
                    f"[vlm-ablation] recomputing stale numeric cache for {arm}: "
                    "scene_role_source is missing or old"
                )
            numeric_results[arm] = evaluate_numeric_arm(
                baseline_dir=baseline_dir,
                arm_dir=arm_dirs[arm],
                prompt_dir=prompt_dir,
                scenes=admitted,
                negative_controls=negatives,
                admission=admission,
                args=args,
                num_blocks=num_blocks,
                output_dir=output_root / "numeric",
            )

    except Exception as exc:
        blocked_reason = str(exc)
    finally:
        qwen_proc = qwen_state["proc"]
        if qwen_proc is not None and qwen_proc.poll() is None:
            qwen_proc.terminate()
            try:
                qwen_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                qwen_proc.kill()
                qwen_proc.wait(timeout=30)

    ledger = build_ledger(
        args=args,
        output_root=output_root,
        video_root=video_root,
        judge_validation=judge_validation,
        gpu=gpu,
        qwen_meta=qwen_meta,
        optimizer_decisions=optimizer_decisions,
        prompt_diffs=prompt_diffs,
        anchor_plan=anchor_plan,
        generation=generation,
        judge_results=judge_results,
        numeric_results=numeric_results,
        negative_control=negative_control_result,
        admission=admission,
        started=started,
        blocked_reason=blocked_reason,
    )
    write_json(output_root / "ledger.json", make_json_safe(ledger))
    write_json(output_root / "optimizer_decisions.json", make_json_safe(optimizer_decisions))
    write_json(output_root / "judge_results.json", make_json_safe(judge_results))
    write_json(output_root / "generation.json", make_json_safe(generation))
    print(f"[vlm-ablation] wrote {Path(output_root / 'ledger.json').resolve()}")
    if blocked_reason:
        print(f"[vlm-ablation] BLOCKED/NULL: {blocked_reason}")
        return 1
    print(f"[vlm-ablation] RESULT: {'PASS' if ledger['pass'] else 'NULL'} winner={ledger.get('winner')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
