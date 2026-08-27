#!/usr/bin/env python3
"""Finalize replay statuses and verify frozen inputs after P0/P1 dumps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "videos/capsule_pilot_20260715"
LAYERS = (0, 7, 14, 21, 29)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_nonempty_chunks(mask_path: Path, *, chunk_frames: int = 8) -> int:
    payload = torch.load(mask_path, map_location="cpu", weights_only=False)
    binary = payload["binary"].to(torch.bool)
    if binary.ndim != 3 or int(binary.shape[0]) != 480:
        raise RuntimeError(
            f"expected a [480,H,W] precomputed mask at {mask_path}, got {tuple(binary.shape)}"
        )
    return sum(
        bool(binary[start : start + chunk_frames].any())
        for start in range(0, int(binary.shape[0]), chunk_frames)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output")
    args = parser.parse_args()
    if any(seed not in (0, 1, 2) for seed in args.seeds):
        raise ValueError("seeds must be a subset of {0,1,2}")
    manifest_path = PILOT_ROOT / "probe_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_runs = []
    for record in manifest["runs"]:
        seed = int(record["seed"])
        if seed not in args.seeds:
            continue
        scene = record["scene"]
        stem = f"probe_seed{seed}__{scene}"
        exit_path = PILOT_ROOT / "logs" / f"{stem}.exit_code"
        runtime_manifest_path = (
            ROOT
            / record["dump_dir"]
            / scene
            / f"seed{seed}"
            / "manifest.json"
        )
        render_dir = ROOT / record["render_dir"]
        videos = sorted(render_dir.glob(f"probe_seed{seed}-rank0-{scene}-*_regular.mp4"))
        exit_code = (
            int(exit_path.read_text().strip()) if exit_path.exists() else None
        )
        runtime_manifest = (
            json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
            if runtime_manifest_path.exists()
            else None
        )
        errors = (runtime_manifest or {}).get("stats", {}).get("errors", [])
        stats = (runtime_manifest or {}).get("stats", {})
        probe_config = (runtime_manifest or {}).get("probe_config", {})
        mask_path_value = probe_config.get("mask_path")
        mask_path = ROOT / mask_path_value if mask_path_value else None
        expected_per_layer = (
            expected_nonempty_chunks(mask_path)
            if mask_path is not None and mask_path.exists()
            else None
        )
        layer_file_counts = {
            str(layer): len(
                list(
                    (
                        runtime_manifest_path.parent
                        / "v"
                        / f"layer_{layer:02d}"
                    ).glob("*.pt")
                )
            )
            for layer in LAYERS
        }
        bytes_total = sum(
            path.stat().st_size
            for path in runtime_manifest_path.parent.rglob("*.pt")
        ) if runtime_manifest_path.parent.exists() else 0
        dump_integrity = bool(
            runtime_manifest
            and expected_per_layer is not None
            and all(
                count == expected_per_layer
                for count in layer_file_counts.values()
            )
            and int(stats.get("dump_files", -1))
            == expected_per_layer * len(LAYERS)
            and int(stats.get("mask_files", -1)) == 60
            and int(stats.get("cross_attention_calls", -1)) == 0
            and not errors
            and bytes_total < 4 * 1024**3
        )
        complete = bool(
            exit_code == 0
            and len(videos) == 1
            and runtime_manifest
            and dump_integrity
        )
        record["status"] = "complete" if complete else "failed_or_missing"
        record["exit_code_file"] = str(exit_path.relative_to(ROOT))
        record["runtime_manifest"] = str(runtime_manifest_path.relative_to(ROOT))
        record["render"] = (
            str(videos[0].relative_to(ROOT)) if len(videos) == 1 else None
        )
        audit_runs.append(
            {
                "seed": seed,
                "scene": scene,
                "complete": complete,
                "exit_code": exit_code,
                "video_count": len(videos),
                "runtime_manifest_exists": runtime_manifest is not None,
                "runtime_stats": stats,
                "dump_integrity": {
                    "passed": dump_integrity,
                    "mask_path": mask_path_value,
                    "expected_files_per_layer": expected_per_layer,
                    "layer_file_counts": layer_file_counts,
                    "bytes": bytes_total,
                    "gib": bytes_total / 1024**3,
                },
            }
        )

    frozen_expected = json.loads(
        (PILOT_ROOT / "frozen_inputs_sha256.json").read_text(encoding="utf-8")
    )
    frozen_mismatches = []
    for relative, expected in frozen_expected.items():
        path = ROOT / relative
        actual = sha256(path) if path.exists() else None
        if actual != expected:
            frozen_mismatches.append(
                {"path": relative, "expected": expected, "actual": actual}
            )
    expected_cells = {
        (seed, scene)
        for seed in args.seeds
        for scene in (
            "bearded_watchmaker_workshop",
            "tattooed_noodle_chef",
            "brown_bear_river",
            "sandy_beach_driftwood",
        )
    }
    actual_cells = [(run["seed"], run["scene"]) for run in audit_runs]
    exact_cell_matrix = bool(
        len(actual_cells) == len(expected_cells)
        and len(set(actual_cells)) == len(actual_cells)
        and set(actual_cells) == expected_cells
    )
    all_complete = bool(
        exact_cell_matrix and all(run["complete"] for run in audit_runs)
    )
    passed = all_complete and not frozen_mismatches
    audit_key = "all" if sorted(args.seeds) == [0, 1, 2] else "_".join(
        f"seed{seed}" for seed in sorted(args.seeds)
    )
    manifest.setdefault("replay_audits", {})[audit_key] = {
        "passed": passed,
        "all_complete": all_complete,
        "exact_cell_matrix": exact_cell_matrix,
        "frozen_inputs_unchanged": not frozen_mismatches,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    audit = {
        "schema_version": 1,
        "passed": passed,
        "all_complete": all_complete,
        "exact_cell_matrix": exact_cell_matrix,
        "expected_cells": len(expected_cells),
        "frozen_inputs_unchanged": not frozen_mismatches,
        "frozen_input_mismatches": frozen_mismatches,
        "runs": audit_runs,
    }
    output_name = args.output or (
        "probe_replay_audit.json"
        if audit_key == "all"
        else f"probe_replay_audit_{audit_key}.json"
    )
    output_path = PILOT_ROOT / output_name
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passed": passed, "complete": sum(r["complete"] for r in audit_runs)}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
