#!/usr/bin/env python3
"""Audit all 12 final P2 arm×scene cells and recheck frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    manifest_path = pilot_root / "p2/p2_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for run in manifest["runs"]:
        arm, scene = run["arm"], run["scene"]
        config_path = ROOT / run["config"]
        videos = sorted((ROOT / run["render_dir"]).glob(f"*-{scene}-*_regular.mp4"))
        runtime_path = ROOT / run["runtime_dir"] / scene / "seed0/manifest.json"
        runtime = (
            json.loads(runtime_path.read_text(encoding="utf-8"))
            if runtime_path.exists()
            else None
        )
        reused = str(run.get("status", "")).startswith("complete_reused")
        stem = config_path.stem
        exit_path = pilot_root / f"p2/logs/{stem}.exit_code"
        exit_code = (
            int(exit_path.read_text().strip()) if exit_path.exists() else None
        )
        execution_ok = reused or exit_code == 0
        errors = (runtime or {}).get("stats", {}).get("errors", [])
        complete = bool(
            execution_ok
            and config_path.exists()
            and len(videos) == 1
            and runtime is not None
            and not errors
        )
        run["status"] = "complete_reused_from_rendered_alpha_scan" if reused and complete else "complete" if complete else "failed_or_missing"
        run["exit_code_file"] = None if reused else str(exit_path.relative_to(ROOT))
        run["render"] = str(videos[0].relative_to(ROOT)) if len(videos) == 1 else None
        run["runtime_manifest"] = str(runtime_path.relative_to(ROOT))
        run["config_sha256"] = sha256(config_path) if config_path.exists() else None
        run["render_sha256"] = sha256(videos[0]) if len(videos) == 1 else None
        records.append(
            {
                "arm": arm,
                "scene": scene,
                "complete": complete,
                "reused_rendered_alpha_scan": reused,
                "exit_code": exit_code,
                "video_count": len(videos),
                "runtime_manifest_exists": runtime is not None,
                "runtime_errors": errors,
            }
        )

    frozen = json.loads(
        (pilot_root / "frozen_inputs_sha256.json").read_text(encoding="utf-8")
    )
    mismatches = []
    for relative, expected in frozen.items():
        path = ROOT / relative
        actual = sha256(path) if path.exists() else None
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    expected_cells = {
        (arm, scene)
        for arm in ("capsule", "full_vector", "random_subspace")
        for scene in (
            "bearded_watchmaker_workshop",
            "tattooed_noodle_chef",
            "brown_bear_river",
            "sandy_beach_driftwood",
        )
    }
    actual_cells = [(record["arm"], record["scene"]) for record in records]
    exact_cell_matrix = bool(
        len(actual_cells) == len(expected_cells)
        and len(set(actual_cells)) == len(actual_cells)
        and set(actual_cells) == expected_cells
    )
    passed = bool(
        exact_cell_matrix
        and all(record["complete"] for record in records)
        and not mismatches
    )
    manifest["audit"] = {
        "passed": passed,
        "complete_cells": sum(record["complete"] for record in records),
        "exact_cell_matrix": exact_cell_matrix,
        "frozen_inputs_unchanged": not mismatches,
        "path": str((pilot_root / "p2/p2_run_audit.json").relative_to(ROOT)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    audit = {
        "schema_version": 1,
        "gate": "P2_run_completeness_and_frozen_inputs",
        "passed": passed,
        "complete_cells": sum(record["complete"] for record in records),
        "expected_cells": 12,
        "exact_cell_matrix": exact_cell_matrix,
        "frozen_inputs_unchanged": not mismatches,
        "frozen_input_mismatches": mismatches,
        "runs": records,
    }
    output = pilot_root / "p2/p2_run_audit.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passed": passed, "complete_cells": audit["complete_cells"]}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
