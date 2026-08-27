#!/usr/bin/env python3
"""Evaluate the v2 G0 probe-transparency gate and brown-bear diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.video_consistency import shot_anchor_centroid_consistency
from evaluation.vbench_consistency import build_clip_image_encoder, build_dino_encoder


SCENES = (
    "bearded_watchmaker_workshop",
    "tattooed_noodle_chef",
    "brown_bear_river",
    "sandy_beach_driftwood",
)
LAYERS = (0, 7, 14, 21, 29)
ARCHIVE = ROOT / "docs/round18_bench_20260701/iter01/numeric/numeric_prompt_only.json"
ARCHIVE_SANITY_TOLERANCE = 1e-2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shot_ranges(scene: str) -> list[tuple[int, int]]:
    durations = [
        int(x)
        for x in (
            ROOT / "example/benchmark_hard_multishot" / scene / "shot_durations.txt"
        ).read_text().split()
    ]
    ranges = []
    start = 0
    for chunks in durations:
        end = start + chunks * 8
        ranges.append((start, end))
        start = end
    return ranges


def sampled_indices(ranges: list[tuple[int, int]], count: int = 3) -> list[int]:
    indices = []
    for start, end in ranges:
        take = max(1, min(count, end - start))
        indices.extend(np.linspace(start, end - 1, num=take).round().astype(int).tolist())
    return indices


def decode_selected(path: Path, indices: list[int]) -> tuple[np.ndarray, dict]:
    wanted = set(indices)
    selected = {}
    digest = hashlib.sha256()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    frame_index = 0
    height = width = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        digest.update(rgb.tobytes())
        height, width = rgb.shape[:2]
        if frame_index in wanted:
            selected[frame_index] = rgb
        frame_index += 1
    cap.release()
    missing = sorted(wanted - set(selected))
    if missing:
        raise RuntimeError(f"video {path} is missing requested frames {missing}")
    return np.stack([selected[index] for index in indices]), {
        "decoded_frame_sha256": digest.hexdigest(),
        "frames": frame_index,
        "height": height,
        "width": width,
    }


def normalized_shot_embeddings(
    frames: np.ndarray,
    encoder,
    shots: int,
    frames_per_shot: int = 3,
) -> np.ndarray:
    features = np.asarray(encoder(frames), dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != frames.shape[0]:
        raise RuntimeError(
            f"encoder returned {features.shape} for {frames.shape[0]} sampled frames"
        )
    features /= np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)
    pooled = []
    for shot in range(shots):
        group = features[shot * frames_per_shot : (shot + 1) * frames_per_shot]
        mean = group.mean(axis=0)
        pooled.append(mean / max(np.linalg.norm(mean), 1e-8))
    return np.stack(pooled)


def per_shot_to_first(features: np.ndarray) -> list[float]:
    normalized = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)
    return [float(value) for value in normalized @ normalized[0]]


def classify_bear_hashes(archive_sha: str, probe_on_sha: str, probe_off_sha: str) -> str:
    """Return the v2 brown-bear decision-tree branch."""
    if probe_off_sha == probe_on_sha:
        return "ALL_EQUAL" if probe_off_sha == archive_sha else "A_STALE_ARCHIVE"
    if probe_off_sha == archive_sha:
        return "B_PROBE_PERTURBATION"
    return "C_SCENE_NONDETERMINISM"


def write_comparison_visual(
    output_dir: Path,
    scene: str,
    left_frames: np.ndarray,
    right_frames: np.ndarray,
    sampled_frame_indices: list[int],
    left_streams: dict[str, list[float]],
    right_streams: dict[str, list[float]],
    *,
    left_label: str,
    right_label: str,
    filename: str,
) -> str:
    """Write the sampled frame from the shot with the largest stream delta."""
    candidates = []
    for stream in ("subject_anchor_to_shot0", "background_anchor_to_shot0"):
        for shot, (left, right) in enumerate(
            zip(left_streams[stream], right_streams[stream])
        ):
            candidates.append((abs(left - right), stream, shot, left, right))
    delta, stream, shot, left_value, right_value = max(candidates)
    sample_offset = shot * 3 + 1
    frame_index = sampled_frame_indices[sample_offset]
    left = cv2.cvtColor(left_frames[sample_offset], cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(right_frames[sample_offset], cv2.COLOR_RGB2BGR)
    panels = []
    for label, frame in ((left_label, left), (right_label, right)):
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 54), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (18, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    canvas = np.concatenate(panels, axis=1)
    footer = np.zeros((72, canvas.shape[1], 3), dtype=np.uint8)
    stream_label = "DINO subject" if stream.startswith("subject") else "CLIP background"
    text = (
        f"{scene} | shot {shot} | frame {frame_index} | {stream_label} "
        f"left={left_value:.4f} right={right_value:.4f} abs_delta={delta:.4f}"
    )
    cv2.putText(
        footer,
        text,
        (18, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output = np.concatenate((canvas, footer), axis=0)
    path = output_dir / filename
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"could not write G0 failure visual: {path}")
    return str(path.relative_to(ROOT))


def locate_video(folder: Path, scene: str) -> Path:
    matches = sorted(folder.glob(f"*-{scene}-*_regular.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {scene} video under {folder}, found {matches}")
    return matches[0]


def archive_records() -> dict[str, dict]:
    payload = json.loads(ARCHIVE.read_text(encoding="utf-8"))
    return {
        record["theme"]: record["baseline_metrics"]
        for record in payload["comparison"]["records"]
        if record.get("theme") in SCENES
    }


def dump_integrity(dump_root: Path, scene: str) -> dict:
    sample_root = dump_root / scene / "seed0"
    manifest_path = sample_root / "manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "reason": f"missing {manifest_path}"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layer_counts = {
        str(layer): len(list((sample_root / "v" / f"layer_{layer:02d}").glob("*.pt")))
        for layer in LAYERS
    }
    bytes_total = sum(path.stat().st_size for path in sample_root.rglob("*.pt"))
    expected_chunks = 60
    missing_subject_chunks = 6 if scene == "sandy_beach_driftwood" else 0
    expected_per_layer = expected_chunks - missing_subject_chunks
    counts_ok = all(count == expected_per_layer for count in layer_counts.values())
    stats = manifest.get("stats", {})
    probe_config = manifest.get("probe_config", {})
    provenance = manifest.get("mask_provenance", {})
    mask_method_ok = probe_config.get("mask_method") == "clipseg_precomputed"
    provenance_ok = bool(
        provenance.get("method") == "transformers_CLIPSeg_precomputed"
        and provenance.get("model") == "CIDAS/clipseg-rd64-refined"
        and provenance.get("scene") == scene
        and int(provenance.get("source_seed", -1)) == 0
        and provenance.get("source_video_sha256")
    )
    passed = bool(
        counts_ok
        and not stats.get("errors")
        and int(stats.get("cross_attention_calls", -1)) == 0
        and int(stats.get("mask_files", -1)) == expected_chunks
        and mask_method_ok
        and provenance_ok
        and bytes_total < 4 * 1024**3
    )
    return {
        "passed": passed,
        "layer_file_counts": layer_counts,
        "expected_files_per_layer": expected_per_layer,
        "bytes": bytes_total,
        "gib": bytes_total / 1024**3,
        "runtime_wall_seconds": stats.get("wall_seconds"),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "stats": stats,
        "mask_method_ok": mask_method_ok,
        "mask_provenance_ok": provenance_ok,
        "mask_provenance": provenance,
        "explicit_empty_mask_shots": [2] if scene == "sandy_beach_driftwood" else [],
    }


def write_report(output_dir: Path, result: dict) -> None:
    lines = [
        "# G0 v2 probe-transparency gate",
        "",
        f"Status: **{result['gate_status'].upper()}**",
        "",
        "The v1 G0 verdict is superseded. G0 infrastructure can pass or block; it cannot produce `verdict=kill`.",
        "The hard gate is decoded-frame SHA-256 equality between probe-on and probe-off. For the three prevalidated scenes, the frozen stage_A artifact is the probe-off reference. Brown bear uses the v2 same-code diagnostic render.",
        "Round18 metric recomputation is a non-gating sanity check at tolerance 1e-2 because the encoder environment can drift (including the observed OpenCLIP QuickGELU mismatch warning).",
        "",
        "| scene | reference | probe transparency SHA | per-shot sanity delta | Round18 sanity delta | dump GiB | pass |",
        "|---|---|:---:|---:|---:|---:|:---:|",
    ]
    for scene, record in result["scenes"].items():
        lines.append(
            f"| {scene} | {record['reference_kind']} "
            f"| {'yes' if record['probe_transparency_exact'] else 'no'} "
            f"| {record['max_per_shot_abs_delta']:.3g} "
            f"| {record['archive_aggregate_max_abs_delta']:.3g} "
            f"| {record['dump_integrity'].get('gib', float('nan')):.3f} "
            f"| {'yes' if record['passed'] else 'no'} |"
        )
    lines += [
        "",
        f"Round18 sanity tolerance: `{result['archive_sanity_tolerance']}` (reported only; not a gate).",
        f"Overhead: **{result['overhead']['status']}** — {result['overhead']['reason']}",
        "Driftwood shot 2 has no driftwood identity phrase or object in the frozen prompt, so it is preserved as an explicit empty mask rather than relabeled.",
    ]
    bear = result.get("bear_diagnostic") or {}
    if bear:
        lines += [
            "",
            "Brown-bear decision tree:",
            "",
            f"- Branch: **{bear.get('branch')}**.",
            f"- stage_A archive SHA: `{bear.get('archive_decoded_sha256')}`.",
            f"- probe-on SHA: `{bear.get('probe_on_decoded_sha256')}`.",
            f"- probe-off SHA: `{bear.get('probe_off_decoded_sha256')}`.",
            f"- Resolution: {bear.get('resolution')}",
        ]
        if bear.get("stage_a_mask_on_current_probe_visual"):
            mask_visual = Path(bear["stage_a_mask_on_current_probe_visual"])
            lines.append(
                f"- Mask alignment QC: [{mask_visual.name}](./{mask_visual.name})."
            )
    comparison_visuals = [
        (scene, record.get("comparison_visual"))
        for scene, record in result["scenes"].items()
        if record.get("comparison_visual")
    ]
    if comparison_visuals:
        lines += ["", "Diagnostic visual evidence:", ""]
        lines.extend(
            f"- `{scene}`: [{Path(path).name}](./{Path(path).name})"
            for scene, path in comparison_visuals
        )
    (output_dir / "G0_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot_root", default="videos/capsule_pilot_20260715"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--archive_sanity_tolerance", type=float, default=ARCHIVE_SANITY_TOLERANCE
    )
    args = parser.parse_args()

    pilot_root = (ROOT / args.pilot_root).resolve()
    output_dir = pilot_root / "g0"
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_json = output_dir / "g0_results.json"
    legacy_report = output_dir / "G0_REPORT.md"
    legacy_snapshot_json = output_dir / "g0_results.v1-superseded.json"
    legacy_snapshot_report = output_dir / "G0_REPORT.v1-superseded.md"
    if legacy_json.exists() and not legacy_snapshot_json.exists():
        payload = json.loads(legacy_json.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 1)) == 1:
            legacy_snapshot_json.write_bytes(legacy_json.read_bytes())
            if legacy_report.exists():
                legacy_snapshot_report.write_text(
                    "# SUPERSEDED BY G0 v2\n\n"
                    "This v1 gate result is retained for audit history and must not be interpreted as the current verdict.\n\n"
                    + legacy_report.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
    subject_encoder = build_dino_encoder(device=args.device)
    background_encoder = build_clip_image_encoder(device=args.device)
    archived = archive_records()
    scene_results = {}

    for scene in SCENES:
        ranges = shot_ranges(scene)
        indices = sampled_indices(ranges)
        baseline_path = locate_video(
            ROOT / "videos/benchmark_hard_multishot/stage_A/baseline_seed0", scene
        )
        probe_on_path = locate_video(pilot_root / "probe_replays/seed0/renders", scene)
        archive_frames, archive_decode = decode_selected(baseline_path, indices)
        probe_on_frames, probe_on_decode = decode_selected(probe_on_path, indices)
        if scene == "brown_bear_river":
            reference_path = locate_video(pilot_root / "g0/bear_probe_off/renders", scene)
            reference_kind = "same-code probe-off diagnostic"
            reference_frames, reference_decode = decode_selected(reference_path, indices)
            branch = classify_bear_hashes(
                archive_decode["decoded_frame_sha256"],
                probe_on_decode["decoded_frame_sha256"],
                reference_decode["decoded_frame_sha256"],
            )
        else:
            reference_path = baseline_path
            reference_kind = "frozen stage_A probe-off artifact"
            reference_frames, reference_decode = archive_frames, archive_decode
            branch = "PREVALIDATED_ARCHIVE_REFERENCE"
        shots = len(ranges)
        archive_subject = normalized_shot_embeddings(
            archive_frames, subject_encoder, shots
        )
        archive_background = normalized_shot_embeddings(
            archive_frames, background_encoder, shots
        )
        if reference_decode["decoded_frame_sha256"] == archive_decode["decoded_frame_sha256"]:
            reference_subject = archive_subject.copy()
            reference_background = archive_background.copy()
        else:
            reference_subject = normalized_shot_embeddings(
                reference_frames, subject_encoder, shots
            )
            reference_background = normalized_shot_embeddings(
                reference_frames, background_encoder, shots
            )
        if probe_on_decode["decoded_frame_sha256"] == reference_decode["decoded_frame_sha256"]:
            probe_on_subject = reference_subject.copy()
            probe_on_background = reference_background.copy()
        else:
            probe_on_subject = normalized_shot_embeddings(
                probe_on_frames, subject_encoder, shots
            )
            probe_on_background = normalized_shot_embeddings(
                probe_on_frames, background_encoder, shots
            )
        archive_streams = {
            "subject_anchor_to_shot0": per_shot_to_first(archive_subject),
            "background_anchor_to_shot0": per_shot_to_first(archive_background),
        }
        reference_streams = {
            "subject_anchor_to_shot0": per_shot_to_first(reference_subject),
            "background_anchor_to_shot0": per_shot_to_first(reference_background),
        }
        probe_on_streams = {
            "subject_anchor_to_shot0": per_shot_to_first(probe_on_subject),
            "background_anchor_to_shot0": per_shot_to_first(probe_on_background),
        }
        probe_transparency_exact = (
            reference_decode["decoded_frame_sha256"]
            == probe_on_decode["decoded_frame_sha256"]
        )
        per_shot_deltas = [
            abs(a - b)
            for key in reference_streams
            for a, b in zip(reference_streams[key], probe_on_streams[key])
        ]
        archive_aggregate = shot_anchor_centroid_consistency(
            archive_subject, archive_background
        )
        reference_aggregate = shot_anchor_centroid_consistency(
            reference_subject, reference_background
        )
        probe_on_aggregate = shot_anchor_centroid_consistency(
            probe_on_subject, probe_on_background
        )
        archive_keys = (
            "subject_anchor_consistency",
            "background_anchor_consistency",
            "anchor_centroid_consistency",
            "subject_anchor_to_shot0_consistency",
            "background_anchor_to_shot0_consistency",
            "anchor_to_shot0_consistency",
            "subject_worst_to_shot0_consistency",
            "background_worst_to_shot0_consistency",
            "worst_anchor_to_shot0_consistency",
            "anchor_drift_aggregate_consistency",
        )
        archive_deltas = {
            key: archive_aggregate[key] - float(archived[scene][key])
            for key in archive_keys
        }
        integrity = dump_integrity(
            pilot_root / "probe_replays/seed0/dumps", scene
        )
        max_per_shot = max(per_shot_deltas)
        max_archive = max(abs(value) for value in archive_deltas.values())
        passed = bool(
            reference_decode["frames"] == probe_on_decode["frames"] == 480
            and probe_transparency_exact
            and integrity.get("passed", False)
        )
        comparison_visual = None
        if scene == "brown_bear_river" and branch == "A_STALE_ARCHIVE":
            comparison_visual = write_comparison_visual(
                output_dir,
                scene,
                archive_frames,
                reference_frames,
                indices,
                archive_streams,
                reference_streams,
                left_label="SUPERSEDED STAGE_A ARCHIVE",
                right_label="CURRENT PROBE-OFF REFERENCE",
                filename="brown_bear_river_archive_mismatch.png",
            )
        elif not probe_transparency_exact:
            comparison_visual = write_comparison_visual(
                output_dir,
                scene,
                reference_frames,
                probe_on_frames,
                indices,
                reference_streams,
                probe_on_streams,
                left_label="PROBE OFF REFERENCE",
                right_label="PROBE ON",
                filename=f"{scene}_probe_transparency_failure.png",
            )
        scene_results[scene] = {
            "passed": passed,
            "reference_kind": reference_kind,
            "reference": str(reference_path.relative_to(ROOT)),
            "probe_on": str(probe_on_path.relative_to(ROOT)),
            "stage_a_archive": str(baseline_path.relative_to(ROOT)),
            "reference_decode": reference_decode,
            "probe_on_decode": probe_on_decode,
            "stage_a_archive_decode": archive_decode,
            "probe_transparency_exact": probe_transparency_exact,
            "decoded_frames_exact": probe_transparency_exact,
            "decision_branch": branch,
            "comparison_visual": comparison_visual,
            "failure_visual": comparison_visual if not passed else None,
            "sampled_frame_indices": indices,
            "reference_per_shot": reference_streams,
            "probe_on_per_shot": probe_on_streams,
            "max_per_shot_abs_delta": max_per_shot,
            "reference_aggregate": reference_aggregate,
            "probe_on_aggregate": probe_on_aggregate,
            "stage_a_archive_aggregate": archive_aggregate,
            "round18_archive_baseline": {
                key: archived[scene][key] for key in archive_keys
            },
            "archive_aggregate_delta": archive_deltas,
            "archive_aggregate_max_abs_delta": max_archive,
            "archive_sanity_pass": max_archive <= args.archive_sanity_tolerance,
            "dump_integrity": integrity,
        }

    bear_record = scene_results["brown_bear_river"]
    branch = bear_record["decision_branch"]
    bear_mask_alignment_path = (
        output_dir / "brown_bear_current_probe_on_stage_a_mask.png"
    )
    bear_mask_alignment_visual = (
        str(bear_mask_alignment_path.relative_to(ROOT))
        if bear_mask_alignment_path.exists()
        else None
    )
    resolution = {
        "ALL_EQUAL": "Current probe-off, probe-on, and frozen archive all match; use the frozen archive reference.",
        "A_STALE_ARCHIVE": "Probe-off equals probe-on while both differ from the frozen archive. The archive is superseded for bear; the same-code probe-off render is the new paired reference.",
        "B_PROBE_PERTURBATION": "Probe-off equals the archive but differs from probe-on. G0 is infrastructure-blocked until the probe hook is fixed and probe-on is rerun.",
        "C_SCENE_NONDETERMINISM": "Probe-off matches neither archive nor probe-on. G0 is infrastructure-blocked pending determinism diagnosis or african_savanna substitution.",
    }[branch]
    gate_pass = all(record["passed"] for record in scene_results.values())
    result = {
        "schema_version": 2,
        "gate": "G0",
        "v1_verdict_superseded": True,
        "gate_pass": gate_pass,
        "gate_status": "pass" if gate_pass else "blocked",
        "archive_sanity_tolerance": args.archive_sanity_tolerance,
        "archive_sanity_is_gating": False,
        "overhead": {
            "status": "exempt",
            "is_gating": False,
            "reason": "The v1 cross-day wall-clock comparison was invalid on a shared host. No same-GPU adjacent probe-on/off timing pair was run for v2.",
        },
        "bear_diagnostic": {
            "branch": branch,
            "archive_decoded_sha256": bear_record["stage_a_archive_decode"]["decoded_frame_sha256"],
            "probe_on_decoded_sha256": bear_record["probe_on_decode"]["decoded_frame_sha256"],
            "probe_off_decoded_sha256": bear_record["reference_decode"]["decoded_frame_sha256"],
            "resolution": resolution,
            "manifest": "videos/capsule_pilot_20260715/g0/bear_diagnostic_manifest.json",
            "stage_a_mask_on_current_probe_visual": bear_mask_alignment_visual,
        },
        "blocked_reasons": [] if gate_pass else [resolution],
        "scenes": scene_results,
    }
    diagnostic_manifest_path = output_dir / "bear_diagnostic_manifest.json"
    diagnostic_manifest = json.loads(
        diagnostic_manifest_path.read_text(encoding="utf-8")
    )
    diagnostic_manifest["decision"] = {
        "branch": branch,
        "resolution": resolution,
        "gate_pass": bear_record["passed"],
        "stage_a_archive": bear_record["stage_a_archive"],
        "stage_a_archive_file_sha256": file_sha256(
            ROOT / bear_record["stage_a_archive"]
        ),
        "stage_a_archive_decoded_sha256": result["bear_diagnostic"][
            "archive_decoded_sha256"
        ],
        "probe_on": bear_record["probe_on"],
        "probe_on_file_sha256": file_sha256(ROOT / bear_record["probe_on"]),
        "probe_on_decoded_sha256": result["bear_diagnostic"][
            "probe_on_decoded_sha256"
        ],
        "probe_off": bear_record["reference"],
        "probe_off_file_sha256": file_sha256(ROOT / bear_record["reference"]),
        "probe_off_decoded_sha256": result["bear_diagnostic"][
            "probe_off_decoded_sha256"
        ],
        "p1_p2_reference": (
            bear_record["reference"] if bear_record["passed"] else None
        ),
        "stage_a_mask_on_current_probe_visual": bear_mask_alignment_visual,
    }
    diagnostic_manifest_path.write_text(
        json.dumps(diagnostic_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "g0_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(output_dir, result)
    print(
        json.dumps(
            {
                "gate_pass": result["gate_pass"],
                "gate_status": result["gate_status"],
                "bear_branch": branch,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
