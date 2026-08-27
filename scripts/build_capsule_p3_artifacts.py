#!/usr/bin/env python3
"""Build 4-up videos, shot strips, entity crops, and per-shot metric curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCENES = (
    "bearded_watchmaker_workshop",
    "tattooed_noodle_chef",
    "brown_bear_river",
    "sandy_beach_driftwood",
)
ROWS = ("baseline", "capsule", "full_vector", "random_subspace")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def shot_ranges(scene: str) -> list[tuple[int, int]]:
    durations = [
        int(value)
        for value in (
            ROOT / f"example/benchmark_hard_multishot/{scene}/shot_durations.txt"
        ).read_text().split()
    ]
    result, start = [], 0
    for chunks in durations:
        end = start + chunks * 8
        result.append((start, end))
        start = end
    return result


def locate(folder: Path, scene: str) -> Path:
    matches = sorted(folder.glob(f"*-{scene}-*_regular.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {scene} video under {folder}: {matches}")
    return matches[0]


def g0_reference_video(pilot_root: Path, scene: str) -> Path:
    g0 = json.loads((pilot_root / "g0/g0_results.json").read_text(encoding="utf-8"))
    if not g0.get("gate_pass"):
        raise RuntimeError("P3 requires a passing G0 v2 reference ledger")
    relative = g0["scenes"][scene].get("reference")
    if not relative:
        return locate(
            ROOT / "videos/benchmark_hard_multishot/stage_A/baseline_seed0", scene
        )
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def video_paths(pilot_root: Path, scene: str) -> dict[str, Path]:
    return {
        "baseline": g0_reference_video(pilot_root, scene),
        **{
            arm: locate(pilot_root / f"p2/{arm}/renders", scene)
            for arm in ROWS[1:]
        },
    }


def read_indices(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    wanted = set(indices)
    result = {}
    capture = cv2.VideoCapture(str(path))
    frame = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if frame in wanted:
            result[frame] = image
        frame += 1
    capture.release()
    if set(result) != wanted:
        raise RuntimeError(f"missing frames in {path}: {sorted(wanted - set(result))}")
    return result


def labeled_cell(image: np.ndarray, label: str, width: int = 320, height: int = 176) -> np.ndarray:
    cell = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(cell, (0, 0), (width, 28), (10, 10, 10), -1)
    cv2.putText(
        cell, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (245, 245, 245), 2, cv2.LINE_AA,
    )
    return cell


def letterbox_crop(image: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    canvas = np.full((176, 320, 3), 32, dtype=np.uint8)
    if box is None:
        cv2.putText(
            canvas, "MASK EMPTY", (76, 96), cv2.FONT_HERSHEY_SIMPLEX,
            0.72, (80, 80, 235), 2, cv2.LINE_AA,
        )
        return canvas
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return canvas
    scale = min(320 / crop.shape[1], 176 / crop.shape[0])
    resized = cv2.resize(
        crop,
        (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
        interpolation=cv2.INTER_CUBIC,
    )
    x = (320 - resized.shape[1]) // 2
    y = (176 - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def build_strips(pilot_root: Path, scene: str, paths: dict[str, Path]) -> None:
    ranges = shot_ranges(scene)
    first = [start for start, _ in ranges]
    middle = [start + (end - start) // 2 for start, end in ranges]
    all_indices = sorted(set(first + middle))
    frames = {row: read_indices(paths[row], all_indices) for row in ROWS}
    header = np.full((42, 112 + 8 * 320, 3), 25, dtype=np.uint8)
    for shot in range(8):
        cv2.putText(
            header, f"shot {shot}", (112 + shot * 320 + 118, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.63, (245, 245, 245), 2, cv2.LINE_AA,
        )
    strip_rows = []
    for row in ROWS:
        label = np.full((176, 112, 3), 25, dtype=np.uint8)
        cv2.putText(label, row, (5, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
        cells = [labeled_cell(frames[row][index], f"f{index}") for index in first]
        strip_rows.append(np.concatenate([label, *cells], axis=1))
    shotstrip = np.concatenate([header, *strip_rows], axis=0)
    qualitative = pilot_root / "qualitative"
    qualitative.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(qualitative / f"{scene}_shotstrip.png"), shotstrip)

    mask_payload = torch.load(
        pilot_root / f"clipseg_masks/seed0/{scene}.pt",
        map_location="cpu", weights_only=False,
    )
    mask = mask_payload["binary"].to(torch.bool)
    boxes = []
    baseline_shape = frames["baseline"][middle[0]].shape
    for index in middle:
        up = cv2.resize(
            mask[index].numpy().astype(np.uint8),
            (baseline_shape[1], baseline_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if not up.any():
            boxes.append(None)
            continue
        ys, xs = np.where(up)
        margin_x = max(8, int(round((xs.max() - xs.min() + 1) * 0.12)))
        margin_y = max(8, int(round((ys.max() - ys.min() + 1) * 0.12)))
        boxes.append(
            (
                max(0, int(xs.min()) - margin_x),
                max(0, int(ys.min()) - margin_y),
                min(baseline_shape[1], int(xs.max()) + 1 + margin_x),
                min(baseline_shape[0], int(ys.max()) + 1 + margin_y),
            )
        )
    crop_rows = []
    for row in ROWS:
        label = np.full((176, 112, 3), 25, dtype=np.uint8)
        cv2.putText(label, row, (5, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
        cells = []
        for shot, (index, box) in enumerate(zip(middle, boxes)):
            cell = letterbox_crop(frames[row][index], box)
            cv2.rectangle(cell, (0, 0), (320, 25), (10, 10, 10), -1)
            cv2.putText(cell, f"shot {shot} f{index}", (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)
            cells.append(cell)
        crop_rows.append(np.concatenate([label, *cells], axis=1))
    crops = np.concatenate([header, *crop_rows], axis=0)
    cv2.imwrite(str(qualitative / f"{scene}_entity_crops.png"), crops)


def ffmpeg_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def build_grid(
    pilot_root: Path,
    scene: str,
    paths: dict[str, Path],
    *,
    layer: int,
    rank: int,
    alpha: float,
    edit_boundaries: list[int],
) -> None:
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for row in ROWS:
        command += ["-i", str(paths[row])]
    filters = []
    labels = []
    for index, row in enumerate(ROWS):
        out = f"q{index}"
        label = ffmpeg_escape(row)
        filters.append(
            f"[{index}:v]scale=640:352,"
            f"drawbox=x=0:y=0:w=220:h=38:color=black@0.75:t=fill,"
            f"drawtext=fontfile={FONT}:text='{label}':x=12:y=8:fontsize=24:fontcolor=white[{out}]"
        )
        labels.append(f"[{out}]")
    filters.append(
        "".join(labels)
        + "xstack=inputs=4:layout=0_0|640_0|0_352|640_352[grid0]"
    )
    current = "grid0"
    starts = [start for start, _ in shot_ranges(scene)][1:]
    for index, frame in enumerate(starts):
        shot = index + 1
        edited = shot in edit_boundaries
        color = "red" if edited else "yellow"
        nxt = f"grid{index + 1}"
        text = (
            f"EDIT shot {shot} | layer={layer} alpha={alpha:g} rank={rank}"
            if edited
            else f"shot {shot} cut (not edited)"
        )
        filters.append(
            f"[{current}]drawbox=x=iw/2-5:y=0:w=10:h=ih:color={color}@0.95:t=fill:"
            f"enable='between(n,{frame},{frame + 5})',"
            f"drawbox=x=0:y=ih-48:w=iw:h=48:color=black@0.70:t=fill:"
            f"enable='between(n,{frame},{frame + 23})',"
            f"drawtext=fontfile={FONT}:text='{ffmpeg_escape(text)}':x=20:y=h-38:"
            f"fontsize=26:fontcolor={color}:enable='between(n,{frame},{frame + 23})'[{nxt}]"
        )
        current = nxt
    grids = pilot_root / "grids"
    grids.mkdir(parents=True, exist_ok=True)
    command += [
        "-filter_complex", ";".join(filters), "-map", f"[{current}]",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", str(grids / f"{scene}_4up.mp4"),
    ]
    subprocess.run(command, check=True)


def build_boundary_sheet(
    pilot_root: Path,
    scene: str,
    paths: dict[str, Path],
    edit_boundaries: list[int],
) -> None:
    """Static ±0.5 s companion for the mandatory real-time boundary review."""
    ranges = shot_ranges(scene)
    labeled_indices = []
    for shot in edit_boundaries:
        cut = ranges[shot][0]
        for offset in (-12, -1, 0, 12):
            index = min(ranges[-1][1] - 1, max(0, cut + offset))
            labeled_indices.append((shot, offset, index))
    wanted = sorted({index for _, _, index in labeled_indices})
    frames = {row: read_indices(paths[row], wanted) for row in ROWS}
    cell_w, cell_h, label_w = 160, 88, 112
    header = np.full(
        (38, label_w + len(labeled_indices) * cell_w, 3), 25, dtype=np.uint8
    )
    for column, (shot, offset, _) in enumerate(labeled_indices):
        cv2.putText(
            header,
            f"s{shot}{offset:+d}",
            (label_w + column * cell_w + 48, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    rows = [header]
    for row in ROWS:
        name = np.full((cell_h, label_w, 3), 25, dtype=np.uint8)
        cv2.putText(
            name, row, (5, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            (245, 245, 245), 1, cv2.LINE_AA,
        )
        cells = []
        for shot, offset, index in labeled_indices:
            cell = cv2.resize(
                frames[row][index], (cell_w, cell_h), interpolation=cv2.INTER_AREA
            )
            color = (30, 30, 220) if offset in (-1, 0) else (30, 180, 230)
            cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), color, 2)
            cells.append(cell)
        rows.append(np.concatenate([name, *cells], axis=1))
    output = pilot_root / f"qualitative/{scene}_boundary_qc.png"
    cv2.imwrite(str(output), np.concatenate(rows, axis=0))


def build_curves(pilot_root: Path, evaluation: dict) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13, 15), sharex=True)
    for row, scene in enumerate(SCENES):
        for arm in ROWS[1:]:
            record = evaluation["g3"]["arms"][arm]["per_scene"][scene]
            identity = {int(key): value for key, value in record["identity_consistency_delta_by_shot"].items()}
            background = {int(key): value for key, value in record["background_preservation_by_shot"].items()}
            axes[row, 0].plot(identity.keys(), identity.values(), marker="o", label=arm)
            axes[row, 1].plot(background.keys(), background.values(), marker="o", label=arm)
        axes[row, 0].axhline(0, color="black", lw=1)
        axes[row, 1].axhline(1, color="black", lw=1)
        axes[row, 0].set_ylabel(scene)
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].grid(alpha=0.25)
    axes[0, 0].set_title("DINO subject anchor delta vs baseline")
    axes[0, 1].set_title("CLIP background preservation vs baseline")
    axes[-1, 0].set_xlabel("shot")
    axes[-1, 1].set_xlabel("shot")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    output = pilot_root / "qualitative/per_shot_dino_clip_curves.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    manifest = json.loads(
        (pilot_root / "p2/p2_run_manifest.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (pilot_root / "p2/evaluation/g2_g3_results.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (pilot_root / "p2/p2_run_audit.json").read_text(encoding="utf-8")
    )
    if (
        not audit.get("passed")
        or not audit.get("exact_cell_matrix")
        or int(audit.get("complete_cells", 0)) != 12
    ):
        raise SystemExit("P3 requires a passing exact 12-cell P2 run audit")
    if "g2" not in evaluation or "g3" not in evaluation:
        raise SystemExit("P3 requires complete G2/G3 evaluation records")
    layer = int(manifest["runs"][0]["layer"])
    rank = int(manifest["runs"][0]["rank"])
    alpha = float(manifest["selected_alpha"])
    for scene in SCENES:
        run = next(
            record
            for record in manifest["runs"]
            if record["arm"] == "capsule" and record["scene"] == scene
        )
        paths = video_paths(pilot_root, scene)
        build_strips(pilot_root, scene, paths)
        build_grid(
            pilot_root,
            scene,
            paths,
            layer=layer,
            rank=rank,
            alpha=alpha,
            edit_boundaries=[int(value) for value in run["boundaries"]],
        )
        build_boundary_sheet(
            pilot_root,
            scene,
            paths,
            [int(value) for value in run["boundaries"]],
        )
    build_curves(pilot_root, evaluation)
    human_path = pilot_root / "qualitative/human_verdict.json"
    if not human_path.exists():
        checks = (
            "identity_recovery",
            "composition_action_leakage",
            "motion_quality",
            "boundary_artifacts",
            "random_placebo",
        )
        human = {
            "schema_version": 1,
            "complete": False,
            "reviewer": None,
            "rating_scale": ["better", "same", "worse", "broken"],
            "required_view_order": [
                "masks/<scene>_mask_qc.png",
                "qualitative/<scene>_entity_crops.png",
                "qualitative/<scene>_shotstrip.png",
                "qualitative/<scene>_boundary_qc.png",
                "grids/<scene>_4up.mp4 at real-time speed",
            ],
            "scenes": {
                scene: {
                    check: {"rating": None, "evidence": None}
                    for check in checks
                }
                for scene in SCENES
            },
            "overall_verdict": None,
            "numeric_human_conflict": None,
            "rationale": None,
        }
        human_path.write_text(
            json.dumps(human, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(pilot_root.relative_to(ROOT))


if __name__ == "__main__":
    main()
