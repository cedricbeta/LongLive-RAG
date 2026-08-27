#!/usr/bin/env python3
"""Offline Track-W test: can boundary telemetry predict next-shot DINO drift?

No rendering is performed.  The script reads the frozen 4-scene x 3-seed capsule probe
dumps/videos, computes only signals that the artifacts actually contain, runs
leave-one-scene-out regressions, and cluster-bootstraps the paired MAE improvement over a
shot-index-only baseline.  Missing requested telemetry fails the Track-W claim and selects
the preregistered fixed-boundary/full-verification fallback even if an available proxy is
numerically predictive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.vbench_consistency import build_dino_encoder


SCENES = (
    "bearded_watchmaker_workshop",
    "tattooed_noodle_chef",
    "brown_bear_river",
    "sandy_beach_driftwood",
)
SEEDS = (0, 1, 2)
LAYERS = (0, 7, 14, 21, 29)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shot_ranges(scene: str) -> list[tuple[int, int]]:
    durations = [
        int(value)
        for value in (ROOT / "example/benchmark_hard_multishot" / scene / "shot_durations.txt")
        .read_text(encoding="utf-8")
        .split()
    ]
    ranges: list[tuple[int, int]] = []
    start = 0
    for chunks in durations:
        end = start + chunks * 8
        ranges.append((start, end))
        start = end
    return ranges


def sampled_indices(scene: str) -> tuple[list[int], list[int]]:
    indices, shots = [], []
    for shot, (start, end) in enumerate(shot_ranges(scene)):
        selected = np.linspace(start, end - 1, num=3).round().astype(int).tolist()
        indices.extend(selected)
        shots.extend([shot] * len(selected))
    return indices, shots


def locate_video(pilot_root: Path, scene: str, seed: int) -> Path:
    matches = sorted(
        (pilot_root / f"probe_replays/seed{seed}/renders").glob(
            f"*-{scene}-*_regular.mp4"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one probe video for {scene}/seed{seed}, found {matches}")
    return matches[0]


def decode_selected(path: Path, indices: list[int]) -> np.ndarray:
    wanted, frames = set(indices), {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    if set(frames) != wanted:
        raise RuntimeError(f"missing frames in {path}: {sorted(wanted - set(frames))}")
    return np.stack([frames[index] for index in indices])


def dino_subject_by_shot(
    *,
    frames: np.ndarray,
    masks: torch.Tensor,
    indices: list[int],
    shots: list[int],
    encoder,
) -> dict[int, np.ndarray]:
    prepared, prepared_shots = [], []
    for frame, frame_index, shot in zip(frames, indices, shots):
        mask = cv2.resize(
            masks[frame_index].numpy().astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if not mask.any():
            continue
        image = frame.copy()
        image[~mask] = 127
        prepared.append(image)
        prepared_shots.append(shot)
    if not prepared:
        return {}
    features = np.asarray(encoder(np.stack(prepared)), dtype=np.float64)
    features /= np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)
    result: dict[int, np.ndarray] = {}
    shot_array = np.asarray(prepared_shots)
    for shot in sorted(set(prepared_shots)):
        mean = features[shot_array == shot].mean(axis=0)
        result[shot] = mean / max(float(np.linalg.norm(mean)), 1e-8)
    return result


def _entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0, None)
    total = float(values.sum())
    if total <= 0 or len(values) <= 1:
        return 0.0
    probs = values / total
    return float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum() / math.log(len(probs)))


def dump_features(sample_root: Path) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    manifest = json.loads((sample_root / "manifest.json").read_text(encoding="utf-8"))
    cross_calls = int(manifest.get("stats", {}).get("cross_attention_calls", -1))
    mask_method = str(manifest.get("probe_config", {}).get("mask_method", ""))
    per_shot_layer: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for layer in LAYERS:
        for path in sorted((sample_root / "v" / f"layer_{layer:02d}").glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            values = payload["v"].float()
            mean_vector = values.mean(dim=0).reshape(-1).numpy().astype(np.float64)
            norms = torch.linalg.vector_norm(values, dim=-1).double().reshape(-1).numpy()
            scores = payload["attention_scores"].float().reshape(-1).numpy().astype(np.float64)
            frames = payload["frame_indices"].long().reshape(-1).numpy()
            per_shot_layer[(int(payload["shot_index"]), layer)].append(
                {
                    "chunk": int(payload["chunk_index"]),
                    "mean_vector": mean_vector,
                    "norm_mean": float(norms.mean()),
                    "norm_std": float(norms.std()),
                    "scores": scores,
                    "frames": frames,
                }
            )

    features: dict[int, dict[str, float]] = {}
    for shot in range(7):  # boundary after this shot, predicting shot+1
        row: dict[str, float] = {"shot_index": float(shot + 1)}
        for layer in LAYERS:
            chunks = sorted(per_shot_layer.get((shot, layer), []), key=lambda item: item["chunk"])
            if not chunks:
                row[f"layer{layer}_v_norm_mean"] = float("nan")
                row[f"layer{layer}_v_norm_cv"] = float("nan")
                row[f"layer{layer}_v_norm_drift"] = float("nan")
                row[f"layer{layer}_v_direction_drift"] = float("nan")
                continue
            norm_means = np.asarray([item["norm_mean"] for item in chunks])
            row[f"layer{layer}_v_norm_mean"] = float(norm_means.mean())
            row[f"layer{layer}_v_norm_cv"] = float(norm_means.std() / max(norm_means.mean(), 1e-8))
            row[f"layer{layer}_v_norm_drift"] = float(
                (norm_means[-1] - norm_means[0]) / max(abs(norm_means[0]), 1e-8)
            )
            first, last = chunks[0]["mean_vector"], chunks[-1]["mean_vector"]
            cosine = float(first @ last / max(np.linalg.norm(first) * np.linalg.norm(last), 1e-8))
            row[f"layer{layer}_v_direction_drift"] = 1.0 - cosine

        # ``attention_scores`` are CLIPSeg salience in these frozen dumps, not
        # model attention (manifest cross_attention_calls == 0).  Preserve the
        # useful per-frame distribution under an honest proxy name.
        chunks = sorted(per_shot_layer.get((shot, 14), []), key=lambda item: item["chunk"])
        frame_mass: dict[int, float] = defaultdict(float)
        for item in chunks:
            for frame, score in zip(item["frames"], item["scores"]):
                frame_mass[int(frame)] += float(score)
        masses = np.asarray([frame_mass[key] for key in sorted(frame_mass)], dtype=np.float64)
        row["segmentation_salience_frame_entropy"] = _entropy(masses)
        row["segmentation_salience_last_frame_share"] = (
            float(masses[-1] / max(masses.sum(), 1e-8)) if masses.size else float("nan")
        )
        features[shot + 1] = row

    availability = {
        "attention_entropy": {
            "available": False,
            "reason": f"cross_attention_calls={cross_calls}; no self-attention weight dump",
        },
        "sink_share": {
            "available": False,
            "reason": "clean-recache entity V payloads contain no attended sink columns",
        },
        "per_frame_mass_distribution": {
            "available": False,
            "proxy_available": True,
            "proxy": "CLIPSeg segmentation-salience distribution",
            "reason": f"mask_method={mask_method}; scores are not model attention mass",
        },
        "v_norm_drift": {"available": True, "layers": list(LAYERS)},
        "identity_word_cross_attention_mass": {
            "available": False,
            "reason": f"cross_attention_calls={cross_calls}; probe used precomputed CLIPSeg masks",
        },
    }
    return features, availability


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float) -> np.ndarray:
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    std[std < 1e-8] = 1.0
    train = np.nan_to_num((train_x - mean) / std)
    test = np.nan_to_num((test_x - mean) / std)
    train = np.column_stack((np.ones(len(train)), train))
    test = np.column_stack((np.ones(len(test)), test))
    penalty = np.eye(train.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(train.T @ train + penalty) @ train.T @ train_y
    return test @ weights


def loso_predictions(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    for scene in SCENES:
        train = [index for index, row in enumerate(rows) if row["scene"] != scene]
        test = [index for index, row in enumerate(rows) if row["scene"] == scene]
        train_x = np.asarray([[rows[i][name] for name in feature_names] for i in train])
        test_x = np.asarray([[rows[i][name] for name in feature_names] for i in test])
        train_y = np.asarray([rows[i]["target_dino_drift"] for i in train])
        predictions[test] = _fit_predict(train_x, train_y, test_x, ridge=1.0)
    return predictions


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    baseline_error: np.ndarray,
    signal_error: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> list[float]:
    clusters = sorted({(row["scene"], row["seed"]) for row in rows})
    by_cluster = {
        cluster: np.asarray(
            [index for index, row in enumerate(rows) if (row["scene"], row["seed"]) == cluster]
        )
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(int(repeats)):
        sampled = [clusters[index] for index in rng.integers(0, len(clusters), len(clusters))]
        indices = np.concatenate([by_cluster[cluster] for cluster in sampled])
        deltas.append(float((baseline_error[indices] - signal_error[indices]).mean()))
    return [float(x) for x in np.quantile(deltas, [0.025, 0.5, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    parser.add_argument("--output", default="videos/agentic_runtime_v1/track_w/results.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap_repeats", type=int, default=5000)
    args = parser.parse_args()

    pilot_root = (ROOT / args.pilot_root).resolve()
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dino = build_dino_encoder(device=args.device)
    rows: list[dict[str, Any]] = []
    availability_records: list[dict[str, Any]] = []
    source_files: dict[str, str] = {}

    for seed in SEEDS:
        for scene in SCENES:
            sample_root = pilot_root / f"probe_replays/seed{seed}/dumps/{scene}/seed{seed}"
            video = locate_video(pilot_root, scene, seed)
            mask_path = pilot_root / f"clipseg_masks/seed{seed}/{scene}.pt"
            manifest_path = sample_root / "manifest.json"
            source_files[str(video.relative_to(ROOT))] = sha256(video)
            source_files[str(mask_path.relative_to(ROOT))] = sha256(mask_path)
            source_files[str(manifest_path.relative_to(ROOT))] = sha256(manifest_path)

            indices, shots = sampled_indices(scene)
            frames = decode_selected(video, indices)
            masks = torch.load(mask_path, map_location="cpu", weights_only=False)["binary"].bool()
            embeddings = dino_subject_by_shot(
                frames=frames, masks=masks, indices=indices, shots=shots, encoder=dino
            )
            features, availability = dump_features(sample_root)
            availability_records.append({"scene": scene, "seed": seed, "signals": availability})
            for boundary in range(1, 8):
                if boundary - 1 not in embeddings or boundary not in embeddings:
                    continue
                target = 1.0 - float(embeddings[boundary - 1] @ embeddings[boundary])
                rows.append(
                    {
                        "scene": scene,
                        "seed": seed,
                        "boundary_shot_index": boundary,
                        "target_dino_drift": target,
                        **features[boundary],
                    }
                )

    signal_names = sorted(
        name
        for name in rows[0]
        if name.startswith("layer") and ("v_norm_" in name or "v_direction_drift" in name)
    )
    target = np.asarray([row["target_dino_drift"] for row in rows])
    baseline_pred = loso_predictions(rows, ["shot_index"])
    signal_pred = loso_predictions(rows, signal_names)
    baseline_error = np.abs(target - baseline_pred)
    signal_error = np.abs(target - signal_pred)
    ci = cluster_bootstrap(
        rows,
        baseline_error,
        signal_error,
        repeats=args.bootstrap_repeats,
        seed=20260716,
    )
    folds = {}
    for scene in SCENES:
        index = np.asarray([i for i, row in enumerate(rows) if row["scene"] == scene])
        folds[scene] = {
            "n": int(len(index)),
            "shot_index_mae": float(baseline_error[index].mean()),
            "available_signal_mae": float(signal_error[index].mean()),
            "mae_improvement": float((baseline_error[index] - signal_error[index]).mean()),
        }

    required_available = {
        name: all(record["signals"][name].get("available", False) for record in availability_records)
        for name in (
            "attention_entropy",
            "sink_share",
            "per_frame_mass_distribution",
            "v_norm_drift",
            "identity_word_cross_attention_mass",
        )
    }
    telemetry_complete = all(required_available.values())
    statistical_pass = bool(ci[0] > 0)
    gate_pass = bool(telemetry_complete and statistical_pass)
    result = {
        "schema_version": 1,
        "track": "W",
        "zero_render": True,
        "scenes": list(SCENES),
        "seeds": list(SEEDS),
        "rows": len(rows),
        "target": "next-shot masked DINO subject drift = 1-cos(previous_shot,current_shot)",
        "required_signal_availability": required_available,
        "availability_records": availability_records,
        "available_model_features": signal_names,
        "evaluation": {
            "protocol": "leave-one-scene-out ridge regression; scene-seed cluster bootstrap",
            "shot_index_baseline_mae": float(baseline_error.mean()),
            "available_signal_mae": float(signal_error.mean()),
            "paired_mae_improvement": float((baseline_error - signal_error).mean()),
            "cluster_bootstrap_improvement_ci95": ci,
            "bootstrap_repeats": args.bootstrap_repeats,
            "folds": folds,
        },
        "telemetry_complete": telemetry_complete,
        "statistical_pass": statistical_pass,
        "gate_pass": gate_pass,
        "decision": "when_supported" if gate_pass else "downgrade_fixed_boundary_full_verification",
        "quality_claim": False,
        "source_sha256": source_files,
        "records": [
            {
                **row,
                "shot_index_prediction": float(baseline_pred[index]),
                "available_signal_prediction": float(signal_pred[index]),
            }
            for index, row in enumerate(rows)
        ],
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("rows", "telemetry_complete", "statistical_pass", "gate_pass", "decision")}, indent=2))
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
