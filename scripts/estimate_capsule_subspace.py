#!/usr/bin/env python3
"""Estimate and gate the typed-memory identity subspace from P0/P1 V dumps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
SCENES = (
    "bearded_watchmaker_workshop",
    "tattooed_noodle_chef",
    "brown_bear_river",
    "sandy_beach_driftwood",
)
LAYERS = (0, 7, 14, 21, 29)


@dataclass
class FrameSample:
    scene: str
    seed: int
    shot: int
    frame: int
    value: torch.Tensor  # [heads, dim], float64 CPU


def shot_for_frame(scene: str, frame: int) -> int:
    durations_path = ROOT / "example/benchmark_hard_multishot" / scene / "shot_durations.txt"
    durations = [int(x) for x in durations_path.read_text().split()]
    end = 0
    for shot, chunks in enumerate(durations):
        end += chunks * 8
        if frame < end:
            return shot
    return len(durations) - 1


def dump_paths(dump_root: Path, scene: str, seed: int, layer: int) -> list[Path]:
    return sorted(
        (dump_root / f"seed{seed}" / "dumps" / scene / f"seed{seed}" / "v" / f"layer_{layer:02d}").glob("*.pt")
    )


def load_frame_samples(dump_root: Path, layer: int) -> list[FrameSample]:
    samples: list[FrameSample] = []
    missing: list[str] = []
    for seed in (0, 1, 2):
        for scene in SCENES:
            paths = dump_paths(dump_root, scene, seed, layer)
            if not paths:
                missing.append(f"seed{seed}/{scene}/layer{layer}")
                continue
            by_frame: dict[int, list[torch.Tensor]] = {}
            for path in paths:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                values = payload["v"].to(torch.float64)
                frames = payload["frame_indices"].to(torch.long)
                for frame in frames.unique(sorted=True).tolist():
                    by_frame.setdefault(int(frame), []).append(values[frames == frame])
            for frame, chunks in sorted(by_frame.items()):
                value = torch.cat(chunks, dim=0).mean(dim=0)
                samples.append(
                    FrameSample(
                        scene=scene,
                        seed=seed,
                        shot=shot_for_frame(scene, frame),
                        frame=frame,
                        value=value,
                    )
                )
    if missing:
        raise FileNotFoundError(
            "P1 requires complete dumps before estimation; missing " + ", ".join(missing)
        )
    return samples


def stack(samples: Iterable[FrameSample]) -> tuple[torch.Tensor, np.ndarray]:
    selected = list(samples)
    return (
        torch.stack([sample.value for sample in selected]),
        np.asarray(
            [SCENES.index(sample.scene) * 3 + sample.seed for sample in selected],
            dtype=np.int64,
        ),
    )


def class_balanced_covariances(x: torch.Tensor, labels: np.ndarray):
    """Return per-head within/between covariance and global means."""
    classes = sorted(set(labels.tolist()))
    class_means = []
    within = torch.zeros(x.shape[1], x.shape[2], x.shape[2], dtype=torch.float64)
    for label in classes:
        group = x[torch.from_numpy(labels == label)]
        mean = group.mean(dim=0)
        class_means.append(mean)
        centered = group - mean
        within += torch.einsum("nhd,nhe->hde", centered, centered) / max(1, group.shape[0] - 1)
    within /= len(classes)
    means = torch.stack(class_means)
    global_mean = means.mean(dim=0)
    centered_means = means - global_mean
    between = torch.einsum("chd,che->hde", centered_means, centered_means) / max(1, len(classes) - 1)
    return within, between, global_mean


def whitened_fisher(
    within: torch.Tensor,
    between: torch.Tensor,
    *,
    max_condition: float = 1.0e6,
):
    heads, dim, _ = within.shape
    identity = torch.eye(dim, dtype=torch.float64)
    eps_grid = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    W, W_inv, eigvecs, eigvals = [], [], [], []
    epsilons, conditions, positive_ranks = [], [], []
    for head in range(heads):
        scale = float(torch.trace(within[head]).item()) / dim
        chosen = None
        for epsilon in eps_grid:
            regularized = within[head] + epsilon * max(scale, 1e-12) * identity
            values, vectors = torch.linalg.eigh(regularized)
            condition = float((values.max() / values.min().clamp_min(1e-18)).item())
            chosen = (epsilon, regularized, values, vectors, condition)
            if condition <= max_condition:
                break
        assert chosen is not None
        epsilon, _, values, vectors, condition = chosen
        white = (vectors * values.rsqrt().unsqueeze(0)) @ vectors.T
        unwhite = (vectors * values.sqrt().unsqueeze(0)) @ vectors.T
        bw = white @ between[head] @ white.T
        fisher_values, fisher_vectors = torch.linalg.eigh((bw + bw.T) * 0.5)
        order = torch.argsort(fisher_values, descending=True)
        fisher_values = fisher_values[order]
        fisher_vectors = fisher_vectors[:, order]
        threshold = max(1e-10, float(fisher_values[0].abs().item()) * 1e-8)
        positive = int((fisher_values > threshold).sum().item())
        W.append(white)
        W_inv.append(unwhite)
        eigvecs.append(fisher_vectors)
        eigvals.append(fisher_values)
        epsilons.append(epsilon)
        conditions.append(condition)
        positive_ranks.append(positive)
    return {
        "W": torch.stack(W),
        "W_inv": torch.stack(W_inv),
        "eigvecs": torch.stack(eigvecs),
        "eigvals": torch.stack(eigvals),
        "epsilon": epsilons,
        "condition": conditions,
        "positive_rank": positive_ranks,
    }


def whiten(x: torch.Tensor, mean: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return torch.einsum("hij,nhj->nhi", W, x - mean.unsqueeze(0))


def nearest_centroid_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(sorted(set(train_labels.tolist())))
    centroids = np.stack([train_features[train_labels == label].mean(axis=0) for label in labels])
    distances = ((test_features[:, None] - centroids[None]) ** 2).mean(axis=-1)
    return labels[distances.argmin(axis=1)]


def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Mean per-class recall without sklearn's bootstrap absent-class warning."""
    recalls = []
    for label in sorted(set(truth.tolist())):
        selected = truth == label
        recalls.append(float((prediction[selected] == label).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def features_for_basis(z: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coefficients = torch.einsum("hdr,nhd->nhr", basis, z)
    return coefficients.flatten(1)


def complement_features(z: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coefficients = torch.einsum("hdr,nhd->nhr", basis, z)
    projected = torch.einsum("hdr,nhr->nhd", basis, coefficients)
    return (z - projected).flatten(1)


def fisher_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    overall = features.mean(axis=0)
    between = 0.0
    within = 0.0
    for label in sorted(set(labels.tolist())):
        group = features[labels == label]
        mean = group.mean(axis=0)
        between += float(((mean - overall) ** 2).mean())
        within += float(((group - mean) ** 2).mean())
    return between / max(within, 1e-12)


def grouped_bootstrap_delta(
    truth: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: list[tuple[str, int, int]],
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(groups))
    group_indices = {group: np.asarray([i for i, value in enumerate(groups) if value == group]) for group in unique}
    deltas = []
    for _ in range(repeats):
        drawn = rng.choice(len(unique), size=len(unique), replace=True)
        indices = np.concatenate([group_indices[unique[i]] for i in drawn])
        if len(set(truth[indices].tolist())) < 2:
            continue
        a = balanced_accuracy(truth[indices], pred_a[indices])
        b = balanced_accuracy(truth[indices], pred_b[indices])
        deltas.append(a - b)
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(x) for x in np.quantile(deltas, [0.025, 0.5, 0.975]))


def grouped_bootstrap_accuracy(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: list[tuple[str, int, int]],
    *,
    repeats: int,
    seed: int,
) -> tuple[float, tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(groups))
    group_indices = {
        group: np.asarray([i for i, value in enumerate(groups) if value == group])
        for group in unique
    }
    values = []
    for _ in range(repeats):
        drawn = rng.choice(len(unique), size=len(unique), replace=True)
        indices = np.concatenate([group_indices[unique[i]] for i in drawn])
        values.append(balanced_accuracy(truth[indices], prediction[indices]))
    array = np.asarray(values, dtype=np.float64)
    return float(array.std(ddof=1)), tuple(
        float(x) for x in np.quantile(array, [0.025, 0.5, 0.975])
    )


def random_basis(heads: int, dim: int, rank: int, generator: torch.Generator) -> torch.Tensor:
    bases = []
    for _ in range(heads):
        matrix = torch.randn(dim, rank, generator=generator, dtype=torch.float64)
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        bases.append(q)
    return torch.stack(bases)


def prepare_loso_folds(
    train_samples: list[FrameSample],
    test_samples: list[FrameSample],
) -> list[dict]:
    """Fit one projector per held-out scene using only the other three scenes."""
    folds = []
    for held_out_scene in SCENES:
        projector_samples = [
            sample for sample in train_samples if sample.scene != held_out_scene
        ]
        calibration_samples = [
            sample for sample in train_samples if sample.scene == held_out_scene
        ]
        held_out_samples = [
            sample for sample in test_samples if sample.scene == held_out_scene
        ]
        x_projector, y_projector = stack(projector_samples)
        x_calibration, y_calibration = stack(calibration_samples)
        x_held_out, y_held_out = stack(held_out_samples)
        expected_projector_classes = {
            SCENES.index(scene) * 3 + seed
            for scene in SCENES
            if scene != held_out_scene
            for seed in (0, 1, 2)
        }
        expected_held_out_classes = {
            SCENES.index(held_out_scene) * 3 + seed for seed in (0, 1, 2)
        }
        if set(y_projector.tolist()) != expected_projector_classes:
            raise RuntimeError(
                f"LOSO projector split for {held_out_scene} has wrong identities"
            )
        if set(y_calibration.tolist()) != expected_held_out_classes:
            raise RuntimeError(
                f"LOSO calibration split for {held_out_scene} has wrong identities"
            )
        if set(y_held_out.tolist()) != expected_held_out_classes:
            raise RuntimeError(
                f"LOSO test split for {held_out_scene} has wrong identities"
            )
        within, between, mean = class_balanced_covariances(
            x_projector, y_projector
        )
        fisher = whitened_fisher(within, between)
        folds.append(
            {
                "scene": held_out_scene,
                "fisher": fisher,
                "mean": mean,
                "z_calibration": whiten(x_calibration, mean, fisher["W"]),
                "y_calibration": y_calibration,
                "z_held_out": whiten(x_held_out, mean, fisher["W"]),
                "y_held_out": y_held_out,
                "projector_samples": len(projector_samples),
                "calibration_samples": len(calibration_samples),
                "test_samples": len(held_out_samples),
                "max_rank": min(16, min(fisher["positive_rank"])),
                "stable": not any(
                    value > 1.0e6 for value in fisher["condition"]
                ),
            }
        )
    return folds


def evaluate_loso_rank(
    folds: list[dict],
    rank: int,
    *,
    random_trials: int,
    random_seed_value: int,
    layer: int,
) -> dict:
    fold_results = []
    random_by_fold = []
    for fold_index, fold in enumerate(folds):
        if not fold["stable"] or rank > fold["max_rank"]:
            fold_results.append(
                {
                    "scene": fold["scene"],
                    "gate_pass": False,
                    "blocked_reason": (
                        "ill-conditioned LOSO covariance"
                        if not fold["stable"]
                        else "rank exceeds LOSO positive Fisher rank"
                    ),
                }
            )
            continue
        basis = fold["fisher"]["eigvecs"][:, :, :rank]
        train_id = features_for_basis(fold["z_calibration"], basis).numpy()
        test_id = features_for_basis(fold["z_held_out"], basis).numpy()
        prediction = nearest_centroid_predict(
            train_id, fold["y_calibration"], test_id
        )
        accuracy = balanced_accuracy(fold["y_held_out"], prediction)
        train_comp = complement_features(fold["z_calibration"], basis).numpy()
        test_comp = complement_features(fold["z_held_out"], basis).numpy()
        complement_accuracy = balanced_accuracy(
            fold["y_held_out"],
            nearest_centroid_predict(
                train_comp, fold["y_calibration"], test_comp
            ),
        )
        generator = torch.Generator().manual_seed(
            random_seed_value + 50000 + layer * 1000 + rank * 10 + fold_index
        )
        random_accuracies = []
        for _ in range(random_trials):
            random = random_basis(
                fold["z_calibration"].shape[1],
                fold["z_calibration"].shape[2],
                rank,
                generator,
            )
            random_train = features_for_basis(
                fold["z_calibration"], random
            ).numpy()
            random_test = features_for_basis(fold["z_held_out"], random).numpy()
            random_accuracies.append(
                balanced_accuracy(
                    fold["y_held_out"],
                    nearest_centroid_predict(
                        random_train, fold["y_calibration"], random_test
                    ),
                )
            )
        random_array = np.asarray(random_accuracies, dtype=np.float64)
        random_by_fold.append(random_array)
        random_p95 = float(np.quantile(random_array, 0.95))
        fold_results.append(
            {
                "scene": fold["scene"],
                "projector_samples": fold["projector_samples"],
                "calibration_samples": fold["calibration_samples"],
                "test_samples": fold["test_samples"],
                "identity_balanced_accuracy": accuracy,
                "complement_balanced_accuracy": complement_accuracy,
                "random_accuracy_mean": float(random_array.mean()),
                "random_accuracy_p95": random_p95,
                "gate_pass": bool(
                    accuracy > complement_accuracy and accuracy > random_p95
                ),
            }
        )
    valid = [record for record in fold_results if "identity_balanced_accuracy" in record]
    if len(valid) != len(SCENES):
        return {
            "rank": rank,
            "folds": fold_results,
            "gate_pass": False,
            "blocked_reason": "one or more LOSO folds were numerically invalid",
        }
    identity_mean = float(
        np.mean([record["identity_balanced_accuracy"] for record in valid])
    )
    complement_mean = float(
        np.mean([record["complement_balanced_accuracy"] for record in valid])
    )
    random_trial_means = np.stack(random_by_fold).mean(axis=0)
    random_p95 = float(np.quantile(random_trial_means, 0.95))
    folds_passing = sum(bool(record["gate_pass"]) for record in valid)
    return {
        "rank": rank,
        "identity_balanced_accuracy_mean": identity_mean,
        "complement_balanced_accuracy_mean": complement_mean,
        "random_accuracy_mean": float(random_trial_means.mean()),
        "random_accuracy_p95": random_p95,
        "folds_passing": folds_passing,
        "folds_required": 3,
        "folds": fold_results,
        "gate_pass": bool(
            folds_passing >= 3
            and identity_mean > complement_mean
            and identity_mean > random_p95
        ),
    }


def pose_time_energy(samples: list[FrameSample], mean: torch.Tensor, W: torch.Tensor, basis: torch.Tensor):
    by_sequence: dict[tuple[str, int, int], list[FrameSample]] = {}
    for sample in samples:
        by_sequence.setdefault((sample.scene, sample.seed, sample.shot), []).append(sample)
    deltas = []
    for sequence in by_sequence.values():
        ordered = sorted(sequence, key=lambda item: item.frame)
        deltas.extend(b.value - a.value for a, b in zip(ordered, ordered[1:]))
    if not deltas:
        raise RuntimeError("pose/time diagnostic requires consecutive sampled frames")
    delta = torch.stack(deltas)
    z = torch.einsum("hij,nhj->nhi", W, delta)
    coefficient = torch.einsum("hdr,nhd->nhr", basis, z)
    identity_energy = float(coefficient.square().sum().item())
    total_energy = float(z.square().sum().item())
    complement_energy = max(0.0, total_energy - identity_energy)
    rank = basis.shape[-1]
    dim = z.shape[-1]
    return {
        "pairs": len(deltas),
        "identity_fraction": identity_energy / max(total_energy, 1e-12),
        "complement_fraction": complement_energy / max(total_energy, 1e-12),
        "identity_energy_per_dimension": identity_energy / max(1, rank * z.shape[1] * z.shape[0]),
        "complement_energy_per_dimension": complement_energy / max(1, (dim - rank) * z.shape[1] * z.shape[0]),
    }


def evaluate_layer(
    samples: list[FrameSample],
    layer: int,
    *,
    random_trials: int,
    bootstrap_repeats: int,
    random_seed_value: int,
):
    train_shots = (0, 2, 4, 6)
    held_out_shots = (1, 3, 5, 7)
    train_samples = [sample for sample in samples if sample.shot in train_shots]
    test_samples = [sample for sample in samples if sample.shot in held_out_shots]
    x_train, y_train = stack(train_samples)
    x_test, y_test = stack(test_samples)
    expected_identities = set(range(len(SCENES) * 3))
    if set(y_train.tolist()) != expected_identities:
        raise RuntimeError(
            "training split does not contain all 12 scene×seed identities: "
            f"found {sorted(set(y_train.tolist()))}"
        )
    if set(y_test.tolist()) != expected_identities:
        raise RuntimeError(
            "held-out split does not contain all 12 scene×seed identities: "
            f"found {sorted(set(y_test.tolist()))}"
        )
    within, between, mean = class_balanced_covariances(x_train, y_train)
    fisher = whitened_fisher(within, between)
    loso_folds = prepare_loso_folds(train_samples, test_samples)
    z_train = whiten(x_train, mean, fisher["W"])
    z_test = whiten(x_test, mean, fisher["W"])
    max_rank = min(
        16,
        min(fisher["positive_rank"]),
        *(fold["max_rank"] for fold in loso_folds),
    )
    unstable_condition = any(value > 1.0e6 for value in fisher["condition"])
    unstable_loso = any(not fold["stable"] for fold in loso_folds)
    if max_rank < 1 or unstable_condition or unstable_loso:
        return {
            "layer": layer,
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "identity_classes": 12,
            "train_shots": list(train_shots),
            "held_out_shots": list(held_out_shots),
            "heads": int(x_train.shape[1]),
            "head_dim": int(x_train.shape[2]),
            "regularization_epsilon_by_head": fisher["epsilon"],
            "within_condition_by_head": fisher["condition"],
            "positive_fisher_rank_by_head": fisher["positive_rank"],
            "fisher_eigenvalues_by_head_top16": (
                fisher["eigvals"][:, :16].cpu().numpy().tolist()
            ),
            "fisher_eigenvalue_mean_top16": (
                fisher["eigvals"][:, :16].mean(dim=0).cpu().numpy().tolist()
            ),
            "ranks": [],
            "gate_pass": False,
            "blocked_reason": (
                "within covariance remained ill-conditioned after regularization"
                if unstable_condition or unstable_loso
                else "no positive Fisher eigenvalue"
            ),
            "leave_one_scene_out": {
                "gate_pass": False,
                "folds": [
                    {
                        "scene": fold["scene"],
                        "stable": fold["stable"],
                        "max_rank": fold["max_rank"],
                    }
                    for fold in loso_folds
                ],
            },
        }, None
    ranks = []
    groups = [(sample.scene, sample.seed, sample.shot) for sample in test_samples]
    for rank in range(1, max_rank + 1):
        basis = fisher["eigvecs"][:, :, :rank]
        train_id = features_for_basis(z_train, basis).numpy()
        test_id = features_for_basis(z_test, basis).numpy()
        pred_id = nearest_centroid_predict(train_id, y_train, test_id)
        accuracy = balanced_accuracy(y_test, pred_id)
        train_comp = complement_features(z_train, basis).numpy()
        test_comp = complement_features(z_test, basis).numpy()
        pred_comp = nearest_centroid_predict(train_comp, y_train, test_comp)
        complement_accuracy = balanced_accuracy(y_test, pred_comp)
        raw_train = (x_train - mean.unsqueeze(0)).flatten(1).numpy()
        raw_test = (x_test - mean.unsqueeze(0)).flatten(1).numpy()
        raw_accuracy = balanced_accuracy(
            y_test, nearest_centroid_predict(raw_train, y_train, raw_test)
        )
        generator = torch.Generator().manual_seed(random_seed_value + layer * 1000 + rank)
        random_accuracies = []
        random_predictions = []
        random_bases = []
        for trial in range(random_trials):
            rb = random_basis(z_train.shape[1], z_train.shape[2], rank, generator)
            random_bases.append(rb)
            train_random = features_for_basis(z_train, rb).numpy()
            test_random = features_for_basis(z_test, rb).numpy()
            prediction = nearest_centroid_predict(train_random, y_train, test_random)
            random_predictions.append(prediction)
            random_accuracies.append(balanced_accuracy(y_test, prediction))
        random_array = np.asarray(random_accuracies)
        median_index = int(np.argsort(random_array)[len(random_array) // 2])
        median_basis = random_bases[median_index]
        train_median_random = features_for_basis(z_train, median_basis).numpy()
        test_median_random = features_for_basis(z_test, median_basis).numpy()
        ci = grouped_bootstrap_delta(
            y_test,
            pred_id,
            random_predictions[median_index],
            groups,
            repeats=bootstrap_repeats,
            seed=random_seed_value + layer + rank,
        )
        identity_bootstrap_se, identity_bootstrap_ci = grouped_bootstrap_accuracy(
            y_test,
            pred_id,
            groups,
            repeats=bootstrap_repeats,
            seed=random_seed_value + 10000 + layer + rank,
        )
        pose = pose_time_energy(test_samples, mean, fisher["W"], basis)
        loso = evaluate_loso_rank(
            loso_folds,
            rank,
            random_trials=random_trials,
            random_seed_value=random_seed_value,
            layer=layer,
        )
        significant = bool(
            accuracy > float(np.quantile(random_array, 0.95))
            and ci[0] > 0.0
        )
        rank_pass = bool(
            significant
            and accuracy > complement_accuracy
            and pose["complement_fraction"] > 0.5
            and loso["gate_pass"]
        )
        ranks.append(
            {
                "rank": rank,
                "identity_balanced_accuracy": accuracy,
                "raw_v_balanced_accuracy": raw_accuracy,
                "complement_balanced_accuracy": complement_accuracy,
                "identity_fisher_ratio": fisher_ratio(test_id, y_test),
                "identity_silhouette": float(silhouette_score(test_id, y_test)),
                "complement_fisher_ratio": fisher_ratio(test_comp, y_test),
                "complement_silhouette": float(
                    silhouette_score(test_comp, y_test)
                ),
                "median_random_fisher_ratio": fisher_ratio(
                    test_median_random, y_test
                ),
                "median_random_silhouette": float(
                    silhouette_score(test_median_random, y_test)
                ),
                "random_accuracy_mean": float(random_array.mean()),
                "random_accuracy_std": float(
                    random_array.std(ddof=1 if random_trials > 1 else 0)
                ),
                "random_accuracy_p95": float(np.quantile(random_array, 0.95)),
                "random_null_percentile": float((random_array < accuracy).mean()),
                "bootstrap_identity_minus_median_random_ci95": list(ci),
                "identity_accuracy_bootstrap_se": identity_bootstrap_se,
                "identity_accuracy_bootstrap_ci95": list(identity_bootstrap_ci),
                "pose_time": pose,
                "leave_one_scene_out": loso,
                "gate_pass": rank_pass,
            }
        )
    passing = [record for record in ranks if record["gate_pass"]]
    result = {
        "layer": layer,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "identity_classes": 12,
        "train_shots": list(train_shots),
        "held_out_shots": list(held_out_shots),
        "heads": int(x_train.shape[1]),
        "head_dim": int(x_train.shape[2]),
        "regularization_epsilon_by_head": fisher["epsilon"],
        "within_condition_by_head": fisher["condition"],
        "positive_fisher_rank_by_head": fisher["positive_rank"],
        "fisher_eigenvalues_by_head_top16": (
            fisher["eigvals"][:, :16].cpu().numpy().tolist()
        ),
        "fisher_eigenvalue_mean_top16": (
            fisher["eigvals"][:, :16].mean(dim=0).cpu().numpy().tolist()
        ),
        "ranks": ranks,
        "leave_one_scene_out_protocol": {
            "projector_fit": "even shots from the other three scenes",
            "held_out_scene_calibration": "even shots establish seed0/1/2 centroids only",
            "held_out_scene_test": "odd shots",
            "required_passing_folds": 3,
        },
        "gate_pass": bool(passing),
    }
    return result, {"fisher": fisher, "mean": mean, "samples": samples}


def donor_for_scene(dump_root: Path, scene: str, layer: int) -> torch.Tensor:
    values = []
    for path in dump_paths(dump_root, scene, 0, layer):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["shot_index"]) == 0:
            values.append(payload["v"].float())
    if not values:
        raise RuntimeError(f"no seed0 shot0 donor tokens for {scene} layer {layer}")
    return torch.cat(values).mean(dim=0)


def write_report(output_dir: Path, result: dict) -> None:
    lines = [
        "# G1 identity-subspace gate",
        "",
        f"Verdict: **{'PASS' if result['gate_pass'] else 'FAIL — stop P2'}**",
        "",
        "Primary held-out split: train shots {0,2,4,6}, test shots {1,3,5,7}; labels are the 12 scene×seed generated identities.",
        "Leave-one-scene-out: fit the projector on the other three scenes; use even shots of the held-out scene only to establish its seed0/1/2 centroids, then classify odd shots. At least 3/4 folds must beat complement and rank-matched random.",
        "Same-prompt cross-seed instances are hard negatives; frame tokens are pooled before evaluation.",
        "Random null: rank-matched orthonormal bases in the same whitened metric.",
        "",
        "| layer | rank | id balanced acc | complement acc | random p95 | bootstrap delta CI | pose complement | LOSO id/comp/random p95 | LOSO folds | pass |",
        "|---:|---:|---:|---:|---:|---|---:|---|---:|:---:|",
    ]
    for layer in result["layers"]:
        for rank in layer.get("ranks", []):
            ci = rank["bootstrap_identity_minus_median_random_ci95"]
            loso = rank["leave_one_scene_out"]
            lines.append(
                f"| {layer['layer']} | {rank['rank']} | {rank['identity_balanced_accuracy']:.3f} "
                f"| {rank['complement_balanced_accuracy']:.3f} | {rank['random_accuracy_p95']:.3f} "
                f"| [{ci[0]:.3f}, {ci[2]:.3f}] | {rank['pose_time']['complement_fraction']:.3f} "
                f"| {loso['identity_balanced_accuracy_mean']:.3f}/{loso['complement_balanced_accuracy_mean']:.3f}/{loso['random_accuracy_p95']:.3f} "
                f"| {loso['folds_passing']}/{loso['folds_required']} "
                f"| {'yes' if rank['gate_pass'] else 'no'} |"
            )
    lines += [
        "",
        "Numerical conditioning by layer:",
        "",
        "| layer | train/test frame samples | epsilon range | max within-cov condition | min positive Fisher rank |",
        "|---:|---:|---|---:|---:|",
    ]
    for layer in result["layers"]:
        eps = layer.get("regularization_epsilon_by_head", [])
        condition = layer.get("within_condition_by_head", [])
        positive = layer.get("positive_fisher_rank_by_head", [])
        lines.append(
            f"| {layer['layer']} | {layer.get('train_samples', 0)}/{layer.get('test_samples', 0)} "
            f"| [{min(eps):.0e}, {max(eps):.0e}] "
            f"| {max(condition):.3g} | {min(positive)} |"
        )
    candidate = result.get("candidate_selection")
    if candidate is not None:
        final_refit = candidate["final_refit"]
        lines += [
            "",
            "Held-out-shot and LOSO selection passed, but edit-artifact construction was "
            "blocked during the mandatory all-data refit; this is not a no-subspace kill: "
            f"{final_refit['failure_reason']}.",
        ]
    lines += [
        "",
        "This is a single-benchmark, three-seed linear probe for an exploratory pilot; it is not paper-level evidence.",
    ]
    (output_dir / "G1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump_root",
        default="videos/capsule_pilot_20260715/probe_replays",
    )
    parser.add_argument(
        "--output_dir",
        default="videos/capsule_pilot_20260715/g1",
    )
    parser.add_argument("--random_trials", type=int, default=100)
    parser.add_argument("--bootstrap_repeats", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=20260715)
    args = parser.parse_args()
    dump_root = (ROOT / args.dump_root).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_results = []
    fitted = {}
    for layer in LAYERS:
        samples = load_frame_samples(dump_root, layer)
        result, fit = evaluate_layer(
            samples,
            layer,
            random_trials=args.random_trials,
            bootstrap_repeats=args.bootstrap_repeats,
            random_seed_value=args.random_seed,
        )
        layer_results.append(result)
        if fit is not None:
            fitted[layer] = fit
    passing = [
        (layer["layer"], rank)
        for layer in layer_results
        for rank in layer.get("ranks", [])
        if rank["gate_pass"]
    ]
    selection = None
    candidate_selection = None
    if passing:
        best_layer, best_record = max(
            passing, key=lambda item: item[1]["identity_balanced_accuracy"]
        )
        best_accuracy = best_record["identity_balanced_accuracy"]
        one_se = best_record["identity_accuracy_bootstrap_se"]
        candidates = []
        for layer, rank in passing:
            if rank["identity_balanced_accuracy"] >= best_accuracy - one_se:
                candidates.append((rank["rank"], layer, rank))
        _, selected_layer, selected_rank_record = min(candidates)
        selected_rank = int(selected_rank_record["rank"])
        selection = {
            "layer": selected_layer,
            "rank": selected_rank,
            "criterion": "smallest rank within one grouped-bootstrap SE of best passing held-out accuracy",
            "best_layer": best_layer,
            "best_identity_balanced_accuracy": best_accuracy,
            "one_standard_error": one_se,
        }

        # Refit the selected projector on all three seeds after the held-out gate.
        all_samples = fitted[selected_layer]["samples"]
        x_all, y_all = stack(all_samples)
        within, between, mean = class_balanced_covariances(x_all, y_all)
        final_fisher = whitened_fisher(within, between)
        selection["final_refit"] = {
            "samples": int(x_all.shape[0]),
            "identity_classes": int(len(set(y_all.tolist()))),
            "regularization_epsilon_by_head": final_fisher["epsilon"],
            "within_condition_by_head": final_fisher["condition"],
            "positive_fisher_rank_by_head": final_fisher["positive_rank"],
            "fisher_eigenvalue_mean_top16": (
                final_fisher["eigvals"][:, :16].mean(dim=0).cpu().numpy().tolist()
            ),
        }
        final_refit_failure = None
        if any(value > 1.0e6 for value in final_fisher["condition"]):
            final_refit_failure = (
                "final all-data within covariance remained ill-conditioned after regularization"
            )
        elif selected_rank > min(final_fisher["positive_rank"]):
            final_refit_failure = (
                "selected rank exceeds the positive Fisher rank after final refit"
            )
        if final_refit_failure is not None:
            selection["final_refit"]["artifact_ready"] = False
            selection["final_refit"]["failure_reason"] = final_refit_failure
            candidate_selection = selection
            selection = None
        else:
            selection["final_refit"]["artifact_ready"] = True
            U = final_fisher["eigvecs"][:, :, :selected_rank]
            generator = torch.Generator().manual_seed(args.random_seed)
            random_R = random_basis(U.shape[0], U.shape[1], selected_rank, generator)
            artifacts_dir = output_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            for scene in SCENES:
                artifact = {
                    "schema_version": 1,
                    "scene": scene,
                    "layer": selected_layer,
                    "rank": selected_rank,
                    "W": final_fisher["W"].float(),
                    "W_inv": final_fisher["W_inv"].float(),
                    "mean": mean.float(),
                    "U": U.float(),
                    "random_R": random_R.float(),
                    "donor": donor_for_scene(dump_root, scene, selected_layer).float(),
                    "random_seed": args.random_seed,
                    "estimator": "per-head whitened Fisher generalized eigendecomposition",
                    "donor_source": "seed0 shot0 clean-recache entity-token mean",
                    "regularization_epsilon_by_head": final_fisher["epsilon"],
                    "within_condition_by_head": final_fisher["condition"],
                    "positive_fisher_rank_by_head": final_fisher["positive_rank"],
                    "fisher_eigenvalues_by_head_top16": (
                        final_fisher["eigvals"][:, :16].float()
                    ),
                }
                torch.save(artifact, artifacts_dir / f"{scene}.pt")

    output = {
        "schema_version": 1,
        "gate": "G1",
        "gate_pass": bool(passing) and selection is not None,
        "verdict_if_stop": (
            None
            if passing and selection is not None
            else "blocked"
            if candidate_selection is not None
            else "kill"
        ),
        "selection": selection,
        "candidate_selection": candidate_selection,
        "identity_definition": "scene×seed generated instance (12 classes)",
        "split": {"train_shots": [0, 2, 4, 6], "held_out_shots": [1, 3, 5, 7]},
        "leave_one_scene_out": {
            "projector_fit_scenes": 3,
            "held_out_scene_centroid_shots": [0, 2, 4, 6],
            "held_out_scene_test_shots": [1, 3, 5, 7],
            "required_passing_folds": 3,
        },
        "random_trials": args.random_trials,
        "bootstrap_repeats": args.bootstrap_repeats,
        "random_seed": args.random_seed,
        "layers": layer_results,
    }
    (output_dir / "g1_results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(output_dir, output)
    print(json.dumps({"gate_pass": output["gate_pass"], "selection": selection}, indent=2))


if __name__ == "__main__":
    main()
