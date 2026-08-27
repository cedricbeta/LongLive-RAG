#!/usr/bin/env python3
"""Final preregistered G1b estimator: within-scene, frame-token V contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import estimate_capsule_subspace as base


SCENES = base.SCENES
LAYERS = base.LAYERS
TRAIN_SHOTS = (0, 2, 4, 6)
TEST_SHOTS = (1, 3, 5, 7)


def _scene_stack(samples: list[base.FrameSample], scene: str):
    selected = [sample for sample in samples if sample.scene == scene]
    if not selected:
        raise RuntimeError(f"no samples for {scene}")
    return (
        torch.stack([sample.value for sample in selected]),
        np.asarray([sample.seed for sample in selected], dtype=np.int64),
        selected,
    )


def fit_within_scene_fisher(
    samples: list[base.FrameSample], scenes: tuple[str, ...] | list[str]
) -> dict:
    """Sum scene-internal seed contrasts so scene offsets never enter S_b."""
    within_parts = []
    between_parts = []
    means = {}
    for scene in scenes:
        x, labels, _ = _scene_stack(samples, scene)
        if set(labels.tolist()) != {0, 1, 2}:
            raise RuntimeError(f"{scene} does not contain seed classes 0/1/2")
        within, between, mean = base.class_balanced_covariances(x, labels)
        within_parts.append(within)
        between_parts.append(between)
        means[scene] = mean
    fisher = base.whitened_fisher(
        torch.stack(within_parts).mean(dim=0),
        torch.stack(between_parts).mean(dim=0),
    )
    return {"fisher": fisher, "means": means, "fit_scenes": list(scenes)}


def whiten_scene(
    x: torch.Tensor, scene: str, means: dict[str, torch.Tensor], W: torch.Tensor
) -> torch.Tensor:
    return base.whiten(x, means[scene], W)


def _predictions_for_basis(
    z_train: torch.Tensor,
    y_train: np.ndarray,
    z_test: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    train_id = base.features_for_basis(z_train, basis).numpy()
    test_id = base.features_for_basis(z_test, basis).numpy()
    pred_id = base.nearest_centroid_predict(train_id, y_train, test_id)
    train_comp = base.complement_features(z_train, basis).numpy()
    test_comp = base.complement_features(z_test, basis).numpy()
    pred_comp = base.nearest_centroid_predict(train_comp, y_train, test_comp)
    return pred_id, pred_comp


def _random_null_for_scene(
    z_train: torch.Tensor,
    y_train: np.ndarray,
    z_test: torch.Tensor,
    y_test: np.ndarray,
    random_bases: list[torch.Tensor],
) -> tuple[np.ndarray, list[np.ndarray]]:
    accuracies = []
    predictions = []
    for basis in random_bases:
        train = base.features_for_basis(z_train, basis).numpy()
        test = base.features_for_basis(z_test, basis).numpy()
        prediction = base.nearest_centroid_predict(train, y_train, test)
        predictions.append(prediction)
        accuracies.append(base.balanced_accuracy(y_test, prediction))
    return np.asarray(accuracies, dtype=np.float64), predictions


def evaluate_loso_diagnostic(
    train_samples: list[base.FrameSample],
    test_samples: list[base.FrameSample],
    *,
    rank: int,
    random_trials: int,
    random_seed: int,
    layer: int,
) -> dict:
    folds = []
    for fold_index, held_scene in enumerate(SCENES):
        fit_scenes = [scene for scene in SCENES if scene != held_scene]
        fit = fit_within_scene_fisher(train_samples, fit_scenes)
        fisher = fit["fisher"]
        if rank > min(fisher["positive_rank"]):
            folds.append(
                {
                    "scene": held_scene,
                    "valid": False,
                    "reason": "rank exceeds positive Fisher rank in diagnostic fit",
                }
            )
            continue
        x_train, y_train, calibration = _scene_stack(train_samples, held_scene)
        x_test, y_test, heldout = _scene_stack(test_samples, held_scene)
        _, _, held_mean = base.class_balanced_covariances(x_train, y_train)
        z_train = base.whiten(x_train, held_mean, fisher["W"])
        z_test = base.whiten(x_test, held_mean, fisher["W"])
        basis = fisher["eigvecs"][:, :, :rank]
        pred_id, pred_comp = _predictions_for_basis(
            z_train, y_train, z_test, basis
        )
        identity_accuracy = base.balanced_accuracy(y_test, pred_id)
        complement_accuracy = base.balanced_accuracy(y_test, pred_comp)
        generator = torch.Generator().manual_seed(
            random_seed + 70000 + layer * 1000 + rank * 10 + fold_index
        )
        random_bases = [
            base.random_basis(z_train.shape[1], z_train.shape[2], rank, generator)
            for _ in range(random_trials)
        ]
        random_accuracy, _ = _random_null_for_scene(
            z_train, y_train, z_test, y_test, random_bases
        )
        folds.append(
            {
                "scene": held_scene,
                "valid": True,
                "projector_fit_scenes": fit_scenes,
                "calibration_samples": len(calibration),
                "test_samples": len(heldout),
                "identity_balanced_accuracy": identity_accuracy,
                "complement_balanced_accuracy": complement_accuracy,
                "random_accuracy_mean": float(random_accuracy.mean()),
                "random_accuracy_p95": float(np.quantile(random_accuracy, 0.95)),
                "diagnostic_win": bool(
                    identity_accuracy > complement_accuracy
                    and identity_accuracy > float(np.quantile(random_accuracy, 0.95))
                ),
            }
        )
    valid = [fold for fold in folds if fold.get("valid")]
    return {
        "gate_role": "diagnostic_only",
        "folds": folds,
        "valid_folds": len(valid),
        "diagnostic_wins": sum(bool(fold["diagnostic_win"]) for fold in valid),
    }


def evaluate_layer(
    samples: list[base.FrameSample],
    layer: int,
    *,
    random_trials: int,
    bootstrap_repeats: int,
    random_seed: int,
) -> tuple[dict, dict]:
    train_samples = [sample for sample in samples if sample.shot in TRAIN_SHOTS]
    test_samples = [sample for sample in samples if sample.shot in TEST_SHOTS]
    fit = fit_within_scene_fisher(train_samples, list(SCENES))
    fisher = fit["fisher"]
    unstable = any(value > 1.0e6 for value in fisher["condition"])
    max_rank = min(8, min(fisher["positive_rank"]))
    scene_data = {}
    for scene in SCENES:
        x_train, y_train, calibration = _scene_stack(train_samples, scene)
        x_test, y_test, heldout = _scene_stack(test_samples, scene)
        scene_data[scene] = {
            "z_train": whiten_scene(x_train, scene, fit["means"], fisher["W"]),
            "y_train": y_train,
            "train_samples": calibration,
            "z_test": whiten_scene(x_test, scene, fit["means"], fisher["W"]),
            "y_test": y_test,
            "test_samples": heldout,
        }
    ranks = []
    if not unstable and max_rank >= 1:
        for rank in range(1, max_rank + 1):
            basis = fisher["eigvecs"][:, :, :rank]
            generator = torch.Generator().manual_seed(
                random_seed + layer * 1000 + rank
            )
            random_bases = [
                base.random_basis(
                    scene_data[SCENES[0]]["z_train"].shape[1],
                    scene_data[SCENES[0]]["z_train"].shape[2],
                    rank,
                    generator,
                )
                for _ in range(random_trials)
            ]
            scene_results = {}
            for scene_index, scene in enumerate(SCENES):
                data = scene_data[scene]
                pred_id, pred_comp = _predictions_for_basis(
                    data["z_train"],
                    data["y_train"],
                    data["z_test"],
                    basis,
                )
                identity_accuracy = base.balanced_accuracy(data["y_test"], pred_id)
                complement_accuracy = base.balanced_accuracy(
                    data["y_test"], pred_comp
                )
                random_accuracy, random_predictions = _random_null_for_scene(
                    data["z_train"],
                    data["y_train"],
                    data["z_test"],
                    data["y_test"],
                    random_bases,
                )
                random_p95 = float(np.quantile(random_accuracy, 0.95))
                p95_index = int(np.argmin(np.abs(random_accuracy - random_p95)))
                groups = [
                    (sample.scene, sample.seed, sample.shot)
                    for sample in data["test_samples"]
                ]
                ci_vs_complement = base.grouped_bootstrap_delta(
                    data["y_test"],
                    pred_id,
                    pred_comp,
                    groups,
                    repeats=bootstrap_repeats,
                    seed=random_seed + 10000 + layer * 100 + rank * 10 + scene_index,
                )
                ci_vs_random_p95 = base.grouped_bootstrap_delta(
                    data["y_test"],
                    pred_id,
                    random_predictions[p95_index],
                    groups,
                    repeats=bootstrap_repeats,
                    seed=random_seed + 20000 + layer * 100 + rank * 10 + scene_index,
                )
                scene_pass = bool(
                    identity_accuracy > complement_accuracy
                    and identity_accuracy > random_p95
                    and ci_vs_complement[0] > 0.0
                    and ci_vs_random_p95[0] > 0.0
                )
                scene_results[scene] = {
                    "identity_balanced_accuracy": identity_accuracy,
                    "complement_balanced_accuracy": complement_accuracy,
                    "random_accuracy_mean": float(random_accuracy.mean()),
                    "random_accuracy_p95": random_p95,
                    "random_null_percentile": float(
                        (random_accuracy < identity_accuracy).mean()
                    ),
                    "cluster_bootstrap_identity_minus_complement_ci95": list(
                        ci_vs_complement
                    ),
                    "cluster_bootstrap_identity_minus_p95_random_ci95": list(
                        ci_vs_random_p95
                    ),
                    "bootstrap_clusters": len(set(groups)),
                    "bootstrap_unit": "scene×seed×shot",
                    "gate_pass": scene_pass,
                }
            scenes_passing = sum(
                bool(record["gate_pass"]) for record in scene_results.values()
            )
            loso = evaluate_loso_diagnostic(
                train_samples,
                test_samples,
                rank=rank,
                random_trials=random_trials,
                random_seed=random_seed,
                layer=layer,
            )
            ranks.append(
                {
                    "rank": rank,
                    "scenes": scene_results,
                    "scenes_passing": scenes_passing,
                    "scenes_required": 3,
                    "mean_identity_balanced_accuracy": float(
                        np.mean(
                            [
                                record["identity_balanced_accuracy"]
                                for record in scene_results.values()
                            ]
                        )
                    ),
                    "mean_complement_balanced_accuracy": float(
                        np.mean(
                            [
                                record["complement_balanced_accuracy"]
                                for record in scene_results.values()
                            ]
                        )
                    ),
                    "mean_random_accuracy_p95": float(
                        np.mean(
                            [
                                record["random_accuracy_p95"]
                                for record in scene_results.values()
                            ]
                        )
                    ),
                    "leave_one_scene_out": loso,
                    "gate_pass": scenes_passing >= 3,
                }
            )
    result = {
        "layer": layer,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "identity_definition": "three generated seed instances within each scene",
        "scene_offsets_enter_between_scatter": False,
        "sample_unit": "per-frame entity-token mean",
        "bootstrap_unit": "scene×seed×shot cluster",
        "heads": int(next(iter(scene_data.values()))["z_train"].shape[1]),
        "head_dim": int(next(iter(scene_data.values()))["z_train"].shape[2]),
        "regularization_epsilon_by_head": fisher["epsilon"],
        "within_condition_by_head": fisher["condition"],
        "positive_fisher_rank_by_head": fisher["positive_rank"],
        "fisher_eigenvalue_mean_top8": (
            fisher["eigvals"][:, :8].mean(dim=0).cpu().numpy().tolist()
        ),
        "ranks": ranks,
        "gate_pass": any(rank["gate_pass"] for rank in ranks),
        "blocked_reason": (
            "within covariance remained ill-conditioned after regularization"
            if unstable
            else "no positive within-scene Fisher direction"
            if max_rank < 1
            else None
        ),
    }
    return result, {"fit": fit, "samples": samples}


def donor_for_scene(
    dump_root: Path, scene: str, layer: int
) -> torch.Tensor:
    return base.donor_for_scene(dump_root, scene, layer)


def final_refit_artifacts(
    dump_root: Path,
    output_dir: Path,
    fitted: dict[int, dict],
    selection: dict,
    random_seed: int,
) -> dict:
    layer = int(selection["layer"])
    rank = int(selection["rank"])
    samples = fitted[layer]["samples"]
    fit = fit_within_scene_fisher(samples, list(SCENES))
    fisher = fit["fisher"]
    if rank > min(fisher["positive_rank"]):
        return {
            "artifact_ready": False,
            "failure_reason": "selected rank exceeds final positive Fisher rank",
        }
    U = fisher["eigvecs"][:, :, :rank]
    generator = torch.Generator().manual_seed(random_seed)
    random_R = base.random_basis(U.shape[0], U.shape[1], rank, generator)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        torch.save(
            {
                "schema_version": 2,
                "scene": scene,
                "layer": layer,
                "rank": rank,
                "W": fisher["W"].float(),
                "W_inv": fisher["W_inv"].float(),
                "mean": fit["means"][scene].float(),
                "U": U.float(),
                "random_R": random_R.float(),
                "donor": donor_for_scene(dump_root, scene, layer).float(),
                "random_seed": random_seed,
                "estimator": "G1b summed within-scene seed Fisher",
                "donor_source": "seed0 shot0 clean-recache entity-token mean",
            },
            artifacts / f"{scene}.pt",
        )
    return {"artifact_ready": True, "artifact_dir": str(artifacts.relative_to(ROOT))}


def write_report(output_dir: Path, result: dict) -> None:
    lines = [
        "# G1b final within-scene identity-subspace gate",
        "",
        f"Verdict: **{'PASS' if result['gate_pass'] else 'FAIL — final linear-capsule close'}**",
        "",
        "Fisher between-scatter is the sum of seed-instance contrasts centered separately inside each scene; scene offsets never enter the between-scatter.",
        "Samples are per-frame means of entity-region V tokens (no shot pooling). Significance uses a cluster bootstrap over scene×seed×shot clusters.",
        "The gate requires the identity projection to beat the orthogonal complement and the rank-matched random p95 with both cluster-bootstrap lower bounds above zero in at least 3/4 scenes. LOSO is diagnostic only.",
        "",
        "| layer | rank | mean id acc | mean complement | mean random p95 | scenes passing | LOSO diagnostic wins | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    order = {14: 0, 0: 1, 7: 2, 21: 3, 29: 4}
    for layer in sorted(result["layers"], key=lambda item: order[item["layer"]]):
        for rank in layer.get("ranks", []):
            lines.append(
                f"| {layer['layer']} | {rank['rank']} "
                f"| {rank['mean_identity_balanced_accuracy']:.3f} "
                f"| {rank['mean_complement_balanced_accuracy']:.3f} "
                f"| {rank['mean_random_accuracy_p95']:.3f} "
                f"| {rank['scenes_passing']}/4 (need ≥{rank['scenes_required']}) "
                f"| {rank['leave_one_scene_out']['diagnostic_wins']}/4 "
                f"| {'yes' if rank['gate_pass'] else 'no'} |"
            )
    lines += [
        "",
        "Layer 14 is reported first because it had the strongest G1 held-out decodability; it receives no gate privilege.",
        "This is the preregistered final estimator iteration. No further estimator variant is permitted in this pilot.",
    ]
    (output_dir / "G1B_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump_root", default="videos/capsule_pilot_20260715/probe_replays"
    )
    parser.add_argument("--output_dir", default="videos/capsule_pilot_20260715/g1b")
    parser.add_argument("--random_trials", type=int, default=100)
    parser.add_argument("--bootstrap_repeats", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=20260716)
    args = parser.parse_args()
    dump_root = (ROOT / args.dump_root).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_results = []
    fitted = {}
    for layer in LAYERS:
        samples = base.load_frame_samples(dump_root, layer)
        result, fit = evaluate_layer(
            samples,
            layer,
            random_trials=args.random_trials,
            bootstrap_repeats=args.bootstrap_repeats,
            random_seed=args.random_seed,
        )
        layer_results.append(result)
        fitted[layer] = fit
    passing = [
        (layer["layer"], rank)
        for layer in layer_results
        for rank in layer.get("ranks", [])
        if rank["gate_pass"]
    ]
    selection = None
    if passing:
        selected_layer, selected_rank = max(
            passing,
            key=lambda item: (
                item[1]["scenes_passing"],
                min(
                    record["cluster_bootstrap_identity_minus_p95_random_ci95"][0]
                    for record in item[1]["scenes"].values()
                    if record["gate_pass"]
                ),
                item[0] == 14,
                -item[1]["rank"],
            ),
        )
        selection = {
            "layer": int(selected_layer),
            "rank": int(selected_rank["rank"]),
            "scenes_passing": int(selected_rank["scenes_passing"]),
            "criterion": "most passing scenes, strongest worst passing-scene random-p95 bootstrap margin, then layer14 and lower rank",
        }
        selection["final_refit"] = final_refit_artifacts(
            dump_root, output_dir, fitted, selection, args.random_seed
        )
    output = {
        "schema_version": 1,
        "gate": "G1b",
        "gate_pass": bool(passing and selection and selection["final_refit"]["artifact_ready"]),
        "verdict_if_linear_capsule": "continue_to_P2" if passing else "close_permanently",
        "selection": selection,
        "identity_definition": "three generated seed instances contrasted within each scene",
        "sample_unit": "per-frame mean of entity-region V tokens",
        "scene_variance_in_between_scatter": False,
        "split": {"train_shots": list(TRAIN_SHOTS), "held_out_shots": list(TEST_SHOTS)},
        "gate_rule": "rank<=8 beats random p95 and orthogonal complement with shot-cluster CI lower bounds >0 in >=3/4 scenes",
        "loso_role": "diagnostic_only",
        "random_trials": args.random_trials,
        "bootstrap_repeats": args.bootstrap_repeats,
        "random_seed": args.random_seed,
        "layers": layer_results,
    }
    (output_dir / "g1b_results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(output_dir, output)
    print(json.dumps({"gate_pass": output["gate_pass"], "selection": selection}, indent=2))


if __name__ == "__main__":
    main()
