#!/usr/bin/env python3
"""Build mandatory G1 spectrum, head, UMAP, cosine, and shot diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from estimate_capsule_subspace import (  # noqa: E402
    SCENES,
    features_for_basis,
    load_frame_samples,
    stack,
    whiten,
)


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    result = json.loads((pilot_root / "g1/g1_results.json").read_text(encoding="utf-8"))
    output_dir = pilot_root / "g1/visuals"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for layer in result["layers"]:
        values = np.maximum(
            np.asarray(layer["fisher_eigenvalue_mean_top16"], dtype=np.float64),
            1e-12,
        )
        ax.semilogy(np.arange(1, len(values) + 1), values, marker="o", label=f"L{layer['layer']}")
    ax.set(xlabel="Fisher component rank", ylabel="mean eigenvalue", title="Whitened Fisher spectrum")
    ax.legend(ncol=3)
    ax.grid(alpha=0.25)
    save(fig, output_dir / "capsule_spectrum.png")

    if not result.get("gate_pass") or not result.get("selection"):
        print("G1 failed: spectrum written; selected-subspace visuals are not defined")
        return
    layer_index = int(result["selection"]["layer"])
    rank = int(result["selection"]["rank"])
    selected = next(item for item in result["layers"] if int(item["layer"]) == layer_index)
    heat = np.asarray(selected["fisher_eigenvalues_by_head_top16"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(heat, cmap="magma", ax=ax, cbar_kws={"label": "Fisher eigenvalue"})
    ax.set(xlabel="component", ylabel="attention head", title=f"Layer {layer_index} head spectrum")
    save(fig, output_dir / "head_heatmap.png")

    artifact = torch.load(
        pilot_root / f"g1/artifacts/{SCENES[0]}.pt",
        map_location="cpu",
        weights_only=False,
    )
    basis = artifact["U"].double()
    heads = basis.shape[0]
    similarity = np.empty((heads, heads), dtype=np.float64)
    for left in range(heads):
        for right in range(heads):
            overlap = basis[left].T @ basis[right]
            similarity[left, right] = float(overlap.square().sum().sqrt() / np.sqrt(rank))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(similarity, vmin=0, vmax=1, cmap="viridis", ax=ax)
    ax.set(title=f"Subspace cosine/overlap across heads (L{layer_index}, r={rank})", xlabel="head", ylabel="head")
    save(fig, output_dir / "subspace_cosine.png")

    samples = load_frame_samples(pilot_root / "probe_replays", layer_index)
    x, labels = stack(samples)
    z = whiten(x, artifact["mean"].double() if "mean" in artifact else x.mean(dim=0), artifact["W"].double())
    coefficients = features_for_basis(z, basis).numpy()
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "token_umap.png requires umap-learn; install the pinned pilot dependency"
        ) from exc
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.15,
        metric="euclidean",
        random_state=20260715,
    )
    embedding = reducer.fit_transform(coefficients)
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = plt.get_cmap("tab20")
    for label in sorted(set(labels.tolist())):
        selected_rows = labels == label
        scene = SCENES[label // 3]
        seed = label % 3
        ax.scatter(
            embedding[selected_rows, 0],
            embedding[selected_rows, 1],
            s=14,
            alpha=0.65,
            color=palette(label),
            label=f"{scene[:12]} s{seed}",
        )
    ax.set(title=f"UMAP of selected capsule coefficients (L{layer_index}, r={rank})", xlabel="UMAP-1", ylabel="UMAP-2")
    ax.legend(fontsize=7, ncol=2, loc="best")
    save(fig, output_dir / "token_umap.png")

    norms = np.linalg.norm(coefficients.reshape(len(samples), heads, rank), axis=(1, 2))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for scene_index, (scene, ax) in enumerate(zip(SCENES, axes.flat)):
        for seed in (0, 1, 2):
            means = []
            for shot in range(8):
                selected_rows = np.asarray(
                    [
                        sample.scene == scene and sample.seed == seed and sample.shot == shot
                        for sample in samples
                    ]
                )
                means.append(float(np.mean(norms[selected_rows])) if selected_rows.any() else np.nan)
            ax.plot(range(8), means, marker="o", label=f"seed{seed}")
        ax.set_title(scene)
        ax.grid(alpha=0.25)
        ax.set_ylabel("coefficient norm")
    axes[-1, 0].set_xlabel("shot")
    axes[-1, 1].set_xlabel("shot")
    axes[0, 0].legend()
    save(fig, output_dir / "projection_norm_by_shot.png")
    print(output_dir.relative_to(ROOT))


if __name__ == "__main__":
    main()
