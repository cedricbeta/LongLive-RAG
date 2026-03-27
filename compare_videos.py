"""
Compare baseline vs RAG-enhanced video generation.
Extracts frames, computes per-frame SSIM/LPIPS against an early reference frame,
and generates a side-by-side comparison image grid.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from torchvision.io import read_video
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_video_frames(path):
    """Load video and return frames as numpy array [T, H, W, C] uint8."""
    frames, _, _ = read_video(str(path), pts_unit='sec')
    return frames.numpy()


def compute_ssim_series(frames, ref_frame_idx=10):
    """Compute SSIM of each frame against a reference frame."""
    ref = frames[ref_frame_idx]
    scores = []
    for i in range(len(frames)):
        # SSIM on grayscale for speed, or channel_axis for color
        score = ssim(ref, frames[i], channel_axis=2, data_range=255)
        scores.append(score)
    return scores


def make_comparison(baseline_path, rag_path, output_dir, prompt_idx):
    """Generate comparison for one prompt."""
    print(f"\n=== Prompt {prompt_idx} ===")
    print(f"  Baseline: {baseline_path}")
    print(f"  RAG:      {rag_path}")

    baseline_frames = load_video_frames(baseline_path)
    rag_frames = load_video_frames(rag_path)

    n_baseline = len(baseline_frames)
    n_rag = len(rag_frames)
    n_frames = min(n_baseline, n_rag)
    print(f"  Frames: baseline={n_baseline}, rag={n_rag}, using={n_frames}")

    # Use frame 10 as reference (early but past initial frames)
    ref_idx = min(10, n_frames - 1)

    # Compute SSIM series
    print("  Computing SSIM (baseline)...")
    baseline_ssim = compute_ssim_series(baseline_frames[:n_frames], ref_idx)
    print("  Computing SSIM (RAG)...")
    rag_ssim = compute_ssim_series(rag_frames[:n_frames], ref_idx)

    # --- Plot 1: SSIM over time ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(baseline_ssim, label='Baseline (no RAG)', color='tab:red', alpha=0.8)
    ax.plot(rag_ssim, label='RAG-KV', color='tab:blue', alpha=0.8)
    ax.set_xlabel('Frame')
    ax.set_ylabel(f'SSIM vs frame {ref_idx}')
    ax.set_title(f'Prompt {prompt_idx}: Visual Consistency Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ssim_plot_path = os.path.join(output_dir, f'prompt{prompt_idx}_ssim.png')
    fig.savefig(ssim_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved SSIM plot: {ssim_plot_path}")

    # --- Plot 2: Side-by-side frame grid ---
    # Sample frames at regular intervals
    sample_indices = np.linspace(0, n_frames - 1, 8, dtype=int)

    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    fig.suptitle(f'Prompt {prompt_idx}: Baseline (top) vs RAG (bottom)', fontsize=14)
    for col, idx in enumerate(sample_indices):
        axes[0, col].imshow(baseline_frames[idx])
        axes[0, col].set_title(f'F{idx}', fontsize=9)
        axes[0, col].axis('off')
        axes[1, col].imshow(rag_frames[idx])
        axes[1, col].set_title(f'F{idx}', fontsize=9)
        axes[1, col].axis('off')
    axes[0, 0].set_ylabel('Baseline', fontsize=11)
    axes[1, 0].set_ylabel('RAG', fontsize=11)
    grid_path = os.path.join(output_dir, f'prompt{prompt_idx}_grid.png')
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved frame grid: {grid_path}")

    # --- Summary stats ---
    # Compare average SSIM in the second half of the video (where drift matters)
    half = n_frames // 2
    baseline_avg = np.mean(baseline_ssim[half:])
    rag_avg = np.mean(rag_ssim[half:])
    diff = rag_avg - baseline_avg
    print(f"  Avg SSIM (frames {half}-{n_frames}): baseline={baseline_avg:.4f}, RAG={rag_avg:.4f}, diff={diff:+.4f}")

    return {
        'prompt_idx': prompt_idx,
        'baseline_ssim_late': baseline_avg,
        'rag_ssim_late': rag_avg,
        'diff': diff,
    }


def main():
    baseline_dir = Path("videos/single")
    rag_dir = Path("videos/single-rag")
    output_dir = Path("videos/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Match videos by index
    baseline_videos = sorted(baseline_dir.glob("rank0-*_lora.mp4"))
    rag_videos = sorted(rag_dir.glob("rank0-*_lora.mp4"))

    assert len(baseline_videos) == len(rag_videos), \
        f"Mismatch: {len(baseline_videos)} baseline vs {len(rag_videos)} RAG videos"

    results = []
    for i, (bv, rv) in enumerate(zip(baseline_videos, rag_videos)):
        r = make_comparison(bv, rv, str(output_dir), i)
        results.append(r)

    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"{'Prompt':<10} {'Baseline SSIM':>15} {'RAG SSIM':>15} {'Diff':>10}")
    print("-" * 50)
    for r in results:
        print(f"  {r['prompt_idx']:<8} {r['baseline_ssim_late']:>15.4f} {r['rag_ssim_late']:>15.4f} {r['diff']:>+10.4f}")
    avg_diff = np.mean([r['diff'] for r in results])
    print("-" * 50)
    print(f"  {'Avg':<8} {'':>15} {'':>15} {avg_diff:>+10.4f}")
    if avg_diff > 0:
        print(f"\nRAG improves late-video consistency by {avg_diff:+.4f} SSIM on average.")
    else:
        print(f"\nRAG shows {avg_diff:+.4f} SSIM difference (no improvement).")


if __name__ == "__main__":
    main()
