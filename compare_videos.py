"""
Compare baseline vs RAG-enhanced video generation.
- SSIM consistency over time
- Early vs late frame comparison (reappearance test)
- Latency comparison from timing.json
- Side-by-side frame grids
"""
import os
import json
import numpy as np
from pathlib import Path
from torchvision.io import read_video
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_video_frames(path):
    """Load video frames as numpy [T, H, W, C] uint8."""
    frames, _, _ = read_video(str(path), pts_unit='sec')
    return frames.numpy()


def compute_ssim_series(frames, ref_idx=10):
    """SSIM of each frame vs a reference frame."""
    ref = frames[ref_idx]
    return [ssim(ref, frames[i], channel_axis=2, data_range=255) for i in range(len(frames))]


def compute_early_late_ssim(frames, early_range=(5, 25), late_start_ratio=0.8):
    """
    Compare early frames to late frames — tests reappearance consistency.
    Returns mean SSIM between each late frame and the best-matching early frame.
    """
    n = len(frames)
    early_frames = frames[early_range[0]:early_range[1]]
    late_start = int(n * late_start_ratio)
    late_frames = frames[late_start:]

    scores = []
    for late_f in late_frames:
        # Best match against any early frame
        best = max(ssim(ef, late_f, channel_axis=2, data_range=255) for ef in early_frames)
        scores.append(best)
    return np.mean(scores), late_start


def make_comparison(baseline_path, rag_path, output_dir, prompt_idx):
    print(f"\n{'='*60}")
    print(f"  Prompt {prompt_idx}")
    print(f"{'='*60}")

    baseline_frames = load_video_frames(baseline_path)
    rag_frames = load_video_frames(rag_path)
    n = min(len(baseline_frames), len(rag_frames))
    print(f"  Frames: {n}")

    ref_idx = min(10, n - 1)

    # --- 1. SSIM over time ---
    print("  Computing SSIM over time...")
    b_ssim = compute_ssim_series(baseline_frames[:n], ref_idx)
    r_ssim = compute_ssim_series(rag_frames[:n], ref_idx)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(b_ssim, label='Baseline', color='tab:red', alpha=0.8)
    ax.plot(r_ssim, label='RAG-KV', color='tab:blue', alpha=0.8)
    ax.set_xlabel('Frame')
    ax.set_ylabel(f'SSIM vs frame {ref_idx}')
    ax.set_title(f'Prompt {prompt_idx}: Visual Consistency Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.savefig(os.path.join(output_dir, f'prompt{prompt_idx}_ssim.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 2. Early-vs-late SSIM (reappearance) ---
    print("  Computing early-vs-late SSIM...")
    b_reappear, late_start = compute_early_late_ssim(baseline_frames[:n])
    r_reappear, _ = compute_early_late_ssim(rag_frames[:n])

    # --- 3. Frame grid ---
    sample_indices = np.linspace(0, n - 1, 8, dtype=int)
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
    fig.savefig(os.path.join(output_dir, f'prompt{prompt_idx}_grid.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 4. Early vs Late frame grid ---
    early_idx = list(range(5, 25, 5))[:4]  # 4 early frames
    late_idx = np.linspace(late_start, n - 1, 4, dtype=int).tolist()  # 4 late frames
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    fig.suptitle(f'Prompt {prompt_idx}: Early (left 4) vs Late (right 4) — Baseline top, RAG bottom', fontsize=12)
    for col, idx in enumerate(early_idx + late_idx):
        label = f'F{idx} {"(early)" if col < 4 else "(late)"}'
        axes[0, col].imshow(baseline_frames[min(idx, n-1)])
        axes[0, col].set_title(label, fontsize=8)
        axes[0, col].axis('off')
        axes[1, col].imshow(rag_frames[min(idx, n-1)])
        axes[1, col].set_title(label, fontsize=8)
        axes[1, col].axis('off')
    fig.savefig(os.path.join(output_dir, f'prompt{prompt_idx}_early_late.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Summary ---
    half = n // 2
    b_late_ssim = np.mean(b_ssim[half:])
    r_late_ssim = np.mean(r_ssim[half:])
    diff = r_late_ssim - b_late_ssim
    reappear_diff = r_reappear - b_reappear

    print(f"  Late-half SSIM:     baseline={b_late_ssim:.4f}  RAG={r_late_ssim:.4f}  diff={diff:+.4f}")
    print(f"  Reappearance SSIM:  baseline={b_reappear:.4f}  RAG={r_reappear:.4f}  diff={reappear_diff:+.4f}")

    return {
        'prompt_idx': prompt_idx,
        'baseline_late_ssim': b_late_ssim,
        'rag_late_ssim': r_late_ssim,
        'consistency_diff': diff,
        'baseline_reappear_ssim': b_reappear,
        'rag_reappear_ssim': r_reappear,
        'reappear_diff': reappear_diff,
    }


def load_timing(folder):
    """Load timing.json from a video output folder."""
    path = os.path.join(folder, "timing.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def print_latency_comparison(baseline_dir, rag_dir):
    """Print latency comparison from timing.json files."""
    b_timing = load_timing(baseline_dir)
    r_timing = load_timing(rag_dir)

    if not b_timing or not r_timing:
        print("\n  (timing.json not found in one or both folders — skipping latency comparison)")
        print("  Re-run both experiments to generate timing data.")
        return

    print(f"\n{'='*60}")
    print("LATENCY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Baseline':>12} {'RAG':>12} {'Overhead':>12}")
    print("-" * 61)

    for i, (bt, rt) in enumerate(zip(b_timing, r_timing)):
        b_total = bt['total_diffusion_ms']
        r_total = rt['total_diffusion_ms']
        overhead_ms = r_total - b_total
        overhead_pct = 100 * overhead_ms / b_total if b_total > 0 else 0
        b_fps = bt['fps']
        r_fps = rt['fps']

        print(f"\n  Prompt {i}:")
        print(f"  {'Total diffusion':<23} {b_total:>10.0f}ms {r_total:>10.0f}ms {overhead_ms:>+10.0f}ms")
        print(f"  {'Avg block':<23} {bt['avg_block_ms']:>10.1f}ms {rt['avg_block_ms']:>10.1f}ms")
        print(f"  {'FPS (pixel frames)':<23} {b_fps:>10.1f}   {r_fps:>10.1f}   {r_fps-b_fps:>+10.1f}")

        if 'rag_overhead_ms' in rt:
            print(f"  {'RAG overhead':<23} {'':>12} {rt['rag_overhead_ms']:>10.1f}ms ({rt['rag_overhead_pct']:.1f}%)")
            print(f"    store: {rt['rag_store_ms']:.1f}ms ({rt['rag_store_calls']} calls)  "
                  f"retrieve: {rt['rag_retrieve_ms']:.1f}ms ({rt['rag_retrieve_calls']} calls)")

    # Averages
    avg_b_fps = np.mean([t['fps'] for t in b_timing])
    avg_r_fps = np.mean([t['fps'] for t in r_timing])
    avg_b_total = np.mean([t['total_diffusion_ms'] for t in b_timing])
    avg_r_total = np.mean([t['total_diffusion_ms'] for t in r_timing])
    avg_overhead = avg_r_total - avg_b_total
    avg_overhead_pct = 100 * avg_overhead / avg_b_total if avg_b_total > 0 else 0

    print(f"\n{'─'*61}")
    print(f"  {'AVERAGE':<23} {avg_b_total:>10.0f}ms {avg_r_total:>10.0f}ms {avg_overhead:>+10.0f}ms ({avg_overhead_pct:+.1f}%)")
    print(f"  {'AVERAGE FPS':<23} {avg_b_fps:>10.1f}   {avg_r_fps:>10.1f}   {avg_r_fps-avg_b_fps:>+10.1f}")
    print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="videos/reappear-baseline", help="Baseline video folder")
    parser.add_argument("--rag", default="videos/reappear-rag", help="RAG video folder")
    parser.add_argument("--output", default="videos/reappear-comparison", help="Output folder")
    cli_args = parser.parse_args()

    baseline_dir = cli_args.baseline
    rag_dir = cli_args.rag
    output_dir = cli_args.output
    os.makedirs(output_dir, exist_ok=True)

    baseline_videos = sorted(Path(baseline_dir).glob("rank0-*_lora.mp4"))
    rag_videos = sorted(Path(rag_dir).glob("rank0-*_lora.mp4"))

    assert len(baseline_videos) == len(rag_videos), \
        f"Mismatch: {len(baseline_videos)} baseline vs {len(rag_videos)} RAG videos"

    # Visual comparison
    results = []
    for i, (bv, rv) in enumerate(zip(baseline_videos, rag_videos)):
        r = make_comparison(bv, rv, output_dir, i)
        results.append(r)

    # Summary table
    print(f"\n{'='*75}")
    print("QUALITY SUMMARY")
    print(f"{'='*75}")
    print(f"{'Prompt':<8} {'Late SSIM (B)':>14} {'Late SSIM (R)':>14} {'Diff':>8}  {'Reappear (B)':>13} {'Reappear (R)':>13} {'Diff':>8}")
    print("-" * 75)
    for r in results:
        print(f"  {r['prompt_idx']:<6} {r['baseline_late_ssim']:>14.4f} {r['rag_late_ssim']:>14.4f} {r['consistency_diff']:>+8.4f}"
              f"  {r['baseline_reappear_ssim']:>13.4f} {r['rag_reappear_ssim']:>13.4f} {r['reappear_diff']:>+8.4f}")
    avg_c = np.mean([r['consistency_diff'] for r in results])
    avg_r = np.mean([r['reappear_diff'] for r in results])
    print("-" * 75)
    print(f"  {'Avg':<6} {'':>14} {'':>14} {avg_c:>+8.4f}  {'':>13} {'':>13} {avg_r:>+8.4f}")
    print(f"{'='*75}")

    # Latency comparison
    print_latency_comparison(baseline_dir, rag_dir)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
