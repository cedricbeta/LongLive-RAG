#!/usr/bin/env python
"""C1 theory curve for docs/attention-downweighting-study.zh.md.

For the Wan 5B head layout (head_dim 128, temporal RoPE dims 44 -> 22
frequency pairs, theta=10000), plot the *identical-content* attention-logit
kernel as a function of temporal distance: a query attending to a key with
the same content but Delta frames away keeps only

    kernel(Delta) = (1 - f_t) + f_t * mean_i cos(theta_i * Delta)

of its Delta=0 logit (isotropic-content expectation; f_t = 44/128 is the
fraction of head dims that rotate with time). The spatial dims (h/w) do not
rotate with temporal distance, hence the (1 - f_t) floor. Real decay depends
on trained q/k statistics and is measured by E1 from the attn_probe dumps;
this figure is the zero-parameter reference the measurement is compared to.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HEAD_DIM = 128
DIM_T = HEAD_DIM - 4 * (HEAD_DIM // 6)  # 44
N_PAIRS = DIM_T // 2  # 22
THETA = 10000.0
F_T = DIM_T / HEAD_DIM

LOCAL_WINDOW_FRAMES = 32  # long_multishot local_attn_size (frames, incl. sink)


def main() -> None:
    freqs = THETA ** (-np.arange(0, DIM_T, 2) / DIM_T)  # [22]
    delta = np.arange(0, 97)
    cos_mean = np.cos(np.outer(delta, freqs)).mean(axis=1)
    kernel = (1 - F_T) + F_T * cos_mean

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(delta, kernel, color="#4c72b0", lw=2,
            label="identical-content kernel (theory)")
    ax.axhline(1 - F_T, color="#4c72b0", ls=":", lw=1,
               label=f"spatial-dims floor = {1-F_T:.2f}")
    ax.axhline(0.0, color="gray", lw=0.8)

    ax.axvspan(0, LOCAL_WINDOW_FRAMES, color="green", alpha=0.06)
    ax.axvline(LOCAL_WINDOW_FRAMES, color="green", ls="--", lw=1.2)
    ax.annotate("local window edge (32 frames)\n= where re-RoPE'd memory sits",
                xy=(LOCAL_WINDOW_FRAMES, kernel[LOCAL_WINDOW_FRAMES]),
                xytext=(40, 0.75), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1, color="green"),
                color="green")

    k32 = kernel[LOCAL_WINDOW_FRAMES]
    ax.annotate(f"kernel(32) = {k32:.2f}\n(~{(1-k32)*100:.0f}% of the matched-content\n logit is rotated away)",
                xy=(8, kernel[8]), xytext=(8, 0.35), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1, color="#4c72b0"))

    ax.set_xlabel("temporal distance Delta (latent frames)")
    ax.set_ylabel("relative logit vs Delta = 0")
    ax.set_title("Wan 5B temporal RoPE: identical-content logit decay "
                 f"({N_PAIRS} freq pairs, theta={THETA:g})", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, lw=0.4)
    ax.set_xlim(0, 96)

    out = Path("docs/figures/attention_downweighting")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rope_temporal_kernel.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    print(f"kernel at 8/16/32/64 frames: "
          f"{kernel[8]:.3f} / {kernel[16]:.3f} / {kernel[32]:.3f} / {kernel[64]:.3f}")


if __name__ == "__main__":
    main()
