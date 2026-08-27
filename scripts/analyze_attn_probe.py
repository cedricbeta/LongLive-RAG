#!/usr/bin/env python
"""Offline analyses E1-E5 for the attention down-weighting study.

Reads the attn_probe dumps written during an instrumented replay
(utils/kv_rag.py attn_probe_*; see docs/attention-downweighting-study.zh.md)
and produces, per scene:

  E1  distance-decay curves      logit(q, rope(k, Delta)) for same-shot /
                                 cross-shot / shuffled key pools, per layer,
                                 with the zero-parameter RoPE theory kernel
                                 overlaid.
  E3  logit waterfall            local-near logit -> position component ->
                                 content component -> injected logit (nats).
  E2  slot counterfactual        softmax mass on the injected block when it
                                 is re-placed at sink-adjacent / mid-window /
                                 current (window-edge) virtual positions.
  E4  phase gap                  current-chunk K at each denoise step vs the
                                 same chunk's clean-recache K (cosine).
  E5  head census                per-head mass ratio injected vs matched
                                 local columns.

All logits are computed the way the model computes them (post-RoPE q.k /
sqrt(d), fp32), on the exact tensors the live attention call saw.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HEAD_DIM = 128
C = HEAD_DIM // 2  # complex dims per head
SPLIT = (C - 2 * (C // 3), C // 3, C // 3)  # (22, 21, 21)


# ---------------------------------------------------------------- rope utils


def load_freqs(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    freqs = torch.load(root / "rope_freqs.pt", map_location="cpu",
                       weights_only=False)["freqs"]
    return freqs.split(list(SPLIT), dim=1)  # complex [1024, c_part] each


def temporal_rot(freqs_t: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotation factors for (possibly non-integer) temporal positions [F] -> [F, 22]."""
    base = torch.angle(freqs_t[1]).to(torch.float64)  # [22]
    ang = positions.to(torch.float64).unsqueeze(1) * base.unsqueeze(0)
    return torch.polar(torch.ones_like(ang), ang)


def rope_full(
    x: torch.Tensor,
    frame_positions: torch.Tensor,
    grid_h: int,
    grid_w: int,
    freqs_parts,
) -> torch.Tensor:
    """Apply 3D RoPE to x [F*h*w, H, 128] with given temporal frame positions [F]."""
    ft, fh, fw = freqs_parts
    F = int(frame_positions.numel())
    xc = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], C, 2)
    )
    rot = torch.cat(
        [
            temporal_rot(ft, frame_positions).view(F, 1, 1, -1).expand(F, grid_h, grid_w, -1),
            fh[:grid_h].view(1, grid_h, 1, -1).expand(F, grid_h, grid_w, -1),
            fw[:grid_w].view(1, 1, grid_w, -1).expand(F, grid_h, grid_w, -1),
        ],
        dim=-1,
    ).reshape(x.shape[0], 1, C)
    return torch.view_as_real(xc * rot).flatten(2).to(torch.float32)


def to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], C, 2))


def scalar_offset(value) -> float:
    """Best-effort scalar view of the dumped temporal_offset."""
    if torch.is_tensor(value):
        v = value.flatten()
        return float(v[0].item()) if v.numel() else 0.0
    return float(value or 0.0)


MAX_Q_ROWS = 256


def subsample_q(q: torch.Tensor) -> torch.Tensor:
    """Cap query rows so [H, Lq, Lk] logit tensors stay in memory."""
    if q.shape[0] <= MAX_Q_ROWS:
        return q
    idx = torch.linspace(0, q.shape[0] - 1, MAX_Q_ROWS).round().long()
    return q.index_select(0, idx)


# ---------------------------------------------------------------- data access


def attn_files(root: Path, layer: int) -> list[Path]:
    return sorted((root / "attn" / f"layer_{layer:02d}").glob("chunk_*_step0.pt"))


def attn_file_step(root: Path, layer: int, chunk: int, step: int) -> Path:
    return root / "attn" / f"layer_{layer:02d}" / f"chunk_{chunk:03d}_step{step}.pt"


def store_file(root: Path, layer: int, chunk: int) -> Path:
    return root / "store_k" / f"layer_{layer:02d}" / f"chunk_{chunk:03d}.pt"


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def store_index(root: Path, layer: int) -> dict[int, dict]:
    """chunk_index -> lightweight meta (shot_index, start/end tokens)."""
    out = {}
    for p in sorted((root / "store_k" / f"layer_{layer:02d}").glob("chunk_*.pt")):
        d = load(p)
        out[int(d["chunk_index"])] = {
            "shot_index": d.get("shot_index"),
            "start_token": int(d["start_token"]),
            "end_token": int(d["end_token"]),
            "path": p,
        }
    return out


# ---------------------------------------------------------------- E1


def pair_spectrum(q_pre: torch.Tensor, k_pre: torch.Tensor,
                  frame_seqlen: int) -> torch.Tensor:
    """Mean complex spectrum S[H, C] over same-spatial-cell (q, k) frame pairs.

    q_pre [Fq*s, H, D], k_pre [Fk*s, H, D] with s = frame_seqlen tokens/frame.
    Pairing same spatial cell makes the spatial rotation cancel exactly, so
    Re(sum_c S_c * rot_t(Delta)_c) is the mean roped logit at temporal
    distance Delta (times sqrt(d) scaling handled by caller).
    """
    s = frame_seqlen
    fq, fk = q_pre.shape[0] // s, k_pre.shape[0] // s
    qc = to_complex(q_pre).reshape(fq, s, q_pre.shape[1], C)
    kc = to_complex(k_pre).reshape(fk, s, k_pre.shape[1], C)
    # mean over frames first (linear), keep spatial cells paired
    S = (qc.mean(0).unsqueeze(-3) * kc.mean(0).conj().unsqueeze(-3)).squeeze(-3)
    return S.mean(0)  # mean over spatial cells -> [H, C]


def logit_vs_delta(S: torch.Tensor, freqs_t: torch.Tensor,
                   deltas: torch.Tensor) -> torch.Tensor:
    """[H] mean logits per Delta -> [len(deltas)] (mean over heads)."""
    rot = temporal_rot(freqs_t, deltas)  # [Nd, 22]
    n_t = SPLIT[0]
    S_t = S[:, :n_t]        # temporal dims rotate
    S_s = S[:, n_t:].real.sum(dim=-1)  # spatial dims: same cell -> rot = 1
    vals = (S_t.unsqueeze(0) * rot.unsqueeze(1)).real.sum(dim=-1) + S_s.unsqueeze(0)
    return vals.mean(dim=1).float() / math.sqrt(HEAD_DIM)


def run_e1(root: Path, layers: list[int], out_dir: Path, scene: str) -> dict:
    freqs_parts = load_freqs(root)
    ft = freqs_parts[0]
    deltas = torch.arange(0, 65)
    theory = ((1 - SPLIT[0] * 2 / HEAD_DIM)
              + (SPLIT[0] * 2 / HEAD_DIM)
              * np.cos(np.outer(deltas.numpy(),
                                np.angle(ft[1].numpy()))).mean(axis=1))

    fig, axes = plt.subplots(1, len(layers), figsize=(3.4 * len(layers), 3.4),
                             sharey=False)
    summary = {}
    for ax, layer in zip(np.atleast_1d(axes), layers):
        sindex = store_index(root, layer)
        if not sindex:
            ax.set_title(f"L{layer}: no store dumps")
            continue
        curves = defaultdict(list)
        for af in attn_files(root, layer):
            meta = load(af)
            fsl = int(meta["frame_seqlen"])
            chunk = int(meta["chunk_index"])
            shot = meta.get("shot_index")
            q = meta["q_pre"].float()
            if q.shape[0] % fsl:
                continue
            # adjacent pool: the chunk generated right before this boundary
            adjacent = [c for c in sindex if c < chunk][-1:]
            # anchor-source pool: an early chunk from the first shot
            shots = sorted({m["shot_index"] for m in sindex.values()
                            if m["shot_index"] is not None})
            cross = [c for c, m in sindex.items()
                     if shots and m["shot_index"] == shots[0]
                     and m["shot_index"] != shot]
            for kind, pool in (("adjacent_past", adjacent), ("anchor_source", cross[:1])):
                for c in pool:
                    k = load(sindex[c]["path"])["k_pre"].float()
                    if k.shape[0] % fsl:
                        continue
                    S = pair_spectrum(q, k, fsl)
                    curves[kind].append(logit_vs_delta(S, ft, deltas))
                    if kind == "anchor_source":
                        # content-destroyed control: shuffle feature dims
                        perm = torch.randperm(k.shape[-1])
                        Ss = pair_spectrum(q, k[..., perm], fsl)
                        curves["shuffled"].append(logit_vs_delta(Ss, ft, deltas))
        colors = {"adjacent_past": "#2a9d8f", "anchor_source": "#e76f51",
                  "shuffled": "#adb5bd"}
        for kind, cs in curves.items():
            arr = torch.stack(cs).numpy()
            ax.plot(deltas, arr.mean(0), color=colors[kind], lw=1.8, label=kind)
            ax.fill_between(deltas, arr.mean(0) - arr.std(0),
                            arr.mean(0) + arr.std(0),
                            color=colors[kind], alpha=0.15)
        if "adjacent_past" in curves:
            base = torch.stack(curves["adjacent_past"]).numpy().mean(0)[0]
            ax.plot(deltas, theory * base, "k--", lw=1, label="RoPE theory")
            summary[f"L{layer}"] = {
                k: torch.stack(v).numpy().mean(0).tolist()
                for k, v in curves.items()
            }
        ax.axvline(32, color="green", ls=":", lw=1)
        ax.set_title(f"layer {layer}", fontsize=9)
        ax.set_xlabel("temporal distance (latent frames)", fontsize=8)
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("mean logit (nats)", fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(f"E1 {scene}: content vs position in the q*k logit "
                 "(same spatial cell pairs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{scene}_e1_distance_decay.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    return summary


# ---------------------------------------------------------------- E3 + E5


def window_spans(meta: dict) -> dict[str, tuple[int, int]]:
    rag = int(meta["rag_tokens"])
    sink = int(meta["effective_sink"])
    pinned = int(meta["pinned_len"])
    total = int(meta["window_tokens"])
    chunk = int(meta["current_end"]) - int(meta["current_start"])
    return {
        "rag": (0, rag),
        "sink": (rag, rag + sink),
        "pinned": (rag + sink, rag + sink + pinned),
        "local_past": (rag + sink + pinned, total - chunk),
        "chunk": (total - chunk, total),
    }


def run_e3_e5(root: Path, layers: list[int], out_dir: Path, scene: str) -> dict:
    """E3: per-span logits + content-at-same-slot comparison. E5: head census.

    Real boundary geometry (measured): at a shot boundary the attended window
    is [rag | sink | pinned | current chunk] -- there is no same-shot local
    past. The reference for "how well does accepted old content score" is
    therefore the PINNED span (previous-shot anchor at its trained slot); the
    content axis is isolated by re-roping the temporally-adjacent chunk's
    clean K into the exact rag slot and comparing logits at equal position.
    """
    freqs_parts = load_freqs(root)
    rows_e3: dict[int, dict] = {}
    head_ratio: dict[int, np.ndarray] = {}
    for layer in layers:
        files = attn_files(root, layer)
        if not files:
            continue
        sindex = store_index(root, layer)
        comp = defaultdict(list)
        ratios = []
        for af in files:
            meta = load(af)
            if "window_k_post" not in meta:
                continue
            spans = window_spans(meta)
            q = subsample_q(meta["q_post"].float())
            k = meta["window_k_post"].float()
            fsl = int(meta["frame_seqlen"])
            rag_n = spans["rag"][1] - spans["rag"][0]
            if rag_n <= 0 or rag_n % fsl:
                continue
            scale = 1.0 / math.sqrt(HEAD_DIM)
            logits = torch.einsum("lhd,mhd->hlm", q, k) * scale  # [H, Lq, Lk]
            w = torch.softmax(logits, dim=-1)

            def span_stat(name):
                lo, hi = spans[name]
                if hi <= lo:
                    return float("nan"), float("nan")
                return (logits[..., lo:hi].mean().item(),
                        w[..., lo:hi].sum(-1).mean().item())

            l_rag, m_rag = span_stat("rag")
            l_sink, m_sink = span_stat("sink")
            l_pin, m_pin = span_stat("pinned")
            l_past, m_past = span_stat("local_past")
            l_chunk, m_chunk = span_stat("chunk")
            ref = "local_past" if spans["local_past"][1] > spans["local_past"][0] \
                else "pinned"
            l_ref = l_past if ref == "local_past" else l_pin

            # content axis at EQUAL position: adjacent chunk's clean K re-roped
            # into the rag slot vs the rag content actually there
            l_adj_slot = float("nan")
            rsf = meta.get("rag_start_frame")
            adjacent = [c for c in sindex if c < int(meta["chunk_index"])]
            if rsf is not None and adjacent:
                kd = load(sindex[adjacent[-1]]["path"])
                k_pre = kd["k_pre"].float()
                gh, gw = int(meta["grid_h"]), int(meta["grid_w"])
                rag_frames = rag_n // fsl
                frames = k_pre.shape[0] // fsl
                use = min(frames, rag_frames)
                if use > 0:
                    sub = k_pre[-use * fsl:]  # newest frames of adjacent chunk
                    toff = scalar_offset(meta.get("temporal_offset", 0.0))
                    pos = torch.arange(use) + int(rsf) + toff
                    k_moved = rope_full(sub, pos, gh, gw, freqs_parts)
                    l_adj_slot = (torch.einsum("lhd,mhd->hlm", q, k_moved)
                                  * scale).mean().item()

            for key, val in (("l_rag", l_rag), ("l_sink", l_sink),
                             ("l_pinned", l_pin), ("l_local_past", l_past),
                             ("l_chunk", l_chunk), ("l_ref", l_ref),
                             ("l_adj_at_ragslot", l_adj_slot),
                             ("m_rag", m_rag), ("m_sink", m_sink),
                             ("m_pinned", m_pin), ("m_chunk", m_chunk)):
                comp[key].append(val)
            comp["ref_kind"].append(ref)

            # E5: per-head mass ratio rag vs reference span
            lo, hi = spans[ref]
            if hi > lo:
                mh_rag = w[..., spans["rag"][0]:spans["rag"][1]].sum(-1).mean(dim=1)
                mh_ref = w[..., lo:hi].sum(-1).mean(dim=1)
                # normalize by span sizes so the ratio is per-column
                mh_rag = mh_rag / rag_n
                mh_ref = mh_ref / (hi - lo)
                ratios.append((mh_rag / mh_ref.clamp_min(1e-12)).numpy())
        if comp["l_rag"]:
            row = {k: float(np.nanmean(v)) for k, v in comp.items()
                   if k != "ref_kind"}
            row["ref_kind"] = comp["ref_kind"][0]
            row["gap_vs_ref"] = row["l_rag"] - row["l_ref"]
            row["content_gap_same_slot"] = row["l_adj_at_ragslot"] - row["l_rag"]
            rows_e3[layer] = row
        if ratios:
            head_ratio[layer] = np.stack(ratios).mean(axis=0)

    # E3 figure: per-span logits + the same-slot content comparison
    if rows_e3:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        ls = sorted(rows_e3)
        x = np.arange(len(ls))
        names = [("l_rag", "injected (rag slot)", "#e76f51"),
                 ("l_adj_at_ragslot", "adjacent content @ same slot", "#2a9d8f"),
                 ("l_pinned", "pinned anchor (trained slot)", "#4c72b0"),
                 ("l_sink", "sink", "#999999"),
                 ("l_chunk", "current chunk (self)", "#c9a227")]
        wdt = 0.8 / len(names)
        for i, (key, label, color) in enumerate(names):
            vals = [rows_e3[l].get(key, np.nan) for l in ls]
            axes[0].bar(x + (i - len(names) / 2 + 0.5) * wdt, vals, wdt,
                        label=label, color=color)
        axes[0].set_xticks(x, [f"L{l}" for l in ls])
        axes[0].set_ylabel("mean logit (nats)")
        axes[0].set_title("per-span mean logits at boundaries")
        axes[0].legend(fontsize=7)
        axes[0].grid(axis="y", alpha=0.3, lw=0.4)
        mnames = [("m_rag", "injected", "#e76f51"),
                  ("m_pinned", "pinned", "#4c72b0"),
                  ("m_sink", "sink", "#999999"),
                  ("m_chunk", "current chunk", "#c9a227")]
        wdt = 0.8 / len(mnames)
        for i, (key, label, color) in enumerate(mnames):
            vals = [rows_e3[l].get(key, np.nan) for l in ls]
            axes[1].bar(x + (i - len(mnames) / 2 + 0.5) * wdt, vals, wdt,
                        label=label, color=color)
        axes[1].set_xticks(x, [f"L{l}" for l in ls])
        axes[1].set_ylabel("mean softmax mass")
        axes[1].set_title("where the mass actually goes")
        axes[1].legend(fontsize=7)
        axes[1].grid(axis="y", alpha=0.3, lw=0.4)
        fig.suptitle(f"E3 {scene}: injected vs privileged spans "
                     "(boundary windows, lambda=0)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{scene}_e3_waterfall.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # E5 figure: per-head per-column mass ratio distribution
    if head_ratio:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ls = sorted(head_ratio)
        ax.boxplot([head_ratio[l] for l in ls],
                   tick_labels=[f"L{l}" for l in ls], showfliers=True)
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylabel("per-head per-column mass: injected / reference span")
        ax.set_title(f"E5 {scene}: head census (each point = one head)")
        ax.grid(axis="y", which="both", alpha=0.3, lw=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"{scene}_e5_head_census.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    return {"e3": {str(k): v for k, v in rows_e3.items()},
            "e5_head_ratio": {str(k): v.tolist() for k, v in head_ratio.items()}}


# ---------------------------------------------------------------- E2


def run_e2(root: Path, layers: list[int], out_dir: Path, scene: str) -> dict:
    freqs_parts = load_freqs(root)
    placements = {}
    for layer in layers:
        rows = defaultdict(list)
        for af in attn_files(root, layer):
            meta = load(af)
            if "window_k_post" not in meta or "rag_k_pre" not in meta:
                continue
            spans = window_spans(meta)
            rag_n = spans["rag"][1] - spans["rag"][0]
            fsl = int(meta["frame_seqlen"])
            if rag_n <= 0 or rag_n % fsl:
                continue
            q = subsample_q(meta["q_post"].float())
            k = meta["window_k_post"].float()
            rag_pre = meta["rag_k_pre"].float()
            gh, gw = int(meta["grid_h"]), int(meta["grid_w"])
            rag_frames = rag_n // fsl
            rsf = meta.get("rag_start_frame")
            if rsf is None:
                continue
            toff = scalar_offset(meta.get("temporal_offset", 0.0))
            sink_frames = int(meta["effective_sink"]) // fsl
            win_frames = (int(meta["window_tokens"]) - rag_n) // fsl
            cur_frame = int(meta["current_end"]) // fsl
            # effective (post-offset) temporal positions per placement:
            #   sink columns keep their ORIGINAL near-zero absolute positions,
            #   so sink-adjacent means right after frame `sink_frames`, without
            #   the current shot's RoPE offset; the live placement (sanity-
            #   checked) is rag_start_frame + current offset.
            starts = {
                "sink_adjacent": float(sink_frames),
                "mid_window": cur_frame + toff - win_frames // 2 - rag_frames,
                "window_edge(current)": float(rsf) + toff,
            }
            scale = 1.0 / math.sqrt(HEAD_DIM)
            for name, sf in starts.items():
                pos = torch.arange(rag_frames) + sf
                k_moved = rope_full(rag_pre, pos, gh, gw, freqs_parts)
                k_cf = torch.cat([k_moved.half().float(), k[rag_n:]], dim=0)
                logits = torch.einsum("lhd,mhd->hlm", q, k_cf) * scale
                w = torch.softmax(logits, dim=-1)
                rows[name].append(w[..., :rag_n].sum(-1).mean().item())
            # sanity: recomputed mass at true placement vs live window
            logits = torch.einsum("lhd,mhd->hlm", q, k) * scale
            w = torch.softmax(logits, dim=-1)
            rows["live_window(sanity)"].append(
                w[..., :rag_n].sum(-1).mean().item())
        if rows:
            placements[layer] = {k: float(np.mean(v)) for k, v in rows.items()}

    if placements:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        ls = sorted(placements)
        names = ["sink_adjacent", "mid_window", "window_edge(current)",
                 "live_window(sanity)"]
        x = np.arange(len(ls))
        wdt = 0.8 / len(names)
        for i, name in enumerate(names):
            vals = [placements[l].get(name, np.nan) for l in ls]
            ax.bar(x + (i - len(names) / 2 + 0.5) * wdt, vals, wdt, label=name)
        ax.set_xticks(x, [f"L{l}" for l in ls])
        ax.set_ylabel("mean mass on injected block")
        ax.set_title(f"E2 {scene}: counterfactual placement of the same "
                     "injected content")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"{scene}_e2_slot_test.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    return {str(k): v for k, v in placements.items()}


# ---------------------------------------------------------------- E4


def run_e4(root: Path, layers: list[int], out_dir: Path, scene: str) -> dict:
    out = {}
    for layer in layers:
        per_step = defaultdict(list)
        for af in attn_files(root, layer):
            meta = load(af)
            chunk = int(meta["chunk_index"])
            clean = None
            sp = store_file(root, layer, chunk)
            if sp.exists():
                clean = load(sp)["k_pre"].float()
            ref = None
            for step in range(0, 8):
                p = attn_file_step(root, layer, chunk, step)
                if not p.exists():
                    break
                d = load(p)
                kc = d["chunk_k_post"].float()
                if ref is None:
                    ref = kc
                cos_prev = torch.nn.functional.cosine_similarity(
                    kc.flatten(0, 1), ref.flatten(0, 1), dim=-1).mean().item()
                per_step[step].append(cos_prev)
            if clean is not None and ref is not None:
                pass  # clean is pre-RoPE; step-to-step drift already informative
        if per_step:
            out[layer] = {int(s): float(np.mean(v)) for s, v in per_step.items()}
    if out:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for layer, d in sorted(out.items()):
            steps = sorted(d)
            ax.plot(steps, [d[s] for s in steps], marker="o",
                    label=f"L{layer}")
        ax.set_xlabel("denoise step (0 = first)")
        ax.set_ylabel("cosine vs step-0 chunk K")
        ax.set_title(f"E4 {scene}: current-chunk key drift across denoise steps")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, lw=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"{scene}_e4_phase_gap.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    return {str(k): v for k, v in out.items()}


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-root", required=True,
                    help="e.g. videos/attn_probe_20260820/<scene>/dumps")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 7, 14, 21, 29])
    ap.add_argument("--out-dir", default="docs/figures/attention_downweighting")
    ap.add_argument("--analyses", nargs="*",
                    default=["e1", "e2", "e3e5", "e4"])
    args = ap.parse_args()

    root = Path(args.dump_root)
    scene = args.scene or root.parent.name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"scene": scene}
    if "e1" in args.analyses:
        summary["e1"] = run_e1(root, args.layers, out_dir, scene)
        print("E1 done")
    if "e3e5" in args.analyses:
        summary.update(run_e3_e5(root, args.layers, out_dir, scene))
        print("E3/E5 done")
    if "e2" in args.analyses:
        summary["e2"] = run_e2(root, args.layers, out_dir, scene)
        print("E2 done")
    if "e4" in args.analyses:
        summary["e4"] = run_e4(root, args.layers, out_dir, scene)
        print("E4 done")

    out_json = out_dir / f"{scene}_analysis_summary.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
