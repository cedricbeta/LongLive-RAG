"""Post-softmax attention-mass probe for LongLive-RAG memory tokens.

Mirrors the rag_attn_mass diagnostic from the local LongLive-RAG repo:
for a subsample of query tokens, compute the exact softmax over the
concatenated [sink | memory | local] context and report the attention
mass landing on each region, compared with the uniform baseline
(region_tokens / total_tokens).
"""
import torch


class AttnMassProbe:
    def __init__(self, layers=None, max_queries=128, seed=0):
        self.layers = set(layers) if layers is not None else None
        self.max_queries = max_queries
        self.records = []
        self.meta = {}
        self.gen = torch.Generator().manual_seed(seed)
        # Region split state, set by the pipeline before each block's forwards:
        # mem_subject_mask/known_mask are flat bool tensors of length mem_len
        # (k_sel frames x frame_seqlen), mem_frame_ids the matching global
        # frame indices per slot.
        self.mem_subject_mask = None
        self.mem_known_mask = None
        self.mem_frame_ids = None
        self.frame_seqlen = 1560
        # per-frame accumulated per-column attention mass (for heatmaps)
        self.frame_col_sum = {}
        self.frame_col_count = {}

    def record(self, layer, q, k, sink_len, mem_len):
        if self.layers is not None and layer not in self.layers:
            return
        b, s, n, d = q.shape
        total = k.shape[1]
        local_len = total - sink_len - mem_len
        if s > self.max_queries:
            idx = torch.randperm(s, generator=self.gen)[: self.max_queries].to(q.device)
            q_sub = q[:, idx]
        else:
            q_sub = q
        with torch.no_grad():
            logits = torch.einsum(
                "bqnd,bknd->bnqk", q_sub.float(), k.float()
            ) * (d ** -0.5)
            attn = logits.softmax(dim=-1)
            mass_sink = attn[..., :sink_len].sum(-1).mean().item()
            mass_mem = attn[..., sink_len : sink_len + mem_len].sum(-1).mean().item()
            mass_local = attn[..., sink_len + mem_len :].sum(-1).mean().item()
            # per-head spread for the memory region
            mem_per_head = attn[..., sink_len : sink_len + mem_len].sum(-1).mean(dim=(0, 2))

            # region split: subject vs background tokens inside injected frames
            subj = dict(mass_mem_subj=None, mass_mem_bg=None,
                        n_subj_tokens=None, n_known_tokens=None)
            mask = self.mem_subject_mask
            if mask is not None and mask.numel() == mem_len and mem_len > 0:
                mem_attn = attn[..., sink_len : sink_len + mem_len]  # [b,n,q,mem]
                mask_dev = mask.to(mem_attn.device)
                known_dev = self.mem_known_mask.to(mem_attn.device)
                bg_dev = known_dev & ~mask_dev
                subj = dict(
                    mass_mem_subj=mem_attn[..., mask_dev].sum(-1).mean().item(),
                    mass_mem_bg=mem_attn[..., bg_dev].sum(-1).mean().item(),
                    n_subj_tokens=int(mask_dev.sum().item()),
                    n_known_tokens=int(known_dev.sum().item()),
                )
                # accumulate per-column mass for heatmaps (mean over b,n,q)
                col = mem_attn.mean(dim=(0, 1, 2)).float().cpu()  # [mem_len]
                fs = self.frame_seqlen
                if self.mem_frame_ids is not None:
                    for si, fid in enumerate(self.mem_frame_ids):
                        seg = col[si * fs : (si + 1) * fs]
                        if seg.numel() != fs:
                            continue
                        if fid in self.frame_col_sum:
                            self.frame_col_sum[fid] += seg
                            self.frame_col_count[fid] += 1
                        else:
                            self.frame_col_sum[fid] = seg.clone()
                            self.frame_col_count[fid] = 1
        self.records.append(
            dict(
                **subj,
                layer=layer,
                sink_len=sink_len,
                mem_len=mem_len,
                local_len=local_len,
                total_len=total,
                mass_sink=mass_sink,
                mass_mem=mass_mem,
                mass_local=mass_local,
                mem_head_max=mem_per_head.max().item(),
                mem_head_min=mem_per_head.min().item(),
                uniform_sink=sink_len / total,
                uniform_mem=mem_len / total,
                uniform_local=local_len / total,
                **self.meta,
            )
        )
