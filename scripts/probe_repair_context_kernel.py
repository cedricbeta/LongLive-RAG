#!/usr/bin/env python3
"""Real native CUDA attention/rolling-cache contracts with synthetic tensors."""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def main(args):
    began = time.monotonic()
    sys.path.insert(0, str(args.checkout.resolve()))
    import torch
    import repair_context as context
    native = importlib.import_module('wan.modules.causal_model_latentmem')
    cross = importlib.import_module('wan.modules.model')
    canonical = native.attention
    original_cross = cross.flash_attention
    for name, expected in context.PINNED_SOURCES.items():
        assert hashlib.sha256((args.checkout / name).read_bytes()).hexdigest() == expected
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    rows = []
    with torch.inference_mode():
        for case, start, end, local, frames in [('rolling', 81, 81, 12, 3),
                ('direct', 9, 9, 9, 3), ('recompute', 9, 12, 12, 3), ('last_single', 120, 120, 12, 1)]:
            model = native.CausalWanSelfAttention(64, 2, local_attn_size=12, sink_size=1, memory_size=6).cuda().bfloat16().eval()
            model.max_attention_size = 24  # Tiny two-token spatial grid; same twelve-frame budget.
            x = torch.randn(1, frames * 2, 64, device='cuda', dtype=torch.bfloat16)
            cache = {'k': torch.randn(1, 24, 2, 32, device='cuda', dtype=torch.bfloat16),
                     'v': torch.randn(1, 24, 2, 32, device='cuda', dtype=torch.bfloat16),
                     'global_end_index': torch.tensor(start * 2 if case != 'recompute' else end * 2, device='cuda'),
                     'local_end_index': torch.tensor(local * 2, device='cuda'),
                     'cpu_k_frames': [torch.randn(1, 1, 2, 2, 32, dtype=torch.bfloat16) for _ in range(69)],
                     'cpu_v_frames': [torch.randn(1, 1, 2, 2, 32, dtype=torch.bfloat16) for _ in range(69)]}
            original_cache = copy.deepcopy(cache)
            freqs = torch.cat([native.rope_params(1024, 12), native.rope_params(1024, 10),
                               native.rope_params(1024, 10)], 1).cuda()
            grid = torch.tensor([[frames, 1, 2]], device='cuda')
            kwargs = dict(x=x, seq_lens=torch.tensor([frames * 2], device='cuda'), grid_sizes=grid,
                          freqs=freqs, block_mask=None, kv_cache=cache, current_start=start * 2,
                          memory_indices=torch.tensor([[0, 1, 2, 3, 4, 5]], device='cuda'))
            captured = []
            def capture(q, k, v):
                captured.append((q.clone(), k.clone(), v.clone()))
                return canonical(q, k, v)
            native.attention = capture
            full, metadata = model(**kwargs)
            native.attention = canonical
            off_trace, on_trace = [], []
            with context.attention_scope(native, 'full_context_v1', frames * 2, 1, off_trace):
                assert native.attention is canonical
                off, off_meta = model(**kwargs)
            with context.attention_scope(native, 'current_block_only_v1', frames * 2, 1, on_trace):
                assert cross.flash_attention is original_cross
                on, on_meta = model(**kwargs)
            q, k, v = captured[0]
            expected = model.o(canonical(q, k[:, -frames * 2:], v[:, -frames * 2:]).flatten(2))
            assert torch.equal(full, off) and torch.equal(on, expected)
            projected_k = model.norm_k(model.k(x)).view(1, frames * 2, 2, 32)
            projected_v = model.v(x).view(1, frames * 2, 2, 32)
            # The actual native rolling/direct/recompute branches all finish at frame 12 here.
            expected_k = native.causal_online_rope(projected_k, grid, freqs,
                relative_frame_indices=torch.arange(12 - frames, 12, device='cuda')).type_as(projected_v)
            assert torch.equal(k[:, -frames * 2:], expected_k)
            assert torch.equal(v[:, -frames * 2:], projected_v)
            def equal(a, b):
                if isinstance(a, torch.Tensor):
                    return torch.equal(a, b)
                if isinstance(a, dict):
                    return a.keys() == b.keys() and all(equal(a[n], b[n]) for n in a)
                if isinstance(a, (list, tuple)):
                    return len(a) == len(b) and all(equal(c, d) for c, d in zip(a, b))
                return a == b
            assert equal(metadata, off_meta) and equal(metadata, on_meta)
            assert equal(cache, original_cache) and native.attention is canonical
            rows.append({'case': case, 'off_output_exact': True, 'on_equals_explicit_current_block': True,
                         'tail_keys_equal_current_projected_roped_keys': True,
                         'tail_values_equal_current_projected_values': True, 'input_cache_exact': True,
                         'returned_cache_update_metadata_exact': True,
                         'full_vs_on_output_equal': torch.equal(full, on), 'on_trace': on_trace})
    record = {'protocol': 'native_attention_context_contract_v1', 'cases': rows,
        'wall_seconds': time.monotonic() - began, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'context_source_sha256': hashlib.sha256((args.checkout / 'repair_context.py').read_bytes()).hexdigest(),
        'native_source_sha256': context.PINNED_SOURCES, 'torch': torch.__version__,
        'cuda_device': torch.cuda.get_device_name(), 'new_video_generations': 0,
        'limits': ['Synthetic random weights and a two-token spatial grid; not a pretrained model or visual efficacy test.',
                   'Runs the pinned native cached forward and actual installed CUDA attention kernel.']}
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    try:
        main(args)
    except Exception as error:
        args.output.write_text(json.dumps({'status': 'execution_failed',
            'error': f'{type(error).__name__}: {error}', 'wall_seconds': time.monotonic() - started,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
        raise
