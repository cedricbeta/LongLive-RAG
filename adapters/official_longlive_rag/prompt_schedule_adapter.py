"""Experimental prompt switches for pinned official LongLive-RAG (no repair).

Refresh the complete resident sink/recent window under the next prompt while
preserving historical CPU retrieval entries and latent descriptors. Disabled
schedules add no generator/text-encoder calls. Timing follows the native decoder.
"""
import hashlib
import time

import torch


def validate_schedule(schedule, initial_prompts, num_frames, block_size):
    if schedule is None:
        return {}
    result = {}
    for row in schedule:
        start = row['start_latent_frame']
        prompts = list(row['prompts'])
        if isinstance(start, bool) or not isinstance(start, int) or start % block_size:
            raise ValueError('Prompt switch must use an integer block-aligned latent index')
        if start < 0 or start >= num_frames or start in result:
            raise ValueError('Out-of-range or duplicate prompt boundary')
        if len(prompts) != len(initial_prompts) or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError('Each boundary needs one nonempty prompt per batch element')
        result[start] = prompts
    if 0 not in result or result[0] != list(initial_prompts):
        raise ValueError('Schedule at zero must equal the original initial prompts')
    return result


def native_decoded_prefix(latent_frames, chunk_size=60):
    """Number of decoded frames preceding an aligned latent boundary.

Each independently decoded chunk contributes 4T-3. This is a boundary timing
account, not an assertion of exact within-chunk VAE causality.
"""
    if latent_frames < 0 or chunk_size <= 0:
        raise ValueError('Invalid latent boundary or decode chunk size')
    chunks = (latent_frames + chunk_size - 1) // chunk_size
    return 4 * latent_frames - 3 * chunks


def _tensor_hash(tensor):
    tensor = tensor.detach().contiguous().cpu().view(torch.uint8)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _history_hash(pipeline):
    # Hash the actual pool contents; list identity alone would miss in-place changes.
    h = hashlib.sha256()
    for layer in pipeline.kv_cache1:
        for key in ('cpu_k_frames', 'cpu_v_frames'):
            for tensor in layer[key]:
                h.update(_tensor_hash(tensor).encode())
    for tensor in pipeline.latent_descriptors:
        h.update(_tensor_hash(tensor).encode())
    return h.hexdigest()


def refresh_prompt(pipeline, output, current_start_frame, prompts, memory_indices, *, audit=False):
    started = time.monotonic()
    tokens = pipeline.frame_seq_length
    caches = pipeline.kv_cache1
    positions = [(int(c['global_end_index']), int(c['local_end_index'])) for c in caches]
    if not positions or len(set(positions)) != 1:
        raise ValueError('Layer cache positions disagree at prompt boundary')
    end, local_end = positions[0]
    if end != current_start_frame * tokens or local_end % tokens or end % tokens:
        raise ValueError('Cache positions do not match the requested boundary')
    resident = local_end // tokens
    sink = min(int(getattr(pipeline.args.model_kwargs, 'sink_size', 0)), resident)
    if not 0 < resident <= current_start_frame:
        raise ValueError('Invalid resident cache length')
    # The cache is compacted as [original sink, latest rolling frames].
    indices = list(range(sink)) + list(range(current_start_frame - (resident - sink), current_start_frame))
    if len(indices) != resident or len(set(indices)) != resident:
        raise ValueError('Invalid compacted context frame mapping')
    counts = [(len(c['cpu_k_frames']), len(c['cpu_v_frames'])) for c in caches]
    if len(set(counts)) != 1:
        raise ValueError('Layer retrieval pool counts disagree')
    if len(pipeline.latent_descriptors) != current_start_frame:
        raise ValueError('Descriptors do not cover the generated history')
    before = _history_hash(pipeline) if audit else None
    old_keys = [_tensor_hash(c['k']) for c in caches] if audit else None
    old_values = [_tensor_hash(c['v']) for c in caches] if audit else None
    device, dtype = caches[0]['k'].device, caches[0]['k'].dtype
    context = output[:, indices].to(device=device, dtype=dtype)
    conditional = pipeline.text_encoder(text_prompts=prompts)
    pipeline._initialize_crossattn_cache(batch_size=context.shape[0], dtype=dtype, device=device)
    # The synthetic contiguous start maps the compacted sequence back to local
    # positions 0..resident-1, while current_end stays at the true global cursor.
    pipeline.generator(noisy_image_or_video=context, conditional_dict=conditional,
        timestep=torch.full(context.shape[:2], getattr(pipeline.args, 'context_noise', 0.0),
                            dtype=torch.int64, device=device),
        kv_cache=caches, crossattn_cache=pipeline.crossattn_cache,
        current_start=(current_start_frame - resident) * tokens,
        sink_recache_after_switch=True, memory_indices=memory_indices)
    after_positions = [(int(c['global_end_index']), int(c['local_end_index'])) for c in caches]
    after_counts = [(len(c['cpu_k_frames']), len(c['cpu_v_frames'])) for c in caches]
    if after_positions != positions or after_counts != counts:
        raise RuntimeError('Prompt recache moved cursors or changed the retrieval pool length')
    after = _history_hash(pipeline) if audit else None
    if audit and before != after:
        raise RuntimeError('Prompt recache modified historical retrieval entries or descriptors')
    report = {'start_latent_frame': current_start_frame,
        'native_decoded_prefix_frames': native_decoded_prefix(current_start_frame),
        'native_fps': 16, 'resident_source_frames': indices,
        'cache_positions_before': positions, 'cache_positions_after': after_positions,
        'retrieval_pool_counts_before': counts, 'retrieval_pool_counts_after': after_counts,
        'descriptor_count': len(pipeline.latent_descriptors),
        'history_sha256_before': before, 'history_sha256_after': after,
        'history_content_audited': audit,
        'resident_k_sha256_before': old_keys,
        'resident_k_sha256_after': [_tensor_hash(c['k']) for c in caches] if audit else None,
        'resident_v_sha256_before': old_values,
        'resident_v_sha256_after': [_tensor_hash(c['v']) for c in caches] if audit else None,
        'cross_attention_initialized': [bool(c['is_init']) for c in pipeline.crossattn_cache],
        'prompt_sha256': [hashlib.sha256(p.encode()).hexdigest() for p in prompts]}
    report['wall_seconds_including_optional_audit'] = time.monotonic() - started
    return conditional, report
