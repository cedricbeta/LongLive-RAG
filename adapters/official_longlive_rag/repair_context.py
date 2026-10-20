"""Experimental, repair-only self-attention context intervention.

The pinned native implementation concatenates sink, retrieved and local K/V,
ending in the complete current block. Only that last block is exposed when
``current_block_only_v1`` is explicitly requested. RoPE and cache construction
stay native. This is a renderer diagnostic, not a planner action or acceptor.
"""
from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
import threading


MODES = ('full_context_v1', 'current_block_only_v1')
PINNED_SOURCES = {
    'wan/modules/causal_model_latentmem.py': '5b0cb6431568cfd6717d636c0a9a1f51bf1f71a4e597e3d04ff9ecb2e5c58449',
    'utils/wan_wrapper.py': '4de4e71f290cc061187478b70b97dce5e7c2395c31bfdfdd9d768f97ef6f15e7',
}
_lock = threading.Lock()


@contextmanager
def attention_scope(module, mode, query_tokens, layers, trace):
    """Process-local single-thread scope; restoration also runs on exceptions.

    FULL does not replace the callable. ON records each actual self-attention
    invocation and rejects unexpected shapes before the real kernel runs.
    This helper itself does not establish the native concatenation semantics;
    private_forward validates those separately against the pinned source.
    """
    if mode not in MODES or query_tokens <= 0 or layers <= 0:
        raise ValueError('Invalid repair context mode or geometry')
    if not _lock.acquire(blocking=False):
        raise RuntimeError('Overlapping repair context scopes are unsupported')
    original = module.attention
    owner = threading.get_ident()
    calls = []

    def current_only(q, k, v):
        if threading.get_ident() != owner:
            raise RuntimeError('Concurrent native self-attention during private repair')
        if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[0] != 1
                or q.shape[1] != query_tokens or k.shape != v.shape
                or k.shape[0] != q.shape[0] or k.shape[2:] != q.shape[2:]
                or k.shape[1] < query_tokens or q.device != k.device or q.device != v.device
                or q.dtype != k.dtype or q.dtype != v.dtype):
            raise RuntimeError('Unexpected private self-attention layout')
        calls.append({'query_tokens': q.shape[1], 'native_key_tokens': k.shape[1],
                      'used_key_tokens': query_tokens})
        if len(calls) > layers:
            raise RuntimeError('Unexpected additional self-attention call')
        return original(q, k[:, -query_tokens:], v[:, -query_tokens:])

    replacement = current_only if mode == 'current_block_only_v1' else original
    try:
        module.attention = replacement
        yield
        if mode == 'current_block_only_v1' and len(calls) != layers:
            raise RuntimeError('Not all native layers used the context intervention')
        if module.attention is not replacement:
            raise RuntimeError('Native attention callable changed inside repair')
        trace.append({'mode': mode, 'layers_expected': layers,
                      'intercepted_calls': calls, 'callable_restored': True})
    finally:
        module.attention = original
        _lock.release()


def private_forward(pipeline, request, trace, **kwargs):
    """Called only from the private masked-sampler loop, once per noise step."""
    mode = request.get('repair_context_mode', 'full_context_v1')
    if mode not in MODES:
        raise ValueError('Unknown repair context mode')
    native = importlib.import_module('wan.modules.causal_model_latentmem')
    root = Path(native.__file__).resolve().parents[2]
    for name, expected in PINNED_SOURCES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Unsupported native context layout: ' + name)
    canonical = importlib.import_module('wan.modules.attention').attention
    if native.attention is not canonical:
        raise RuntimeError('Native attention is already replaced')
    modules = [m for m in pipeline.generator.model.modules()
               if type(m) is native.CausalWanSelfAttention]
    frames = kwargs['noisy_image_or_video'].shape[1]
    tokens = frames * pipeline.frame_seq_length
    caches = kwargs['kv_cache']
    if (len(modules) != 30 or len(caches) != 30 or pipeline.frame_seq_length != 1560
            or frames not in (1, 2, 3) or kwargs['current_start'] <= 0
            or kwargs.get('sink_recache_after_switch', False)):
        raise ValueError('Unsupported private repair geometry')
    # Reject a current block which might be truncated or overlap protected sink.
    # The native replay has already denoised this same block, without recache.
    for module, cache in zip(modules, caches):
        if (module.sink_size != 1 or module.memory_size != 6
                or module.max_attention_size != 12 * pipeline.frame_seq_length):
            raise ValueError('Unsupported private repair attention configuration')
        local_budget = module.max_attention_size - (module.sink_size + module.memory_size) * pipeline.frame_seq_length
        current_end = kwargs['current_start'] + tokens
        delta = current_end - int(cache['global_end_index'])
        local_end = int(cache['local_end_index']) + delta
        evicted = max(0, local_end - module.max_attention_size)
        local_start = local_end - evicted - tokens
        if tokens > local_budget or local_start < module.sink_size * pipeline.frame_seq_length:
            raise ValueError('Current block is not complete in the native local window')
    with attention_scope(native, mode, tokens, len(modules), trace):
        return pipeline.generator(**kwargs)
