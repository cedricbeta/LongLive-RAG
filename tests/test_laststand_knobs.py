# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""GPU-free smoke tests for the last-stand appended-injection knobs.

Covers the three default-off mechanisms added for the appended-KV last-stand
round (docs/attention-downweighting-study.zh.md):

1. inject_active_steps  -- step-gated injection: when off, retrieval output is
   bit-identical to the incumbent path and the attention layer never enters
   the step-tracking branch; when on, only the listed denoise steps allow
   injection.
2. inject_head_filter   -- per-head additive mask: shape/semantics of the
   [1, H, 1, L] bias, softmax effect, and broadcast against the persistent
   [1, 1, 1, L] bias.
3. store_phase_matched_layers -- the clean-recache store of a listed layer
   replaces the injected K with the same chunk's captured denoise-step-0 K,
   while V and the retrieval summary stay clean.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kv_rag import HEAD_FILTER_NEG_LOGIT, KVRAGConfig, KVRAGMemory

CPU = torch.device("cpu")
FSL = 4     # tokens per frame
HEADS = 2
DIM = 3


def _mem(**kw) -> KVRAGMemory:
    params = dict(enabled=True, top_k=1)
    params.update(kw)
    return KVRAGMemory(KVRAGConfig(**params))


def _store(
    memory,
    *,
    layer=0,
    frames=2,
    start=0,
    k=None,
    v=None,
    heads=HEADS,
    dim=DIM,
):
    t = frames * FSL
    if k is None:
        k = torch.randn(1, t, heads, dim)
    if v is None:
        v = torch.randn(1, t, heads, dim)
    memory.add(
        layer=layer, k_pre=k, k_post=k, v=v,
        start_token=start, end_token=start + t,
        frame_seqlen=FSL, h=heads, w=2, frames=frames,
    )
    return k, v


class TestConfigParsing(unittest.TestCase):
    def test_defaults_are_off(self):
        cfg = KVRAGConfig.from_config({"enabled": True})
        self.assertIsNone(cfg.inject_active_steps)
        self.assertIsNone(cfg.inject_head_filter)
        self.assertIsNone(cfg.store_phase_matched_layers)
        self.assertFalse(cfg.laststand_step_tracking)
        self.assertIsNone(cfg.head_filter_for_layer(29))
        self.assertIn("inject_active_steps=None", cfg.describe())

    def test_parsing_and_normalization(self):
        cfg = KVRAGConfig.from_config({
            "enabled": True,
            "inject_active_steps": [1, 0, 1],
            "inject_head_filter": {29: [23, 1], "14": 5},
            "store_phase_matched_layers": 14,
        })
        self.assertEqual(cfg.inject_active_steps, (0, 1))
        self.assertEqual(cfg.inject_head_filter, ((14, (5,)), (29, (1, 23))))
        self.assertEqual(cfg.store_phase_matched_layers, (14,))
        self.assertTrue(cfg.laststand_step_tracking)
        self.assertEqual(cfg.head_filter_for_layer(29), (1, 23))
        self.assertEqual(cfg.head_filter_for_layer(14), (5,))
        self.assertIsNone(cfg.head_filter_for_layer(0))

    def test_validation_fails_fast(self):
        with self.assertRaises(ValueError):
            KVRAGMemory(KVRAGConfig(enabled=True, inject_active_steps=()))
        with self.assertRaises(ValueError):
            KVRAGMemory(KVRAGConfig(enabled=True, inject_active_steps=(-1,)))
        with self.assertRaises(ValueError):
            KVRAGMemory(KVRAGConfig(enabled=True, inject_head_filter=((29, ()),)))
        with self.assertRaises(ValueError):
            KVRAGMemory(KVRAGConfig(
                enabled=True, store_phase_matched_layers=(14,), reinject_rope=False,
            ))


class TestActiveStepsOff(unittest.TestCase):
    """Knob off -> incumbent behavior, bit for bit."""

    def _fill_and_retrieve(self, memory):
        torch.manual_seed(7)
        k0, v0 = _store(memory, start=0)
        k1, v1 = _store(memory, start=8)
        q = torch.randn(1, FSL, HEADS, DIM)
        return memory.retrieve(
            layer=0, query=q, current_start=1000, frame_seqlen=FSL,
            dtype=torch.float32, device=CPU,
        )

    def test_retrieval_bitwise_identical_when_off(self):
        base = self._fill_and_retrieve(_mem())
        explicit_off = self._fill_and_retrieve(_mem(inject_active_steps=None))
        self.assertIsNotNone(base)
        self.assertTrue(torch.equal(base[0], explicit_off[0]))  # rag_k
        self.assertTrue(torch.equal(base[1], explicit_off[1]))  # rag_v

    def test_off_never_enters_step_tracking_branch(self):
        # causal_model gates the whole new branch on laststand_step_tracking,
        # so the default config never touches the counter or capture path.
        m = _mem()
        self.assertFalse(m.laststand_step_tracking)
        self.assertTrue(m.inject_allowed_at_step(None))
        self.assertEqual(m.stats["active_step_inject_skips"], 0)

    def test_step_counter_and_gating(self):
        m = _mem(inject_active_steps=(0, 1))
        self.assertTrue(m.laststand_step_tracking)
        m.set_runtime_context(chunk_index=5, phase="denoise")
        allowed = []
        for _ in range(4):
            step = m.denoise_step_index(29)
            allowed.append(m.inject_allowed_at_step(step))
        self.assertEqual(allowed, [True, True, False, False])
        # uncond / recache phases are never counted or gated
        m.set_runtime_context(chunk_index=5, phase="denoise_uncond")
        self.assertIsNone(m.denoise_step_index(29))
        m.set_runtime_context(chunk_index=5, phase="clean_recache")
        self.assertIsNone(m.denoise_step_index(29))
        # a new chunk restarts at step 0
        m.set_runtime_context(chunk_index=6, phase="denoise")
        self.assertEqual(m.denoise_step_index(29), 0)
        # counters are per layer
        self.assertEqual(m.denoise_step_index(14), 0)
        m.note_active_step_skip()
        self.assertEqual(m.stats["active_step_inject_skips"], 1)


class TestHeadFilterBias(unittest.TestCase):
    def test_shape_and_semantics(self):
        m = _mem(inject_head_filter=((29, (1, 3)),))
        bias = m.build_head_filter_bias(
            layer=29, total_tokens=10, rag_tokens=3, num_heads=4,
            device=CPU, dtype=torch.float32,
        )
        self.assertEqual(tuple(bias.shape), (1, 4, 1, 10))
        # whitelisted heads see everything unchanged
        self.assertTrue(torch.all(bias[0, 1] == 0))
        self.assertTrue(torch.all(bias[0, 3] == 0))
        # masked heads: NEG on the injected columns only
        for h in (0, 2):
            self.assertTrue(torch.all(bias[0, h, 0, :3] == HEAD_FILTER_NEG_LOGIT))
            self.assertTrue(torch.all(bias[0, h, 0, 3:] == 0))
        self.assertEqual(m.stats["head_filter_bias_calls"], 1)
        self.assertEqual(m.stats["head_filter_masked_heads"], 2)

    def test_unlisted_layer_and_degenerate_inputs_return_none(self):
        m = _mem(inject_head_filter=((29, (1,)),))
        self.assertIsNone(m.build_head_filter_bias(
            layer=14, total_tokens=10, rag_tokens=3, num_heads=4, device=CPU))
        self.assertIsNone(m.build_head_filter_bias(
            layer=29, total_tokens=10, rag_tokens=0, num_heads=4, device=CPU))
        m_all = _mem(inject_head_filter=((29, (0, 1)),))
        self.assertIsNone(m_all.build_head_filter_bias(
            layer=29, total_tokens=10, rag_tokens=3, num_heads=2, device=CPU))

    def test_softmax_effect_masks_only_masked_heads(self):
        torch.manual_seed(3)
        heads, lq, lk, rag = 4, 2, 8, 3
        q = torch.randn(1, heads, lq, DIM)
        k = torch.randn(1, heads, lk, DIM)
        logits = q @ k.transpose(-1, -2) / math.sqrt(DIM)
        m = _mem(inject_head_filter=((29, (1, 3)),))
        bias = m.build_head_filter_bias(
            layer=29, total_tokens=lk, rag_tokens=rag, num_heads=heads, device=CPU)
        w_plain = torch.softmax(logits, dim=-1)
        w_masked = torch.softmax(logits + bias, dim=-1)
        # masked heads receive (numerically) zero mass on injected columns
        self.assertLess(float(w_masked[0, 0, :, :rag].max()), 1e-30)
        self.assertLess(float(w_masked[0, 2, :, :rag].max()), 1e-30)
        # whitelisted heads are bit-identical to the unbiased softmax
        self.assertTrue(torch.equal(w_masked[0, 1], w_plain[0, 1]))
        self.assertTrue(torch.equal(w_masked[0, 3], w_plain[0, 3]))

    def test_broadcast_against_persistent_bias(self):
        m = _mem(inject_head_filter=((29, (0,)),))
        head_bias = m.build_head_filter_bias(
            layer=29, total_tokens=6, rag_tokens=2, num_heads=2, device=CPU)
        persistent = torch.full((1, 1, 1, 6), 0.5)
        combined = persistent + head_bias
        self.assertEqual(tuple(combined.shape), (1, 2, 1, 6))
        self.assertTrue(torch.all(combined[0, 0] == 0.5))
        self.assertTrue(
            torch.all(combined[0, 1, 0, :2] == 0.5 + HEAD_FILTER_NEG_LOGIT))


class TestPhaseMatchedStore(unittest.TestCase):
    LAYERS = (3, 14)

    def _mem(self, phase_matched=(14,)):
        return _mem(layers=self.LAYERS, store_phase_matched_layers=phase_matched)

    def _run_chunk(self, m, *, layer, start=0, frames=2, capture=True):
        t = frames * FSL
        clean_k = torch.randn(1, t, HEADS, DIM)
        clean_v = torch.randn(1, t, HEADS, DIM)
        step0_k = clean_k + 10.0  # clearly distinguishable noisy-phase key
        m.set_runtime_context(chunk_index=0, phase="denoise")
        if capture:
            step = m.denoise_step_index(layer)
            m.capture_phase_matched_k(
                layer=layer, step=step, k_pre=step0_k,
                start_token=start, end_token=start + t,
            )
        m.set_runtime_context(chunk_index=0, phase="clean_recache")
        m.add(
            layer=layer, k_pre=clean_k, k_post=None, v=clean_v,
            start_token=start, end_token=start + t,
            frame_seqlen=FSL, h=HEADS, w=2, frames=frames,
            phase="clean_recache",
        )
        return clean_k, clean_v, step0_k

    def _retrieve(self, m, layer):
        q = torch.randn(1, FSL, HEADS, DIM)
        return m.retrieve(
            layer=layer, query=q, current_start=1000, frame_seqlen=FSL,
            dtype=torch.float32, device=CPU,
        )

    def test_listed_layer_stores_step0_k_and_clean_v(self):
        torch.manual_seed(11)
        m = self._mem()
        clean_k, clean_v, step0_k = self._run_chunk(m, layer=14)
        rag_k, rag_v, selected = self._retrieve(m, 14)
        self.assertTrue(torch.equal(rag_k, step0_k))      # K: noisy step-0 phase
        self.assertFalse(torch.equal(rag_k, clean_k))
        self.assertTrue(torch.equal(rag_v, clean_v))      # V: clean content
        self.assertEqual(m.stats["phase_matched_k_captures"], 1)
        self.assertEqual(m.stats["phase_matched_k_replacements"], 1)

    def test_summary_still_computed_from_clean_k(self):
        torch.manual_seed(13)
        m = self._mem()
        clean_k, clean_v, _ = self._run_chunk(m, layer=14)
        control = _mem(layers=self.LAYERS)
        control.add(
            layer=14, k_pre=clean_k, k_post=None, v=clean_v,
            start_token=0, end_token=clean_k.shape[1],
            frame_seqlen=FSL, h=HEADS, w=2, frames=2, phase="clean_recache",
        )
        test_entry = m.entries_by_layer[14][0]
        control_entry = control.entries_by_layer[14][0]
        self.assertTrue(torch.equal(test_entry.summary, control_entry.summary))
        self.assertFalse(torch.equal(test_entry.k, control_entry.k))

    def test_unlisted_layer_keeps_clean_k(self):
        torch.manual_seed(17)
        m = self._mem(phase_matched=(14,))
        clean_k, clean_v, _ = self._run_chunk(m, layer=3)  # capture is a no-op
        rag_k, rag_v, _ = self._retrieve(m, 3)
        self.assertTrue(torch.equal(rag_k, clean_k))
        self.assertTrue(torch.equal(rag_v, clean_v))
        self.assertEqual(m.stats["phase_matched_k_captures"], 0)
        self.assertEqual(m.stats["phase_matched_k_replacements"], 0)

    def test_capture_is_consumed_once(self):
        torch.manual_seed(19)
        m = self._mem()
        self._run_chunk(m, layer=14, start=0)
        # second chunk stored WITHOUT a fresh capture falls back to clean K
        clean_k, clean_v, _ = self._run_chunk(m, layer=14, start=8, capture=False)
        second = m.entries_by_layer[14][1]
        self.assertTrue(torch.equal(second.k, clean_k))
        self.assertEqual(m.stats["phase_matched_k_replacements"], 1)
        self.assertGreaterEqual(m.stats["phase_matched_k_misses"], 1)

    def test_token_range_mismatch_keeps_clean_k(self):
        torch.manual_seed(23)
        m = self._mem()
        t = 2 * FSL
        m.set_runtime_context(chunk_index=0, phase="denoise")
        step = m.denoise_step_index(14)
        m.capture_phase_matched_k(
            layer=14, step=step, k_pre=torch.randn(1, t, HEADS, DIM),
            start_token=100, end_token=100 + t,  # different chunk span
        )
        clean_k = torch.randn(1, t, HEADS, DIM)
        m.set_runtime_context(chunk_index=0, phase="clean_recache")
        m.add(
            layer=14, k_pre=clean_k, k_post=None, v=torch.randn(1, t, HEADS, DIM),
            start_token=0, end_token=t, frame_seqlen=FSL, h=HEADS, w=2, frames=2,
            phase="clean_recache",
        )
        self.assertTrue(torch.equal(m.entries_by_layer[14][0].k, clean_k))
        self.assertEqual(m.stats["phase_matched_k_replacements"], 0)


if __name__ == "__main__":
    unittest.main()
