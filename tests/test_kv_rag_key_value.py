# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the decoupled retrieval key/value representations (AC-3/3.1/5).

Covers the two new keys (``subject_identity`` identity-aware prototype on the K/V
tensors; ``semantic`` external caption-text key) and the new compressed
``top_frame`` value, plus the fail-fast contracts (unknown names; a flag missing
its companion) and the backward-compat guarantee that the default ``pooled+raw``
representation is unchanged. No GPU, no checkpoint, no network.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kv_rag import KEY_MODES, VALUE_MODES, KVRAGConfig, KVRAGMemory


def _mem(**overrides) -> KVRAGMemory:
    cfg = KVRAGConfig(enabled=True, layers=(0,), top_k=2, frame_aligned_store=True, **overrides)
    return KVRAGMemory(cfg)


def _frame_aligned_kv(frames=2, frame_seqlen=4, heads=2, dim=4, seed=0):
    torch.manual_seed(seed)
    t = frames * frame_seqlen
    k = torch.randn(1, t, heads, dim)
    v = torch.randn(1, t, heads, dim)
    return k, v, frames, frame_seqlen


def _store(mem, k, v, frames, frame_seqlen, **kw):
    mem.add(layer=0, k_pre=k, k_post=k, v=v, start_token=0, end_token=k.shape[1],
            frame_seqlen=frame_seqlen, h=2, w=2, frames=frames, **kw)


class TestModesRegistered(unittest.TestCase):
    def test_new_keys_and_value_present(self):
        self.assertIn("subject_identity", KEY_MODES)
        self.assertIn("semantic", KEY_MODES)
        self.assertIn("top_frame", VALUE_MODES)

    def test_unknown_key_fails_fast(self):
        with self.assertRaises(ValueError):
            KVRAGMemory(KVRAGConfig(enabled=True, retrieval_key_mode="does_not_exist"))

    def test_unknown_value_fails_fast(self):
        with self.assertRaises(ValueError):
            KVRAGMemory(KVRAGConfig(enabled=True, retrieval_value_mode="nonsense"))


class TestSubjectIdentityKey(unittest.TestCase):
    # A fixed, SHARED background direction (same across all scenes) that fills most
    # tokens at a moderate per-token norm; the subject is a couple of HIGH-norm
    # tokens. With many shared-background tokens the unweighted pooled mean is
    # dominated by the background (so pooled cannot tell two subjects apart), while
    # subject_identity selects the high-norm subject tokens and subtracts the
    # background, isolating the subject.
    _BG = F.normalize(torch.randn(2, 4, generator=torch.Generator().manual_seed(3)), dim=-1)

    def _scene(self, subject, frames=8, frame_seqlen=4, noise_seed=7):
        heads, dim = 2, 4
        t = frames * frame_seqlen
        g = torch.Generator().manual_seed(noise_seed)
        x = self._BG.view(1, 1, heads, dim).expand(1, t, heads, dim).clone()
        x = x + 0.01 * torch.randn(1, t, heads, dim, generator=g)  # shared background
        x[:, 0] = subject  # two high-norm subject tokens
        x[:, 1] = subject
        return x

    def test_identity_isolates_subject_from_shared_background(self):
        # AC-2/AC-3 non-vacuity at the KEY level: two scenes that SHARE a large
        # background but differ in subject are separated by subject_identity (which
        # subtracts the background) far more than by plain pooled (which the shared
        # background dominates).
        g = torch.Generator().manual_seed(1)
        subj_a = torch.randn(2, 4, generator=g) * 5.0
        subj_b = torch.randn(2, 4, generator=g) * 5.0
        scene_a = self._scene(subj_a, noise_seed=7)
        scene_a2 = self._scene(subj_a, noise_seed=8)  # same subject, re-rolled background
        scene_b = self._scene(subj_b, noise_seed=9)

        ident = _mem(retrieval_key_mode="subject_identity", retrieval_key_top_m=2)
        pooled = _mem(retrieval_key_mode="pooled")

        def margin(mem):
            qa = mem._compute_key(scene_a)
            same = mem._compute_key(scene_a2)
            diff = mem._compute_key(scene_b)
            return mem._score(qa, same) - mem._score(qa, diff)

        # identity separates same vs different subject by a wider margin than pooled.
        self.assertGreater(margin(ident), margin(pooled))
        # and the prototype is per-head [B, H, D] like pooled (so _score_batch works).
        self.assertEqual(ident._compute_key(scene_a).shape, (1, 2, 4))

    def test_subject_identity_differs_from_pooled(self):
        scene = self._scene(torch.randn(2, 4) * 5.0)
        ident = _mem(retrieval_key_mode="subject_identity", retrieval_key_top_m=2)._compute_key(scene)
        pooled = _mem(retrieval_key_mode="pooled")._compute_key(scene)
        self.assertFalse(torch.allclose(ident, pooled, atol=1e-4))

    def test_subject_identity_scores_by_cosine_regardless_of_similarity(self):
        # The identity prototype is scored by plain cosine independent of the global
        # similarity setting: an l2 config must NOT change its scores. (P2 fix.)
        q = F.normalize(torch.randn(1, 2, 4), dim=-1)
        a = F.normalize(torch.randn(1, 2, 4), dim=-1)
        b = F.normalize(torch.randn(1, 2, 4), dim=-1)

        def _fake(summary):
            return type("E", (), {"summary": summary, "persistent": False})()

        cos = _mem(retrieval_key_mode="subject_identity", similarity="cosine")
        l2 = _mem(retrieval_key_mode="subject_identity", similarity="l2")
        s_cos = cos._score_candidates(q, [_fake(a), _fake(b)])
        s_l2 = l2._score_candidates(q, [_fake(a), _fake(b)])
        self.assertAlmostEqual(s_cos[0], s_l2[0], places=6)
        self.assertAlmostEqual(s_cos[1], s_l2[1], places=6)
        # and the score IS cosine (dot of normalized summaries)
        self.assertAlmostEqual(s_l2[0], float((q * a).sum(-1).mean()), places=6)


class TestSemanticKey(unittest.TestCase):
    def test_fails_fast_without_context(self):
        mem = _mem(retrieval_key_mode="semantic")
        k, v, frames, fseq = _frame_aligned_kv()
        with self.assertRaises(RuntimeError):
            _store(mem, k, v, frames, fseq)

    def test_context_key_is_normalized_and_used(self):
        mem = _mem(retrieval_key_mode="semantic")
        mem.set_context_key(torch.tensor([3.0, 0.0, 0.0, 0.0]))
        key = mem._compute_key(torch.randn(1, 8, 2, 4))  # tensor is ignored
        self.assertEqual(key.shape, (1, 4))
        self.assertAlmostEqual(float(key.norm()), 1.0, places=5)

    def test_context_key_pools_to_D_over_all_leading_dims(self):
        # [D], [seq, D] and [blocks, seq, D] T5 prompt_embeds all reduce to the
        # SAME shape-independent pooled [1, D] caption vector (not [seq*D]).
        mem = _mem(retrieval_key_mode="semantic")
        D = 4
        for shape in [(D,), (6, D), (3, 6, D)]:
            mem.set_context_key(torch.randn(*shape))
            self.assertEqual(tuple(mem._context_key.shape), (1, D))

    def test_context_key_is_mean_pool_not_flatten(self):
        # All tokens identical -> the pooled key equals that (normalized) token,
        # regardless of seq/block dims. A flatten-to-[seq*D] bug would fail this.
        mem = _mem(retrieval_key_mode="semantic")
        tok = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mem.set_context_key(tok.repeat(3, 6, 1))  # [3, 6, 4] all-equal tokens
        expected = F.normalize(tok, dim=-1).reshape(1, -1)
        self.assertTrue(torch.allclose(mem._context_key, expected, atol=1e-5))
        # a pre-pooled [D] of the same content yields the SAME key (shape-independent).
        mem2 = _mem(retrieval_key_mode="semantic")
        mem2.set_context_key(tok)
        self.assertTrue(torch.allclose(mem._context_key, mem2._context_key, atol=1e-5))

    def test_same_caption_matches_higher_than_different(self):
        # Store an entry under perspective 0's caption embedding, then retrieve as
        # a later perspective whose caption embedding is close vs far.
        mem = _mem(retrieval_key_mode="semantic")
        emb0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        mem.set_context_key(emb0)
        k, v, frames, fseq = _frame_aligned_kv()
        _store(mem, k, v, frames, fseq)
        stored = mem.entries_by_layer[0][0].summary

        close = F.normalize(torch.tensor([1.0, 0.2, 0.0, 0.0]), dim=-1).reshape(1, -1)
        far = F.normalize(torch.tensor([0.0, 0.0, 1.0, 0.0]), dim=-1).reshape(1, -1)
        self.assertGreater(mem._score(close, stored), mem._score(far, stored))

    def test_retrieve_returns_entry_using_context_not_query(self):
        mem = _mem(retrieval_key_mode="semantic")
        mem.set_context_key(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        k, v, frames, fseq = _frame_aligned_kv()
        _store(mem, k, v, frames, fseq)
        # New perspective: a different (but present) caption; query tensor content
        # is irrelevant for a semantic key.
        mem.set_context_key(torch.tensor([0.9, 0.1, 0.0, 0.0]))
        out = mem.retrieve(layer=0, query=torch.randn(1, 4, 2, 4), current_start=100,
                           frame_seqlen=fseq, dtype=torch.float32, device=torch.device("cpu"))
        self.assertIsNotNone(out)
        rag_k, rag_v, selected = out
        self.assertEqual(len(selected), 1)

    def test_clear_resets_context_key(self):
        mem = _mem(retrieval_key_mode="semantic")
        mem.set_context_key(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        mem.clear()
        self.assertIsNone(mem._context_key)


class TestTopFrameValue(unittest.TestCase):
    def test_requires_frame_aligned_store(self):
        with self.assertRaises(ValueError):
            KVRAGMemory(KVRAGConfig(enabled=True, retrieval_value_mode="top_frame",
                                    frame_aligned_store=False))

    def test_keeps_single_highest_norm_frame(self):
        mem = _mem(retrieval_value_mode="top_frame")
        # Build a 3-frame slice where frame 1 has by far the largest token norm.
        frames, fseq, heads, dim = 3, 4, 2, 4
        k = torch.randn(1, frames * fseq, heads, dim) * 0.1
        k[:, fseq:2 * fseq] += 10.0  # frame index 1 is the subject-salient frame
        v = torch.randn(1, frames * fseq, heads, dim)
        _store(mem, k, v, frames, fseq)
        entry = mem.entries_by_layer[0][0]
        self.assertEqual(entry.frames, 1)
        self.assertEqual(entry.num_tokens, fseq)
        # the kept frame should be the high-norm one (mean close to +10 offset).
        self.assertGreater(float(entry.k.mean()), 1.0)

    def test_decoupling_semantic_key_with_top_frame_value(self):
        # AC-3.1: a semantic key can index a compressed top_frame value.
        mem = _mem(retrieval_key_mode="semantic", retrieval_value_mode="top_frame")
        mem.set_context_key(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        k, v, frames, fseq = _frame_aligned_kv(frames=3, frame_seqlen=4)
        _store(mem, k, v, frames, fseq)
        self.assertEqual(mem.entries_by_layer[0][0].frames, 1)


class TestBackwardCompat(unittest.TestCase):
    def test_default_is_pooled_raw(self):
        cfg = KVRAGConfig(enabled=True)
        self.assertEqual(cfg.retrieval_key_mode, "pooled")
        self.assertEqual(cfg.retrieval_value_mode, "raw")

    def test_raw_value_keeps_all_frames(self):
        mem = _mem()  # pooled + raw
        k, v, frames, fseq = _frame_aligned_kv(frames=3, frame_seqlen=4)
        _store(mem, k, v, frames, fseq)
        entry = mem.entries_by_layer[0][0]
        self.assertEqual(entry.frames, 3)
        self.assertEqual(entry.num_tokens, 12)

    def test_pooled_key_unaffected_by_new_modes(self):
        # The pooled summary is byte-identical to the legacy _summarize output.
        mem = _mem()
        x = torch.randn(1, 8, 2, 4)
        self.assertTrue(torch.allclose(mem._compute_key(x), mem._summarize(x)))


if __name__ == "__main__":
    unittest.main()
