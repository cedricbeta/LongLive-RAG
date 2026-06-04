# Retrieval Key/Value Study (single-scene multi-perspective KV-RAG)

This note records the decoupled retrieval **key** (index) and **value** (payload)
representations implemented in `utils/kv_rag.py` and ranks them on a GPU-free
viewpoint-invariance probe. The probe is a **proxy** available inside the
unattended loop; it is **not** the authoritative selector. The authoritative
selection is the frame-level comparison on rendered video (the `AC-6` offline
metric, enforced by the `AC-7` gate), which is now a single on-demand command
(see **Status**). No rendered-video winner is claimed here yet: the default
stays the byte-identical baseline (`pooled` + `raw`) and `salient_set` is the
*recommended candidate to confirm on video*, not a measured winner.

## Why the key matters here

A perspective change keeps the scene but rearranges where things appear in the
frame. The retrieval **key** must therefore be *viewpoint-invariant* (match the
same scene across camera angles) while the **value** stays faithful enough to
condition generation. The two are decoupled (`AC-3.1`) so a viewpoint-robust key
can index a faithful raw-K/V payload.

## Implemented representations

KEY (`retrieval_key_mode`):
| mode | description | viewpoint-invariant | extra cost | train-free |
|------|-------------|---------------------|------------|------------|
| `pooled` *(baseline)* | mean-pool over tokens, per head | yes (orderless), but blends subject+background | none | yes |
| `moment` | mean+std orderless signature | yes | none | yes |
| `multi_centroid` | rank tokens by activation, split into N region centroids, best-pair match | yes, **and region-aware** (subject vs background) | tiny (argsort + N means) | yes |
| `salient_set` | bounded set of top-M salient tokens, symmetric mutual-best (Chamfer) cosine | yes, **fine-grained** (per-element, not blended) | tiny (top-M + Chamfer) | yes |
| `positional` | position-weighted pooling | **no** (order-sensitive) — reference "bad key" | none | yes |

`salient_set` was added on the strength of the loop's Codex review, which noted
that `multi_centroid`'s magnitude buckets are not semantic regions and average
all buckets equally; per-element mutual-best matching over the salient token set
is the recommended train-free, GPU-free upgrade.

VALUE (`retrieval_value_mode`):
| mode | description | injected length | faithfulness |
|------|-------------|-----------------|--------------|
| `raw` *(baseline)* | stored frame-aligned K/V slice | grows with top_k×frames | full |
| `mean_frame` | slice collapsed to one representative frame (stays re-RoPE'able) | bounded (1 frame) | lossy/compressed |

## Viewpoint-invariance probe (GPU-free)

Probe construction (see `tests/test_kv_rag_scene_memory.py::TestViewpointInvarianceProbe`):
the *same scene* = the same set of distinctive scene elements, **rearranged** in
the frame (a camera move); a *different scene* = different elements. A good key
scores same-scene/new-view well above a different scene; the order-sensitive key
cannot even recognise the same scene across views.

Reproducible scores (seed 0), `score(same-scene new view)` vs `score(different scene)`:

| mode | same-scene, new view | different scene | margin |
|------|----------------------|-----------------|--------|
| `pooled` | 1.000 | -0.172 | 1.172 |
| `moment` | 1.000 | 0.618 | 0.382 |
| `multi_centroid` | 0.776 | 0.359 | 0.417 |
| `salient_set` | 1.000 | 0.210 | **0.790** |
| `positional` | **0.172** | -0.234 | 0.406 |

Reading:
- `pooled`, `moment`, `multi_centroid`, `salient_set` all rank the same scene
  above a different scene — they are viewpoint-invariant (permutation-invariant),
  confirming the baseline mean-pool is already orderless.
- `salient_set` recognises the same scene perfectly (1.000) while keeping a wide
  margin (0.790) and matching at the level of individual scene elements rather
  than a blended vector — the strongest discriminator among the region-aware keys.
- `moment` is a **weak discriminator**: its std term scores even unrelated scenes
  highly (0.618), shrinking the margin — not selected.
- `positional` collapses to 0.172 on the same scene: it fails to recognise the
  scene across views. This is the deliberate negative control proving the probe
  is non-vacuous (it can reject a bad key), satisfying the `AC-3` negative test.

## Selection

- **Default = `pooled` key + `raw` value.** This reproduces today's behaviour
  byte-for-byte (`AC-4`); disabling the new switches changes nothing.
- **Recommended multi-view configuration = `salient_set` key + `raw` value.**
  The synthetic probe rewards `pooled` with the widest margin, but that probe is
  a *pure spatial permutation*; it under-represents real perspective changes where
  the subject also changes scale/appearance and only part of the background
  persists. `salient_set` matches individual salient scene elements across views
  with symmetric mutual-best cosine, so the same subject can match across angles
  instead of being averaged into one blended vector — the failure mode the plan
  calls out for `pooled`, and the per-element fix the Codex review recommended over
  `multi_centroid`'s coarse magnitude buckets. It is train-free, GPU-free, and
  adds only a top-M selection + Chamfer score. `multi_centroid` remains available
  as a coarser, cheaper region-aware alternative.
- `mean_frame` value is offered as a **bounded** alternative when injected length
  must be capped; `raw` stays the faithful default.

## Anti-cheating tie-in (`AC-3.2` negative)

Selection must not be made on the scene-consistency score alone. A representation
that maximised consistency by effectively replaying shot 0 would be caught by the
companion metrics in the offline evaluator (`evaluation/video_consistency.py`):
`inter_shot_composition_diversity` and `within_shot_motion` collapse toward zero
under a "copy shot 0" degeneracy (verified in
`tests/test_cross_perspective_metric.py::test_copy_collapse_is_visible`). The
final on-video selection at the milestone gate must show a consistency win
**without** regressing those companions beyond tolerance.

## Status

- Key/value interface, alternatives, scene-aware scoring, and the probe: **done,
  unit-tested, GPU-free.**
- Rendered-video evaluation harness (resolver, exact shot boundaries,
  prompt-adherence guard, pass/fail gate): **done, unit-tested, GPU-free**
  (`evaluation/multiview_prompts.py`, `evaluation/video_consistency.py`,
  `scripts/run_kv_rag_ablation.py --mode cross_perspective`).
- On-rendered-video ranking / winner selection via the `AC-6` metric: **pending a
  GPU render** (the unattended loop must not spend a multi-minute 5B render per
  round). It is now a single on-demand command and is the authoritative selector.

### How to produce the rendered ranking (on a GPU box)

Render baseline + each candidate on the same two-prompt subset and read the
per-prompt consistency / diversity / motion / adherence deltas the gate prints:

```bash
# winner candidates to compare: pooled+raw (baseline), salient_set+raw
# (recommended), salient_set+mean_frame (bounded payload).
python scripts/run_kv_rag_ablation.py \
  --config_path configs/inference_kv_rag.yaml \
  --mode cross_perspective \
  --prompts_dir example/multiview_prompts \
  --prompt_subset frying_egg_closeup,african_savanna \
  --score_adherence --adherence_tolerance 0.02 \
  --generator_ckpt <ckpt> --lora_ckpt <lora> \
  --output_root videos/kv_rag_gate
```

The modified variant defaults to the recommended multi-view settings
(`scene_memory_enabled`, `boundary_inject_anchors=2`, `scene_score_bonus=0.1`,
`retrieval_key_mode=salient_set`). Re-run with the kv_rag config's
`retrieval_key_mode` / `retrieval_value_mode` set to each candidate to rank them.
Record the per-prompt deltas here and promote the winner only if its consistency
gain does **not** collapse `inter_shot_composition_diversity` / `within_shot_motion`
or regress `prompt_adherence_*` beyond tolerance.
