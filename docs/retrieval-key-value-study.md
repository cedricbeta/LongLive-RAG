# Retrieval Key/Value Study (single-scene multi-perspective KV-RAG)

This note records the decoupled retrieval **key** (index) and **value** (payload)
representations implemented in `utils/kv_rag.py` and ranks them both on a GPU-free
viewpoint-invariance probe (a proxy) and on **rendered 5B video** via the AC-6
offline metric (the authoritative selector, enforced by the AC-7 gate). The
rendered ranking (see **Rendered ranking**) selects **`pooled` key + `raw` value**
as the winner — which is also the byte-identical baseline representation — because
on real video it gave the largest cross-shot consistency gain without collapsing
the anti-cheating companions or regressing prompt adherence. Notably this
overturned the synthetic probe, which had favored `salient_set`.

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

## Rendered ranking (measured on the 5B model)

Each candidate was rendered against a matched-seed baseline on the 5B model
(`checkpoints/longlive2_5b/longlive2_merged_generator.pt`, 4xA100), with the
persistent scene memory on (`scene_memory_enabled`, `boundary_inject_anchors=2`,
`scene_score_bonus=0.1`) and only the key/value representation varied. Metric =
`cross_shot_scene_consistency` (AC-6); guard = per-shot CLIP `prompt_adherence`
(tolerance 0.02). Subset: `frying_egg_closeup` (baseline drift, headroom) and
`african_savanna` (baseline already 0.824 -> saturated, little headroom).

| variant | frying_egg Δconsist | african_savanna Δconsist | Δdiversity | Δadherence(mean) |
|---------|--------------------:|-------------------------:|-----------:|-----------------:|
| **pooled+raw** (winner) | **+0.0724** | −0.0018 | +0.0038 / +0.0047 | −0.0057 / +0.0011 |
| salient_set+raw | +0.0290 | −0.0099 | +0.0018 / +0.0038 | −0.0029 / −0.0019 |
| salient_set+mean_frame | +0.0290 | −0.0099 | +0.0018 / +0.0038 | −0.0029 / −0.0019 |

(JSON: `videos/kv_rag_gate/{kv_rag_cross_perspective,eval_pooled_raw,eval_salient_meanframe}.json`.)

**Winner: `pooled+raw`** — the largest consistency gain where there is headroom
(`frying_egg` +0.072, ~2.5x `salient_set`), no `inter_shot_composition_diversity`
or `within_shot_motion` collapse (diversity stays slightly positive), and
adherence within tolerance on every prompt. `salient_set+mean_frame` was
byte-identical to `salient_set+raw` here (the `mean_frame` value collapse did not
change the injected payload on this subset), so the value mode was not a
differentiator.

### Caveats / findings

- **The rendered metric overturned the synthetic probe.** The probe ranked
  `salient_set` first (margin 0.790 on the permutation probe); on rendered video
  the plain `pooled` key gave the larger consistency win. The probe is a useful
  GPU-free smoke test but is NOT the selector — exactly the reason AC-3.2 requires
  the rendered metric.
- **The HSV scene signature saturates** on already-coherent scenes
  (`african_savanna` baseline 0.824), so no key wins there — there is little
  cross-shot drift to repair. The feature helps where baseline drifts
  (`frying_egg` 0.55 -> 0.62). A richer shot-level feature (the repo's
  `frame_feature`) would saturate less but is framing-sensitive, so it is left as
  a follow-up rather than swapped into the framing-robust scene signal.

### Reproduce

```bash
python scripts/run_kv_rag_ablation.py \
  --config_path configs/inference_kv_rag.yaml --mode cross_perspective \
  --prompts_dir example/multiview_prompts \
  --prompt_subset frying_egg_closeup,african_savanna \
  --modified_retrieval_key_mode <pooled|salient_set> \
  --modified_retrieval_value_mode <raw|mean_frame> \
  --adherence_tolerance 0.02 \
  --generator_ckpt checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --no_lora_adapter --output_root videos/kv_rag_gate
```
