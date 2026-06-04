# Retrieval Key/Value Study (single-scene multi-perspective KV-RAG)

This note records the decoupled retrieval **key** (index) and **value** (payload)
representations implemented in `utils/kv_rag.py` and ranks them both on a GPU-free
viewpoint-invariance probe (a proxy) and on **rendered 5B video** via the AC-6
offline metric (the authoritative selector, enforced by the AC-7 gate).

**Prior cross-shot context, not the current cross-video answer:** an earlier
cross-shot protocol selected **`pooled` key + `raw` value** as its winner because
it gave the largest cross-shot consistency gain in that older setup. That result
is retained below only as history. **Current-plan conclusion:** see **Round 1:
current cross-video conclusion (multiview_vbench, full AC-2 backbones)** below.

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

## Pre-Round-1 defaults and synthetic hypothesis

- **Implementation default = `pooled` key + `raw` value.** This reproduces
  today's behaviour byte-for-byte (`AC-4`); disabling the new switches changes
  nothing. This is a default/control, not the current cross-video conclusion.
- **Synthetic probe hypothesis = `salient_set` key + `raw` value.**
  The synthetic probe rewards `pooled` with the widest margin, but that probe is
  a *pure spatial permutation*; it under-represents real perspective changes where
  the subject also changes scale/appearance and only part of the background
  persists. `salient_set` matches individual salient scene elements across views
  with symmetric mutual-best cosine, so the same subject can match across angles
  instead of being averaged into one blended vector — the failure mode the plan
  calls out for `pooled`, and the per-element fix the Codex review recommended over
  `multi_centroid`'s coarse magnitude buckets. It is train-free, GPU-free, and
  adds only a top-M selection + Chamfer score. Round 1 rendered gates supersede
  this synthetic hypothesis as the current selector evidence.
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

## Prior cross-shot rendered ranking (measured on the 5B model)

This is prior cross-shot context from an earlier protocol, not the current
cross-video `multiview_vbench` answer. Each candidate was rendered against a
matched-seed baseline on the 5B model
(`checkpoints/longlive2_5b/longlive2_merged_generator.pt`, 4xA100), with the
persistent scene memory on (`scene_memory_enabled`, `boundary_inject_anchors=2`,
`scene_score_bonus=0.1`) and only the key/value representation varied. Metric =
`cross_shot_scene_consistency` (AC-6); guard = per-shot CLIP `prompt_adherence`
(tolerance 0.02). Subset: `frying_egg_closeup` (baseline drift, headroom) and
`african_savanna` (baseline already 0.824 -> saturated, little headroom).

| variant | frying_egg Δconsist | african_savanna Δconsist | Δdiversity | Δadherence(mean) |
|---------|--------------------:|-------------------------:|-----------:|-----------------:|
| **pooled+raw** (prior winner) | **+0.0724** | −0.0018 | +0.0038 / +0.0047 | −0.0057 / +0.0011 |
| salient_set+raw | +0.0290 | −0.0099 | +0.0018 / +0.0038 | −0.0029 / −0.0019 |
| salient_set+mean_frame | +0.0290 | −0.0099 | +0.0018 / +0.0038 | −0.0029 / −0.0019 |

(JSON: `videos/kv_rag_gate/{kv_rag_cross_perspective,eval_pooled_raw,eval_salient_meanframe}.json`.)

**Prior cross-shot result: `pooled+raw`** - the largest consistency gain where there is headroom
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

## Round 1: current cross-video conclusion (multiview_vbench, full AC-2 backbones)

Date: 2026-06-04.

**Current conclusion:** the retrieval **key** only acts as a selector under
content-match, not under boundary force-injection. Among the rendered
content-match finalists, **`semantic+raw` is the leading key candidate** because
it has the largest mean `aggregate_consistency` gain against the no-scene-memory
baseline. **No configuration passes the full AC-4 guarded gate**: every finalist
fails the motion guard on `african_savanna`, so this is a render-confirmed honest
null, not a passing winner.

The per-perspective scene-memory substrate was corrected in this round:
perspective 0 seeds the persistent anchors, and later perspectives force-inject
those anchors at chunk 0. The live substrate verification is not serialized in
the two metric JSON files; the Round 1 force-inject commit records
`boundary_injections > 0` and `stored_scene_entries=10`, with only perspective 0
seeding persistent scene entries.

Both Round 1 gates were run on the 5B model with the full AC-2 backbones loaded:
`subject_dino.loaded=true`, `background_clip.loaded=true`, and
`identity.loaded=true`, with `subject_kind=auto`. Both used mandatory adherence,
diversity, and motion guards: `adherence_tolerance=0.02`,
`diversity_tolerance=0.05`, and `motion_tolerance=0.3`. In both JSONs,
`winner=null` and `is_null_result=true`.

### Gate A: force-inject finalists

Source: `docs/multiview_gate_results/round1_finalists.json`.

Settings: `boundary_inject_anchors=2`, `scene_score_bonus=0.1`,
`scene_memory_enabled=true`, and `perspective_coverage` is 4 for
`african_savanna` and 4 for `frying_egg_closeup`. This is the force-inject
round0gate run.

Under boundary force-injection, the three rendered finalists
`pooled+raw`, `semantic+raw`, and `subject_identity+raw` have identical JSON
metric records. The Round 1 force-inject commit also records that the rendered
videos were byte-identical by md5. This is the key finding for Gate A: with
force-injection and a small anchor pool, the retrieval key is not the lever.

All three finalists have the same gate outcome:

| finalist | mean `aggregate_consistency` delta | scene wins | failed guard | passed |
|----------|-----------------------------------:|-----------:|--------------|--------|
| `pooled+raw` | -0.0011045587908575794 | 1/2 | motion on `african_savanna` | false |
| `semantic+raw` | -0.0011045587908575794 | 1/2 | motion on `african_savanna` | false |
| `subject_identity+raw` | -0.0011045587908575794 | 1/2 | motion on `african_savanna` | false |

On the failing scene, `dynamic_degree` drops from 2.550567817563812 to
2.1392649033417306 (`delta=-0.4113029142220812`). The aggregate consistency
change there is positive but tiny: 0.7479458969866849 to 0.7480506237720738
(`delta=0.00010472678538886449`). This is an honest null, not a passing result.

### Gate B: content-match key sweep

Source: `docs/multiview_gate_results/round1_keysweep_contentmatch.json`.

Settings: `boundary_inject_anchors=0`, `scene_score_bonus=0.1`,
`scene_memory_enabled=true`, and `perspective_coverage` is 2 for
`african_savanna` and 2 for `frying_egg_closeup`. This is the pure
content-match round1keysweep run.

With boundary force-injection disabled, the keys differentiate. The Round 1
content-match commit records distinct md5s, and the JSON ranking separates the
finalists by rendered mean `aggregate_consistency` delta:

| rank | finalist | mean `aggregate_consistency` delta | scene wins | `african_savanna` delta | `frying_egg_closeup` delta | failed guard | passed |
|-----:|----------|-----------------------------------:|-----------:|------------------------:|---------------------------:|--------------|--------|
| 1 | `semantic+raw` | 0.007539604896712293 | 2/2 | 0.011825278263029926 | 0.0032539315303946603 | motion on `african_savanna` | false |
| 2 | `pooled+raw` | 0.006561657897123652 | 2/2 | 0.007683491271069487 | 0.005439824523177816 | motion on `african_savanna` | false |
| 3 | `subject_identity+raw` | 0.005023037300745825 | 2/2 | 0.007782215482724508 | 0.002263859118767142 | motion on `african_savanna` | false |

The leading `semantic+raw` candidate improves the strongest aggregate result on
`african_savanna`: `aggregate_consistency` 0.733069493017568 to
0.7448947712805979 (`delta=0.011825278263029926`),
`subject_consistency` 0.4062790368804213 to 0.43541673900429917,
`background_consistency` 0.8167224942174938 to 0.8587366724416364,
and `appearance_style` 0.9797304188933249 to 0.984180419296064.
It slightly lowers `overall_consistency` from 0.32250483334064484 to
0.3220709413290024 and `subject_identity_consistency` from
0.8758832763358713 to 0.8729876301095633.

The metric vector previously summarized as `subject_consistency`
0.4062790368804213 to 0.42157645325174514,
`background_consistency` 0.8167224942174938 to 0.8430124942600754,
`appearance_style` 0.9797304188933249 to 0.9860756981207562,
`overall_consistency` 0.32250483334064484 to 0.327101394534111,
`subject_identity_consistency` 0.8758832763358713 to 0.8713149203609201, and
`aggregate_consistency` 0.733069493017568 to 0.7407529842886375 belongs to
`pooled+raw` in this JSON, not `semantic+raw`.

The honest-null blocker is motion. For the leading `semantic+raw` row on
`african_savanna`, `dynamic_degree` drops from 6.173149074117342 to
5.85896157224973 (`delta=-0.3141875018676119`), which fails the
`motion_tolerance=0.3` guard. Every Gate B finalist has
`motion_failures=["african_savanna"]`, `motion_ok=false`, and `passed=false`.

> Caveat (semantic key implementation): the Gate B `semantic` finalist was
> rendered with the pre-Round-4 semantic key, which (a) keyed the unconditional
> CFG bank on the negative prompt (fixed in Round 3) and (b) pooled the caption
> embedding as a flattened `[seq*D]` vector rather than the documented `[D]`
> (fixed in Round 4). Both were consistent WITHIN that run (constant seq length,
> same scene), so `semantic` was a functional, differentiating key here, but its
> exact margin over `pooled` (+0.0075 vs +0.0066) is provisional and should be
> re-confirmed with the corrected `[D]` scene-caption key. The honest-null
> conclusion is unaffected: the blocker is the motion cost, which is
> key-independent (all three finalists fail the same `african_savanna` motion
> guard).

Next hypotheses:
- Re-confirm the `semantic` finalist with the corrected `[D]` scene-caption key.
- Reduce the motion cost with motion-preserving injection or fewer injected
  anchors.
- Revisit whether `motion_tolerance=0.3` is too strict for high-baseline-motion
  scenes such as `african_savanna`.
- Study value modes and larger candidate pools.

### Round 1 reproduce

Gate A, boundary force-inject on by default:

```bash
python scripts/run_kv_rag_ablation.py \
  --config_path configs/inference_kv_rag_round0gate.yaml \
  --mode multiview_vbench \
  --prompts_dir example/multiview_prompts \
  --prompt_subset frying_egg_closeup,african_savanna \
  --max_perspectives 4 \
  --finalists pooled:raw,semantic:raw,subject_identity:raw \
  --vbench_subject \
  --vbench_background \
  --vbench_identity \
  --subject_kind auto \
  --adherence_tolerance 0.02 \
  --diversity_tolerance 0.05 \
  --motion_tolerance 0.3 \
  --generator_ckpt checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --no_lora_adapter \
  --metrics_json docs/multiview_gate_results/round1_finalists.json \
  --output_root videos/round1_finalist_gate
```

Gate B, pure content-match:

```bash
python scripts/run_kv_rag_ablation.py \
  --config_path configs/inference_kv_rag_round1keysweep.yaml \
  --mode multiview_vbench \
  --prompts_dir example/multiview_prompts \
  --prompt_subset frying_egg_closeup,african_savanna \
  --max_perspectives 2 \
  --modified_boundary_inject_anchors 0 \
  --finalists pooled:raw,semantic:raw,subject_identity:raw \
  --vbench_subject \
  --vbench_background \
  --vbench_identity \
  --subject_kind auto \
  --adherence_tolerance 0.02 \
  --diversity_tolerance 0.05 \
  --motion_tolerance 0.3 \
  --generator_ckpt checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --no_lora_adapter \
  --metrics_json docs/multiview_gate_results/round1_keysweep_contentmatch.json \
  --output_root videos/round1_keysweep
```

## Round 0 history: cross-video VBench protocol (multiview_vbench)

Date: 2026-06-04.

Round 0 is retained as history only. It is superseded by the Round 1
full-backbone, guarded gates above and is not the current cross-video answer.

This round extends the sweep to 7 retrieval keys x 3 retrieval values = 21
cells. The two new keys are `semantic` (caption-text context supplied via
`memory.set_context_key(...)`) and `subject_identity` (per-head subject prototype
on K/V tensors). The new value is `top_frame`, a bounded single-frame payload.

| key | key signal | `raw` | `mean_frame` | `top_frame` |
|-----|------------|-------|--------------|-------------|
| `pooled` | mean-pooled pre-RoPE K summary | cell | cell | cell |
| `moment` | mean + std token summary | cell | cell | cell |
| `multi_centroid` | activation-ranked token buckets | cell | cell | cell |
| `salient_set` | top-M high-norm token set | cell | cell | cell |
| `positional` | position-weighted pooled K summary; negative control | screen-only | screen-only | screen-only |
| `subject_identity` | per-head subject prototype on K/V tensors | cell | cell | cell |
| `semantic` | caption-text context key via `memory.set_context_key(...)` | cell | cell | cell |

### Offline-proxy pre-filter

The GPU-free offline screen is a pre-filter, not the selector
(`BL-20260604-rendered-metric-over-probe`). The JSON `offline_screen` ranked key
margins as:

| rank | key | margin | result |
|------|-----|-------:|--------|
| 1 | `pooled` | 1.172 | shortlisted |
| 2 | `subject_identity` | 1.161 | shortlisted |
| 3 | `semantic` | 0.861 | shortlisted |
| 4 | `salient_set` | 0.790 | shortlisted |
| 5 | `multi_centroid` | 0.417 | shortlisted |
| 6 | `positional` | 0.406 | FAILS - negative control |
| 7 | `moment` | 0.382 | shortlisted |

Shortlist excludes `positional`: `pooled`, `subject_identity`, `semantic`,
`salient_set`, `multi_centroid`, and `moment`.

### Rendered AC-2 result

The authoritative selector is the rendered AC-2 suite. For the rendered finalist
`subject_identity+raw` against the no-scene-memory baseline on the 5B model, the
gate PASSED with 2/2 `aggregate_consistency` wins:

| scene | baseline | modified | delta |
|-------|---------:|---------:|------:|
| `african_savanna` | 0.7524 | 0.7548 | +0.0024 |
| `frying_egg_closeup` | 0.7689 | 0.7707 | +0.0018 |

Per-dimension deltas: `appearance_style` african +0.0080 / frying +0.0010;
`overall_consistency` african -0.0005 / frying +0.0032; `temporal_style` ~flat;
`inter_video_diversity` african +0.0057 / frying -0.0001 (no collapse).
`prompt_adherence` stayed within tolerance.

### Caveats / risks

- This gate scored GPU-free dimensions only. DINO subject, CLIP background, and
  ArcFace/DINO-patch identity backbones were NOT loaded (`backbones` are all
  false in the JSON), so `aggregate_consistency` here is only
  mean(`temporal_style`, `appearance_style`, `overall_consistency`).
- Coverage is minimal: 2 scenes x 2 perspectives, with `num_output_frames=16`
  (2 blocks).
- `dynamic_degree` dropped notably: african 2.61 -> 2.22 and frying 2.99 ->
  2.78. This is a reduction in motion, not a collapse (`inter_video_diversity`
  held), but it is a flagged risk.

At the time, this was a real but SMALL, narrow-coverage Round 0 signal, not a
final winner. Its proposed next step was to add a `dynamic_degree`
non-regression guard, then render the other finalists (`semantic+raw`,
`pooled+raw` control) with the full semantic backbones across more
scenes/perspectives before declaring any render-confirmed winner. Round 1 above
is that superseding guarded result, and it is an honest null.

> Update (same round): the `dynamic_degree` non-regression guard is now
> implemented (`evaluate_multiview_vbench_gate(..., motion_tolerance=...)`, exposed
> as `--motion_tolerance`). It is OFF by default so this committed result -- which
> did not enforce it -- is reported faithfully rather than retroactively failed;
> future gates can enable it.

### Reproduce

```bash
python scripts/run_kv_rag_ablation.py \
  --config_path configs/inference_kv_rag_round0gate.yaml \
  --mode multiview_vbench \
  --prompts_dir example/multiview_prompts \
  --prompt_subset frying_egg_closeup,african_savanna \
  --max_perspectives 2 \
  --modified_retrieval_key_mode subject_identity \
  --modified_retrieval_value_mode raw \
  --adherence_tolerance 0.02 \
  --diversity_tolerance 0.05 \
  --generator_ckpt checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --no_lora_adapter \
  --metrics_json docs/multiview_gate_results/round0_subject_identity_2persp.json \
  --output_root videos/round0_vbench_gate
```
