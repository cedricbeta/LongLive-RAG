# KV-RAG AC-3 Candidate Matrix and Screen

This is the implementation spec for the LongLive-RAG single-scene,
cross-video consistency key/value sweep. It extends the existing decoupled
retrieval interface in `utils/kv_rag.py` and uses the AC-2 VBench-style
cross-video suite in `evaluation/vbench_consistency.py` as the rendered gate.

The prior rendered study (`docs/retrieval-key-value-study.md`) is binding:
`BL-20260604-rendered-metric-over-probe` showed the GPU-free synthetic probe
mis-ranked keys (`pooled+raw` beat `salient_set+raw` by about 2.5x on rendered
5B video). The offline screen below is therefore a labeled pre-filter only. It
must never select a winner without rendered AC-2 confirmation.

## Candidate Matrix

Keys (`retrieval_key_mode`):

| key | indexing signal | score | sweep role |
|---|---|---|---|
| `pooled` | current mean-pooled pre-RoPE K summary | cosine / configured batch scorer | incumbent rendered winner and baseline-compatible control |
| `moment` | mean + std token summary | cosine / configured batch scorer | cheap orderless variant; prior probe showed weak discrimination |
| `multi_centroid` | activation-ranked token buckets, centroid per bucket | best-centroid cosine | coarse region-aware visual key |
| `salient_set` | top-M high-norm token set | symmetric mutual-best cosine | fine-grained visual key; prior proxy favorite, rendered loser |
| `positional` | position-weighted pooled K summary | cosine / configured batch scorer | negative control; expected to fail viewpoint invariance |
| `semantic` | L2-normalized caption/text embedding from the active `conditional_dict["prompt_embeds"]` | plain cosine | decoupled text/scene key, set per perspective with `memory.set_context_key(...)` |
| `subject_identity` | per-head subject prototype from attention tensors: top-M high-norm tokens minus per-head background mean, L2-normalized | plain cosine | visual identity-aware key for subject continuity across perspective changes |

Values (`retrieval_value_mode`):

| value | payload | injected length | sweep role |
|---|---|---:|---|
| `raw` | stored frame-aligned K/V slice | full stored slice | faithful default; isolates key effects |
| `mean_frame` | average across stored frames, keeping one frame of tokens | 1 frame | existing bounded compressed value |
| `top_frame` | single highest-norm stored frame, keeping its original frame tokens | 1 frame | new bounded value; frame-aligned and re-RoPE'able |

Full sweep: every key above crossed with every value above, for 21 labeled
candidate cells. `positional` cells are screen-only controls and must not become
render finalists except when debugging the screen itself.

Recommended first render finalists:

| finalist | why |
|---|---|
| `pooled+raw` | rendered incumbent from the prior study; required control for the probe-misranking lesson |
| `semantic+raw` | tests the new decoupled text/scene key while keeping the faithful value fixed |
| `subject_identity+raw` | tests the new identity-aware visual key while keeping the faithful value fixed |

If only two candidates can be rendered, keep `pooled+raw` and render whichever
of `semantic+raw` or `subject_identity+raw` passes the screen with the larger
same-scene margin. `top_frame` is swept in the pre-filter and should graduate in
the next render pass paired with the best rendered key, so the first pass does
not confound new-key behavior with compression loss.

## Offline Proxy Screen

The GPU-free signal is the existing viewpoint-invariance permutation probe:

1. Build labeled synthetic tensors for a reference view `A`, a same-scene
   rearranged view `B`, and a different scene `D`.
2. For each candidate key, compute `score(A, B)` and `score(A, D)`.
3. Rank by `margin = score(A, B) - score(A, D)`, with `score(A, B) > 0.6` and
   `margin > 0.1` as the minimum nondegenerate pass.
4. Require `positional` to fail the same-view threshold; this keeps the probe
   non-vacuous.
5. For values, record only structural facts: stored token count, frame alignment,
   and whether the payload remains re-RoPE'able. The proxy does not rank value
   quality.

`semantic` is screened with the same labels but with external context vectors
instead of K-token summaries. The harness must call `memory.set_context_key(...)`
for each synthetic perspective. If no context key is set, `semantic` fails fast
instead of silently falling back to visual K. GPU-free semantic fixtures can be
deterministic synthetic vectors or cached prompt embeddings; they only verify
the context-key path and cosine scoring, not real text-encoder quality.

`subject_identity` is screened on the attention-tensor fixture. The prototype is
computed per head by selecting the top-M token vectors by norm, subtracting the
per-head background mean, then L2-normalizing. Candidate scoring is plain cosine
averaged over batch and heads, independent of `similarity`.

## What The Screen Can Tell Us

The screen can reject keys that are clearly position-sensitive, cross-scene
contaminating, missing required context, shape-incompatible, nondeterministic,
or not frame-aligned for compressed values. It is useful for pruning broken
candidate cells before expensive renders.

The screen cannot tell us whether a key/value changes the model's actual
attention distribution, denoising trajectory, subject identity, prompt
adherence, diversity, or final rendered pixels. It also under-represents real
perspective changes where the subject changes scale, self-occludes, rotates,
or changes appearance, and where only part of the background persists.

Known failure mode: the probe can reward a representation for pure token
permutation invariance even when that representation is too coarse for real
video. This is exactly what happened when `salient_set` looked best offline but
`pooled+raw` won on rendered 5B video.

## Graduation To Rendered AC-2

The screen produces a ranked shortlist, not a winner. A candidate graduates only
if it:

- passes the nondegenerate same-scene margin;
- does not fail shape/context/value-alignment checks;
- is not the `positional` negative control;
- has its exact key/value settings recorded for the render job.

Final selection is made only after per-perspective 5B renders are scored by the
AC-2 suite: `subject_consistency`, `background_consistency`,
`subject_identity_consistency`, `temporal_style`, `appearance_style`,
`overall_consistency`, `aggregate_consistency`, plus the anti-cheat companions
`inter_video_diversity`, `dynamic_degree`, `motion_smoothness`, and the
`prompt_adherence` guard. A consistency gain that collapses diversity/motion or
regresses adherence beyond tolerance is not a win.

The rendered JSON must record the chosen `retrieval_key_mode`,
`retrieval_value_mode`, key hyperparameters (`retrieval_key_top_m`,
`retrieval_key_centroids`), scene-memory settings, seeds, scene subset, and
per-scene metric deltas. Proxy-only promotion is rejected by design.

## Decoupling And Fail-Fast Rules

AC-3.1 remains the interface contract: the retrieval key is the index and the
retrieval value is the payload. They are chosen independently. A `semantic` or
`subject_identity` key can index a faithful `raw` K/V value, and a compressed
`top_frame` value can be paired with any key.

Implementation requirements:

- Add the new names to `KEY_MODES` and `VALUE_MODES`; unknown names keep failing
  fast in `KVRAGMemory._validate_config`.
- `semantic` uses only the active external context key set by
  `memory.set_context_key(...)`, derived from the current perspective's
  `conditional_dict["prompt_embeds"]`; missing context is an error.
- `subject_identity` uses the attention tensor summary path and scores by plain
  cosine, not L2 or a configurable alternative.
- `top_frame` keeps exactly one whole frame of K/V tokens and reports
  `stored_frames = 1`, so the retrieved payload stays frame-aligned and
  re-RoPE'able.
- Defaults remain `pooled+raw`; disabling the new modes preserves baseline
  retrieval behavior.
