# Typed Memory Capsule pilot: method transfer

Date: 2026-07-15  
Status: P-1 design freeze completed before capsule instrumentation or rendering

## Scope and fixed decisions

This document transfers experimental discipline and mathematical templates from
established representation-editing methods into the Wan2.2-TI2V-5B pilot. It
does not reopen the pilot's fixed decisions:

- estimate and edit only clean-recache self-attention **V** vectors;
- do not modify K or RoPE;
- edit entity tokens already present in the active local window, without
  appending attention columns;
- use the pooled shot-0 entity prototype as donor;
- keep the estimator low-rank, closed-form, and training-free;
- use the same in-place delivery path for capsule, full-vector, and random arms;
- report motion, background, and diversity leakage separately.

The model has 24 attention heads with head dimension 128. The implementation
must preserve head structure. A single configured rank `r <= 16` is used, with
one whitened projector per `(layer, head)`. This follows the repository's
existing observation that attention heads are independent feature spaces; it
also avoids treating a concatenation of 24 unrelated head coordinates as one
Euclidean space.

## Method-transfer map

| Source template | Validated operation/evaluation pattern | Capsule-pilot transfer | Frozen consequence |
|---|---|---|---|
| [Activation Addition (ActAdd)](https://arxiv.org/abs/2308.10248), [Representation Engineering (RepE)](https://arxiv.org/abs/2310.01405) | Construct contrastive activation directions and intervene at inference time; intervention strength and layer choice materially affect efficacy and side effects. | The donor-conditioned update is activation steering whose direction depends on the current target. Scan the five probe layers, then scan `alpha in {0.25, 0.5, 1.0}` on one scene while running guards. | Layer selection is made by held-out G1, not convenience. Alpha is fixed after the one-scene stability scan and is not retuned per scene. |
| [INLP](https://aclanthology.org/2020.acl-main.647/) | Remove linearly decodable concept information using projections onto classifier null spaces. | Treat identity as linearly represented only if held-out identity decoding is concentrated in a low-rank subspace and reduced in its complement. | G1 is a falsification gate. Failure stops all P2 rendering. INLP itself is not used because its iterative classifiers violate the pilot's closed-form/no-training estimator constraint. |
| [LEACE](https://arxiv.org/abs/2306.03819) | Whiten correlated features, project in the whitened metric, and transport back; this yields a minimum-distortion closed-form edit rather than a naive raw-space Euclidean projection. | Whiten by regularized within-identity covariance, select Fisher directions in whitened coordinates, edit there, then unwhiten. | `P_id` is orthogonal in whitened coordinates and generally oblique in raw V coordinates. A raw Euclidean `UU^T` is forbidden for the capsule arm. |
| [ROME](https://arxiv.org/abs/2202.05262), [MEMIT](https://arxiv.org/abs/2210.07229) | Locate the layers carrying an association before editing; report efficacy, generalization, and specificity. | Use G1 to locate a causal candidate layer without changing weights. Organize G3 under the same three headings. | The selected layer must be declared before P2 full runs. Neighbor preservation becomes the unaggregated leakage vector plus existing guards. |
| [Asyrp / h-space](https://arxiv.org/abs/2210.10960) | Edit a semantic internal space through an asymmetric diffusion path; editing behavior depends on where/when an intervention enters the reverse process. | The pilot edits clean context cached at timestep zero, not the noisy generation-time hidden state. The edited context V is reused across denoising sigma values. | No timestep schedule is added in this pilot. This is a design advantage of clean-recache capsules; a schedule is deferred unless future work edits generation-time hidden states. |
| [StyleAligned](https://arxiv.org/abs/2312.02133) | Share self-attention K/V across images; stronger/full sharing improves alignment but can reduce prompt alignment and diversity. | Existing whole-frame/region KV transfer is the causal-video analogue. The capsule asks whether sharing only an identity component of V avoids content/layout leakage. | Full-vector `P=I` is a required mechanistic control, delivered through the same token-local in-place path as the capsule. |
| [ConsiStory](https://arxiv.org/abs/2402.03286) | Share subject-specific attention activations without training and localize subject patches using attention associated with subject text. | First-choice P0 masks come from the model's own subject-word cross-attention maps; external CLIPSeg and manual boxes are fallbacks only. | Cross-attention diagnostics are computed in a sampled/side `einsum` path outside compiled flash attention. Mask provenance and QC are mandatory. |
| [IP-Adapter](https://arxiv.org/abs/2308.06721), [PhotoMaker](https://arxiv.org/abs/2312.04461) | Learned adapters or identity encoders can supply strong image/identity conditioning. | They are a related trained upper-bound family, not a transferable estimator for this pilot. | No adapter, identity encoder, finetuning, or new rendering for training data is introduced. |

## Final estimator and edit formula

All statistics below are computed separately for probe layer `ell` and attention
head `h`; those subscripts are omitted. Each clean-recache entity token is a
column vector `v in R^128`. Samples are grouped by generated-instance identity
and by shot/frame so train/validation splits never separate tokens from the same
frame.

### 1. Covariances

An identity class is one generated instance, keyed by `(scene, seed)`. In
particular, seed 0/1/2 generations of the same prompt are **different
identities in matched semantic context**, as required by the pilot's hard
negative construction; they must never be merged into one class. For identity
`c`, let `mu_c` be the training mean and let `mu` be the global training mean.
With class-balanced weights, compute

```text
Sigma_within = mean_c mean_{v in c} (v - mu_c)(v - mu_c)^T
Sigma_between = mean_c (mu_c - mu)(mu_c - mu)^T.
```

Regularize only the within covariance:

```text
Sigma_w = Sigma_within + epsilon * trace(Sigma_within)/d * I,
d = 128.
```

`epsilon` is selected from a short, declared numerical-stability grid using only
the training fold. The chosen value, condition number, effective numerical rank,
and sample counts must be written to the G1 manifest. No label-dependent
hyperparameter is selected on the held-out fold.

### 2. Whitening and Fisher directions

Use a symmetric eigendecomposition of `Sigma_w` in float64 on CPU and define

```text
W     = Sigma_w^(-1/2)           # whitening
W_inv = Sigma_w^(+1/2)           # inverse transform
B_w   = W Sigma_between W^T.
```

Let `U_r` contain the `r <= 16` leading orthonormal eigenvectors of `B_w`, with
rank capped by its positive numerical eigenvalues. The identity-component
operator is

```text
Q_id = U_r U_r^T                         # orthogonal in whitened coordinates
P_id = W_inv Q_id W.                     # idempotent, generally oblique in raw V
```

Equivalently, if `A = W U_r` are within-metric-normalized generalized Fisher
directions satisfying

```text
Sigma_between A = Sigma_w A Lambda,
A^T Sigma_w A = I,
```

then `P_id = Sigma_w A A^T`. The explicit whiten/edit/unwhiten implementation is
preferred over materializing `P_id`; it is better conditioned and makes the
metric contract testable.

### 3. Donor-conditioned in-place edit

For each edited boundary, pool shot-0 donor entity tokens per head:

```text
v_bar_donor[h] = mean_{shot-0 donor entity tokens} v[h].
```

For every target entity token `i` in the active window, and only at the selected
layer/head, compute in float32 and cast back to the cache dtype:

```text
delta_w = W (v_bar_donor - v_i)
v_i' = v_i + alpha * W_inv U_r U_r^T delta_w.
```

Centering does not appear in this equation because it cancels in the
donor-minus-target difference. K, token positions, RoPE, sequence length, and
all non-entity V rows remain bitwise untouched.

### 4. Control arms

All arms call the same boundary editor and differ only in `Q`:

```text
capsule:          Q = U_r U_r^T
full-vector:      Q = I                  (so W_inv Q W = I)
random-subspace:  Q = R_r R_r^T
```

`R_r` is generated once per layer/head from a manifest-recorded seed, QR
orthonormalized in whitened coordinates, and has exactly the capsule rank.
Random directions and the capsule therefore use the same covariance metric,
rank, and alpha; their realized projected-update norms are logged rather than
assumed equal. The random basis, boundary list, masks, donor pool, and cache
write sites are frozen before rendering.

## Layer, rank, and alpha selection discipline

1. Probe exactly layers `{0, 7, 14, 21, 29}`.
2. Evaluate ranks from the positive Fisher spectrum, capped at 16. Select the
   smallest rank within one standard error of the best held-out identity score,
   subject to the pose/time leakage check.
3. Select a single `(layer, rank)` from held-out G1 only. Do not inspect P2
   renders to change it.
4. On the declared watchmaker scene, render `alpha = {0.25, 0.5, 1.0}` after a
   real-V numerical prefilter. Review a synchronized baseline/three-alpha 4-up
   and contact sheet, then choose one alpha before the four-scene run; do not
   tune per arm or per scene. A zero-render norm check is not an alpha choice.
5. Run the frozen Round-18 guard suite at every rendered alpha (diversity 10%,
   RAFT motion 20%, adherence 0.02, invariant margin 0), in addition to cache,
   frame, and boundary checks. NaN, crash, cache-contract violation, guard
   regression, or visible boundary artifacts reject that setting.

## G1 held-out test contract

The primary split is held-out shots, grouped at the frame level: even-indexed
shots `{0,2,4,6}` form identity prototypes and odd-indexed shots `{1,3,5,7}`
are held out. The labels are all 12 `(scene, seed)` generated identities, so the
same-prompt cross-seed pairs are hard negatives. Tokens from one frame never
cross folds. This satisfies the frozen G1 choice of held-out shots; no seed is
held out because a seed is part of the generated-instance identity label.

As a scene-generalization check, also run four leave-one-scene-out (LOSO)
folds. In each fold, estimate the Fisher projector from even-indexed shots of
the other three scenes only. Use even-indexed shots of the held-out scene only
to form its three seed-specific centroids, then test those centroids on the
held-out scene's odd-indexed shots. The held-out scene therefore contributes no
samples to subspace estimation or rank selection. Report capsule, complement,
and manifest-seeded rank-matched random-subspace accuracy for every fold.

For every layer/rank, report:

- nearest-class-centroid balanced accuracy on frame/shot-pooled V in the
  identity subspace, the raw V, the full whitened orthogonal complement, and at
  least 100 manifest-seeded rank-matched random subspaces;
- Fisher separation and silhouette score for the same representations;
- the random-null percentile and a grouped bootstrap 95% confidence interval,
  resampling frames/shots rather than individual tokens;
- the fraction and per-dimension density of within-identity pose/time delta
  energy in `Q_id` versus `I-Q_id`.

G1 passes only if at least one layer has identity-subspace performance above the
95th percentile of the random null, above the orthogonal complement on the
declared identity metrics, and pose/time delta energy predominantly in the
complement. The selected rank must additionally beat the complement and the
aggregate random-null threshold in at least three of four LOSO folds, with its
mean LOSO accuracy above both controls. Otherwise the pilot stops with
`verdict=kill` and P2 renders remain zero. If these statistical gates pass but
the mandatory all-data refit is ill-conditioned or cannot retain the selected
rank, the outcome is instead infrastructure-blocked; that artifact-construction
failure does not overturn the observed held-out subspace evidence.

## G3 report schema

G3 is reported as three sections, preserving per-scene and per-shot records.
No leakage scalar may be synthesized.

### `efficacy`

| Field | Definition |
|---|---|
| `dino_subject_anchor_to_shot0` | Existing per-shot subject stream for all four scenes. |
| `face_crop_dino_anchor_to_shot0` | Additional per-shot crop stream for the two human scenes. |
| `paired_delta_vs_baseline` | Capsule minus same-seed baseline at each edited drift shot. |
| `paired_delta_vs_random` | Capsule minus same-seed random arm at each edited drift shot. |
| `human_identity_rating` | `{better,same,worse,broken}` plus one sentence naming the decisive shot and visible trait. |

### `generalization`

| Field | Definition |
|---|---|
| `scenes_improved` | Count/list of scenes whose edited drift shots improve over baseline and random. |
| `boundaries_improved` | Count/list of edited boundaries after which improvement persists. |
| `fixed_hyperparameters` | One layer, rank, alpha, mask method, and donor rule used across the full run. |
| `human_prompt_following` | Per scene, whether later shots retain their own prompt composition/action rather than shot-0 layout. |

### `specificity`

| Field | Definition |
|---|---|
| `raft_motion_delta` | Per-shot capsule-minus-baseline motion, compared separately with full-vector. |
| `clip_background_delta` | Per-shot capsule-minus-baseline background similarity/change, compared separately with full-vector. |
| `inter_shot_diversity_delta` | Per-shot capsule-minus-baseline diversity, compared separately with full-vector. |
| `guard_results` | Existing guard results, NaN/crash status, and artifact flags. |
| `human_motion_rating` | Real-time 4-up judgment for freezing, sluggishness, or ghosting. |
| `human_boundary_artifact_rating` | Judgment over the one-second interval around each edited boundary. |
| `placebo_rating` | Whether random-subspace is visually indistinguishable from baseline. |

The final decision is:

- **promising** only when capsule identity is better than baseline and random on
  drift shots, generalizes beyond one scene, and its three leakage coordinates
  are materially smaller than full-vector, with supporting human inspection;
- **kill** when capsule is no better than random, leakage matches full-vector,
  or improvement is confined to one scene.

If numeric G3 and `human_verdict` conflict, the report records both and uses the
human conclusion for the final verdict. This is a single-seed exploratory pilot
and explicitly does not constitute paper-level evidence.
