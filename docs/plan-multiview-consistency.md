# Single-Scene Multi-Perspective Consistency for LongLive-RAG

## Goal Description

Improve the KV-cache RAG so that multi-shot videos depicting ONE scene from DIFFERENT
camera perspectives stay coherent across shots. When successive shots are different
viewpoints of the same environment, the spatial layout, geometry, lighting, and the
placement/identity of the subjects and objects must read as the *same place seen from a
new angle* -- not a freshly hallucinated scene at every cut -- while the camera framing
and composition are still allowed to change.

The work targets the seam between the KV-RAG memory and the multi-shot boundary
handling. Today a shot boundary re-pins the attention sink to the new shot's first frame
(`multi_shot_sink`), applies a per-shot RoPE phase offset (`multi_shot_rope_offset`), and
optionally clears the KV cache (`shot_clean_recache`). These reset the very anchors that
establish a shared environment, and KV-RAG retrieval (content-matched on pre-RoPE
summaries) can fail to surface same-scene tokens once the framing changes. The core
design tension to resolve: at a perspective change, KEEP the scene but ALLOW the
viewpoint to change. The whole KV state is currently treated as one bucket that is reset
wholesale at cuts.

A first-class part of this work is determining the best RAG design for this setting --
in particular what the retrieval *key* (the lookup/index signal matched against the
current query) and the retrieval *value* (the payload injected back into attention)
should be. The current design indexes by a mean-pooled, per-head, pre-RoPE attention-key
summary (one vector per stored chunk) and injects the raw frame-aligned K/V slices; both
choices are treated here as OPEN and must be evaluated, because matching the *same scene
from a different camera perspective* is exactly where the key representation matters most.

The deliverable must be verifiable cheaply enough to run inside an unattended RLCR
(`--yolo`) loop: GPU-free unit tests plus an offline metric, with full-model inference
reserved for milestone gates only.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification.

- AC-1: KVRAGMemory exposes a persistent "scene-memory" partition that survives shot
  boundaries, distinct from per-shot state that may still reset.
  - Positive Tests (expected to PASS):
    - After storing clean tokens in shot 0 and crossing a simulated shot boundary, the
      scene-memory partition still contains the shot-0 anchor entries.
    - The per-shot partition is cleared at the boundary when `shot_clean_recache` is on,
      while the scene-memory partition is unaffected.
  - Negative Tests (expected to FAIL):
    - With the feature flag disabled, no persistent partition exists and behavior is
      byte-identical to the current single-bucket reset (regression guard).
    - Storing more than the configured scene-memory capacity must not grow memory without
      bound (eviction policy is enforced, not silently ignored).

- AC-2: At a shot boundary, scene-anchor tokens are force-injected into the attention
  window of the new shot, re-RoPE'd to in-distribution relative positions.
  - Positive Tests:
    - On the first chunk after a boundary, the attention key/value sequence length
      includes the injected scene-anchor tokens at the expected offset.
    - Injected tokens carry RoPE positions consistent with the virtual-frame placement
      used by the existing retrieval injection (no out-of-distribution positions).
  - Negative Tests:
    - Disabling injection yields no added tokens at the boundary (flag isolation).
    - Injection must not duplicate tokens already present via the normal sink/window
      (no double-counting of the same frame).

- AC-3: The retrieval key and value representations are an explicit, evaluated design
  choice, and the selected key surfaces same-scene tokens after a viewpoint change.
  - Positive Tests (expected to PASS):
    - At least two retrieval-KEY representations and at least two retrieval-VALUE
      representations are implemented behind a config switch and selectable without code
      changes (e.g. key in {pooled pre-RoPE attention key [baseline], alternative}; value
      in {raw frame-aligned K/V [baseline], alternative}).
    - Viewpoint-invariance probe: for two synthetic chunks that share scene/subject content
      but differ in framing/viewpoint, the chosen key scores them as more similar than a
      genuinely different scene at equal nominal content distance.
    - Given a query from a new framing of the same scene, top-k retrieval returns at least
      one same-scene anchor; scoring is deterministic for a fixed input.
  - Negative Tests (expected to FAIL):
    - Cross-scene contamination is rejected: a different scene's tokens are not
      preferentially retrieved over same-scene tokens at equal content distance.
    - A key representation that is NOT viewpoint-invariant (e.g. raw post-RoPE keys) fails
      the viewpoint-invariance probe -- the probe must distinguish good keys from bad,
      not pass vacuously.
  - AC-3.1: A pluggable interface separates "how an entry is indexed" (key) from "what is
    injected" (value), so key and value can be chosen independently (decoupled retrieval).
    - Positive: selecting a semantic/external key while keeping raw-K/V values works.
    - Negative: an unknown key/value name fails fast with a clear error.
  - AC-3.2: A written comparison selects the key/value representation on the offline metric
    (AC-6) with a recorded rationale (consistency win, cost, train-free preference).
    - Positive: the comparison ranks the implemented variants and names a winner.
    - Negative: a variant that regresses per-shot prompt adherence is not selected on
      consistency score alone (anti-cheating tie-in to AC-7 guardrails).

- AC-4: Backward compatibility -- all new behavior is gated behind config flags; the
  single-shot path and the existing multi-shot path are unchanged when the new flags are
  off.
  - Positive Tests:
    - Existing inference configs (no new flags) produce identical control flow through
      the multi-shot scheduler (verified by the existing tests / a golden trace).
    - The default key/value representation is the current one (pooled pre-RoPE key + raw
      K/V), so disabling the new switch reproduces baseline retrieval exactly.
  - Negative Tests:
    - A config that sets a new flag without its required companion value fails fast with a
      clear error rather than silently misbehaving.

- AC-5: GPU-free unit tests cover memory bucketing, persistence across a cut, injection
  ordering, RoPE re-positioning, retrieval determinism, and the key/value representations.
  - Positive Tests:
    - The unit-test suite runs on CPU with no model checkpoint and passes.
  - Negative Tests:
    - A deliberately broken injection offset is caught by the ordering/RoPE assertions.

- AC-6: An offline cross-perspective scene-consistency metric is added to
  `scripts/evaluate_kv_rag.py`, runnable on `example/multiview_prompts/`.
  - Positive Tests:
    - The script computes a per-video cross-shot consistency score from pre-rendered
      frames and emits JSON, baseline vs. modified, without requiring a GPU render in the
      same process.
  - Negative Tests:
    - The metric rejects degenerate "copy shot 0" outputs by also reporting per-shot
      prompt-adherence / viewpoint-variation so a collapse is visible, not hidden.

- AC-7 (milestone quality gate): On at least two vendored sample prompts, the modified
  pipeline improves cross-perspective scene consistency over the baseline WITHOUT
  regressing per-shot prompt adherence beyond a configured tolerance.
  - Positive Tests:
    - Consistency score (modified) > consistency score (baseline) on >= 2 prompts.
  - Negative Tests:
    - Prompt-adherence (modified) does not drop below baseline minus tolerance on any
      evaluated prompt (anti-cheating guard).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
Decoupled memory with two partitions -- a persistent "scene" partition (survives cuts,
force-injected at boundaries) and a per-shot "composition/framing" partition (resets per
shot) -- plus retrieval scoring biased toward same-scene anchors, plus optional
reference-frame conditioning that keeps shot 0's clean anchors as a persistent sink. A
pluggable, decoupled retrieval key/value layer is included, with several candidate KEY
representations (pooled pre-RoPE key [baseline], spatially/region-structured or
multi-centroid keys, and a viewpoint-invariant semantic key from a frozen encoder or a
lightweight learned projection) and several VALUE representations (raw frame-aligned K/V
[baseline], pooled/centroid anchors, subject-masked or compressed K/V), with the final
choice selected empirically on the offline metric. Full GPU-free unit coverage and a
complete cross-perspective metric integrated into the existing ablation harness.

### Lower Bound (Minimum Acceptable Scope)
A single persistent scene-anchor sink, populated from shot 0's clean recache and
force-injected at every subsequent shot boundary (re-RoPE'd like existing retrieved
tokens), gated behind one config flag, with GPU-free unit tests (AC-1, AC-2, AC-4, AC-5)
and a working-but-minimal metric (AC-6). At minimum, the retrieval key/value study
compares the current representation (pooled pre-RoPE key + raw K/V) against one
alternative key and one alternative value and records the choice (AC-3). AC-7 (measured
win) remains stretch within the lower bound.

### Allowed Choices
- Can use: the existing KV-RAG infrastructure (`KVRAGMemory`, `KVRAGEntry`, the attention
  store/retrieve hooks), new config flags with backward-compatible defaults, the existing
  re-RoPE virtual-frame injection mechanism, existing evaluation/ablation scripts.
- Can use for the key/value: pooled / spatially-structured / multi-centroid summaries; a
  decoupled semantic key (DINO / CLIP on a decoded clean anchor frame, or a caption-text
  embedding) computed once at store time; pooled/centroid, subject-masked, or
  FP4-compressed K/V values (the repo already supports KV compression). A lightweight
  learned retrieval projection is allowed ONLY as a small add-on that trains without
  retraining the 5B base model.
- Can use for the metric: an off-the-shelf frozen visual encoder (DINO / CLIP) and,
  optionally, a face-embedding model for human subjects -- pinned and downloaded once.
- Cannot use: any change that requires retraining the 5B base model from scratch, an
  external document/corpus retrieval system, or that breaks existing config flags or the
  single-shot path. A learned key/value cannot become the DEFAULT if it requires
  base-model retraining. No new mandatory heavyweight dependency in the per-round test
  path (the per-round unit tests must stay GPU-free and checkpoint-free).

> Design-axis note: the injection *mechanism* (re-RoPE into a virtual frame block placed
> before the local window) and the `KVRAGEntry` container layout are stable -- reuse them
> rather than inventing a new injection path. However, the retrieval *key* (what populates
> `summary` and how the query is matched) and the retrieval *value* (what is stored as the
> injected payload) are explicitly OPEN and are a primary exploration axis of this plan;
> upper and lower bounds do NOT converge for them.

## Feasibility Hints and Suggestions

> Reference only -- conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
```
# In KVRAGMemory: add an optional persistent partition.
class KVRAGMemory:
    scene_entries: list[KVRAGEntry]   # NOT cleared on shot boundary
    shot_entries:  list[KVRAGEntry]   # cleared per existing reset policy

    def store(entry, *, persistent: bool): ...
    def reset_shot(): shot_entries.clear()           # scene_entries untouched
    def retrieve(query, k):
        # key_fn(query) vs key_fn(entry) similarity over scene_entries + shot_entries;
        # optionally up-weight scene_entries; return value_fn(entry) payloads.
        ...

# In the multi-shot scheduler (pipeline/causal_diffusion_inference.py):
on shot boundary:
    if scene_memory_enabled:
        anchors = memory.scene_entries top-N
        force_inject(anchors)         # re-RoPE into the virtual frame block, like retrieval
    apply existing multi_shot_sink / rope_offset / clean_recache as before

# Populate scene_entries from shot 0's CLEAN recache (t=0), mirroring how KV-RAG
# already stores clean slices -- just route shot-0 anchors into the persistent partition.
```

### Current baseline (what the key/value are today)
Confirmed in `utils/kv_rag.py`:
- Retrieval KEY: `KVRAGEntry.summary` = `_summarize(k_pre)` -- the pre-RoPE attention KEY
  tensor `[B,T,H,D]` mean-pooled over tokens to `[B,H,D]` (per-head) and L2-normalized
  (`_summarize`, `_score_batch`). One summary vector per stored chunk per layer; the query
  is summarized the same way and matched by cosine. Position-invariant (pre-RoPE) but NOT
  explicitly viewpoint-invariant, and chunk-level (the mean blends subject + background).
- Retrieval VALUE: the raw frame-aligned per-token K/V slices (`entry.k`, `entry.v`),
  optionally downsampled to whole frames and re-RoPE'd at injection (`reinject_rope`).
- Granularity: store per chunk/layer; retrieve top_k chunks; matching is chunk-level.

### Retrieval key/value design space (to explore and select)
KEY (the lookup/index signal):
- Pooled pre-RoPE attention key (baseline) -- coarse, view-sensitive mean.
- Hidden-state (residual) summary instead of the attention-key projection.
- Spatially-structured or multi-centroid keys (several vectors per chunk: foreground vs
  background) to enable subject-region matching across views.
- Viewpoint-invariant semantic key from a frozen encoder (DINO/CLIP) on the decoded clean
  anchor frame, computed once at store time -- decoupled from the attention key space.
- Caption/text-derived key (embed the shot caption) -- cheap, explicitly scene-aligned.
- Lightweight learned retrieval projection (contrastive, view-invariant) -- powerful but
  needs small add-on training; train-free options preferred.

VALUE (the injected payload):
- Raw frame-aligned K/V (baseline) -- faithful but per-layer and grows with top_k.
- Pooled/centroid anchors -- bounded injected length; emphasize persistent scene/subject.
- Subject-masked K/V -- inject identity/scene, not background (serves keep-subject/change-view).
- Compressed K/V (the repo already has FP4 KV compression) -- cheaper storage.
- Clean hidden states reprojected to K/V per layer at inject time -- avoids per-layer store.

Central tension: the KEY should be viewpoint-invariant (match the same scene across
camera angles) while the VALUE stays faithful enough to condition generation. Decoupling
the key from the value (AC-3.1) lets a semantic/invariant key index a faithful raw-K/V
payload -- likely the most promising starting point to evaluate.

### Relevant References
- `utils/kv_rag.py` - KVRAGConfig / KVRAGMemory / KVRAGEntry; `add` (store), `retrieve`,
  `_summarize` (current key), `_score_batch` (matching). Primary site for the persistent
  partition, eviction, and the pluggable key/value interface.
- `wan_5b/modules/causal_model.py` - `CausalWanSelfAttention` store/retrieve/inject hooks
  and the re-RoPE virtual-frame injection. Primary site for force-injection at boundaries.
- `pipeline/causal_diffusion_inference.py` - per-shot scheduling, `_is_shot_boundary`,
  `multi_shot_sink` / `rope_temporal_offset` / `shot_clean_recache`, `_reset_kv_rag`.
  Primary site for boundary-time orchestration and flag gating.
- `utils/dataset.py` - `MultiTextConcatDataset` directory mode (`0.json..N.json` +
  `shot_durations.txt`) -- matches the vendored `example/multiview_prompts/` set.
- `scripts/evaluate_kv_rag.py` - extend with the cross-perspective consistency metric.
- `scripts/run_kv_rag_ablation.py` - baseline-vs-modified generation driver for the
  milestone gate (AC-7).
- `configs/inference_kv_rag.yaml` - where the new flags (incl. key/value selection) live
  with safe defaults.
- `example/multiview_prompts/` - vendored NVlabs single-scene multi-perspective prompts
  (11 themes, 6-8 shots each); the standing evaluation set.

## Dependencies and Sequence

### Milestones
1. Persistent scene memory (data layer): AC-1, AC-4, AC-5(partial)
   - Phase A: add the scene partition + eviction to `KVRAGMemory`, flag-gated.
   - Phase B: route shot-0 clean-recache anchors into the persistent partition.
2. Boundary-time injection (attention layer): AC-2, AC-5
   - Phase A: force-inject scene anchors at shot boundaries via the existing re-RoPE path.
   - Phase B: ordering/positioning assertions; ensure no duplication with sink/window.
3. Retrieval key/value design study: AC-3
   - Step 1: add a pluggable, decoupled key/value interface (config switch) over the
     existing `_summarize` / `entry.k,v` path; default reproduces the baseline.
   - Step 2: implement >= 2 key and >= 2 value representations (including a
     viewpoint-invariant key) + scene-aware scoring/up-weighting.
   - Step 3: unit tests (determinism + viewpoint-invariance probe); compare on the offline
     metric and select the winner.
4. Evaluation (offline): AC-6
   - Step 1: cross-perspective consistency metric + anti-cheating companion metrics.
   - Step 2: wire into `evaluate_kv_rag.py`; consume pre-rendered frames.
5. Milestone quality gate (expensive, few seeds): AC-7
   - Step 1: render baseline + modified on >= 2 vendored prompts via the ablation driver.
   - Step 2: report consistency win + prompt-adherence non-regression.

Milestone 2 depends on 1. Milestone 3 depends on 1 (and is evaluated via 4). Milestone 4
depends on 1 (and 2/3 for meaningful eval). Milestone 5 depends on 3 and 4 (and a working 2).

## Task Breakdown

Each task includes exactly one routing tag: `coding` (Claude) or `analyze` (Codex).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Add flag-gated persistent scene partition + eviction to KVRAGMemory | AC-1, AC-4 | coding | - |
| task2 | Route shot-0 clean-recache anchors into the persistent partition | AC-1 | coding | task1 |
| task3 | Force-inject scene anchors at shot boundaries via existing re-RoPE virtual-frame path | AC-2 | coding | task2 |
| task4 | GPU-free unit tests: persistence-across-cut, injection ordering, RoPE re-positioning, retrieval determinism | AC-5 | coding | task3 |
| task5 | Review injection placement for OOD RoPE positions / duplication risk vs sink+window | AC-2 | analyze | task3 |
| task6 | Scene-aware retrieval scoring/up-weighting | AC-3 | coding | task11 |
| task7 | Cross-perspective consistency metric + anti-cheating companion metrics | AC-6 | coding | - |
| task8 | Wire metric into evaluate_kv_rag.py; consume pre-rendered frames; emit JSON | AC-6 | coding | task7 |
| task9 | Milestone gate: render baseline+modified on >=2 vendored prompts; report win + non-regression | AC-7 | analyze | task3, task8 |
| task10 | Design study: enumerate/analyze candidate (key, value) representations; recommend a shortlist with rationale (viewpoint invariance, cost, train-free) | AC-3 | analyze | task1 |
| task11 | Add pluggable, decoupled retrieval key/value interface (config switch); >=2 key reps and >=2 value reps incl. a viewpoint-invariant key; default = baseline | AC-3, AC-3.1, AC-4 | coding | task10 |
| task12 | Unit tests per representation: determinism + viewpoint-invariance probe (good keys pass, view-sensitive keys fail) | AC-3, AC-5 | coding | task11 |
| task13 | Compare implemented key/value variants on the offline metric; select with recorded rationale | AC-3.2 | analyze | task11, task8 |

## Claude-Codex Deliberation

### Agreements
- The persistent scene partition must be flag-gated with backward-compatible defaults.
- Per-round verification must remain GPU-free and checkpoint-free; full render is a
  milestone-only gate.
- Injected scene anchors must reuse the existing re-RoPE virtual-frame placement to stay
  in-distribution.
- The retrieval key/value representation is an open design axis and must be selected
  empirically, not assumed; the default representation reproduces today's behavior.

### Resolved Disagreements
- (To be populated by the RLCR loop's Codex review.) This plan was hand-authored from a
  repo exploration rather than produced by `gen-plan`'s Claude-Codex deliberation, so the
  loop's review pass is expected to exercise and refine these design decisions --
  especially the key/value representation choice.

### Convergence Status
- Final Status: `partially_converged` (pending Codex review during the loop and the user
  decisions below).

## Pending User Decisions

- DEC-1: Consistency-metric backbone.
  - Claude Position: DINO features on the shared scene region (robust to framing changes),
    with optional ArcFace for human subjects.
  - Codex Position: (pending)
  - Tradeoff Summary: DINO is viewpoint-robust and general; CLIP is cheaper but more
    semantic; ArcFace is identity-specific but only applies to faces.
  - Decision Status: PENDING

- DEC-2: Scene-memory retention policy.
  - Claude Position: keep shot-0 anchors as a fixed persistent sink (bounded, simple).
  - Codex Position: (pending)
  - Tradeoff Summary: fixed shot-0 sink is simple and cheap but may not represent later
    scene state; a rolling/most-salient policy adapts but adds eviction complexity.
  - Decision Status: PENDING

- DEC-3: Milestone-gate compute budget (how many prompts / seeds / resolution for AC-7).
  - Claude Position: 2 prompts x 1 seed at the existing inference resolution per gate.
  - Codex Position: (pending)
  - Tradeoff Summary: more prompts/seeds give a stronger signal but cost GPU time the
    unattended loop should not spend every round.
  - Decision Status: PENDING

- DEC-4: Retrieval KEY representation.
  - Claude Position: start from the baseline pooled pre-RoPE key, then add a decoupled
    viewpoint-invariant key (DINO/CLIP on the clean anchor frame, or multi-centroid
    pre-RoPE keys) and select on the metric. Prefer train-free.
  - Codex Position: (pending)
  - Tradeoff Summary: baseline is free but view-sensitive; a semantic key is view-robust
    but adds an encoder + decode at store time; a learned key is strongest but needs
    add-on training.
  - Decision Status: PENDING

- DEC-5: Retrieval VALUE representation.
  - Claude Position: keep raw frame-aligned K/V as the faithful default; evaluate
    pooled/centroid anchors and subject-masked K/V to bound injected length and emphasize
    the persistent scene/subject.
  - Codex Position: (pending)
  - Tradeoff Summary: raw K/V is faithful but heavy; pooled/masked/compressed is cheaper
    and more targeted but lossy.
  - Decision Status: PENDING

- DEC-6: Key/value coupling and matching granularity.
  - Claude Position: decouple the key from the value (a semantic key indexing a raw-K/V
    payload) and move from chunk-level toward region/centroid-level matching.
  - Codex Position: (pending)
  - Tradeoff Summary: decoupling + finer granularity should improve cross-view recall but
    adds bookkeeping and store-time cost.
  - Decision Status: PENDING

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- Use descriptive, domain-appropriate naming (e.g. `scene_memory`, `persistent_anchors`,
  `inject_scene_anchors`, `retrieval_key_mode`, `retrieval_value_mode`) rather than plan
  identifiers.
- New behavior is additive and flag-gated; defaults preserve current outputs, including
  the default retrieval key/value representation.
