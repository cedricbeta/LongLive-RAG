# Cross-Video Scene Consistency for LongLive-RAG (Multi-Prompt, Independent Renders)

## Goal Description

For each scene we generate MULTIPLE independent videos -- one per prompt
(`example/multiview_prompts/<scene>/0.json .. N.json` -> `0.mp4 .. N.mp4`). Every prompt is
a different description/viewpoint of the SAME place and subject. The objective: those
*separately generated* videos must read as the same scene -- same environment geometry,
same subject identity, same lighting/style -- not a freshly hallucinated place each time,
while each video still follows its own prompt.

The lever is the KV-cache RAG. A persistent "scene memory" partition is seeded from a
reference perspective (perspective 0's clean anchors), preserved across independent
`pipeline.inference()` calls (`preserve_scene_memory` + `_reset_kv_rag_keep_scene`), and
force-injected into the first chunk of every subsequent perspective through the existing
re-RoPE virtual-frame path. This substrate ALREADY EXISTS and is unit-tested
(`utils/kv_rag.py`, `pipeline/causal_diffusion_inference.py`, `utils/dataset.py`
`MultiViewPerspectiveDataset`); this plan does NOT rebuild it -- it treats it as the
established, flag-gated baseline and a regression guard.

The OPEN scientific question -- the center of this plan -- is: **which retrieval KEY (the
signal a new perspective matches against the scene memory) and retrieval VALUE (the
payload injected back) maximize cross-video consistency**, measured on REAL renders with a
VBench-style metric suite, not a synthetic proxy. A prior round established on the 5B model
that a GPU-free synthetic probe mis-ranked the keys -- plain `pooled` beat `salient_set` on
rendered video (~2.5x the consistency gain) -- so selection here is render-confirmed by
mandate, never proxy-only.

This plan also strengthens the metric itself. Cross-video subject (DINO) and background
(CLIP) consistency are extended with (a) subject-IDENTITY matching (the SAME subject across
videos, not merely a similar-looking one) and (b) additional VBench dimensions adapted to
the cross-video setting, with anti-cheating companions so a "make every video
identical/frozen" collapse cannot score a fake win.

Verification stays cheap per round: GPU-free unit tests + a GPU-free offline screen on
every round. Real per-perspective renders + the VBench suite are reserved for milestone
gates (the 5B checkpoint and GPU are available in this environment).

### Prior-round context (honest baseline this plan builds on)

- The independent-render + seeded-scene-memory protocol and the decoupled key/value
  interface are implemented and tested (rounds 0-2).
- The earlier *within-one-video cross-shot* approach, measured on the 5B model, did NOT
  robustly beat baseline (won on 1/2 and 1/3 prompts; AC-7 failed). The cross-video VBench
  protocol (committed `a4afab4`) is the agreed replacement and is BUILT BUT NEVER MEASURED.
- Lesson carried forward (`BL-20260604-rendered-metric-over-probe`): never promote a
  retrieval representation on a synthetic proxy alone -- confirm on rendered video.

## Acceptance Criteria

Each criterion has positive (expected PASS) and negative (expected FAIL) tests for
deterministic verification.

- AC-1: The independent-render + persistent seeded scene-memory substrate is an explicit,
  flag-gated baseline that is preserved as a regression guard.
  - Positive Tests:
    - With `multiview_per_perspective` on, perspective 0 stores persistent anchors and
      perspectives 1..N retrieve them across SEPARATE `inference()` calls
      (`_reset_kv_rag_keep_scene` keeps the scene partition; per-shot partition resets).
    - With the new flags off, control flow and outputs are byte-identical to baseline.
  - Negative Tests:
    - Flags off -> no persistent partition exists (single-bucket behavior; regression).
    - Storing beyond `scene_memory_max_entries` evicts rather than growing unbounded.

- AC-2: A cross-video VBench-style consistency SUITE scores a scene's per-perspective
  videos, extended with subject-identity matching and additional VBench dimensions.
  - Positive Tests:
    - The suite computes, across a scene's videos: subject consistency (DINO),
      background consistency (CLIP), subject-IDENTITY consistency (ArcFace embedding for
      human subjects; DINO-patch / keypoint identity for non-human subjects, selected per
      scene), and the adapted VBench dims `temporal_style`, `appearance_style`,
      `overall_consistency` (CLIP text-video), `motion_smoothness`. It emits a per-dimension
      JSON breakdown plus an aggregate, baseline vs. modified.
    - The pure cross-video aggregation math is GPU-free unit-tested on synthetic
      embeddings for every dimension and for identity; backbones load only under a
      milestone flag (one-time DINO/CLIP/ArcFace download).
  - Negative Tests (non-vacuous + anti-cheat):
    - A degenerate "all videos identical / frozen" input scores high on raw consistency
      but is flagged by `inter_video_diversity` and `dynamic_degree` (collapse visible).
    - The identity metric scores same-subject synthetic embeddings strictly higher than
      different-subject ones at equal global-feature distance (identity != mere similarity).

- AC-3: The best retrieval key/value is selected by a HYBRID procedure: a GPU-free offline
  screen narrows candidates; finalists are render-confirmed on the AC-2 suite.
  - Positive Tests:
    - An offline, GPU-free screen ranks the implemented (key, value) candidates and is
      LABELED a proxy/pre-filter (not the selector). The candidate set includes the
      existing keys {`pooled`, `moment`, `multi_centroid`, `salient_set`, `positional`
      [view-sensitive control]} plus >= 1 decoupled semantic key (DINO/CLIP on the decoded
      reference anchor frame, or a caption-text embedding) and >= 1 identity-aware key; the
      value set includes {`raw`, `mean_frame`} plus >= 1 compressed/subject-masked value.
    - The top finalists (>= 2) are rendered per-perspective on the 5B model and scored on
      the AC-2 suite; the WINNER is chosen on the rendered metric and is attributable
      (settings recorded in the gate JSON).
  - AC-3.1 (decoupling): key and value are chosen independently; a semantic/identity key
    can index a faithful raw-K/V value. Unknown key/value names fail fast with a clear error.
  - Negative Tests:
    - A candidate promoted on the offline proxy WITHOUT a rendered confirmation is rejected
      (the `rendered-metric-over-probe` guard).
    - Default key/value (current baseline) reproduces baseline retrieval exactly (AC-5 tie).

- AC-4: Measured cross-video win (milestone gate) -- with an HONEST null-result path.
  - Positive Tests:
    - On the evaluated scenes, the selected config raises the AC-2 cross-video consistency
      aggregate over the no-scene-memory baseline on >= ceil(N/2) scenes, WITHOUT regressing
      per-video prompt adherence beyond tolerance or collapsing inter-video diversity. The
      gate JSON records per-scene deltas + the modified settings.
  - Negative Tests (anti-cheating / honesty -- the review MUST enforce these):
    - A consistency gain bought by a prompt-adherence regression or a diversity/motion
      collapse does NOT count as a pass.
    - If no config clears the bar, a FABRICATED or cherry-picked win is the failure mode:
      the acceptable deliverable is a recorded, evidence-backed null/partial result plus a
      stated next hypothesis (e.g. baseline saturation on already-coherent scenes, metric
      choice, or mechanism). "Method does not yet win, here is the evidence and the next
      step" is a valid completion; "method wins" without committed rendered JSON is not.

- AC-5: Backward compatibility + GPU-free per-round verification.
  - Positive Tests:
    - Existing configs (no new flags) produce identical control flow; defaults reproduce
      baseline. The per-round unit suite runs on CPU with no checkpoint and passes.
  - Negative Tests:
    - A new flag set without its required companion value fails fast.
    - A deliberately broken cross-video aggregation (e.g. wrong normalization) is caught by
      the AC-2 unit assertions.

- AC-6: A research ledger records the sweep.
  - Positive Tests:
    - `docs/retrieval-key-value-study.md` lists, per candidate, the offline-proxy rank AND
      the rendered AC-2 breakdown (subject/background/identity/dims/companions), names the
      winner with rationale, and gives the exact one-command reproduction.
  - Negative Tests:
    - A claimed winner with no committed rendered JSON evidence is rejected.

- AC-7: Known-limitation transparency (no silent caps).
  - Positive Tests:
    - The single-GPU per-perspective eval is documented as the standing path; SP-pipeline
      scene-memory mirroring and the DDP scene-contiguity caveat are recorded as explicit
      known limitations.
  - Negative Tests:
    - A coverage bound (scene subset, seed count, finalist count) that is silently dropped
      rather than logged is a defect.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A full cross-video VBench suite (subject/background/identity + temporal_style/
appearance_style/overall_consistency/motion_smoothness + anti-cheat companions), a candidate
key/value matrix that adds a decoupled semantic key (frozen DINO/CLIP on a decoded anchor
frame OR caption-text embedding) and an identity-aware key plus compressed/subject-masked
values, a hybrid offline-screen -> rendered-confirm selection harness, and a milestone gate
executed on the 5B model across a principled (non-cherry-picked) scene subset with the
winner and rationale recorded. A lightweight learned retrieval projection is allowed ONLY as
a small train-free-of-the-base-model add-on and may not become the default if it needs base
retraining.

### Lower Bound (Minimum Acceptable Scope)
Extend the existing cross-video metric with subject-identity matching and at least
`overall_consistency` + one more VBench dim (AC-2); add at least one decoupled semantic OR
identity-aware key beyond the existing set (AC-3); run the hybrid screen and render-confirm
at least the top-2 finalists on >= 2 non-saturated scenes (AC-3, AC-4); record the result
honestly in the ledger (AC-6). The existing pooled/salient_set/multi_centroid keys and
raw/mean_frame values count toward the candidate set. GPU-free unit coverage (AC-5) and the
limitation note (AC-7) are mandatory.

### Allowed Choices
- Can use: the existing KV-RAG infra (`KVRAGMemory`, `KVRAGEntry`, `KVRAGConfig`, the
  decoupled `retrieval_key_mode`/`retrieval_value_mode` switches), the re-RoPE virtual-frame
  injection, `MultiViewPerspectiveDataset` + `preserve_scene_memory`, the ablation/gate
  driver (`scripts/run_kv_rag_ablation.py --mode multiview_vbench`), and the evaluation
  modules (`evaluation/vbench_consistency.py`, `evaluation/video_consistency.py`).
- Can use for the metric: frozen DINO (`facebook/dino-vits16`), CLIP (`ViT-B/32`), an
  ArcFace face-embedding model for human subjects, optical-flow proxies for dynamics --
  pinned and downloaded once, milestone-only; the cross-video aggregation stays NumPy +
  GPU-free unit-tested.
- Can use for the key/value: pooled / moment / multi_centroid / salient_set summaries; a
  decoupled semantic key (DINO/CLIP on a decoded clean anchor frame, or caption-text
  embedding) computed once at store time; identity-aware keys; raw / mean_frame /
  centroid / subject-masked / compressed values (the repo already supports KV compression).
- Cannot use: any change that retrains the 5B base model, an external document/corpus
  retrieval system, breaking existing config flags or the single-shot path, a new MANDATORY
  heavyweight dependency in the per-round test path (per-round tests stay GPU-free and
  checkpoint-free), or selecting a representation on the synthetic proxy alone.

> Design-axis note: the injection MECHANISM (re-RoPE into a virtual frame block before the
> local window; `virtual_frame_start()`) and the `KVRAGEntry` layout are STABLE -- reuse
> them. The retrieval KEY (`summary` + how a query is matched) and the retrieval VALUE
> (what is stored/injected) are the variables under study.

### Relevant References
- `utils/kv_rag.py` -- `KVRAGConfig`/`KVRAGMemory`/`KVRAGEntry`; `add` (store), `retrieve`,
  `_compute_key` (key modes), `_apply_value_mode` (value modes), scene partition,
  `reset_shot`, `_reset_kv_rag_keep_scene`, `virtual_frame_start`. Primary key/value site.
- `pipeline/causal_diffusion_inference.py` -- `preserve_scene_memory`,
  `_reset_kv_rag_keep_scene`, boundary force-inject orchestration.
- `utils/dataset.py` -- `MultiViewPerspectiveDataset` (one sample per perspective; emits
  `scene_name`, `perspective_index`, `is_first_perspective`).
- `inference.py` -- per-prompt independent render loop; `multiview_per_perspective`.
- `evaluation/vbench_consistency.py` -- cross-video subject(DINO)+background(CLIP) +
  anti-cheat companions; EXTEND with identity + extra dims here.
- `evaluation/video_consistency.py` -- `discover_videos`, `pair_cross_perspective_dirs`,
  the `<prefix>-rankN-<scene>-pP-seedS_model` naming, CLIP prompt-adherence.
- `scripts/run_kv_rag_ablation.py` -- baseline-vs-modified render/gate driver with
  `--modified_retrieval_key_mode/_value_mode/...` overrides; `--mode multiview_vbench`.
- `scripts/evaluate_kv_rag.py` -- offline metric entry; `--mode multiview_vbench`.
- `docs/retrieval-key-value-study.md` -- the research ledger (extend).
- `docs/multiview_gate_results/*.json` -- committed rendered evidence.
- `example/multiview_prompts/` -- the standing scene set (11 scenes, 4-8 prompts each).

## Dependencies and Sequence

### Milestones
1. Metric suite (offline, GPU-free math): AC-2, AC-5
   - Phase A: subject-identity matching (ArcFace human / DINO-patch non-human) + synthetic
     unit tests (same vs different subject is non-vacuous).
   - Phase B: additional VBench dims (temporal_style, appearance_style, overall_consistency,
     motion_smoothness) adapted to cross-video aggregation + unit tests; JSON breakdown.
2. Candidate key/value space: AC-3, AC-3.1, AC-5
   - Phase A: design the candidate matrix + the offline-proxy screen as an explicit
     pre-filter (analyze).
   - Phase B: implement the decoupled semantic key + identity-aware key (+ any new value);
     default reproduces baseline; fail-fast; unit tests.
3. Hybrid selection: AC-3
   - Phase A: offline screen ranks all candidates (GPU-free).
   - Phase B: render the finalists per-perspective on the 5B model; score on the AC-2 suite.
4. Milestone gate + ledger: AC-4, AC-6
   - Phase A: execute the gate on a principled scene subset; record per-scene deltas +
     adherence/diversity non-regression; honest null path if no win.
   - Phase B: write the ledger ranking + winner/rationale + reproduction command.
5. Limitation note: AC-7.

Milestone 2 depends on 1 (selection is judged by it). Milestone 3 depends on 1 and 2.
Milestone 4 depends on 3. Milestone 5 is independent documentation.

## Task Breakdown

Each task carries exactly one routing tag: `coding` (Claude) or `analyze` (Codex).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Add subject-IDENTITY cross-video matching (ArcFace human / DINO-patch non-human, per-scene subject selector) to `evaluation/vbench_consistency.py`; GPU-free synthetic unit tests (same > different) | AC-2 | coding | - |
| task2 | Add adapted cross-video VBench dims (temporal_style, appearance_style, overall_consistency, motion_smoothness); per-dim JSON breakdown + aggregate; GPU-free unit tests | AC-2 | coding | - |
| task3 | Design the candidate (key, value) matrix incl. a decoupled semantic key + identity-aware key; specify the offline-proxy screen as a labeled pre-filter (NOT the selector) with its known failure modes | AC-3 | analyze | - |
| task4 | Implement the decoupled semantic key (DINO/CLIP on decoded anchor frame OR caption-text) + identity-aware key (+ any new value rep); default = baseline; fail-fast on unknown names; unit tests | AC-3, AC-3.1, AC-5 | coding | task3 |
| task5 | Hybrid sweep harness: offline screen ranks all candidates; finalists rendered per-perspective via `run_kv_rag_ablation.py --mode multiview_vbench`; settings recorded in gate JSON | AC-3 | coding | task1, task2, task4 |
| task6 | Execute the milestone gate on a principled non-saturated scene subset; record per-scene consistency deltas + adherence/diversity non-regression; honest null path | AC-4 | coding | task5 |
| task7 | GPU-free unit tests across new metric dims + key/value reps + backward-compat regression | AC-5 | coding | task1, task2, task4 |
| task8 | Research ledger: rank candidates on the RENDERED AC-2 metric, name the winner + rationale, add the one-command reproduction; reject proxy-only claims | AC-6 | analyze | task5, task6 |
| task9 | Document SP-pipeline scene-memory mirroring + DDP scene-contiguity as explicit known limitations; single-GPU eval as the standing path | AC-7 | analyze | - |

## Claude-Codex Deliberation

### Agreements
- The independent-render + seeded-scene-memory substrate and the decoupled key/value
  interface are established; this plan studies the key/value choice, it does not rebuild
  the mechanism.
- Selection of any retrieval representation MUST be confirmed on rendered video via the
  AC-2 VBench suite; the GPU-free offline screen is a pre-filter only.
- Per-round verification stays GPU-free and checkpoint-free; renders are milestone-only.
- A null/partial result, recorded honestly with committed rendered evidence, is an
  acceptable completion; a fabricated or proxy-only "win" is not.

### Resolved Disagreements
- (To be populated by the RLCR loop's Codex review.) The prior cross-shot framing failed
  its gate on the 5B model; this plan adopts the cross-video VBench protocol as primary and
  is expected to be exercised/refined by the loop's review pass, especially the candidate
  key/value matrix and the win/null bar in AC-4.

### Convergence Status
- Final Status: `partially_converged` (pending the loop's Codex review).

## Pending User Decisions (resolved)

- DEC-1: Consistency-metric backbones. RESOLVED -> cross-video subject (DINO) + background
  (CLIP) + subject-IDENTITY (ArcFace for humans, DINO-patch/keypoint for non-humans) +
  the adapted VBench dims (temporal_style, appearance_style, overall_consistency,
  motion_smoothness), with the diversity/dynamics/adherence anti-cheats retained.
- DEC-2: Scene-memory retention/seed policy. RESOLVED (default, treated as a sweep axis) ->
  seed persistent anchors from reference perspective 0; bounded `scene_memory_max_entries`;
  salient/rolling retention may be compared as part of the key/value study, not a blocker.
- DEC-3: Experiment compute model. RESOLVED -> HYBRID: a GPU-free offline screen each round
  narrows candidates; only finalists get a real per-perspective render + the AC-2 VBench
  suite at milestone gates.
- DEC-4: Retrieval KEY/VALUE selection. RESOLVED -> empirical, render-confirmed. Sweep the
  existing keys plus a decoupled semantic key and an identity-aware key; values raw/
  mean_frame plus a compressed/subject-masked option; pick the winner on the rendered
  AC-2 metric (the synthetic proxy is a pre-filter, never the selector).
