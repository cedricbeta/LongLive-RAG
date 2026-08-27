# Typed Memory Capsule feasibility pilot

Final outcome: **close linear capsule — channel effective**

G1b still failed against the rank-matched random null, permanently closing the low-rank linear capsule; at least one P=I intervention changed subject appearance, so causal V-channel validity is retained as a future-work pivot.

> This is an exploratory four-scene probe with at most three replay seeds and a single-seed causal intervention. It does not constitute paper-level evidence.

## Protocol v2/v3 and supersession

- The v1 `verdict=kill` is **superseded** and reclassified as `G0-infra-blocked`.
- G0 is a probe-transparency infrastructure gate. A G0 failure can only block and trigger diagnosis; it cannot kill the research hypothesis.
- Under protocol v2, only G1 (no held-out linear identity subspace) or G3 (no causal advantage over random / leakage comparable to full-vector) could produce `verdict=kill`; protocol v3 adds the terminal G1b×channel decision matrix below.
- Decoded-frame SHA-256 equality is the hard transparency check. Round18 metric recomputation is a non-gating sanity check at tolerance `1e-2`.
- Cross-day wall-clock overhead is deleted. A same-GPU adjacent probe-on/off timing pair is optional; shared-host anomalies may be recorded and exempted.
- Protocol v3 preserves the original G1 result but tests its stronger-than-application claim once more with the preregistered within-scene G1b estimator; no further estimator variant is allowed.
- Protocol v3 independently tests causal channel reachability with four P=I full-vector arms. G1b and channel validity are combined only by the frozen decision matrix.

## Fixed-scope integrity

- Frozen prompt/video audit: **PASS**.
- Model edit: clean-recache self-attention V only; K and RoPE untouched; no appended columns.
- Feature flags remain default-off (`capsule_probe.enabled=false`, `capsule_edit.enabled=false`).
- Method transfer and formula: [METHOD_TRANSFER.md](METHOD_TRANSFER.md).
- Frozen SHA-256 ledger: [videos/capsule_pilot_20260715/frozen_inputs_sha256.json](../../videos/capsule_pilot_20260715/frozen_inputs_sha256.json).

## Mask selection and QC

Cross-attention was tested first and rejected because the watchmaker overlay fragmented onto clocks and bench regions. Partial runs were stopped and preserved under `rejected_mask_v0/`; none entered the estimator. The accepted fallback is CLIPSeg, with explicit manual visibility exclusions for known seed1/2 driftwood false positives.

- QC decision: **CLIPSeg_accepted**, 12 scene×seed sheets accepted.
- Audit narrative: [MASK_QC_REPORT.md](../../videos/capsule_pilot_20260715/masks/MASK_QC_REPORT.md).
- Rejected-mask failure frame: [cross_attention_watchmaker_rejection.png](../../videos/capsule_pilot_20260715/rejected_mask_v0/cross_attention_watchmaker_rejection.png).
- Pixel/token map: 704×1280 pixels → 22×40 latent cells; one token cell is 32×32 decoded pixels; flattening is `frame*880 + y*40 + x`.

## G0 v2 — probe transparency

Status: **PASS**. Round18 sanity tolerance: `0.01` (non-gating). Overhead: `exempt`.

| scene | probe-off reference | probe transparency SHA | max per-shot sanity delta | Round18 sanity delta | dump GiB |
|---|---|:---:|---:|---:|---:|
| bearded_watchmaker_workshop | frozen stage_A probe-off artifact | yes | 0 | 9.13e-05 | 0.881 |
| tattooed_noodle_chef | frozen stage_A probe-off artifact | yes | 0 | 0.000177 | 0.757 |
| brown_bear_river | same-code probe-off diagnostic | yes | 0 | 4.32e-05 | 0.947 |
| sandy_beach_driftwood | frozen stage_A probe-off artifact | yes | 0 | 0.000242 | 0.766 |

Detailed G0 report: [G0_REPORT.md](../../videos/capsule_pilot_20260715/g0/G0_REPORT.md).
- Bear mask alignment on the current replay: [brown_bear_current_probe_on_stage_a_mask.png](../../videos/capsule_pilot_20260715/g0/brown_bear_current_probe_on_stage_a_mask.png).
- Superseded v1 machine record: [g0_results.v1-superseded.json](../../videos/capsule_pilot_20260715/g0/g0_results.v1-superseded.json).
- Superseded v1 narrative: [G0_REPORT.v1-superseded.md](../../videos/capsule_pilot_20260715/g0/G0_REPORT.v1-superseded.md).

G0 diagnostic comparisons (人工必看):

- `brown_bear_river`: [brown_bear_river_archive_mismatch.png](../../videos/capsule_pilot_20260715/g0/brown_bear_river_archive_mismatch.png)

G0 visual reviewer: `Codex visual review (model-assisted; not independent human)`.

- `brown_bear_river` (stale_archive): The same-code probe-off and probe-on replays have identical decoded-frame SHA-256 hashes, while both differ from the frozen stage_A archive; this confirms a stale archive rather than probe-induced drift. The stage_A-derived mask over the current probe-on replay still broadly covers the bear in all eight reviewed shots, including the changed pose in shot 1.

## G1 — held-out identity subspace

Verdict: **FAIL — stop P2**. Labels are 12 scene×seed generated identities; same-prompt cross-seed instances are hard negatives. Train shots are (0, 2, 4, 6), held-out shots are (1, 3, 5, 7). LOSO fits the projector on three scenes and requires at least 3/4 held-out scenes to beat complement and random.


Detailed G1 report: [G1_REPORT.md](../../videos/capsule_pilot_20260715/g1/G1_REPORT.md).

## G1b — final within-scene estimator

Verdict: **FAIL — close low-rank linear capsule**. Scene offsets are removed before accumulating seed-instance between-scatter; samples are per-frame entity-token means, and significance is cluster-bootstrapped by scene×seed×shot.

Passing layer×rank candidates: **0**. The gate requires ≥3/4 scenes to beat both the orthogonal complement and rank-matched random p95 with both cluster-bootstrap lower bounds above zero. LOSO is diagnostic only.

Detailed G1b report: [G1B_REPORT.md](../../videos/capsule_pilot_20260715/g1b/G1B_REPORT.md).

This is the final preregistered estimator iteration; no G1c/v4 is permitted.

## Causal V-channel mini-test

Verdict: **channel_effective**. Four seed0 P=I arms cover two scenes × {layer-14-only, all-five-layer} edits at α=1.0.

| arm | reachability | max |DINO subject Δ| | metric effect | human-visible subject change | effective |
|---|:---:|---:|:---:|:---:|:---:|
| bearded_watchmaker_workshop__all5 | yes | 0.0677 | yes | yes | yes |
| bearded_watchmaker_workshop__layer14 | yes | 0.0319 | yes | yes | yes |
| sandy_beach_driftwood__all5 | yes | 0.0934 | yes | yes | yes |
| sandy_beach_driftwood__layer14 | yes | 0.0407 | yes | yes | yes |

Guard/motion/composition changes are causal evidence in this mini-test, not failure gates.
All four α=1.0 arms remained visually coherent; the preregistered α=0.5 collapse fallback was not used.
Signed per-shot DINO direction and selected guard deltas:

- `bearded_watchmaker_workshop__all5` — DINO: shot 5: -0.0043, shot 6: +0.0677, shot 7: -0.0231; guard Δ: dynamic degree +1.9489, composition diversity -0.0019, palette agreement +0.0052.
- `bearded_watchmaker_workshop__layer14` — DINO: shot 5: +0.0037, shot 6: +0.0300, shot 7: -0.0319; guard Δ: dynamic degree +5.8166, composition diversity -0.0050, palette agreement -0.0125.
- `sandy_beach_driftwood__all5` — DINO: shot 3: -0.0934, shot 4: +0.0235, shot 5: +0.0575, shot 6: +0.0011, shot 7: -0.0255; guard Δ: dynamic degree -14.1789, composition diversity -0.0122, palette agreement +0.0643.
- `sandy_beach_driftwood__layer14` — DINO: shot 3: -0.0141, shot 4: -0.0407, shot 5: -0.0087, shot 6: +0.0406, shot 7: +0.0341; guard Δ: dynamic degree -1.4620, composition diversity -0.0037, palette agreement +0.0959.

Visual reviewer: `Codex model-assisted visual review (not an independent human reviewer)`. Arm-level evidence:

- `bearded_watchmaker_workshop__all5`: Shot 7 visibly changes the watchmaker's face, beard shape, head angle, loupe/headgear, and held object; shot 5 also changes watch-face details. The scene remains coherent rather than collapsed. [2-up](../../videos/capsule_pilot_20260715/channel_test/visuals/bearded_watchmaker_workshop__all5_2up.png); [subject closeups](../../videos/capsule_pilot_20260715/channel_test/visuals/bearded_watchmaker_workshop__all5_subject_closeups.png)
- `bearded_watchmaker_workshop__layer14`: Shot 7 visibly removes/changes the head-mounted loupe and changes the watchmaker's facial pose, beard rendering, and action; shot 6 changes hair and apron-strap detail. The scene remains coherent. [2-up](../../videos/capsule_pilot_20260715/channel_test/visuals/bearded_watchmaker_workshop__layer14_2up.png); [subject closeups](../../videos/capsule_pilot_20260715/channel_test/visuals/bearded_watchmaker_workshop__layer14_subject_closeups.png)
- `sandy_beach_driftwood__all5`: Shot 3 visibly changes the fish and driftwood geometry; shots 4-7 substantially change the driftwood/fire subject configuration, and shot 7 changes the person framing. The beach scene remains coherent rather than collapsed. [2-up](../../videos/capsule_pilot_20260715/channel_test/visuals/sandy_beach_driftwood__all5_2up.png); [subject closeups](../../videos/capsule_pilot_20260715/channel_test/visuals/sandy_beach_driftwood__all5_subject_closeups.png)
- `sandy_beach_driftwood__layer14`: Shot 4 changes the fish/driftwood arrangement and red subject detail; shots 5-7 visibly change the driftwood pile, flames, and the person's pose/framing. These are nontrivial but remain coherent. [2-up](../../videos/capsule_pilot_20260715/channel_test/visuals/sandy_beach_driftwood__layer14_2up.png); [subject closeups](../../videos/capsule_pilot_20260715/channel_test/visuals/sandy_beach_driftwood__layer14_subject_closeups.png)

Detailed channel report: [CHANNEL_REPORT.md](../../videos/capsule_pilot_20260715/channel_test/CHANNEL_REPORT.md).
Machine result: [channel_results.json](../../videos/capsule_pilot_20260715/channel_test/channel_results.json); human verdict: [human_verdict.json](../../videos/capsule_pilot_20260715/channel_test/human_verdict.json).

## Final evidence-chain summary

1. **G0 replay transparency survived:** probe-on/off decoded frames match exactly; the bear discrepancy was a stale archive, not probe-induced drift.
2. **G1 killed the stronger universal claim:** no pooled 12-class layer×rank candidate beat complement, random null, and gating LOSO.
3. **G1b also died:** the final within-scene, no-shot-pooling estimator still produced no candidate passing ≥3/4 scenes.
4. **The causal V channel was effective:** the four P=I arms produced at least one identity-metric or visible subject change.

The decision matrix is terminal. No additional estimator variant or v4 is allowed.

## Terminal decision-matrix application

| causal channel | G1b pass | G1b fails against random |
|---|---|---|
| effective | enter full three-arm P2 | permanently close low-rank linear capsule; retain channel validity and end pilot |
| ineffective | close capsule (channel overrides estimator) | close capsule; double-negative confirmation |

Observed cell: **channel effective × G1b fail** → **close linear capsule — channel effective**.

## Operations and incident controls

- Main and channel orchestrators hold the same non-blocking `logs/orchestrator.lock`; resume scripts probe that lock before starting.
- Queue claim files are protected by `flock`, and a nonzero config is archived with a `.failed.<timestamp>` suffix before exactly one retry; the second failure is fatal.
- Long-render admission uses `memory.free >= 61440 MiB` (60 GiB).
- Long renders run in tmux, write per-config logs/exit codes, and retain frozen-input SHA auditing.
- Exercised retry evidence: [channel__bearded_watchmaker_workshop__all5.exit_code.failed.20260716T120345](../../videos/capsule_pilot_20260715/logs/channel__bearded_watchmaker_workshop__all5.exit_code.failed.20260716T120345); each corresponding log/runtime/render failure trace was archived before the sole retry.

## Future work (outside this pilot)

The low-rank linear capsule is permanently closed; this is not an invitation to try another estimator variant. The retained causal-channel result supports only two distinct follow-up families: (1) a nonlinear or learned capsule, evaluated as a new study with its own controls, or (2) whole-vector prototype steering that explicitly accepts the composition/motion leakage measured here. Neither path reopens this pilot or creates a v4.

## Reproduction ledger

```bash
python scripts/prepare_capsule_clipseg_masks.py --seeds 0 1 2
python scripts/build_capsule_mask_qc.py --mask_source precomputed --seeds 0 1 2 --human_qc accepted
python scripts/build_capsule_pilot_configs.py
# tmux: scripts/run_capsule_probe_replays.sh GPU CONFIG...
python scripts/build_capsule_g0_bear_diagnostic.py
# tmux: scripts/run_capsule_g0_bear_diagnostic.sh GPU
python scripts/audit_capsule_probe_replays.py
python scripts/evaluate_capsule_g0.py
# If G0 is blocked, diagnose/fix/substitute before continuing.
# Only if G0 passes:
python scripts/estimate_capsule_subspace.py --random_trials 100 --bootstrap_repeats 1000
python scripts/build_capsule_g1_visuals.py
# Protocol v3 final dual track (G1 remains preserved):
python scripts/estimate_capsule_subspace_g1b.py --random_trials 100 --bootstrap_repeats 1000
python scripts/build_capsule_channel_test.py
# tmux: scripts/orchestrate_capsule_channel_test.sh
python scripts/evaluate_capsule_channel_test.py --device cuda:0
# Fill channel_test/human_verdict.json after viewing every 2-up and subject-closeup sheet
python scripts/finalize_capsule_channel_test.py
# Only if G1b and the causal channel both pass:
python scripts/calibrate_capsule_alpha.py
# tmux: scripts/run_capsule_alpha_scan.sh GPU
python scripts/evaluate_capsule_alpha_scan.py --device cuda:0
# Fill p2/alpha_scan/visual_verdict.json after viewing the contact sheet and 4-up
python scripts/finalize_capsule_alpha_scan.py
python scripts/build_capsule_p2_configs.py
python scripts/promote_capsule_alpha_render.py
python scripts/list_capsule_p2_remaining_configs.py
# tmux: scripts/run_capsule_p2.sh GPU REMAINING_CONFIG...
python scripts/audit_capsule_p2_runs.py
python scripts/evaluate_capsule_p2.py
python scripts/build_capsule_p3_artifacts.py
python scripts/build_capsule_pilot_report.py
```

- Probe config/command manifest: [videos/capsule_pilot_20260715/probe_run_manifest.json](../../videos/capsule_pilot_20260715/probe_run_manifest.json).
- Probe audit: [videos/capsule_pilot_20260715/probe_replay_audit.json](../../videos/capsule_pilot_20260715/probe_replay_audit.json).
- Bear G0 diagnostic manifest: [videos/capsule_pilot_20260715/g0/bear_diagnostic_manifest.json](../../videos/capsule_pilot_20260715/g0/bear_diagnostic_manifest.json).
- G0 visual review: [videos/capsule_pilot_20260715/g0/visual_review.json](../../videos/capsule_pilot_20260715/g0/visual_review.json).
- Evaluation environment: [environment_versions.txt](environment_versions.txt).
- G1b final estimator: [videos/capsule_pilot_20260715/g1b/g1b_results.json](../../videos/capsule_pilot_20260715/g1b/g1b_results.json).
- Channel-test manifest/results/human review: [manifest](../../videos/capsule_pilot_20260715/channel_test/channel_run_manifest.json), [results](../../videos/capsule_pilot_20260715/channel_test/channel_results.json), [human verdict](../../videos/capsule_pilot_20260715/channel_test/human_verdict.json).
