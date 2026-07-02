# Frozen hard-scene benchmark for long-multishot consistency

Fixed validation set agreed in the 2026-06 group meeting: stop re-running broad
sweeps; freeze a small set of *hard* multi-shot scenes and iterate methods
(prompt-refine, KV selection, multi-pass refine, token-level KV) against it.

Prompt folders: `example/benchmark_hard_multishot/` (versioned in git — the
Round 17 Stage-A subset under `videos/` was gitignored and therefore not
durable). All scenes carry authored `global.json` invariant/contrast captions,
so the invariant guard is never NaN on this set.

## Scene selection

Main scenes were chosen by **baseline anchor-drift (seed 0, 480-frame render)**
from the Round 17 Stage-A drift audit
(`docs/round17_20260618_fast/fast5_admission.json`): high drift = real headroom
for any consistency intervention, avoiding the near-ceiling regime that made
Rounds 13–14 unfalsifiable.

Multi-seed Stage A ran 2026-07-01 (seeds 0-2, 30 renders): **all 9 main
scenes admitted**, negative control sanity-checked. Frozen per-scene numbers
(drift = mean over seeds; win threshold = 2 x cross-seed baseline sigma,
`threshold_source: baseline_seed_sigma`):

| scene | baseline drift (seeds 0-2) | win threshold (2σ) | role |
| --- | --- | --- | --- |
| sandy_beach_driftwood | 0.612 | 0.045 | main |
| bearded_watchmaker_workshop | 0.570 | 0.050 | main, human fine-detail (new) |
| indoor_tender_moment | 0.543 | 0.044 | main |
| aquamarine_underwater | 0.509 | 0.135 | main (noisiest — hard to win here) |
| vintage_archival_scene | 0.500 | 0.097 | main |
| brown_bear_river | 0.478 | 0.073 | main |
| tattooed_noodle_chef | 0.449 | 0.066 | main, human fine-detail (new) |
| warm_indoor_dining | 0.446 | 0.109 | main |
| african_savanna | 0.410 | 0.018 | main (tightest floor — most sensitive) |
| shimmering_puzzle_surface | 0.287 | 0.118 | negative control (`negative_control: true`) |

The two new scenes are authored specifically around **fine human identity
details** (beard/eyebrows/mole/glasses; tattoo/earring/bandana) because the
Round 16/17 GPT-judge wins concentrated on exactly this kind of detail, which
whole-frame embedding metrics are suspected to miss. Both landed squarely in
the hard regime (0.570 / 0.449).

Deliberately excluded: `skateboarder_high_motion`, `sunlit_balcony_tour`,
`cozy_red_room` (drift 0.25/0.20/0.11 — too close to the metric's stable
regime), `frying_egg_*` (superseded as controls by the flagged
`shimmering_puzzle_surface` folder).

## Frozen protocol

- Generator: Wan2.2-TI2V-5B, `configs/inference_kv_rag_long_multishot.yaml`,
  480 frames = 60 blocks x 8, `local_attn_size 32`, shot plan
  `12 12 6 6 6 6 6 6` blocks (per-scene `shot_durations.txt` wins).
- **Seeds 0, 1, 2 for every arm** — no more single-seed verdicts. The paired
  per-scene noise floor comes from cross-seed baseline spread
  (`noise_sigma_multiplier 2.0`); a win must clear the floor, not a fixed
  absolute delta.
- Admission floors: baseline drift >= 0.02, inter-shot diversity >= 0.03.
- Guards unchanged: RAFT motion, diversity, CLIP adherence, invariant contrast
  probe, prompt lint (similarity floor 0.12), negative control must fire.
- Gate metric: `anchor_drift_aggregate_consistency` (drift-sensitive), as in
  Round 17.

## Stage A (multi-seed baselines + admission)

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> python scripts/run_kv_rag_ablation.py \
  --config_path configs/inference_kv_rag_long_multishot.yaml \
  --mode long_multishot \
  --prompts_dir example/benchmark_hard_multishot \
  --blocks_per_shot 6 \
  --baseline_seeds 0,1,2 \
  --strategy_stage A \
  --motion_tolerance 0.2 --diversity_tolerance 0.1 --adherence_tolerance 0.02 \
  --prompt_similarity_floor 0.12 --baseline_drift_floor 0.02 \
  --admission_diversity_floor 0.03 --noise_sigma_multiplier 2.0 \
  --clip_device cuda:0 \
  --min_admitted_main_scenes 7 \
  --output_root videos/benchmark_hard_multishot/stage_A \
  --metrics_json docs/benchmark_hard_multishot/stage_A_drift_audit.json
```

Existing renders are reused per-stem (Round 17 seed-0 renders were preseeded
into `videos/benchmark_hard_multishot/stage_A/baseline_seed0/`, and
`frame_strategy_A` seed-1/2 renders for the three scenes they cover); only
missing scene x seed combinations render.

## Results ledger

- Stage-A multi-seed drift audit: `docs/benchmark_hard_multishot/stage_A_drift_audit.json`
- Every subsequent method round on this benchmark should link its ledger here.
