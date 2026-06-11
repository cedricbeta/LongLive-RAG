# Long-Video Multi-Shot Consistency Plan

Date: 2026-06-11. This file used to describe the cross-video multiview plan.
That plan is frozen after the Round 5 honest null. The active plan is the
long-video multi-shot pivot in `docs/multiview-consistency-research-plan.md`.

## Current Goal

Test one question under one trustworthy gate:

Does KV-RAG scene memory improve cross-shot consistency in one long multi-shot
video, using principled retrieval key/value signals?

No substrate rebuild is in scope. The landed substrate is:

- `global.json` scene/subject/object anchoring in `MultiTextConcatDataset`.
- `--mode long_multishot` in `scripts/run_kv_rag_ablation.py`.
- Wan VAE chunked decode fallback for long videos.

New flags off preserve baseline flow. Multiview remains only as a regression
surface; it is not the mainline research target.

## Metric And Guards

Unit = shot from `shot_durations.txt`.

Primary metric:

- DINO subject anchor per shot.
- CLIP background anchor per shot.
- Mean cosine to the per-scene centroid for each dimension.
- Gate aggregate: `anchor_centroid_consistency =
  mean(subject_anchor_consistency, background_anchor_consistency)`.

Proxy names:

- `palette_agreement`
- `motion_profile_agreement`

Guarded failures:

- Motion freeze: `dynamic_degree` non-regression with relative tolerance, RAFT
  for gate scoring and Farneback only in CPU tests.
- Copied shots: `inter_shot_composition_diversity` non-collapse.
- Prompt ignoring: per-shot CLIP adherence mean and min non-regression.
- Object-state overwrite: invariant-vs-contrast CLIP margin non-regression.
- Prompt trust: CLIP text-similarity matrix for shot captions in the gate JSON;
  below-floor scenes are rejected as ill-posed prompt data.

Any missing scorer or unscorable input fails closed with `winner=null` and
`blocked_reason`.

## Candidate Set

- Heuristic continuity baseline: `subject_identity+raw`.
- Principled key: `attention_native+raw`, Quest-style live-query `q.k` scoring
  against stored per-entry key min/max bounds, per layer/head, with no external
  embedding. Reference: Quest, https://arxiv.org/abs/2406.10774.
- Principled value: `subject_identity+attention_mass`, retaining the frame whose
  tokens received the most post-softmax attention during clean recache, following
  the H2O/SnapKV attention-mass family. References:
  https://arxiv.org/abs/2306.14048 and https://arxiv.org/abs/2404.14469.

Pose-keyed memory is out of scope: WorldMem and VMem rely on camera/world pose
or surfel-indexed view memory, and this pipeline has no pose stream. References:
https://arxiv.org/abs/2504.12369 and
https://openaccess.thecvf.com/content/ICCV2025/papers/Li_VMem_Consistent_Interactive_Video_Scene_Generation_with_Surfel-Indexed_View_Memory_ICCV_2025_paper.pdf.

## Gate Command

Current mechanism-sweep gate:

```bash
CUDA_VISIBLE_DEVICES=3 PYTORCH_ALLOC_CONF=expandable_segments:True \
python scripts/run_kv_rag_ablation.py \
  --config_path configs/inference_kv_rag_long_multishot.yaml \
  --mode long_multishot \
  --mechanism_sweep \
  --prompts_dir example/long_multishot_prompts:example/multiview_prompts \
  --prompt_subset african_savanna,brown_bear_river,sunlit_balcony_tour,skateboarder_high_motion,shimmering_puzzle_surface \
  --baseline_seeds 0,1,2 \
  --admission_diversity_floor 0.03 \
  --noise_sigma_multiplier 2.0 \
  --motion_tolerance 0.2 \
  --diversity_tolerance 0.1 \
  --adherence_tolerance 0.02 \
  --generator_ckpt checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --no_lora_adapter \
  --metrics_json docs/multiview_gate_results/round14_scene_memory_mechanism_sweep.json \
  --output_root videos/round14_scene_memory_mechanism_sweep
```

Round 13 principled key/value gate:

```bash
CUDA_VISIBLE_DEVICES=3 PYTORCH_ALLOC_CONF=expandable_segments:True \
python scripts/run_kv_rag_ablation.py \
  --config_path configs/inference_kv_rag_long_multishot.yaml \
  --mode long_multishot \
  --prompts_dir example/long_multishot_prompts \
  --prompt_subset frying_egg_same_event,sunlit_balcony_tour,skateboarder_high_motion,frying_egg_closeup \
  --finalists subject_identity:raw,attention_native:raw,subject_identity:attention_mass \
  --motion_tolerance 0.2 \
  --diversity_tolerance 0.1 \
  --adherence_tolerance 0.02 \
  --generator_ckpt checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --no_lora_adapter \
  --metrics_json docs/multiview_gate_results/round13_long_multishot_principled_gate.json \
  --output_root videos/round13_long_multishot_principled_gate6
```

Current recorded result:
`docs/multiview_gate_results/round14_scene_memory_mechanism_sweep.json`.
It is a rendered honest null: all scorer backbones loaded, four non-control
scenes passed prompt lint and diversity admission, the negative control was
sanity-checked, and no mechanism arm cleared the 2-sigma centroid noise floor on
any non-control scene (`0/4` wins for all arms). Best arm:
`B_rolling_boundary`, mean delta `-0.002154`.
