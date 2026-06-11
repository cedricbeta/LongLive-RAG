# Research Plan v2 - Long-Video Multi-Shot Consistency with KV-RAG

Date: 2026-06-11. This supersedes the older cross-video multi-view target.
Prompt folders under `example/long_multishot_prompts/<scene>/` are ordered shot
captions for one long video, so the mainline question is cross-shot consistency
inside a single rollout.

## Research Question

Under one trustworthy gate, does KV-RAG scene memory improve cross-shot
consistency in one long multi-shot video, using principled retrieval key/value
signals rather than ad-hoc summaries?

The previous cross-video study remains an honest null: Round 5 found
`subject_identity+raw` leading on consistency but failing the motion guard. The
new target is not to rebuild that substrate. Multiview remains frozen except for
fail-closed bug fixes; `long_multishot` is the mainline.

## Data Protocol

Each scene folder is:

```text
example/long_multishot_prompts/<scene>/
  global.json
  0.json
  1.json
  ...
  shot_durations.txt
```

`global.json` contains scene/subject/object invariants plus invariant and
contrast captions for the object-state probe. Numbered JSON files contain only
local camera/action text. `MultiTextConcatDataset` prepends the global caption to
each shot at render time, preserving scene-cut prefixes and `shot_durations.txt`
chunk boundaries.

Standing gate set:

- `frying_egg_same_event`
- `sunlit_balcony_tour`
- `skateboarder_high_motion`
- `frying_egg_closeup` as `negative_control=true`

The negative control has inverted interpretation: if KV-RAG gains consistency
there without adherence loss, that is text-override evidence, never a pass.

## Metric

Unit = shot, using `shot_durations.txt`.

Primary metric:

- Per-shot subject anchor: DINO image embedding (`facebook/dino-vits16`, same
  backbone family as VBench subject consistency).
- Per-shot background anchor: CLIP image embedding (`ViT-B/32`, same backbone
  family as VBench background consistency).
- Per-scene score: mean cosine from each shot embedding to the scene centroid
  (EntityBench-style centroid form).
- Gate metric: `anchor_centroid_consistency =
  mean(subject_anchor_consistency, background_anchor_consistency)`.

Proxy metrics are labeled as proxies only:

- `palette_agreement`: HSV palette agreement across shots.
- `motion_profile_agreement`: luminance-change histogram agreement across shots.

No proxy uses a VBench dimension name it does not implement.

## Guards

All guards are enforced by the long gate. Any missing/unscorable guard input
produces `winner=null`, `is_null_result=true`, and `blocked_reason`.

- Motion: `dynamic_degree` non-regression with relative tolerance
  `modified >= baseline * (1 - tol)`. Gate backend is torchvision RAFT;
  Farneback is used only for CPU tests.
- Diversity: `inter_shot_composition_diversity` must not collapse.
- Adherence: per-shot CLIP image-text score must preserve both mean and min.
- Invariant probe: per-shot CLIP margin
  `score(invariant_caption) - score(contrast_caption)` must not regress.
- Prompt trust: pairwise CLIP text-similarity matrix of numbered shot captions is
  serialized in the gate JSON; below-floor scenes are rejected as ill-posed data.

## Principled Key/Value Candidates

Baseline finalist retained for continuity:

- `subject_identity+raw`: heuristic leading key from Round 5, not principled.

New flag-gated candidates:

- `attention_native+raw`: Quest-style query-aware key. Stored entries keep
  per-entry min/max key bounds per layer/head; live queries score relevance by
  the attention-native `q.k` upper bound against those key bounds. No external
  embedding is used. Reference: Quest, "Query-Aware Sparsity for Efficient
  Long-Context LLM Inference", ICML 2024 / arXiv 2406.10774,
  https://arxiv.org/abs/2406.10774.
- `subject_identity+attention_mass`: H2O/SnapKV-style value retention. The stored
  payload is the frame whose tokens received the most post-softmax attention
  during clean recache, not the highest key-norm frame. References: H2O,
  https://arxiv.org/abs/2306.14048; SnapKV,
  https://arxiv.org/abs/2404.14469.

Pose-keyed memory is out of scope because this pipeline has no camera-pose
stream. WorldMem and VMem retrieve view memories with world/pose or surfel-index
signals; that mechanism is not applicable here without adding pose estimation.
References: WorldMem, https://arxiv.org/abs/2504.12369; VMem,
https://openaccess.thecvf.com/content/ICCV2025/papers/Li_VMem_Consistent_Interactive_Video_Scene_Generation_with_Surfel-Indexed_View_Memory_ICCV_2025_paper.pdf.

## Gate Command

```bash
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
  --metrics_json docs/multiview_gate_results/round13_long_multishot_principled_blocked.json \
  --output_root videos/round13_long_multishot_principled
```

PASS requires a finalist to win `anchor_centroid_consistency` on at least
`ceil(N/2)` lint-passing non-control scenes, zero guard failures, and a sane
negative control. A committed JSON null with a named blocker or next hypothesis
is a valid result; JSON-less claims are not.
