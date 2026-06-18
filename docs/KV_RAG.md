# KV-RAG Temporal Consistency

This repository now has an opt-in KV-cache retrieval path for Wan2.2-TI2V-5B causal inference. Baseline inference is unchanged unless `inference.kv_rag.enabled: true`.

## Where It Hooks In

- Generation loop: `pipeline/causal_diffusion_inference.py` generates latent chunks, denoises each chunk over scheduler timesteps, writes the chunk to `output`, then reruns the generator at timestep 0 to recache clean context.
- KV-cache flow: `wan_5b/modules/causal_model.py` projects Q/K/V in `CausalWanSelfAttention`, inserts the new K/V into the per-layer rolling cache, builds the local/sink attention window, and returns cache-update metadata to `CausalWanModel._apply_cache_updates`.
- Frame/segment conditioning: `MultiTextConcatDataset` returns one prompt per latent chunk. The pipeline picks `conditional_dict_list[chunk_index]`, resets cross-attention cache per chunk, and uses scene-cut prompt prefixes to drive multi-shot sink/RoPE behavior.

## Method

KV-RAG stores compact historical K/V slices from clean recache calls. During later denoising calls, each enabled layer summarizes the current query, retrieves the top-k most similar historical entries from the same layer, and prepends those K/V tokens to the normal sliding/sink attention window.

The default store point is clean timestep-0 recache, not noisy denoising steps. This keeps the memory bank focused on stable temporal context. Retrieval during recache is configurable but disabled by default.

### Position handling (re-RoPE)

The active path is absolute-RoPE inference. Naively reusing the cached keys would prepend slices that still carry their *original* absolute-RoPE phase, so the query-to-memory relative position would be an arbitrarily large offset the model never saw in training (the local window is bounded). To avoid that, KV-RAG stores **pre-RoPE** keys and, at injection time, re-RoPEs them into a virtual frame block placed immediately before the local window (`reinject_rope: true`, the default). The query-to-memory relative positions then stay in the same range the model already handles for its local window. This requires frame-aligned storage (`frame_aligned_store: true`) so a slice can be re-RoPE'd as a `(frames, h, w)` grid. Setting both flags `false` reproduces the legacy stale-absolute-RoPE injection.

### Retrieval signal

Retrieval scores the current query against stored keys on their **pre-RoPE** content (`summary_prerope: true`) so the same content at a different time still matches, and keeps per-head structure when scoring (`summary_per_head: true`) instead of flattening all heads into one vector. Scoring is vectorized across candidates in a single batched op.

Relative-RoPE mode is still skipped by KV-RAG (retrieval returns nothing). Ulysses sequence-parallel inference also disables KV-RAG for now.

## How KV-RAG Works End To End

Despite the name, this is intra-video KV memory with content-based retrieval, not retrieval from an external corpus. The memory bank is cleared at the start of every `pipeline.inference(...)` call, so each generated video is independent.

The implementation is split across three files:

- `utils/kv_rag.py`: `KVRAGConfig`, `KVRAGEntry`, and `KVRAGMemory`.
- `pipeline/causal_diffusion_inference.py`: reset/store/retrieve scheduling and separate CFG memories.
- `wan_5b/modules/causal_model.py`: per-layer attention integration and re-RoPE injection.

### 1. Generation loop

Each video is generated chunk by chunk. For a chunk, retrieval happens during normal noisy denoising calls. Storage happens once afterward, during the clean timestep-0 recache pass.

```text
pipeline.inference(...)                         [pipeline/causal_diffusion_inference.py]
|
+-- _reset_kv_rag()       clear memory for this video
|
+-- for each latent chunk c = 0, 1, 2, ...:
    |
    |   DENOISE
    |   for t in scheduler_timesteps:
    |       generator.forward(chunk_c, t, kv_rag_retrieve=True)
    |       scheduler.step(...) updates the latent chunk
    |
    +-- write the completed chunk into the output latent video
    |
    |   CLEAN RECACHE
    |   generator.forward(chunk_c, t=0, kv_rag_store=True)
    |       refresh normal causal KV cache
    |       append clean chunk features to KV-RAG memory
    |
    +-- optional verbose stats, including rag_attn_mass
```

There is one memory bank per CFG branch:

```text
kv_rag_pos   conditional branch
kv_rag_neg   unconditional branch, only when guidance is active
```

The key design choice is that later noisy chunks retrieve from earlier clean recache states. That keeps the memory source more stable than storing transient noisy denoising states.

### 2. Store path

Inside each enabled self-attention layer, the model has both pre-RoPE and post-RoPE keys. Normal LongLive caching still uses the post-RoPE keys. KV-RAG stores the pre-RoPE keys by default so they can be matched and re-positioned later.

```text
CausalWanSelfAttention.forward(...)                  [wan_5b/modules/causal_model.py]

q, k, v = projections(x)
|  |
|  +-- k is pre-RoPE content key
|
+-- roped_query = RoPE(q, current positions)
+-- roped_key   = RoPE(k, current positions)

cache_update_info = {
    "new_k": roped_key,        # normal causal cache
    "new_v": v,
    "rag_k_prerope": k,        # KV-RAG memory
    "frame_seqlen": ...,
    "grid_h": ...,
    "grid_w": ...,
    "num_new_frames": ...,
}

CausalWanModel._apply_cache_updates(...)
    -> KVRAGMemory.add(k_pre=k, k_post=roped_key, v=v, geometry=...)
```

`KVRAGMemory.add(...)` then builds a compact per-layer entry:

```text
stored key source    pre-RoPE K, when reinject_rope=True
summary source       pre-RoPE K, when summary_prerope=True
token selection      whole-frame downsample, when frame_aligned_store=True
summary shape        normalized per-head mean, [B, H, D]
entry payload        K, V, summary, frame geometry, token range, chunk metadata
```

There are two different "keys" in the retrieval system:

```text
entry.k        the actual attention key tokens returned by retrieval
entry.summary  the compact key used only for ranking candidate entries
```

With the default config, `entry.k` is the selected **pre-RoPE** key slice from the clean recache pass. It is not the post-RoPE key used by the normal causal KV cache. The post-RoPE key is still passed into `KVRAGMemory.add(...)` as `k_post`, but it is only used for the legacy path when `reinject_rope: false`.

The stored value tensor is always the matching `v` slice. It is never used for retrieval scoring; it is carried along with the selected key tokens and used only when the attention softmax reads from the retrieved memory.

The memory is a FIFO list per enabled transformer layer:

```text
entries_by_layer = {
    0:  [entry(chunk0), entry(chunk1), ...],
    7:  [entry(chunk0), entry(chunk1), ...],
    14: [entry(chunk0), entry(chunk1), ...],
}
```

### 3. Retrieve and inject path

During denoising of a later chunk, each enabled layer retrieves from its own layer's FIFO memory.

The retrieval query is also a compact summary, not the full attention matrix. With the default config, the current layer uses the **pre-RoPE** query `q`:

```text
query_summary = normalize(mean over tokens of q)
```

For the common `[B, T, H, D]` tensor layout, summaries keep the head axis:

```text
q or k_pre      [B, T, H, D]
mean over T     [B, H, D]
normalize D     [B, H, D]
```

That means matching compares each head in its own subspace instead of flattening all heads into one vector.

```text
CausalWanSelfAttention.forward(...) during denoise

1. Match on pre-RoPE content

   query_summary = normalize(mean_tokens(q))
   # q is the pre-RoPE query when summary_prerope=True

2. Score candidates

   candidates = entries from the same layer
   score = cosine(query_summary, entry.summary)
   selected = top_k candidates after min_frame_gap filtering

3. Gather memory tokens

   rag_k = concat(selected.k)    # pre-RoPE K by default
   rag_v = concat(selected.v)

4. Re-RoPE into a virtual block before the local window

   rag_frames = rag_k_tokens / frame_seqlen
   rag_start_frame = local_window_start_frame - rag_frames
   roped_rag_k = causal_rope_apply(rag_k, start_frame=rag_start_frame, ...)

5. Prepend to the existing attention window

   window_k = concat([roped_rag_k, window_k], dim=1)
   window_v = concat([rag_v,       window_v], dim=1)
   output = attention(roped_query, window_k, window_v)
```

Candidate filtering and scoring are:

```text
min_end = current_start - min_frame_gap * frame_seqlen
candidates = entries where entry.end_token <= min_end

for each candidate i:
    score_i = mean_over_batch_and_heads(
        sum_over_head_dim(query_summary * entry_i.summary)
    )

selected = highest score_i entries, up to top_k
```

So retrieval is content-based within the same transformer layer, with a temporal safety filter to avoid retrieving the current or too-recent chunk. The selected entries return full K/V token slices:

```text
rag_k = concat(selected entry.k along token dimension)
rag_v = concat(selected entry.v along token dimension)
```

The retrieved history goes through the same softmax as the normal sink and sliding-window cache. KV-RAG adds historical context; it does not replace the LongLive causal cache, attention sink, multi-shot sink, or clean recache behavior.

### 4. Why re-RoPE matters

If a stored key keeps its old absolute-RoPE phase, a current query may see it as hundreds of frames away. That relative offset is outside the bounded local window the model normally uses, so the retrieved block can receive little useful attention.

```text
Legacy stale-RoPE injection

past key keeps original frame 200
current query is near frame 1000
relative distance is about 800 frames
=> out-of-distribution for a local-window attention path
```

The default path stores pre-RoPE keys and assigns them fresh virtual positions immediately before the current local window:

```text
Re-RoPE injection

virtual rag block:  frames 965..979
local window:       frames 980..1000
current query:      near frame 1000
relative distance:  local-window scale
=> retrieved memory can compete in the normal attention softmax
```

This is why `reinject_rope: true` and `frame_aligned_store: true` are paired. Re-RoPE needs whole-frame geometry so the retrieved slice can be treated as a valid `(frames, h, w)` token grid. Setting both flags to `false` gives the legacy stale-RoPE behavior for ablation.

### 5. Diagnostics and limitations

When `verbose: true`, stats include `rag_attn_mass`, a sampled estimate of the post-softmax attention mass landing on retrieved tokens. Compare it with the uniform baseline:

```text
uniform_baseline = rag_tokens / total_attention_tokens
```

Interpretation:

- near zero or well below uniform: retrieved tokens are effectively ignored.
- near uniform: retrieved tokens are present, but not clearly preferred over the rest of the window.
- consistently above uniform: the model is allocating extra attention to retrieved history.

`rag_attn_mass` is a mechanism diagnostic, not a quality metric. A higher value only says the retrieved block is influencing attention; use the evaluation script to check whether that influence improves identity or temporal consistency.

Current limitations:

- Retrieval is recomputed during denoising calls, so selected entries can change across timesteps.
- Relative-RoPE mode and Ulysses sequence-parallel inference still bypass KV-RAG.
- The bank is self-retrieval within one video. External reference-bank retrieval would be a separate extension.

When disabled, `_kv_rag_call_kwargs(...)` returns an empty dict and no RAG objects are passed into the model. The baseline inference path therefore keeps the same cache updates and attention window construction.

## Config

Use `configs/inference_kv_rag.yaml` or add this block to an existing inference yaml:

```yaml
inference:
  kv_rag:
    enabled: true
    layers: [0, 7, 14, 21, 29]
    top_k: 2
    max_entries: 32
    max_frames_per_entry: 1
    # Legacy compatibility alias. Used only when max_frames_per_entry is absent.
    max_tokens_per_entry: 1024
    min_frame_gap: 0
    retrieve_during_denoise: true
    retrieve_during_recache: false
    store_after_recache: true
    token_policy: uniform
    similarity: cosine
    store_on_cpu: false
    verbose: true
    summary_prerope: true       # match on pre-RoPE content (position-invariant)
    summary_per_head: true      # keep per-head structure when scoring
    reinject_rope: true         # re-RoPE retrieved keys before the local window
    frame_aligned_store: true   # store whole frames so re-RoPE is well-defined
    require_frame_aligned: true # gate mode: count/drop non-frame-aligned payloads
    persistent_logit_bias_lambda: 0.0 # verdict lever; default keeps normal attention
```

Key knobs:

- `layers`: layer indices to store/retrieve. Use `"all"` or omit for every layer, but memory use increases quickly.
- `top_k`: number of historical entries retrieved per layer call.
- `max_frames_per_entry`: frame cap per stored chunk/layer. `0` stores full chunks. This is the frame-level contract used by long-regime gates.
- `max_tokens_per_entry`: legacy token cap, kept for old configs. With `frame_aligned_store`, it is rounded down to whole frames. It is ignored when `max_frames_per_entry` is set.
- `max_entries`: per-layer FIFO entry limit.
- `min_frame_gap`: excludes entries whose end frame is too close to the current chunk.
- `store_on_cpu`: saves GPU memory at the cost of host-to-device copies during retrieval.
- `summary_prerope` / `summary_per_head`: retrieval-signal quality. Defaults on.
- `reinject_rope` / `frame_aligned_store`: in-distribution injection. Defaults on; set both `false` for the legacy stale-RoPE path (useful for an A/B ablation).
- `require_frame_aligned`: gate mode. Non-frame-aligned store/inject payloads are counted in diagnostics and dropped so the rendered gate can fail closed instead of silently using a token-subset fallback.
- `persistent_logit_bias_lambda`: disabled by default. The post-verdict lever sets this to `1` or `2` to add that value to attention logits for injected persistent whole-frame columns only; diagnostics count manipulated frames and re-RoPE frame injections.

## Running An Ablation

Generate baseline and KV-RAG outputs with matched seeds:

```bash
python scripts/run_kv_rag_ablation.py \
  --config_path configs/inference.yaml \
  --output_root videos/kv_rag_ablation
```

Evaluate existing output folders only:

```bash
python scripts/evaluate_kv_rag.py \
  --baseline_dir videos/kv_rag_ablation/baseline \
  --kv_rag_dir videos/kv_rag_ablation/kv_rag \
  --output_json videos/kv_rag_ablation/kv_rag_metrics.json
```

Metrics include adjacent-frame L1/PSNR/SSIM, optical-flow warp L1/SSIM, frame-feature adjacent cosine, frame-feature-to-first cosine, and appearance-to-first SSIM.

## Three-Stage Validation Prompts

Use `example/kv_rag_3stage_prompts` for identity/appearance stress tests. Each sample has three stages:

1. Subject or object faces the camera.
2. The same subject or object turns away, moves to profile, or is viewed from the rear.
3. The same subject or object returns to the original recognizable view.

The included samples are human-only: a man turning around and back, a woman turning to profile and back, a chef turning toward a stove and back, and a skateboarder turning away and back. Each stage repeats stable identity anchors such as hair, face details, clothing, accessories, distinctive marks, colors, and props.

For smooth-turn validation, this config disables the scene-cut prefix:

```yaml
inference:
  scene_cut_prefix: ""
  multi_shot_sink: false
  multi_shot_rope_offset: 0
```

This matters because `The scene transitions.` tells the dataset/model that a new shot starts at stage boundaries. That is useful for multi-shot testing, but it encourages visible cuts. The human validation captions now describe one continuous uncut shot across all three stage captions.

Run the three-stage set with:

```bash
python scripts/run_kv_rag_ablation.py \
  --config_path configs/inference_kv_rag_3stage.yaml \
  --output_root videos/kv_rag_3stage_ablation \
  --generator_ckpt checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --no_lora_adapter
```

The config uses 144 latent frames with `num_frame_per_block: 8`, producing 18 chunks. Each example has `shot_durations.txt` set to `6 6 6`, so the three stages receive equal temporal budget.

Output videos use readable names from the caption folder plus the ablation variant, for example:

```text
videos/kv_rag_3stage_ablation/baseline/baseline-rank0-000_man_turnaround-seed0_regular.mp4
videos/kv_rag_3stage_ablation/kv_rag/kv_rag-rank0-000_man_turnaround-seed0_regular.mp4
```

If you use an unmerged LoRA setup, keep the `adapter` block in your config and pass:

```bash
python scripts/run_kv_rag_ablation.py \
  --config_path configs/inference_kv_rag_3stage.yaml \
  --output_root videos/kv_rag_3stage_ablation \
  --generator_ckpt /path/to/base_or_generator_checkpoint.pt \
  --lora_ckpt /path/to/lora_checkpoint.pt
```
