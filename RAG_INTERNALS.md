# RAG-KV Internals: How Caching, Eviction, and Retrieval Work

## Table of Contents
1. [LongLive Baseline: KV Cache Without RAG](#1-longlive-baseline-kv-cache-without-rag)
2. [RAG Layer: The Retrieval Bank](#2-rag-layer-the-retrieval-bank)
3. [End-to-End Flow: 3-Segment Prompt Switching](#3-end-to-end-flow-3-segment-prompt-switching)
4. [Attention Assembly With RAG](#4-attention-assembly-with-rag)
5. [Diversity-Aware Eviction in the Bank](#5-diversity-aware-eviction-in-the-bank)
6. [30 Banks vs 1 Bank](#6-30-banks-vs-1-bank)

---

## 1. LongLive Baseline: KV Cache Without RAG

### 1.1 Architecture Constants

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_transformer_blocks` | 30 | Number of self-attention layers |
| `num_heads` | 12 | Attention heads per layer |
| `head_dim` | 128 | Dimension per head |
| `frame_seq_length` | 1560 | Tokens per latent frame (spatial resolution) |
| `num_frame_per_block` | 3 | Frames generated per denoising block |
| `local_attn_size` | 12 | Max frames in local attention window |
| `sink_size` | 3 | Permanently anchored initial frames |
| `denoising_steps` | [1000, 750, 500, 250] | Noise levels for 4-step distillation |

### 1.2 KV Cache Structure (Per Layer)

Each of the 30 transformer layers has its own KV cache:

```
kv_cache[layer_idx] = {
    "k": Tensor [B, cache_size, 12, 128],    # cached keys
    "v": Tensor [B, cache_size, 12, 128],    # cached values
    "global_end_index": int,                  # total tokens processed so far
    "local_end_index": int,                   # write cursor within cache buffer
}
```

Where `cache_size = local_attn_size × frame_seq_length = 12 × 1560 = 18720 tokens`.

The cache buffer is a fixed-size circular buffer. The two indices track:
- **`global_end_index`**: How many total tokens have passed through (monotonically increasing). Used to compute position embeddings and detect recomputation.
- **`local_end_index`**: Where the next write goes within the physical buffer. Wraps around via rolling.

### 1.3 Cache Layout

The cache is partitioned into two regions:

```
┌──────────────────────────────────────────────────────────┐
│  SINK (frames 0,1,2)  │       LOCAL WINDOW              │
│  3 × 1560 = 4680 tok  │  (up to 9 frames × 1560 tok)   │
│  NEVER evicted         │  FIFO: oldest evicted first     │
└──────────────────────────────────────────────────────────┘
  positions [0..4679]      positions [4680..18719]
```

- **Sink region**: The first `sink_size` frames (frames 0, 1, 2). These are written once during the first block and then **permanently protected** — they provide global long-range context anchoring.
- **Local window**: A sliding window of the most recent frames. When full, the oldest frame is evicted (shifted left) to make room for new frames.

### 1.4 What Happens Each Block (3 Frames)

Each block of 3 frames goes through **5 forward passes** of the entire 30-layer transformer:

```
Block for frames [F, F+1, F+2]:
  Pass 1: Denoise at t=1000 (heaviest noise)
  Pass 2: Denoise at t=750
  Pass 3: Denoise at t=500
  Pass 4: Denoise at t=250  → final denoised output
  Pass 5: Clean context update (t≈0) → updates KV cache for next block
```

Passes 1–3 are **intermediate** — they progressively denoise the 3 frames. Each pass runs Q/K/V through all 30 layers, attending to the existing cache.

Pass 4 produces the final clean frames.

Pass 5 re-runs the clean frames through the model at near-zero noise. This is the **authoritative cache update** — the KV entries written here represent the "true" content of those frames and will be used by all future blocks.

### 1.5 Eviction: The `roll_and_insert` Operation

When the local window is full and new tokens arrive, eviction happens in `CausalWanSelfAttention.forward()`:

```python
# Triggered when: local_end_index + num_new_tokens > cache_size
num_evicted_tokens = num_new_tokens + local_end_index - cache_size
num_rolled_tokens  = local_end_index - num_evicted_tokens - sink_tokens
```

Concretely, with 3 new frames (4680 tokens) arriving:

**Before (cache full, 12 frames):**
```
[sink: F0,F1,F2] [F28,F29,F30,F31,F32,F33,F34,F35,F36,F37,F38,F39]
                   ↑ oldest                                    newest ↑
```

**Eviction + Roll + Insert:**
```
Step 1: Identify evicted tokens = F28,F29,F30 (oldest 3 frames = 4680 tokens)
Step 2: Roll remaining left:  [F31..F39] shifts to start of local region
Step 3: Insert new frames:    [F40,F41,F42] at the end
```

**After:**
```
[sink: F0,F1,F2] [F31,F32,F33,F34,F35,F36,F37,F38,F39,F40,F41,F42]
```

Frames F28, F29, F30 are **gone from the cache forever** — unless RAG captures them.

### 1.6 The `is_recompute` Flag

```python
is_recompute = (current_end <= global_end_index) and (current_start > 0)
```

This is `True` when the model is re-processing frames it has already seen (e.g., during the denoising loop where the same 3 frames are run 4 times, or during recache after a prompt switch). When `is_recompute=True`:
- Cache indices (`global_end_index`, `local_end_index`) are **not updated**
- RAG storage is **skipped** (to avoid storing noisy intermediate versions)
- Sink region is **write-protected**

Only the final clean context update (Pass 5) and the very first processing of new frames have `is_recompute=False`, which is when cache indices advance and evictions are committed.

---

## 2. RAG Layer: The Retrieval Bank

### 2.1 One Bank Per Layer

We create 30 `KVRetrievalBank` instances, one per transformer layer:

```python
self.retrieval_banks = [
    KVRetrievalBank(max_frames=256, frame_seq_length=1560)
    for _ in range(30)
]
```

### 2.2 Bank Storage Format

Each bank stores a list of frames. Per frame:

```
keys:      [B, 1560, 12, 128]    # the K vectors for all 1560 spatial tokens
values:    [B, 1560, 12, 128]    # the V vectors
embedding: [B, 1536]             # retrieval key: mean-pooled K, flattened (12×128)
frame_id:  int                   # global frame index in the video
```

The **embedding** is computed as:
```python
emb = k_frame.mean(dim=1).flatten(start_dim=1)  # [B, 1560, 12, 128] → [B, 1536]
```

This collapses the 1560 spatial tokens into a single vector by averaging, then flattens the head dimensions. This 1536-dim vector serves as the "fingerprint" for similarity-based retrieval.

### 2.3 Storage: `store_evicted_frames()`

Called from `causal_model.py` right before the FIFO roll, when `retrieval_bank is not None` and `num_evicted_tokens > 0` and `not is_recompute`:

```python
# In CausalWanSelfAttention.forward(), line 250-257:
evicted_k = kv_cache["k"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
evicted_v = kv_cache["v"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
evicted_start_frame = (global_end_index - local_end_index + sink_tokens) // frame_seqlen
retrieval_bank.store_evicted_frames(evicted_k, evicted_v, evicted_start_frame)
```

The bank then splits the evicted chunk into individual frames:
```python
for i in range(num_frames):
    k_frame = evicted_k[:, i*1560 : (i+1)*1560]  # one frame
    emb = k_frame.mean(dim=1).flatten(start_dim=1)  # retrieval embedding
    self.keys.append(k_frame)
    self.values.append(v_frame)
    self.embeddings.append(emb)
    self.frame_ids.append(start_frame_id + i)
```

### 2.4 Retrieval: `retrieve()`

Called from `causal_model.py` before attention computation, when `retrieval_bank is not None` and `retrieval_top_k > 0` and `len(bank) > 0` and `not is_recompute`:

**Step 1 — Query embedding:**
```python
query_emb = current_query.mean(dim=1).flatten(start_dim=1)  # [B, 1536]
```
Same mean-pooling as storage, applied to the current block's query vectors.

**Step 2 — Exclude frames already in cache:**
```python
exclude = sink_frame_ids | local_window_frame_ids
# e.g., {0,1,2} | {31,32,...,42} — no point retrieving what's already in attention
```

**Step 3 — Cosine similarity:**
```python
sim = cosine_similarity(query_emb, bank_embeddings)  # [B, num_bank_frames]
top_k_indices = sim.topk(k=6)  # select 6 most similar
```

**Step 4 — Return concatenated KVs:**
```python
retrieved_k = cat([bank.keys[i] for i in top_k_indices])  # [B, 6×1560, 12, 128]
retrieved_v = cat([bank.values[i] for i in top_k_indices]) # [B, 6×1560, 12, 128]
```

Retrieved frames are sorted by `frame_id` to preserve temporal order.

---

## 3. End-to-End Flow: 3-Segment Prompt Switching

Config: `switch_frame_indices: 40, 80`, `num_output_frames: 120`

```
Segment 0 (frames 0–39):   "woman facing camera at lake"
Segment 1 (frames 40–79):  "woman turns away, looking at lake"
Segment 2 (frames 80–119): "woman turns back to face camera"
```

### 3.1 Segment 0 — Frames 0 to 39

**RAG state**: `rag_active=False`, `effective_top_k=0`
→ Banks receive evicted frames (storage ON), but no retrieval happens.

```
Block 0 (frames 0,1,2):
  5 passes through model. Cache fills: sink=[0,1,2], local=[0,1,2]
  No eviction (cache not full yet). Bank: empty.

Block 1 (frames 3,4,5):
  Cache: sink=[0,1,2], local=[0,1,2,3,4,5]. Bank: empty.

...

Block 4 (frames 12,13,14):
  Cache full: sink=[0,1,2] + local=[3..14] = 15 frames capacity reached.
  Still fits (sink 3 + local 12 = 15, cache holds 12 local). Bank: empty.

  Wait — actually local_attn_size=12 means the local region holds 12 frames.
  Sink has 3 frames. Cache buffer = 15 frames × 1560 = 23400? No:
  cache_size = local_attn_size × frame_seq_length = 12 × 1560 = 18720 tokens.
  Sink occupies 3 × 1560 = 4680 of those 18720.
  Remaining for local window: 18720 - 4680 = 14040 tokens = 9 frames.

  So at block 4 (frame 12), we've written 13 frames of KVs (0..12) into 
  an 18720-token buffer. Sink uses 4680, local has 13*1560 - 4680 = 15600.
  15600 > 14040 → eviction starts at block 4!

Block 4 (frames 12,13,14):
  Evicts frames 3,4,5 → stored in bank. Bank: [3,4,5]

Block 5 (frames 15,16,17):
  Evicts frames 6,7,8 → bank. Bank: [3,4,5,6,7,8]

...

Block 13 (frames 39):
  By now, bank has accumulated ~24 frames from segment 0.
  Cache: sink=[0,1,2], local window=[last 9 frames of seg 0]
```

### 3.2 Prompt Switch at Frame 40

`_recache_after_switch()` runs:

1. **`global_sink=true`** → KV cache is NOT cleared. Sink [0,1,2] preserved.
2. Cross-attention cache cleared and marked uninitialized.
3. Recache: frames `max(0, 40-12)=28` through `39` (the last 12 frames) are re-run through the model with segment 1's text embedding as cross-attention conditioning.
4. This is a recompute (`is_recompute=True`), so:
   - Cache indices don't change
   - No evictions → no bank storage during recache
   - Sink region write-protected
5. Cross-attention cache cleared again after recache.

### 3.3 Segment 1 — Frames 40 to 79

**RAG state**: `rag_active=False`, `effective_top_k=0`
→ Storage ON, retrieval OFF.

Each block generates 3 new frames. As the cache fills and evicts:
- Remaining segment 0 frames in the local window get evicted → stored in bank
- Then segment 1 frames get evicted → also stored in bank

By end of segment 1, the bank contains: segment 0 frames (3–~30) + segment 1 frames (~40–~70).

### 3.4 Prompt Switch at Frame 80

Same recache process as 3.2, but with segment 2's text embedding.

### 3.5 Segment 2 — Frames 80 to 119

**RAG state**: `rag_active=True`, `effective_top_k=6`
→ Storage ON, **retrieval ON**.

Now each block:
1. Generates query for frames [F, F+1, F+2]
2. Computes query embedding via mean-pooling
3. Searches bank for 6 most similar frames (excluding those in sink + local window)
4. Retrieved KVs are injected into attention (see section 4)

Since segment 2's prompt says "woman turns back to face camera" — the query embeddings should be most similar to segment 0's stored frames (woman facing camera), pulling those entity-consistent KVs into attention.

---

## 4. Attention Assembly With RAG

Without RAG, attention sees:
```
[sink: 3 frames × 1560 tok] + [local window: up to 9 frames × 1560 tok]
= up to 18720 tokens
```

With RAG, the attention input becomes:
```
[sink: 4680 tok] + [retrieved: 6 × 1560 = 9360 tok] + [local: budget remaining]
```

The local window budget is reduced to accommodate retrieved tokens:
```python
rag_tokens = retrieved_k.shape[1]  # 6 × 1560 = 9360
local_budget = max_attention_size - sink_tokens - rag_tokens
# = 18720 - 4680 - 9360 = 4680 tokens = 3 frames
```

So with `top_k=6`, the local window shrinks from 9 frames to 3 frames. The model attends to:
```
┌────────────┬────────────────────────────┬──────────────┐
│ Sink       │ Retrieved (from bank)      │ Local window │
│ F0,F1,F2   │ 6 most similar past frames │ 3 recent     │
│ 4680 tok   │ 9360 tok                   │ 4680 tok     │
└────────────┴────────────────────────────┴──────────────┘
                         Total: 18720 tokens
```

The attention computation itself is standard scaled dot-product:
```python
x = attention(query, k_cat, v_cat)  # flash attention
```

---

## 5. Diversity-Aware Eviction in the Bank

When the bank exceeds `max_frames` (256), instead of evicting the oldest frame (FIFO), we evict the **most redundant** frame.

### Algorithm

```python
def _find_most_redundant(self):
    embs = normalize(stack(self.embeddings))  # [N, 1536]
    sim_matrix = embs @ embs.T               # [N, N] pairwise cosine sim
    sim_matrix.fill_diagonal_(-inf)           # ignore self
    max_sim = sim_matrix.max(dim=1)           # each frame's nearest neighbor sim
    return max_sim.argmax()                   # frame with highest nn similarity
```

**Intuition**: If frame A has similarity 0.95 to frame B, removing A loses very little information — frame B already represents nearly the same content. In contrast, a frame with max neighbor similarity 0.3 is unique and should be kept.

**Why this matters**: In a long video, many consecutive frames are near-identical (e.g., 30 frames of a static pose). FIFO would keep the latest 256, but those might all be from the most recent segment. Diversity-aware eviction ensures that distinctive frames from earlier segments (like the entity's first appearance) persist in the bank even when flooded with repetitive later frames.

---

## 6. 30 Banks vs 1 Bank

### Current Design: 30 Independent Banks

Each transformer layer has its own bank. Layer 5 stores and retrieves from layer 5's KV representations; layer 20 from layer 20's. The retrieval decision (which frame IDs to select) is made independently per layer.

**Pros:**
- Each layer retrieves what's most relevant in its own feature space
- Early layers can retrieve texture-similar frames; late layers can retrieve semantically-similar frames

**Cons:**
- 30× memory for embeddings and stored KVs
- 30× cosine similarity computations
- Different layers may retrieve different frame IDs, meaning there's no coherent "scene" being recalled — layer 3 might pull frame 10 while layer 25 pulls frame 35

### Alternative: Shared Decision, Per-Layer Storage

Use one "pilot layer" (e.g., layer 15) to decide which frame IDs to retrieve, then pull the corresponding KVs from all 30 layers:

```
1 embedding bank (layer 15's embeddings) → decides frame IDs [3, 7, 12]
30 KV stores → each returns its own KVs for frames [3, 7, 12]
```

**Pros:**
- 1 cosine similarity computation instead of 30
- All layers attend to the same past frames → coherent retrieval
- Smaller embedding storage

**Cons:**
- Layer 15's similarity may not be optimal for all layers

### What the Retrieval Logs Show

From the empirical analysis of `retrieval_log.json`:

- **Early in generation** (few frames in bank): All 30 layers agree perfectly on retrieved frames
- **Later in generation** (full bank): Layers diverge significantly — layer 0 and layer 29 may share only 1 out of 6 retrieved frames

This suggests the shared-decision design would be a reasonable simplification for early-to-mid generation, but per-layer banks may capture more nuance later. The retrieval log lets you make this tradeoff empirically.
