# RAG-KV: Retrieval-Augmented KV Cache for LongLive

## The Problem

LongLive generates video **frame by frame**. Each new frame can only "see" a limited window of past frames through the KV cache:

```
Frame:    0  1  2  3  4  5  6  ...  30  31  32  33  34  ...  50
          ├──────┤                        ├───────────────────┤
          sink (3)                        local window (12)
          always kept                     slides forward

          Frames 3-29 are GONE. The model has no memory of them.
```

This causes **visual drift** — frame 50 might render a character's face slightly differently than frame 10, because it can't see frame 10 anymore.

## The Solution

Save evicted frames. Retrieve relevant ones when needed.

```
                    ┌───────────────────────┐
                    │    Retrieval Bank      │
                    │  (stores evicted KVs)  │
                    │                       │
                    │  F3  F4  F5 ... F29   │
                    └──────────┬────────────┘
                               │
                          retrieve top-k
                          most similar
                               │
                               ▼
  ┌─────────┐  ┌───────────┐  ┌───────────────────────────────┐
  │ Sink    │  │ Retrieved │  │ Local Window                  │
  │ F0-F2   │  │ F10, F22  │  │ F39  F40  F41 ... F50        │
  └─────────┘  └───────────┘  └───────────────────────────────┘
        └──────────┴──────────────────┘
                        │
                  All concatenated
                  for attention
                        │
                        ▼
              Attention(Q_current, K_all, V_all)
```

## How It Works: 3 Steps

### Step 1: Store (on eviction)

When the sliding window moves forward, frames fall off the left edge.
**Before they're lost, we save them to the bank.**

```
Before eviction:
  Cache: [F0 F1 F2 | F8  F9  F10  F11 ... F19]
                      ↑
                      about to be evicted

  → bank.store(F8's keys and values, frame_id=8)

After eviction:
  Cache: [F0 F1 F2 | F9  F10  F11  F12 ... F20]
                                              ↑ new frame inserted
```

Each stored frame is saved as:
- **Full KV tokens**: `[1560 tokens, 12 heads, 128 dims]` (for attention)
- **Retrieval embedding**: mean of key tokens `[12 × 128]` (for search)

### Step 2: Retrieve (before attention)

When computing attention for a new frame, we search the bank:

```
1. Compute query embedding:
   query_emb = mean(current_frame_query)        →  [1536-dim vector]

2. Compare against all stored frames:
   similarity = cosine(query_emb, each stored embedding)

3. Pick top-k most similar (e.g., k=3):
   result = [Frame 10, Frame 15, Frame 22]      →  their full KV tokens
```

**Why cosine similarity on keys?** Keys already encode "what information this frame offers to attention." Similar keys = visually similar content = likely the same entity.

### Step 3: Inject (during attention)

Concatenate retrieved KVs into the attention computation:

```
Without RAG:
  Attention(Q, K=[sink + local], V=[sink + local])
  → sees 15 frames

With RAG:
  Attention(Q, K=[sink + retrieved + local], V=[sink + retrieved + local])
  → sees 18 frames (3 from the past that are most relevant)
```

## What This Achieves

```
Without RAG (frame 50 generating):

  "What did the character look like?"
  → Can only check frames 0-2 (sink) and 38-49 (local window)
  → Gradual drift from the original appearance

With RAG (frame 50 generating):

  "What did the character look like?"
  → Checks frames 0-2 (sink) + 38-49 (local window)
  → ALSO retrieves frames 10, 22, 35 (most similar past frames)
  → Anchored to earlier appearance → less drift
```

## Architecture

```
LongLive has 30 transformer layers.
Each layer has its own KV cache and its own retrieval bank.

Layer 0:  Cache ←→ Bank₀   (low-level features: edges, colors)
Layer 1:  Cache ←→ Bank₁
  ...
Layer 15: Cache ←→ Bank₁₅  (mid-level: textures, shapes)
  ...
Layer 29: Cache ←→ Bank₂₉  (high-level: objects, semantics)

Each bank operates independently — a retrieval in layer 5
does NOT affect layer 20.
```

## SSIM Evaluation

We measure **visual consistency** (not quality) by comparing every frame against an early reference:

```
Pick reference = Frame 10 (early, stable)

SSIM(frame_i) = structural_similarity(frame_10, frame_i)

  1.0 = identical to frame 10
  0.7 = somewhat similar
  0.0 = completely different

Plot SSIM over time:

  1.0 ┤
      │╲
      │ ╲____
  0.8 ┤      ╲___________  ← RAG (drifts less)
      │        ╲__________
  0.6 ┤                   ╲______  ← Baseline (drifts more)
      │
  0.4 ┤
      └──────────────────────────→ Frame
      0   50  100  150  200  250

We report the average SSIM over the second half of the video,
where drift is most pronounced.
```

## Results

| Prompt | Baseline | RAG | Improvement |
|--------|:--------:|:---:|:-----------:|
| Elderly man close-up | 0.687 | 0.743 | +0.057 |
| Clocktower orbit | 0.634 | 0.712 | +0.077 |
| Cat on armchair | 0.720 | 0.744 | +0.025 |
| **Average** | | | **+0.053** |

Biggest gain on the clocktower (complex details, camera orbits back to same view) and the face (fine-grained features that drift easily).

## Config

```yaml
# In configs/longlive_inference.yaml
rag_enabled: true       # Turn RAG on/off
rag_top_k: 3            # Frames to retrieve per layer per step
rag_bank_size: 256      # Max frames stored before oldest dropped
```

## Files

| File | Change |
|------|--------|
| `wan/modules/kv_retrieval_bank.py` | New — the bank: store, retrieve |
| `wan/modules/causal_model.py` | Capture evictions, inject retrievals into attention |
| `utils/wan_wrapper.py` | Pass `retrieval_banks` through model forward |
| `pipeline/causal_inference.py` | Create 30 banks, pass to generator |
| `configs/longlive_inference.yaml` | Config knobs |
