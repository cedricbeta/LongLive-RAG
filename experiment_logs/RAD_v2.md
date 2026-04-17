@ -0,0 +1,427 @@
# RAG v2 Implementation Plan for LongLive-RAG

## Objective

Upgrade the current KV-retrieval prototype into a cleaner, more scalable memory system that:

- preserves identity and object consistency over long videos
- reduces retrieval noise from early denoising passes
- lowers retrieval overhead
- generalizes beyond the current hand-tuned walkaway/reappearance benchmarks

This plan assumes we keep the core LongLive causal generation design and improve the retrieval layer around it.

## Current State

The current implementation has the following properties:

- one `KVRetrievalBank` per transformer layer
- each bank stores evicted KV tensors at frame granularity
- retrieval is performed independently in each layer
- query embeddings are derived from the current attention query tensor
- retrieved history is injected as `[sink] + [retrieved] + [local window]`
- interactive prompt-switch logic manually gates retrieval by segment index

This works as a proof of concept, but it has several limitations:

- retrieval is expensive because all layers search independently
- different layers may retrieve different frame IDs, reducing coherence
- retrieval queries may come from noisy intermediate states
- every evicted frame is a candidate, even if it is redundant or low-value
- whole-frame retrieval is coarse and background-heavy
- gating logic is benchmark-specific rather than general

## Target v2 Design

RAG v2 should separate retrieval into two roles:

1. Shared retrieval index
   - decides which historical frame IDs are worth recalling
   - uses cleaner and more semantic features

2. Per-layer KV payload store
   - stores the actual K/V tensors for each layer
   - fetches payloads for the selected frame IDs

This preserves the correct per-layer KV representation while avoiding redundant per-layer search.

## Design Principles

- Keep K/V payloads per layer. Raw K/V tensors are layer-specific and should not be shared across layers.
- Share retrieval decisions across layers when possible. Frame selection should be globally consistent unless there is strong evidence that layer-specific selection helps.
- Use cleaner features for retrieval than the current earliest valid generation pass.
- Prefer keyframe-style memory over indiscriminate storage of all evicted frames.
- Make retrieval conditional on confidence rather than fixed prompt-segment heuristics.
- Measure both quality and latency at every phase.

## Proposed Architecture

### 1. Memory Split: Index vs Payload

Introduce two memory components:

- `FrameMemoryIndex`
  - global per video
  - stores one embedding per historical frame
  - returns top-k frame IDs for a query

- `LayerKVStore`
  - one per transformer layer
  - stores full K/V tensors keyed by frame ID
  - returns K/V payloads for selected frame IDs

Expected effect:

- cheaper retrieval
- more coherent retrieved history across layers
- simpler logging and ablations

### 2. Cleaner Query Features

Replace the current query source with one of the following:

- preferred: a late denoising pass feature
- fallback: the clean recache pass feature
- optional future extension: a learned retrieval projection head

Expected effect:

- less noisy retrieval
- better identity recall
- fewer false matches

### 3. Keyframe-Oriented Storage

Do not store every evicted frame equally. Add an insertion policy that favors:

- identity-rich frames
- large pose changes
- frames dissimilar to already stored memory
- frames with strong salience scores

Expected effect:

- more useful memory under fixed bank size
- lower background redundancy

### 4. Confidence-Based Retrieval Gating

Replace fixed segment-based rules with retrieval controls based on:

- similarity score threshold
- margin between top-1 and top-2 candidates
- prompt-change magnitude
- local-window sufficiency

Expected effect:

- fewer incorrect recalls
- better generalization to prompt changes outside the benchmark setup

### 5. Selective Injection

Ablate whether retrieval is needed:

- in all layers
- only in mid/high semantic layers
- only in a small set of designated retrieval layers

Expected effect:

- lower cost
- less interference in low-level layers

## Implementation Phases

## Phase 0: Baseline Audit and Instrumentation

Goal:

- make the current system measurable before major changes

Tasks:

- add clearer logging for:
  - query pass type
  - selected frame IDs
  - similarity scores
  - retrieval hit rates
  - per-layer vs shared selection differences
- export retrieval stats per prompt and per segment
- record memory occupancy and eviction reasons

Files likely affected:

- `wan/modules/kv_retrieval_bank.py`
- `wan/modules/causal_model.py`
- `pipeline/causal_inference.py`
- `pipeline/interactive_causal_inference.py`

Success criteria:

- reproducible timing and retrieval logs
- easy comparison between current and future variants

## Phase 1: Shared Frame Index

Goal:

- keep per-layer KV payload stores, but move frame selection into a shared index

Tasks:

- add a new module, for example:
  - `wan/modules/frame_memory_index.py`
- define storage record:
  - `frame_id`
  - `embedding`
  - metadata such as segment ID, timestamp, insertion score
- keep `LayerKVStore` functionality in the existing bank or split it out from `KVRetrievalBank`
- update the causal model so retrieval becomes:
  1. query shared index once
  2. get selected `frame_ids`
  3. fetch K/V from each layer using those IDs

Files likely affected:

- new: `wan/modules/frame_memory_index.py`
- refactor: `wan/modules/kv_retrieval_bank.py`
- refactor: `wan/modules/causal_model.py`
- wiring: `pipeline/causal_inference.py`
- wiring: `pipeline/interactive_causal_inference.py`

Success criteria:

- frame selection is shared across layers
- output quality is no worse than current RAG
- retrieval overhead decreases

## Phase 2: Cleaner Query Source

Goal:

- stop using the earliest non-recompute query as the main retrieval signal

Tasks:

- identify a cleaner feature source:
  - late denoising pass
  - clean recache pass
  - cached latent feature after final denoising
- plumb this feature into the shared index query path
- support ablation via config:
  - `rag_query_source: early | late | clean`

Files likely affected:

- `wan/modules/causal_model.py`
- `pipeline/causal_inference.py`
- `pipeline/interactive_causal_inference.py`
- config files under `configs/`

Success criteria:

- improved retrieval precision on reappearance cases
- reduced low-confidence retrievals

## Phase 3: Better Memory Insertion Policy

Goal:

- store better frames, not just more frames

Tasks:

- implement insertion scoring using a combination of:
  - novelty to current memory
  - embedding salience
  - optional prompt relevance
- add configurable policies:
  - `store_all_evicted`
  - `novelty_threshold`
  - `topk_salient_only`
  - `reservoir_with_diversity`
- preserve the current diversity-aware eviction as one option

Files likely affected:

- `wan/modules/frame_memory_index.py`
- `wan/modules/kv_retrieval_bank.py`
- configs under `configs/`

Success criteria:

- same or better consistency with smaller memory budget
- lower retrieval redundancy

## Phase 4: Confidence-Based Gating

Goal:

- replace hand-coded benchmark gating with general retrieval gating

Tasks:

- compute retrieval confidence metrics:
  - top-1 score
  - score margin
  - entropy or concentration across candidates
- add decision rules such as:
  - disable retrieval below confidence threshold
  - reduce `top_k` when confidence is low
  - skip retrieval when local memory already covers a similar frame
- keep prompt-segment gating only as an optional benchmark mode

Files likely affected:

- `wan/modules/frame_memory_index.py`
- `wan/modules/causal_model.py`
- `pipeline/interactive_causal_inference.py`
- configs under `configs/`

Success criteria:

- fewer hallucinated recalls during empty-scene intervals
- better behavior across prompt-switch scenarios

## Phase 5: Selective Layer Injection

Goal:

- determine where retrieval actually helps

Tasks:

- add config for retrieval layers:
  - `all`
  - `mid_only`
  - `high_only`
  - explicit layer list
- ablate retrieval in low-level layers vs semantic layers
- optionally use shared frame IDs but inject K/V only in selected layers

Files likely affected:

- `wan/modules/causal_model.py`
- `pipeline/causal_inference.py`
- `pipeline/interactive_causal_inference.py`
- configs under `configs/`

Success criteria:

- reduced latency
- equal or better identity consistency

## Phase 6: Fine-Grained Memory

Goal:

- move beyond whole-frame retrieval

Tasks:

- explore storing:
  - salient token subsets
  - pooled object slots
  - face or subject-centric region memory if a detector-free heuristic is available
- compare:
  - full-frame KV injection
  - sparse token injection
  - hybrid approach

This phase is more experimental and should only start after the shared-index design is stable.

Success criteria:

- better identity preservation with less background leakage

## Configuration Plan

Add the following config options gradually:

```yaml
rag_enabled: true
rag_mode: shared_index
rag_top_k: 4
rag_query_source: clean
rag_embedding: norm_weighted
rag_index_max_frames: 256
rag_payload_max_frames: 256
rag_store_policy: novelty_threshold
rag_retrieval_threshold: 0.35
rag_retrieval_margin: 0.05
rag_injection_layers: mid_high
rag_benchmark_gating: false
```

## Evaluation Plan

Use the existing benchmark scripts as the starting point, then expand.

### Quality Metrics

- late-half SSIM
- early-vs-late reappearance SSIM
- prompt-switch consistency metrics
- qualitative comparison videos

### Retrieval Metrics

- retrieval precision proxy
- average similarity of chosen frames
- diversity of retrieved frame IDs
- retrieval confidence histogram
- percentage of retrieval calls skipped by gating

### Performance Metrics

- total diffusion wall time
- retrieval overhead
- memory size per layer
- number of retrieval calls
- average top-k size after gating

## Recommended Milestone Order

Implement in this order:

1. Phase 0: instrumentation
2. Phase 1: shared index
3. Phase 2: cleaner query source
4. Phase 4: confidence-based gating
5. Phase 5: selective layer injection
6. Phase 3: smarter storage policy
7. Phase 6: fine-grained memory

Reasoning:

- shared index and cleaner queries address the biggest structural weaknesses first
- gating and selective injection reduce risk and cost early
- smarter storage and fine-grained memory are valuable, but should be built on top of a stable retrieval path

## Risks

- retrieval may become too conservative and fail to recall useful identity frames
- shared frame selection may underperform if some layers genuinely need different memories
- cleaner query extraction may increase implementation complexity or memory pressure
- sparse or region-based memory may introduce instability if not aligned with the model's attention patterns

## Open Questions

- Which layer or layer range gives the best retrieval signal for identity consistency?
- Is shared frame selection always better than layer-specific selection?
- Which query source is best: early, late, or clean recache?
- Does sparse retrieval outperform full-frame retrieval at the same compute budget?
- Should retrieval be conditioned on prompt similarity or only on visual memory signals?

## First Concrete Deliverable

The first practical v2 milestone should be:

- a shared frame index
- per-layer K/V payload stores
- configurable query source
- confidence threshold gating
- benchmark scripts updated to compare baseline vs v2

This is the smallest upgrade that meaningfully improves both engineering quality and research clarity.