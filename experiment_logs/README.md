# RAG v2 Experiment Logs

## Overview

These logs document experiments on the KV-retrieval (RAG) system for LongLive, a causal video generation model. The goal is to maintain entity consistency across long videos by retrieving relevant historical KV cache frames and injecting them into attention.

All experiments use the Wan 1.3B backbone with 30 transformer layers, 4-step denoising (t=1000,750,500,250), and a sliding-window KV cache with sink tokens.

## Benchmark Setup

**Walkaway benchmark** (3 segments, 40 frames each):
- Segment 0: entity present (e.g., "a woman walking in a park")
- Segment 1: empty scene (e.g., "an empty park with trees")
- Segment 2: entity returns (same prompt as segment 0)

RAG is active only in segment 2. Oracle retrieval = any frame from segment 0.

## Experiment Summary

### Baselines

| Experiment | Description |
|---|---|
| `walkaway-baseline` | No RAG, no global sink. Pure sliding-window attention. |
| `walkaway-rag` | RAG with `norm_weighted` embedding, early query (step 0), per-layer retrieval. Original v1 system. |

### Embedding Strategy Ablations

| Experiment | Embedding | Avg Precision | Avg Margin | Finding |
|---|---|---|---|---|
| `walkaway-rag` | norm_weighted | ~0.45* | ~0.02* | Background-dominated; entity features diluted across 1560 tokens. |
| `walkaway-rag-salient-topk` | salient_topk | 0.57 | 0.010 | Pools top-256 tokens by L2 norm. Similar retrieval quality to norm_weighted. Late segment 2 drifts to retrieving empty-scene frames (seg 1), causing abrupt visual changes. |
| `walkaway-rag-max-mean` | max_mean | (not yet run) | - | Concatenates max-pool and mean-pool vectors. |

\* walkaway-rag was run before the audit instrumentation was added; values estimated from reappear-rag-v2 which used the same embedding.

### Query Source Ablations

| Experiment | Query Source | Finding |
|---|---|---|
| `walkaway-rag` | early (step 0, t=1000) | Best results. Step 0 is the most influential injection point. |
| `walkaway-rag-late` | late (step 3, t=250) | Worse than early. By step 3, the denoising trajectory is already committed. Retrieval at t=250 is too late to steer generation. |
| `walkaway-rag-clean` | clean (recache pass, t=0) | Worse than early. Cleanest query signal, but retrieval during recache contaminates hidden state propagation. First reappearance frame gets no retrieval (cache cleared at segment boundary). |

### Random Baseline

| Experiment | Description |
|---|---|
| `reappear-rag-v2-random-baseline` | Random frame retrieval instead of similarity-based. Performance was similar to embedding-based retrieval, confirming the embedding quality is the bottleneck. |

### Structural Change: Shared Frame Selection (implemented today)

Previous behavior: each of the 30 transformer layers independently searched its own retrieval bank via cosine similarity. Cross-layer agreement (Jaccard similarity of retrieved frame ID sets) was ~0.10 -- nearly random.

New behavior: layer 0 performs the similarity search and all other layers retrieve KV tensors for the same frame IDs. This ensures cross-layer consistency (Jaccard = 1.0) and eliminates 29 redundant similarity searches.

**Interaction with embedding quality**: shared selection amplifies both correct and incorrect retrievals. When the leader layer picks wrong frames (e.g., empty-scene frames), all 30 layers inject those wrong frames, making visual disruptions stronger than the old per-layer noise. This was observed in the `walkaway-rag-salient-topk` run where late segment 2 frames showed abrupt scene changes.

## Key Findings

1. **Embedding quality is the bottleneck.** All tested embedding strategies (norm_weighted, salient_topk) produce margins of ~0.01, meaning the top candidate is barely distinguishable from random. The random baseline performs comparably to embedding-based retrieval.

2. **Step 0 (t=1000) is the optimal retrieval point.** It is the only non-recompute denoising step, and injection at the highest noise level has the strongest influence on the output trajectory. Late and clean query sources both performed worse.

3. **Shared frame selection trades noise for coherence.** Per-layer retrieval was incoherent (Jaccard ~0.10) but errors were diluted across layers. Shared selection makes all layers consistent, which is better when retrieval is correct but worse when it is wrong.

4. **Retrieval drifts toward segment 1 in late segment 2.** As the query advances past frame 110, retrieved frames shift from segment 0 (entity) to segment 1 (empty scene). This is a direct consequence of poor embedding discriminability.

## Next Steps

- Run `walkaway-rag-max-mean` to complete the embedding ablation
- Implement confidence-based retrieval gating (Phase 4): skip retrieval when margin < threshold to suppress bad retrievals
- Make shared vs per-layer selection configurable for A/B testing
- Investigate whether a different leader layer (not layer 0) gives better retrieval signal

## File Structure

Each experiment folder contains:
- `config.yaml` — the YAML config used to run the experiment
- `timing.json` — wall-clock timing breakdown
- `retrieval_log.json` — per-layer, per-frame retrieval decisions (frame IDs, similarities)
- `retrieval_audit.json` — aggregate metrics (oracle precision, embedding quality, insertion stats)

See `RAD_v2.md` for the full v2 implementation plan.
