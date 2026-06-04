# Known Limitations: Multi-View KV-RAG Evaluation

This document records explicit AC-7 limits for the current multi-view scene-memory path. These are not silent caps.

## 1. SP scene-memory mirroring is not supported

**Limitation:** The persistent KV-RAG scene partition is process-local. `CausalDiffusionInferencePipeline` owns separate `kv_rag_pos` and `kv_rag_neg` `KVRAGMemory` banks, and scene anchors are kept by `_reset_kv_rag_keep_scene()` when `preserve_scene_memory` is true. Under sequence-parallel or other multi-GPU sharded-attention paths, those scene anchors are not mirrored, broadcast, or all-gathered across ranks. The Ulysses SP inference pipeline currently warns and disables KV-RAG rather than treating SP scene memory as supported.

**Why it exists:** KV-RAG storage/retrieval is wired through the local pipeline banks and model calls. The scene partition survives local resets via `_scene_memory_active`, but there is no collective that synchronizes persistent entries across SP ranks or tensor/sequence shards.

**Safe standing path / workaround:** Treat single-GPU per-perspective inference as the standing evaluation path for cross-perspective scene memory. Do not use SP or multi-GPU sharded attention for KV-RAG scene-memory evaluation until scene-anchor mirroring/all-gather is implemented and tested.

## 2. DDP can break scene contiguity

**Limitation:** `MultiViewPerspectiveDataset` orders samples scene-major, then perspective-index, so all perspectives for one scene are contiguous. `inference.py` relies on that ordering: it preserves scene memory only when the current `scene_name` equals the previous sample's `scene_name`. When `torch.distributed` is initialized, `inference.py` uses `DistributedSampler(dataset, shuffle=False, drop_last=True)`, which can shard a scene's contiguous perspectives across ranks and can drop tail samples.

**Why it exists:** The sampler is index-based, not scene-aware. A rank may see only every Nth perspective, or the last perspectives/scenes may be removed to make shard lengths even, so `_prev_scene_name` no longer represents a complete contiguous scene on that rank.

**Safe standing path / workaround:** Use single-GPU evaluation for `multiview_per_perspective` scene memory, or replace the sampler with scene-aligned sharding that assigns whole scenes, in perspective order, to one rank and never drops required perspectives.

## 3. Coverage bounds are logged, not hidden

**Limitation:** Multi-view gate runs may intentionally evaluate fewer perspectives, fewer scenes, or GPU-free dimensions only. These reductions are explicit coverage bounds, not silent behavior.

**Why it exists:** `scripts/run_kv_rag_ablation.py --mode multiview_vbench` records the rendered perspective count per scene from `--max_perspectives`, the selected scene subset from `--prompt_subset`, the modified finalist key/value settings, and the loaded backbone set in the gate output. DINO subject, CLIP background, and ArcFace/insightface or DINO-patch identity backbones are optional milestone downloads; when absent, the aggregate uses GPU-free dimensions such as temporal style, appearance style, and overall consistency, and records that fact in `gate["backbones"]`.

**Safe standing path / workaround:** Read `multiview_vbench_gate.json` before interpreting a gate result. Confirm `gate["perspective_coverage"]`, `gate["modified_settings"]`, `gate["backbones"]`, the scene count/subset, and any dry-run note. To run the full milestone gate, render all intended perspectives/scenes and enable the required VBench/identity backbones; otherwise report the logged reduced coverage exactly as written.
