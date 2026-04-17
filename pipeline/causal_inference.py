# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch
import os

import time
import json
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from wan.modules.kv_retrieval_bank import KVRetrievalBank

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation, log_gpu_memory
from utils.debug_option import DEBUG
import torch.distributed as dist

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        if DEBUG:
            print(f"args.model_kwargs: {args.model_kwargs}")
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # hard code for Wan2.1-T2V-1.3B
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = args.model_kwargs.local_attn_size

        # Normalize to list if sequence-like (e.g., OmegaConf ListConfig)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Decide the device for output based on low_memory (CPU for low-memory mode; otherwise GPU)
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        # RAG: Initialize per-layer retrieval banks if enabled
        rag_enabled = getattr(self.args, "rag_enabled", False)
        rag_top_k = getattr(self.args, "rag_top_k", 3)
        rag_bank_size = getattr(self.args, "rag_bank_size", 256)
        rag_embedding = getattr(self.args, "rag_embedding", "norm_weighted")
        rag_random_baseline = getattr(self.args, "rag_random_baseline", False)
        rag_query_source = getattr(self.args, "rag_query_source", "early")
        if rag_enabled:
            self.retrieval_banks = [
                KVRetrievalBank(
                    max_frames=rag_bank_size,
                    frame_seq_length=self.frame_seq_length,
                    embedding_strategy=rag_embedding,
                    random_mode=rag_random_baseline,
                )
                for _ in range(self.num_transformer_blocks)
            ]
            mode_str = "RANDOM BASELINE" if rag_random_baseline else f"embedding={rag_embedding}"
            print(f"[RAG] Enabled: top_k={rag_top_k}, bank_size={rag_bank_size}, {mode_str}, query_source={rag_query_source}")
        else:
            self.retrieval_banks = None

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 2: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        block_wall_times = []  # wall-clock ms per block (always collected)
        torch.cuda.synchronize()
        gen_wall_start = time.perf_counter()
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            torch.cuda.synchronize()
            block_wall_t0 = time.perf_counter()

            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames]

            # For 'clean' query source: serve cached retrieval during denoising
            if rag_query_source == "clean" and self.retrieval_banks is not None:
                for bank in self.retrieval_banks:
                    bank.use_cache = True

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # Per-step retrieval top_k based on query source
                is_last_step = (index == len(self.denoising_step_list) - 1)
                if rag_query_source == "late":
                    step_top_k = rag_top_k if is_last_step else 0
                    if self.retrieval_banks is not None:
                        for bank in self.retrieval_banks:
                            bank.force_retrieve = is_last_step
                else:
                    step_top_k = rag_top_k

                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        retrieval_banks=self.retrieval_banks,
                        retrieval_top_k=step_top_k
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        retrieval_banks=self.retrieval_banks,
                        retrieval_top_k=step_top_k
                    )
            # Reset force_retrieve after late-mode denoising loop
            if rag_query_source == "late" and self.retrieval_banks is not None:
                for bank in self.retrieval_banks:
                    bank.force_retrieve = False

            # Step 2.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # For 'clean' query source: fresh retrieval during recache pass
            if rag_query_source == "clean" and self.retrieval_banks is not None:
                for bank in self.retrieval_banks:
                    bank.use_cache = False
                    bank.force_retrieve = True

            # Step 2.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                retrieval_banks=self.retrieval_banks,
                retrieval_top_k=rag_top_k
            )

            # Reset force_retrieve after recache
            if rag_query_source == "clean" and self.retrieval_banks is not None:
                for bank in self.retrieval_banks:
                    bank.force_retrieve = False

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            torch.cuda.synchronize()
            block_wall_times.append((time.perf_counter() - block_wall_t0) * 1000)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 3: Decode the output
        if getattr(self.args.model_kwargs, "use_infinite_attention", False):
            video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)
        else:
            video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        # --- Latency summary (always printed) ---
        torch.cuda.synchronize()
        gen_wall_total = (time.perf_counter() - gen_wall_start) * 1000
        avg_block_ms = sum(block_wall_times) / len(block_wall_times) if block_wall_times else 0
        pixel_frames = (num_output_frames - 1) * 4 + 1
        fps = pixel_frames / (gen_wall_total / 1000) if gen_wall_total > 0 else 0

        timing_stats = {
            "rag_enabled": rag_enabled,
            "num_latent_frames": num_output_frames,
            "num_pixel_frames": pixel_frames,
            "total_diffusion_ms": round(gen_wall_total, 2),
            "avg_block_ms": round(avg_block_ms, 2),
            "fps": round(fps, 2),
            "per_block_ms": [round(t, 2) for t in block_wall_times],
        }

        if rag_enabled and self.retrieval_banks is not None:
            total_store = sum(b.store_time_ms for b in self.retrieval_banks)
            total_retrieve = sum(b.retrieve_time_ms for b in self.retrieval_banks)
            total_store_calls = sum(b.store_calls for b in self.retrieval_banks)
            total_retrieve_calls = sum(b.retrieve_calls for b in self.retrieval_banks)
            bank_sizes = [len(b) for b in self.retrieval_banks]
            timing_stats["rag_store_ms"] = round(total_store, 2)
            timing_stats["rag_retrieve_ms"] = round(total_retrieve, 2)
            timing_stats["rag_store_calls"] = total_store_calls
            timing_stats["rag_retrieve_calls"] = total_retrieve_calls
            timing_stats["rag_bank_sizes"] = bank_sizes
            timing_stats["rag_overhead_ms"] = round(total_store + total_retrieve, 2)
            timing_stats["rag_overhead_pct"] = round(100 * (total_store + total_retrieve) / gen_wall_total, 2) if gen_wall_total > 0 else 0

        print(f"\n{'='*50}")
        print(f"LATENCY SUMMARY ({'RAG' if rag_enabled else 'Baseline'})")
        print(f"{'='*50}")
        print(f"  Latent frames:    {num_output_frames}")
        print(f"  Pixel frames:     {pixel_frames}")
        print(f"  Diffusion total:  {gen_wall_total:.0f} ms")
        print(f"  Avg per block:    {avg_block_ms:.1f} ms")
        print(f"  Generation FPS:   {fps:.1f} (pixel frames / diffusion time)")
        if rag_enabled and self.retrieval_banks is not None:
            print(f"  RAG store:        {timing_stats['rag_store_ms']:.1f} ms ({timing_stats['rag_store_calls']} calls)")
            print(f"  RAG retrieve:     {timing_stats['rag_retrieve_ms']:.1f} ms ({timing_stats['rag_retrieve_calls']} calls)")
            print(f"  RAG overhead:     {timing_stats['rag_overhead_ms']:.1f} ms ({timing_stats['rag_overhead_pct']:.1f}%)")
        print(f"{'='*50}\n")

        # Save timing to JSON (append mode per prompt)
        output_folder = getattr(self.args, 'output_folder', '.')
        os.makedirs(output_folder, exist_ok=True)
        timing_path = os.path.join(output_folder, "timing.json")
        existing = []
        if os.path.exists(timing_path):
            with open(timing_path, 'r') as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        existing.append(timing_stats)
        with open(timing_path, 'w') as f:
            json.dump(existing, f, indent=2)

        # Save per-layer retrieval logs
        if rag_enabled and self.retrieval_banks is not None:
            retrieval_log = {}
            for layer_idx, bank in enumerate(self.retrieval_banks):
                if bank.retrieval_log:
                    retrieval_log[f"layer_{layer_idx}"] = bank.retrieval_log
            if retrieval_log:
                log_path = os.path.join(output_folder, "retrieval_log.json")
                existing_logs = []
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        try:
                            existing_logs = json.load(f)
                        except json.JSONDecodeError:
                            existing_logs = []
                existing_logs.append(retrieval_log)
                with open(log_path, 'w') as f:
                    json.dump(existing_logs, f, indent=2)
                print(f"  Retrieval log saved to {log_path}")

        # Retrieval audit
        if rag_enabled and self.retrieval_banks is not None:
            audit = self._compute_retrieval_audit()
            if audit:
                audit_path = os.path.join(output_folder, "retrieval_audit.json")
                existing_audits = []
                if os.path.exists(audit_path):
                    with open(audit_path, 'r') as f:
                        try:
                            existing_audits = json.load(f)
                        except json.JSONDecodeError:
                            existing_audits = []
                existing_audits.append(audit)
                with open(audit_path, 'w') as f:
                    json.dump(existing_audits, f, indent=2)
                print(f"\n  [AUDIT] Retrieval audit saved to {audit_path}")
                if "cross_layer_agreement" in audit:
                    cla = audit["cross_layer_agreement"]
                    print(f"  [AUDIT] Cross-layer agreement: {cla['overall_avg_jaccard']:.4f} avg Jaccard ({cla['num_query_frames']} queries)")
                if "embedding_quality" in audit:
                    eq = audit["embedding_quality"]
                    print(f"  [AUDIT] Embedding quality: top1={eq['avg_top1_score']:.4f}, margin={eq.get('avg_margin', 'N/A')}, entropy={eq.get('avg_entropy', 'N/A')}")
                if "insertion_stats" in audit:
                    ins = audit["insertion_stats"]
                    print(f"  [AUDIT] Insertions: {ins['num_insertions']} frames, avg_novelty={ins['avg_novelty']:.4f}")

        if return_latents:
            return video, output.to(noise.device)
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                # Local attention: cache only needs to store the window
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Global attention: default cache for 21 frames (backward compatibility)
                kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model's global default (32760 for Wan, 28160 for 5B).
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if local_attn_size_value == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass

    def _compute_retrieval_audit(self, oracle_frame_ids=None, oracle_query_min_frame=None):
        """Compute retrieval quality metrics for Phase 0 instrumentation.

        Returns a dict with:
        - cross_layer_agreement: Jaccard similarity of frame ID sets across layers
        - oracle_comparison: precision of retrieval vs. known-good frame IDs
        - embedding_quality: top1 score, margin, entropy statistics
        - insertion_stats: novelty and embedding norm of stored frames
        """
        if self.retrieval_banks is None:
            return {}

        audit = {}

        # --- 1. Cross-layer agreement ---
        per_layer_by_qf = {}
        for layer_idx, bank in enumerate(self.retrieval_banks):
            for entry in bank.retrieval_log:
                qf = entry.get("query_frame")
                if qf is None or entry.get("mode") == "random":
                    continue
                per_layer_by_qf.setdefault(qf, {})[layer_idx] = set(entry["retrieved_frames"])

        agreement_per_frame = []
        for qf in sorted(per_layer_by_qf.keys()):
            sets = list(per_layer_by_qf[qf].values())
            if len(sets) < 2:
                continue
            jaccards = []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    union = len(sets[i] | sets[j])
                    jaccards.append(len(sets[i] & sets[j]) / union if union > 0 else 0.0)
            avg_j = sum(jaccards) / len(jaccards)
            agreement_per_frame.append({"query_frame": qf, "jaccard": round(avg_j, 4)})

        if agreement_per_frame:
            overall = sum(e["jaccard"] for e in agreement_per_frame) / len(agreement_per_frame)
            audit["cross_layer_agreement"] = {
                "overall_avg_jaccard": round(overall, 4),
                "num_query_frames": len(agreement_per_frame),
                "per_frame": agreement_per_frame,
            }

        # --- 2. Oracle comparison ---
        if oracle_frame_ids is not None:
            oracle_set = set(oracle_frame_ids)
            precisions = []
            for qf in sorted(per_layer_by_qf.keys()):
                if oracle_query_min_frame is not None and qf < oracle_query_min_frame:
                    continue
                layer_precisions = []
                for layer_idx, fid_set in per_layer_by_qf[qf].items():
                    if len(fid_set) > 0:
                        p = len(fid_set & oracle_set) / len(fid_set)
                        layer_precisions.append(p)
                if layer_precisions:
                    precisions.append({
                        "query_frame": qf,
                        "avg_precision": round(sum(layer_precisions) / len(layer_precisions), 4),
                    })

            if precisions:
                overall_p = sum(e["avg_precision"] for e in precisions) / len(precisions)
                audit["oracle_comparison"] = {
                    "oracle_frames": oracle_frame_ids,
                    "avg_precision": round(overall_p, 4),
                    "num_query_frames": len(precisions),
                    "per_frame": precisions,
                }

        # --- 3. Embedding quality summary ---
        all_top1 = []
        all_margins = []
        all_entropies = []
        for bank in self.retrieval_banks:
            for entry in bank.retrieval_log:
                if entry.get("top1_score") is not None:
                    all_top1.append(entry["top1_score"])
                if entry.get("margin") is not None:
                    all_margins.append(entry["margin"])
                if entry.get("entropy") is not None:
                    all_entropies.append(entry["entropy"])

        if all_top1:
            audit["embedding_quality"] = {
                "avg_top1_score": round(sum(all_top1) / len(all_top1), 4),
                "min_top1_score": round(min(all_top1), 4),
                "max_top1_score": round(max(all_top1), 4),
                "avg_margin": round(sum(all_margins) / len(all_margins), 4) if all_margins else None,
                "avg_entropy": round(sum(all_entropies) / len(all_entropies), 4) if all_entropies else None,
                "num_retrievals": len(all_top1),
            }

        # --- 4. Insertion stats ---
        all_novelties = []
        all_emb_norms = []
        for bank in self.retrieval_banks:
            for entry in bank.insertion_log:
                all_novelties.append(entry["novelty"])
                all_emb_norms.append(entry["embedding_norm"])

        if all_novelties:
            audit["insertion_stats"] = {
                "avg_novelty": round(sum(all_novelties) / len(all_novelties), 4),
                "min_novelty": round(min(all_novelties), 4),
                "avg_embedding_norm": round(sum(all_emb_norms) / len(all_emb_norms), 4),
                "num_insertions": len(all_novelties),
            }

        return audit