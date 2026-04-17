# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch
import time
import json
import os

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.causal_inference import CausalInferencePipeline
from wan.modules.kv_retrieval_bank import KVRetrievalBank
import torch.distributed as dist
from utils.debug_option import DEBUG


class InteractiveCausalInferencePipeline(CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)

    # Internal helpers — matches original LongLive logic with RAG support
    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict,
                               retrieval_banks=None, retrieval_top_k=3):
        if retrieval_banks is not None:
            for bank in retrieval_banks:
                bank.clear_retrieval_cache()

        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()

        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        # recache
        if current_start_frame == 0:
            return

        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames

        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        # move to gpu
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        print(f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")

        # prepare blockwise causal mask
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size
        )

        context_timestep = torch.ones([batch_size, num_recache_frames],
                                    device=device, dtype=torch.int64) * self.args.context_noise

        self.generator.model.block_mask = block_mask

        # recache
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=not self.global_sink,
                retrieval_banks=retrieval_banks,
                retrieval_top_k=retrieval_top_k,
            )

        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block


        # encode all prompts
        print(text_prompts_list)
        cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # initialize caches
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size,
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

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0  # current segment index
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        # RAG retrieval control: disabled for the first two segments (initial
        # appearance + empty-scene walkaway), enabled from segment 2 onward
        # where the entity returns and can benefit from recalling earlier KVs.
        rag_active = False
        if rag_enabled:
            print(f"[RAG] Segment 0: retrieval OFF (initial segment)")

        if DEBUG:
            print("[MultipleSwitch] all_num_frames", all_num_frames)
            print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)

        block_wall_times = []
        torch.cuda.synchronize()
        gen_wall_start = time.perf_counter()

        for current_num_frames in all_num_frames:
            torch.cuda.synchronize()
            block_wall_t0 = time.perf_counter()

            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1
                self._recache_after_switch(
                    output, current_start_frame, cond_list[segment_idx],
                    retrieval_banks=self.retrieval_banks,
                    retrieval_top_k=rag_top_k,
                )

                # Enable RAG retrieval from segment 2 onward — this is when
                # the entity returns and benefits from recalling earlier KVs.
                # Segment 1 is the empty-scene gap and should NOT retrieve,
                # otherwise the old entity can be pulled back too early.
                if segment_idx >= 2:
                    rag_active = True
                    if rag_enabled:
                        print(f"[RAG] Segment {segment_idx}: retrieval ON")
                else:
                    rag_active = False
                    if rag_enabled:
                        print(f"[RAG] Segment {segment_idx}: retrieval OFF")

                if DEBUG:
                    print(
                        f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}"
                    )
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
                print(f"segment_idx: {segment_idx}")
                print(f"text_prompts_list[segment_idx]: {text_prompts_list[segment_idx]}")
            cond_in_use = cond_list[segment_idx]

            noisy_input = noise[
                :, current_start_frame : current_start_frame + current_num_frames
            ]

            # Always pass retrieval_banks so evicted KVs are stored.
            # retrieval_top_k controls whether retrieval actually happens:
            # 0 = store only (no retrieval), >0 = store + retrieve.
            effective_top_k = rag_top_k if rag_active else 0

            # For 'clean' query source: serve cached retrieval during denoising
            if rag_query_source == "clean" and self.retrieval_banks is not None:
                for bank in self.retrieval_banks:
                    bank.use_cache = True

            # ---------------- Spatial denoising loop ----------------
            for index, current_timestep in enumerate(self.denoising_step_list):
                # Per-step retrieval top_k based on query source
                is_last_step = (index == len(self.denoising_step_list) - 1)
                if rag_query_source == "late":
                    step_top_k = effective_top_k if is_last_step else 0
                    if self.retrieval_banks is not None:
                        for bank in self.retrieval_banks:
                            bank.force_retrieve = is_last_step
                else:
                    step_top_k = effective_top_k

                timestep = (
                    torch.ones([batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64)
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        retrieval_banks=self.retrieval_banks,
                        retrieval_top_k=step_top_k,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        retrieval_banks=self.retrieval_banks,
                        retrieval_top_k=step_top_k,
                    )

            # Reset force_retrieve after late-mode denoising loop
            if rag_query_source == "late" and self.retrieval_banks is not None:
                for bank in self.retrieval_banks:
                    bank.force_retrieve = False

            # Record output
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # For 'clean' query source: fresh retrieval during recache pass
            if rag_query_source == "clean" and self.retrieval_banks is not None:
                for bank in self.retrieval_banks:
                    bank.use_cache = False
                    bank.force_retrieve = True

            # rerun with clean context to update cache
            # Always pass retrieval_banks here so evicted KVs are still
            # stored in the bank even when retrieval is disabled.
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                retrieval_banks=self.retrieval_banks,
                retrieval_top_k=rag_top_k,
            )

            # Reset force_retrieve after recache
            if rag_query_source == "clean" and self.retrieval_banks is not None:
                for bank in self.retrieval_banks:
                    bank.force_retrieve = False

            torch.cuda.synchronize()
            block_wall_times.append((time.perf_counter() - block_wall_t0) * 1000)

            # Update frame pointer
            current_start_frame += current_num_frames

        # Standard decoding
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        # --- Latency summary ---
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
        print(f"  Generation FPS:   {fps:.1f}")
        if rag_enabled and self.retrieval_banks is not None:
            print(f"  RAG store:        {timing_stats['rag_store_ms']:.1f} ms ({timing_stats['rag_store_calls']} calls)")
            print(f"  RAG retrieve:     {timing_stats['rag_retrieve_ms']:.1f} ms ({timing_stats['rag_retrieve_calls']} calls)")
            print(f"  RAG overhead:     {timing_stats['rag_overhead_ms']:.1f} ms ({timing_stats['rag_overhead_pct']:.1f}%)")
        print(f"{'='*50}\n")

        # Save timing to JSON
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

        # Retrieval audit with oracle comparison
        if rag_enabled and self.retrieval_banks is not None:
            oracle_frame_ids = None
            oracle_query_min_frame = None
            if len(switch_frame_indices) >= 2:
                oracle_frame_ids = list(range(switch_frame_indices[0]))
                oracle_query_min_frame = switch_frame_indices[1]

            audit = self._compute_retrieval_audit(
                oracle_frame_ids=oracle_frame_ids,
                oracle_query_min_frame=oracle_query_min_frame,
            )
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
                if "oracle_comparison" in audit:
                    oc = audit["oracle_comparison"]
                    print(f"  [AUDIT] Oracle precision: {oc['avg_precision']:.4f} (oracle frames: {oc['oracle_frames']}, {oc['num_query_frames']} queries)")
                if "embedding_quality" in audit:
                    eq = audit["embedding_quality"]
                    print(f"  [AUDIT] Embedding quality: top1={eq['avg_top1_score']:.4f}, margin={eq.get('avg_margin', 'N/A')}, entropy={eq.get('avg_entropy', 'N/A')}")
                if "insertion_stats" in audit:
                    ins = audit["insertion_stats"]
                    print(f"  [AUDIT] Insertions: {ins['num_insertions']} frames, avg_novelty={ins['avg_novelty']:.4f}")

        if return_latents:
            return video, output
        return video
