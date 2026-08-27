# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0

from tqdm import tqdm
from typing import List, Optional
import os
import statistics
import threading
import torch
import math

_LLV2_TIME = os.environ.get("LLV2_TIME") == "1"
_LLV2_DUMP_LATENT_DIR = os.environ.get("LLV2_DUMP_LATENT_DIR", "").strip()
# LLV2_PROFILE format: "<call_idx>:<wait>:<warmup>:<active>", e.g. "0:20:2:2"
_LLV2_PROFILE_SPEC = os.environ.get("LLV2_PROFILE", "").strip()
_LLV2_PROFILE_OUTPUT_DIR = os.environ.get("LLV2_PROFILE_OUTPUT_DIR", "").strip()
_LLV2_PROFILE_CALL_COUNTER = 0
from wan_5b.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan_5b.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder, build_vae_5b
from utils.dataset import DEFAULT_SCENE_CUT_PREFIX
from utils.config import section_get, wan_default_config
from utils.kv_rag import KVRAGConfig, KVRAGMemory
from utils.capsule import CapsuleEditConfig, CapsuleProbeConfig, CapsuleRuntime
from utils.agent_runtime import AgentRuntimeConfig, CausalShotSession, NormativeGraph


def _plain_get(cfg, key, default=None):
    """Read a key from a dict / OmegaConf node / namespace uniformly."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        value = getattr(cfg, key, default)
    except Exception:
        return default
    return default if value is None else value


class CausalDiffusionInferencePipeline(torch.nn.Module):
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
        model_name = getattr(args.model_kwargs, "model_name", "Wan2.2-TI2V-5B")
        if "5B" not in model_name:
            raise ValueError(f"Only Wan2.2-TI2V-5B is supported in this release, got {model_name}")
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = build_vae_5b(args) if vae is None else vae

        # iter-33: optionally compile the VAE decoder (cuda:2). The Python
        # `for step in range(total_steps)` wrapper in cached_decode keeps the
        # mutating feat_cache in eager; per-step self.decoder() call hits the
        # compiled version. Opt-in via LLV2_COMPILE_VAE=1; default off.
        if os.environ.get("LLV2_COMPILE_VAE", "0") == "1":
            try:
                inner = getattr(self.vae, "model", None)
                if inner is not None and hasattr(inner, "decoder"):
                    inner.decoder = torch.compile(
                        inner.decoder,
                        backend="inductor",
                        mode="max-autotune-no-cudagraphs",
                        fullgraph=False,
                        dynamic=False,
                    )
                    print("[torch.compile] VAE.decoder wrapped")
            except Exception as exc:
                print(f"[torch.compile][warn] VAE compile setup failed: {exc}")

        # Step 2: Initialize scheduler
        self.num_train_timesteps = getattr(args, "num_train_timestep", 1000)
        self.sampling_steps = section_get(args, "inference", "sampling_steps", 50)
        self.sample_solver = 'unipc'
        self.shift = getattr(args, "timestep_shift",
                             getattr(args.model_kwargs, "timestep_shift", 5.0))

        
        self.frame_seq_length = math.prod(args.image_or_video_shape[-2:]) // 4
        self.model_name = model_name
        self.num_transformer_blocks = wan_default_config[self.model_name]["num_transformer_blocks"]

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.quantize_kv = getattr(args, "kv_quant", False)
        self.kv_quant_scale_rule = getattr(args, "kv_quant_scale_rule", "mse")
        self.kv_quant_backend = getattr(args, "kv_quant_backend", "cuda")
        self.independent_first_frame = section_get(args, "inference", "independent_first_frame", False)
        self.local_attn_size = section_get(
            args, "inference", "local_attn_size", -1, aliases=("inference_local_attn_size",)
        )
        if self.local_attn_size == -1:
            self.local_attn_size = getattr(args, "model_kwargs", {}).get("local_attn_size", -1)
        self.sink_size = section_get(
            args, "inference", "sink_size", None, aliases=("inference_sink_size",)
        )
        if self.sink_size is None:
            _model_sink = getattr(args, "model_kwargs", {}).get("sink_size", None)
            if _model_sink is not None:
                self.sink_size = _model_sink
        if self.sink_size is None:
            self.sink_size = 0
        self.scene_cut_prefix = section_get(args, "inference", "scene_cut_prefix", DEFAULT_SCENE_CUT_PREFIX)
        self.multi_shot_sink = section_get(args, "inference", "multi_shot_sink", False)
        self.shot_clean_recache = section_get(args, "inference", "shot_clean_recache", False)
        self.global_sink_size = self.sink_size if self.multi_shot_sink else 0
        self.multi_shot_rope_offset = section_get(
            args,
            "inference",
            "multi_shot_rope_offset",
            0.0,
        )
        self.guidance_scale = section_get(args, "inference", "guidance_scale", getattr(args, "guidance_scale", 1.0))
        self.negative_prompt = section_get(args, "inference", "negative_prompt", getattr(args, "negative_prompt", ""))
        self.streaming_vae = section_get(args, "inference", "streaming_vae", getattr(args, "streaming_vae", False))
        self.async_vae = section_get(args, "inference", "async_vae", getattr(args, "async_vae", False))
        self.vae_decode_chunk_size = int(section_get(args, "inference", "vae_decode_chunk_size", 0))
        vae_device = section_get(args, "inference", "vae_device", getattr(args, "vae_device", None))
        self.vae_device = torch.device(vae_device) if vae_device else None

        # The legacy inference path remains untouched when this default-off flag
        # is absent/false.  Active sessions are constructed explicitly through
        # build_agent_session(), preventing an enabled flag from silently running
        # a non-transactional full-video generation.
        agent_runtime_raw = section_get(
            args, "inference", "agent_runtime", getattr(args, "agent_runtime", None)
        )
        self.agent_runtime_config = AgentRuntimeConfig.from_config(agent_runtime_raw)
        self.agent_runtime_enabled = self.agent_runtime_config.enabled
        if (
            self.agent_runtime_enabled
            and self.agent_runtime_config.require_streaming_vae_off
            and self.streaming_vae
        ):
            raise ValueError(
                "agent runtime v1 is shot-buffered and requires streaming_vae=false; "
                "strict streaming is a later degradation mode"
            )

        if self.quantize_kv:
            from utils.quant import LongLiveQuantizationConfig

            self.kv_quant_config = LongLiveQuantizationConfig(
                scale_rule=self.kv_quant_scale_rule,
                backend=self.kv_quant_backend,
                type="kv",
            )
        else:
            self.kv_quant_config = None
        self._dit_model.kv_quant_config = self.kv_quant_config

        if self.streaming_vae and self.vae_device is not None:
            vae_mode = "streaming-pipeline"
        elif self.streaming_vae and self.async_vae:
            vae_mode = "streaming-async"
        elif self.streaming_vae:
            vae_mode = "streaming"
        else:
            vae_mode = "batch"
        print(
            f"KV inference with {self.num_frame_per_block} frames per block "
            f"(kv_quant={self.quantize_kv}, vae_decode={vae_mode})"
        )

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.inference_t_scale = getattr(args, "inference_t_scale", None)
        self.use_relative_rope = getattr(args, "use_relative_rope", False)
        self._rope_method_override = getattr(args, "rope_method", None)
        self._original_seq_len_override = getattr(args, "original_seq_len", None)

        kv_rag_raw = section_get(args, "inference", "kv_rag", getattr(args, "kv_rag", None))
        self.kv_rag_config = KVRAGConfig.from_config(kv_rag_raw)
        self.kv_rag_enabled = self.kv_rag_config.enabled
        self.scene_memory_rolling = bool(self.kv_rag_config.scene_memory_rolling)
        self.scene_memory_injection_schedule = self.kv_rag_config.scene_memory_injection_schedule
        self.kv_rag_pos = KVRAGMemory(self.kv_rag_config) if self.kv_rag_enabled else None
        self.kv_rag_neg = KVRAGMemory(self.kv_rag_config) if self.kv_rag_enabled else None
        if self.kv_rag_enabled:
            print(f"[KV-RAG] {self.kv_rag_config.describe()}")

        # SDEdit-style masked renoise repair of one shot's chunks (see
        # docs/attention-downweighting-study.zh.md follow-up): after the normal
        # denoise of an edit-range chunk, add noise back to an intermediate
        # sigma inside the subject mask only and re-denoise with a repair
        # prompt; background latents are re-pinned to the first-pass result at
        # every step, and the clean recache stores the EDITED chunk so later
        # chunks inherit the repaired history. Inert unless configured.
        edit_renoise_raw = section_get(
            args, "inference", "edit_renoise", getattr(args, "edit_renoise", None)
        )
        self.edit_renoise = None
        if edit_renoise_raw is not None and bool(
            _plain_get(edit_renoise_raw, "enabled", False)
        ):
            self.edit_renoise = {
                "chunk_start": int(_plain_get(edit_renoise_raw, "chunk_start", -1)),
                "chunk_end": int(_plain_get(edit_renoise_raw, "chunk_end", -1)),
                "entry_index": int(_plain_get(edit_renoise_raw, "entry_index", 2)),
                "entry_sigma": _plain_get(edit_renoise_raw, "entry_sigma", None),
                "mask_path": str(_plain_get(edit_renoise_raw, "mask_path", "")),
                "mask_threshold": float(
                    _plain_get(edit_renoise_raw, "mask_threshold", 0.35)
                ),
                "mask_dilate": int(_plain_get(edit_renoise_raw, "mask_dilate", 1)),
                "prompt": str(_plain_get(edit_renoise_raw, "prompt", "")),
                "noise_seed": int(_plain_get(edit_renoise_raw, "noise_seed", 4242)),
                # cut-paste-SDEdit: initialize the masked region from a pasted
                # golden-subject latent instead of the drifted content, then
                # harmonize with a mid-sigma renoise.
                "collage": bool(_plain_get(edit_renoise_raw, "collage", False)),
                "donor_chunk": int(_plain_get(edit_renoise_raw, "donor_chunk", -1)),
                "donor_frame": int(_plain_get(edit_renoise_raw, "donor_frame", 0)),
                "donor_mask_path": str(_plain_get(edit_renoise_raw, "donor_mask_path", "")),
                "donor_flip": bool(_plain_get(edit_renoise_raw, "donor_flip", False)),
                # freeze the pasted donor core at its clean value (zero noise)
                # and repaint only the seam ring + rest of the mask
                "paste_pin": bool(_plain_get(edit_renoise_raw, "paste_pin", False)),
            }
            self._edit_masks = None
            self._edit_conditional_dict = None
            self._edit_donor = None
            print(f"[edit_renoise] {self.edit_renoise}")

        # Runtime-installed masked-renoise repair plan (agent edit_shot action).
        # Strictly inert unless set_runtime_edit_plan() is called; the static
        # edit_renoise config block above keeps priority and its exact legacy
        # behavior when both are present.
        self._runtime_edit_plan = None

        capsule_probe_raw = section_get(
            args, "inference", "capsule_probe", getattr(args, "capsule_probe", None)
        )
        capsule_edit_raw = section_get(
            args, "inference", "capsule_edit", getattr(args, "capsule_edit", None)
        )
        self.capsule_probe_config = CapsuleProbeConfig.from_config(capsule_probe_raw)
        self.capsule_edit_config = CapsuleEditConfig.from_config(capsule_edit_raw)
        self.capsule_runtime = None
        if self.capsule_probe_config.enabled or self.capsule_edit_config.enabled:
            latent_h = int(args.image_or_video_shape[-2])
            latent_w = int(args.image_or_video_shape[-1])
            self.capsule_runtime = CapsuleRuntime(
                self.capsule_probe_config,
                self.capsule_edit_config,
                frame_seqlen=self.frame_seq_length,
                grid_h=latent_h // 2,
                grid_w=latent_w // 2,
            )
            print(
                "[capsule] "
                f"probe={self.capsule_probe_config.enabled}, "
                f"edit={self.capsule_edit_config.enabled}, "
                f"grid={latent_h // 2}x{latent_w // 2}"
            )

    def build_agent_session(
        self,
        *,
        scene_dir,
        backend,
        controller,
        verifier,
        ledger_path,
        acquire_lock=True,
    ):
        """Create the opt-in transactional driver around a pipeline shot backend.

        The backend owns model-specific provisional generation and uses
        ``PipelineBoundaryAdapter`` for exact state capture.  Keeping construction
        explicit makes feature-off inference byte-for-byte identical and makes it
        impossible to mistake the legacy monolithic call for an agent session.
        """
        if not self.agent_runtime_enabled:
            raise ValueError("set inference.agent_runtime.enabled=true before building a session")
        scene = NormativeGraph.compile_scene(scene_dir)
        return CausalShotSession(
            self.agent_runtime_config,
            scene,
            backend,
            controller,
            verifier,
            ledger_path=ledger_path,
            acquire_lock=acquire_lock,
        )

    @property
    def _dit_model(self):
        """Return the underlying CausalWanModel, unwrapping PeftModel if present.

        After LoRA wrapping, ``self.generator.model`` is a PeftModel whose
        structure is PeftModel -> LoraModel (.base_model) -> CausalWanModel
        (.model).  Direct attribute writes on PeftModel do NOT propagate to
        CausalWanModel, so any runtime overrides (t_scale, rope_method, …)
        must target the unwrapped model returned by this property.
        """
        model = self.generator.model
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            return model.base_model.model
        return model

    @property
    def _scene_memory_active(self) -> bool:
        """True when the persistent scene-memory partition is in use."""
        return self.kv_rag_enabled and self.kv_rag_config.scene_memory_enabled

    def _reset_kv_rag(self):
        if not self.kv_rag_enabled:
            return
        if self.kv_rag_pos is not None:
            self.kv_rag_pos.clear()
        if self.kv_rag_neg is not None:
            self.kv_rag_neg.clear()

    def _reset_kv_rag_keep_scene(self):
        """Reset per-video transient KV-RAG state but KEEP the persistent scene
        anchors, so a new perspective video of the same scene retrieves from the
        accumulated scene memory. Falls back to a full reset when scene memory is
        not active (nothing to preserve)."""
        if not self._scene_memory_active:
            self._reset_kv_rag()
            return
        for bank in (self.kv_rag_pos, self.kv_rag_neg):
            if bank is not None:
                bank.reset_shot()          # clear per-shot partition only
                bank.set_boundary_inject(False)
                # The scene anchors now belong to a PRIOR perspective's token clock
                # (this perspective restarts at token 0), so they must skip the
                # live-window overlap dedup or the boundary force-inject filters them.
                bank.mark_scene_cross_clock()

    def _reset_kv_rag_shot(self):
        """Drop the per-shot KV-RAG partition at a shot boundary, keeping the
        persistent scene anchors. No-op unless scene memory is active."""
        if not self._scene_memory_active:
            return
        if self.kv_rag_pos is not None:
            self.kv_rag_pos.reset_shot()
        if self.kv_rag_neg is not None:
            self.kv_rag_neg.reset_shot()

    def _set_kv_rag_boundary_inject(self, active: bool):
        """Toggle boundary force-injection of scene anchors on both banks."""
        if not self._scene_memory_active:
            return
        if self.kv_rag_pos is not None:
            self.kv_rag_pos.set_boundary_inject(active)
        if self.kv_rag_neg is not None:
            self.kv_rag_neg.set_boundary_inject(active)

    def _decode_vae_chunk_bcthw(self, chunk_bcthw, vae_scale):
        """Decode one latent chunk to pixel space as [B, C, T, H, W].

        LightVAE exposes ``cached_decode`` for streaming. The default Wan VAE
        does not, so fall back to the wrapper's chunked decode API without the
        cache. This keeps long Wan-VAE videos off the final full-video decode
        path that OOMs on 384-frame runs.
        """
        if hasattr(self.vae.model, "cached_decode"):
            return self.vae.model.cached_decode(chunk_bcthw, vae_scale).float().clamp_(-1, 1)
        chunk_btchw = chunk_bcthw.permute(0, 2, 1, 3, 4).contiguous()
        chunk_size = self.vae_decode_chunk_size
        if chunk_size <= 0:
            chunk_size = int(chunk_btchw.shape[1])
        decoded_btchw = self.vae.decode_to_pixel_chunk(
            chunk_btchw,
            use_cache=False,
            chunk_size=max(1, chunk_size),
        )
        return decoded_btchw.permute(0, 2, 1, 3, 4).contiguous()

    def _set_kv_rag_context_key(self, conditional_dict):
        """Supply the per-perspective caption-text key for the semantic key mode.

        For ``retrieval_key_mode == "semantic"`` the retrieval index is the SCENE
        each perspective DESCRIBES -- the conditional (scene-caption) text
        embedding, reused (no new backbone) and mean-pooled to one vector. We key
        BOTH the conditional and the unconditional (CFG) banks on this SAME scene
        caption: the unconditional negative prompt is scene-independent and
        identical across perspectives, so keying ``kv_rag_neg`` on it would give
        every negative entry one shared, scene-blind summary that cannot match the
        current scene (and would tie-select unrelated/boosted entries). Inert for
        every other key mode (the default path is untouched).
        """
        if not self.kv_rag_enabled or self.kv_rag_config.retrieval_key_mode != "semantic":
            return
        scene_embeds = conditional_dict.get("prompt_embeds") if conditional_dict is not None else None
        if scene_embeds is None:
            return
        if self.kv_rag_pos is not None:
            self.kv_rag_pos.set_context_key(scene_embeds)
        if self.kv_rag_neg is not None:
            self.kv_rag_neg.set_context_key(scene_embeds)

    @staticmethod
    def _memory_boundary_active(
        is_shot_boundary,
        force_scene_memory_boundary,
        chunk_index,
        scene_memory_injection_schedule: str = "boundary",
    ) -> bool:
        """Whether the scene anchors should be force-injected for this chunk.

        Fires on an intra-video shot cut (``is_shot_boundary``) OR on the first
        chunk of a NEW perspective in the per-perspective path
        (``force_scene_memory_boundary and chunk_index == 0``). The latter is the
        fix for `boundary_injections == 0`: a new perspective is a separate
        ``inference()`` call whose chunk 0 is never a prompt shot-cut, so without
        this the perspective-0 anchors were never force-injected.
        """
        schedule = str(scene_memory_injection_schedule or "boundary").lower().replace("-", "_")
        if schedule == "every_chunk":
            return True
        return bool(is_shot_boundary or (force_scene_memory_boundary and chunk_index == 0))

    @staticmethod
    def _should_store_persistent(
        scene_memory_active,
        seed_scene_memory,
        scene_shot_index,
        *,
        scene_memory_rolling: bool = False,
        is_shot_end: bool = False,
    ) -> bool:
        """Whether this chunk's clean-recache slice seeds the PERSISTENT partition.

        Only the reference perspective (``seed_scene_memory``, i.e. perspective 0)
        at its first shot seeds the scene anchors; later perspectives retrieve /
        force-inject them but store only transient per-shot entries (so generated
        later views do not continuously reseed the partition).
        """
        if not (scene_memory_active and seed_scene_memory):
            return False
        if scene_memory_rolling:
            return bool(is_shot_end)
        return scene_shot_index == 0

    def _kv_rag_call_kwargs(
        self,
        bank,
        *,
        retrieve: bool,
        store: bool = False,
        chunk_index: int | None = None,
        shot_index: int | None = None,
        phase: str | None = None,
        persistent: bool = False,
    ):
        kwargs = {}
        if self.capsule_runtime is not None:
            self.capsule_runtime.set_runtime_context(
                chunk_index=chunk_index, shot_index=shot_index, phase=phase
            )
            kwargs["capsule_runtime"] = self.capsule_runtime
        if not self.kv_rag_enabled or bank is None:
            return kwargs
        bank.set_runtime_context(chunk_index=chunk_index, shot_index=shot_index, phase=phase)
        kwargs.update({
            "kv_rag": bank,
            "kv_rag_retrieve": bool(retrieve),
            "kv_rag_store": bool(store),
            "kv_rag_meta": {
                "chunk_index": chunk_index,
                "shot_index": shot_index,
                "phase": phase,
                "persistent": bool(persistent),
            },
        })
        return kwargs

    def kv_rag_diagnostics(self) -> dict[str, object]:
        """JSON-serializable KV-RAG diagnostics for the last inference call."""
        if not self.kv_rag_enabled:
            return {}
        return {
            "config": {
                "scene_memory_enabled": bool(self.kv_rag_config.scene_memory_enabled),
                "scene_memory_rolling": bool(self.kv_rag_config.scene_memory_rolling),
                "scene_memory_injection_schedule": self.kv_rag_config.scene_memory_injection_schedule,
                "boundary_inject_anchors": int(self.kv_rag_config.boundary_inject_anchors),
                "retrieval_key_mode": self.kv_rag_config.retrieval_key_mode,
                "retrieval_value_mode": self.kv_rag_config.retrieval_value_mode,
                "attention_diagnostic": bool(self.kv_rag_config.attention_diagnostic),
            },
            "pos": self.kv_rag_pos.export_diagnostics() if self.kv_rag_pos is not None else {},
            "neg": self.kv_rag_neg.export_diagnostics() if self.kv_rag_neg is not None else {},
        }

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
        preserve_scene_memory: bool = False,
        seed_scene_memory: bool = True,
        force_scene_memory_boundary: bool = False,
        sample_name: str | None = None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts[0]
        )
        conditional_dict_list = [
            {"prompt_embeds": conditional_dict["prompt_embeds"][i:i+1]}
            for i in range(conditional_dict["prompt_embeds"].shape[0])
        ]
        use_cfg = self.guidance_scale != 1.0
        if use_cfg:
            unconditional_dict = self.text_encoder(
                text_prompts=[self.negative_prompt] * batch_size
            )
        else:
            unconditional_dict = None

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        # Cross-perspective protocol: when generating one video per perspective of
        # the same scene, keep the persistent scene anchors across perspectives so
        # later views retrieve the established scene; reset everything otherwise.
        if preserve_scene_memory and self._scene_memory_active:
            # Keep the persistent scene anchors; the new perspective retrieves them
            # during denoise (they are causality-exempt and up-weighted by
            # scene_score_bonus), conditioning this viewpoint on the established scene.
            self._reset_kv_rag_keep_scene()
        else:
            self._reset_kv_rag()

        # Supply this perspective's caption-text key when the semantic key mode is
        # active (no-op otherwise). Set AFTER the scene reset so it is not cleared.
        self._set_kv_rag_context_key(conditional_dict)

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                if use_cfg:
                    self.crossattn_cache_neg[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["pinned_start"].fill_(-1)
                self.kv_cache_pos[block_index]["pinned_len"].zero_()
                if "pinned_abs_start" in self.kv_cache_pos[block_index]:
                    self.kv_cache_pos[block_index]["pinned_abs_start"].fill_(-1)
                    self.kv_cache_pos[block_index]["pinned_abs_len"].zero_()
                if use_cfg:
                    self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_neg[block_index]["pinned_start"].fill_(-1)
                    self.kv_cache_neg[block_index]["pinned_len"].zero_()
                    if "pinned_abs_start" in self.kv_cache_neg[block_index]:
                        self.kv_cache_neg[block_index]["pinned_abs_start"].fill_(-1)
                        self.kv_cache_neg[block_index]["pinned_abs_len"].zero_()

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0

        # Save model state before overriding for inference.
        # Use _dit_model to reach the real CausalWanModel (PeftModel wrapping
        # intercepts attribute writes, so self.generator.model.xxx would land
        # on the wrapper instead of the model that reads them in forward()).
        dit = self._dit_model
        prev_local_attn_size = dit.local_attn_size
        prev_t_scale = getattr(dit, 't_scale', 1.0)
        prev_rope_method = getattr(dit, 'rope_method', 'linear')
        prev_original_seq_len = getattr(dit, 'original_seq_len', None)
        prev_use_relative_rope = getattr(dit, 'use_relative_rope', False)
        prev_rope_temporal_offset = getattr(dit, 'rope_temporal_offset', 0.0)
        prev_max_attention_sizes = {}
        prev_sink_sizes = {}
        prev_global_sink_sizes = {}
        for name, module in self.generator.model.named_modules():
            if hasattr(module, 'max_attention_size'):
                prev_max_attention_sizes[name] = module.max_attention_size
            if hasattr(module, 'sink_size'):
                prev_sink_sizes[name] = module.sink_size
            if hasattr(module, 'global_sink_size'):
                prev_global_sink_sizes[name] = module.global_sink_size

        dit.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {dit.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if self.sink_size is not None:
            self._set_all_modules_sink_size(self.sink_size)
            print(f"[inference] sink_size set to: {self.sink_size}"
                  f"{', multi_shot_sink enabled (pinned position)' if self.multi_shot_sink else ''}"
                  f"{', shot_clean_recache enabled' if self.shot_clean_recache else ''}")

        # Propagate the internally derived global sink length.
        self._set_all_modules_global_sink_size(self.global_sink_size)
        if self.global_sink_size and self.global_sink_size > 0:
            print(f"[inference] auto_global_sink_size set to: {self.global_sink_size} "
                  f"(first {self.global_sink_size} frames permanently anchored)")

        if self.inference_t_scale is not None:
            dit.t_scale = self.inference_t_scale
            print(f"[inference] t_scale overridden to: {dit.t_scale}")

        if self._rope_method_override is not None:
            dit.rope_method = self._rope_method_override
        if self._original_seq_len_override is not None:
            dit.original_seq_len = self._original_seq_len_override
        print(f"[inference] rope_method={dit.rope_method}, "
              f"original_seq_len={dit.original_seq_len}")

        dit.use_relative_rope = self.use_relative_rope
        if self.use_relative_rope:
            print(f"[inference] use_relative_rope enabled")

        dit.rope_temporal_offset = 0.0
        if self.multi_shot_rope_offset != 0.0:
            print(f"[inference] multi_shot_rope_offset={self.multi_shot_rope_offset} "
                  f"(multi-shot RoPE offset enabled)")

        capsule_error = None
        if self.capsule_runtime is not None:
            raw_for_manifest = (
                text_prompts[0]
                if isinstance(text_prompts[0], (list, tuple))
                else text_prompts
            )
            self.capsule_runtime.start_sample(
                sample_name, int(getattr(self.args, "seed", 0)), raw_for_manifest
            )
        try:
            raw_prompts = text_prompts[0] if isinstance(text_prompts[0], (list, tuple)) else text_prompts
            return self._inference_inner(
                noise=noise, batch_size=batch_size, num_frames=num_frames,
                num_channels=num_channels, height=height, width=width,
                num_blocks=num_blocks, num_input_frames=num_input_frames,
                num_output_frames=num_output_frames, output=output,
                conditional_dict=conditional_dict,
                conditional_dict_list=conditional_dict_list,
                unconditional_dict=unconditional_dict,
                use_cfg=use_cfg, initial_latent=initial_latent,
                return_latents=return_latents,
                current_start_frame=current_start_frame,
                cache_start_frame=cache_start_frame,
                raw_prompts=raw_prompts,
                seed_scene_memory=seed_scene_memory,
                force_scene_memory_boundary=force_scene_memory_boundary,
            )
        except Exception as exc:
            capsule_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if self.capsule_runtime is not None:
                self.capsule_runtime.finalize_sample(capsule_error)
            dit.local_attn_size = prev_local_attn_size
            dit.t_scale = prev_t_scale
            dit.rope_method = prev_rope_method
            dit.original_seq_len = prev_original_seq_len
            dit.use_relative_rope = prev_use_relative_rope
            dit.rope_temporal_offset = prev_rope_temporal_offset
            for name, module in self.generator.model.named_modules():
                if name in prev_max_attention_sizes:
                    try:
                        module.max_attention_size = prev_max_attention_sizes[name]
                    except Exception:
                        pass
                if name in prev_sink_sizes:
                    try:
                        module.sink_size = prev_sink_sizes[name]
                    except Exception:
                        pass
                if name in prev_global_sink_sizes:
                    try:
                        module.global_sink_size = prev_global_sink_sizes[name]
                    except Exception:
                        pass

    def _inference_inner(
        self, noise, batch_size, num_frames, num_channels, height, width,
        num_blocks, num_input_frames, num_output_frames, output,
        conditional_dict, conditional_dict_list, unconditional_dict,
        use_cfg, initial_latent, return_latents,
        current_start_frame, cache_start_frame,
        raw_prompts=None,
        seed_scene_memory=True,
        force_scene_memory_boundary=False,
    ):

        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    **self._kv_rag_call_kwargs(
                        self.kv_rag_pos,
                        retrieve=self.kv_rag_config.retrieve_during_recache,
                        store=self.kv_rag_config.store_after_recache,
                        chunk_index=-1,
                        phase="initial_first_frame",
                    ),
                )
                if use_cfg:
                    self.generator(
                        noisy_image_or_video=initial_latent[:, :1],
                        conditional_dict=unconditional_dict,
                        timestep=timestep * 0,
                        kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length,
                        **self._kv_rag_call_kwargs(
                            self.kv_rag_neg,
                            retrieve=self.kv_rag_config.retrieve_during_recache,
                            store=self.kv_rag_config.store_after_recache,
                            chunk_index=-1,
                            phase="initial_first_frame_uncond",
                        ),
                    )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    **self._kv_rag_call_kwargs(
                        self.kv_rag_pos,
                        retrieve=self.kv_rag_config.retrieve_during_recache,
                        store=self.kv_rag_config.store_after_recache,
                        chunk_index=-1 - block_index,
                        phase="initial_context",
                    ),
                )
                if use_cfg:
                    self.generator(
                        noisy_image_or_video=current_ref_latents,
                        conditional_dict=unconditional_dict,
                        timestep=timestep * 0,
                        kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length,
                        **self._kv_rag_call_kwargs(
                            self.kv_rag_neg,
                            retrieve=self.kv_rag_config.retrieve_during_recache,
                            store=self.kv_rag_config.store_after_recache,
                            chunk_index=-1 - block_index,
                            phase="initial_context_uncond",
                        ),
                    )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        # Multi-shot RoPE offset: track current shot index for phase offset.
        current_shot_index = 0
        # Scene-memory shot index: incremented at every shot boundary regardless
        # of the RoPE-offset setting; chunk shot 0 supplies the persistent anchors.
        scene_shot_index = 0
        phi = self.multi_shot_rope_offset
        self._dit_model.rope_temporal_offset = 0.0
        streaming_decode = self.streaming_vae and not return_latents
        pipeline_vae = streaming_decode and self.vae_device is not None
        async_vae = streaming_decode and self.async_vae and not pipeline_vae
        if streaming_decode:
            vae_dev = self.vae_device if pipeline_vae else noise.device
            vae_scale = [
                self.vae.mean.to(device=vae_dev, dtype=noise.dtype),
                1.0 / self.vae.std.to(device=vae_dev, dtype=noise.dtype),
            ]
            self.vae.model.clear_cache()
            video_chunks = []
            if async_vae:
                vae_stream = torch.cuda.Stream(device=noise.device)
                prev_vae_done = None
            if pipeline_vae:
                vae_thread_error = []
                vae_thread_chunks = []
                vae_work_queue = []
                vae_queue_lock = threading.Lock()
                vae_work_ready = threading.Event()
                vae_all_done = threading.Event()

                def _vae_thread_fn():
                    try:
                        while True:
                            vae_work_ready.wait()
                            vae_work_ready.clear()
                            while True:
                                with vae_queue_lock:
                                    if not vae_work_queue:
                                        break
                                    item = vae_work_queue.pop(0)
                                if item is None:
                                    vae_all_done.set()
                                    return
                                decoded = self._decode_vae_chunk_bcthw(item, vae_scale)
                                # Pinned-memory DtoH: pageable copy hits ~0.2 GB/s
                                # (1.5s per 313MB chunk → ~80s/prompt at end); pinned
                                # path runs at PCIe limit (~25 GB/s = ~12ms / chunk).
                                if decoded.device.type == "cpu":
                                    vae_thread_chunks.append(decoded)
                                else:
                                    pinned = torch.empty(
                                        decoded.shape, dtype=decoded.dtype,
                                        device="cpu", pin_memory=True,
                                    )
                                    pinned.copy_(decoded, non_blocking=True)
                                    torch.cuda.synchronize(decoded.device)
                                    vae_thread_chunks.append(pinned)
                    except Exception as exc:
                        vae_thread_error.append(exc)
                        vae_all_done.set()

                vae_bg_thread = threading.Thread(target=_vae_thread_fn, daemon=True)
                vae_bg_thread.start()

        _block_events = [] if _LLV2_TIME else None
        global _LLV2_PROFILE_CALL_COUNTER
        _call_idx = _LLV2_PROFILE_CALL_COUNTER
        _LLV2_PROFILE_CALL_COUNTER += 1
        _prof = None
        _prof_trace_path = None
        if _LLV2_PROFILE_SPEC and _LLV2_PROFILE_OUTPUT_DIR:
            _parts = _LLV2_PROFILE_SPEC.split(":")
            _target_call = int(_parts[0]) if len(_parts) > 0 else 0
            _wait_n = int(_parts[1]) if len(_parts) > 1 else 20
            _warmup_n = int(_parts[2]) if len(_parts) > 2 else 2
            _active_n = int(_parts[3]) if len(_parts) > 3 else 2
            if _call_idx == _target_call:
                from torch.profiler import profile as _tp_profile, ProfilerActivity, schedule
                os.makedirs(_LLV2_PROFILE_OUTPUT_DIR, exist_ok=True)
                _prof_trace_path = os.path.join(
                    _LLV2_PROFILE_OUTPUT_DIR, f"trace_call{_call_idx}.json"
                )
                _prof = _tp_profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=schedule(
                        wait=_wait_n, warmup=_warmup_n, active=_active_n, repeat=1
                    ),
                    record_shapes=False,
                    with_stack=False,
                )
                _prof.start()
                print(
                    f"[LLV2_PROFILE] enabled for call={_call_idx} "
                    f"wait={_wait_n} warmup={_warmup_n} active={_active_n} "
                    f"-> {_prof_trace_path}",
                    flush=True,
                )
        for chunk_index, current_num_frames in enumerate(all_num_frames):
            if _LLV2_TIME:
                _ev_s = torch.cuda.Event(enable_timing=True)
                _ev_e = torch.cuda.Event(enable_timing=True)
                _ev_s.record()
            conditional_dict = conditional_dict_list[chunk_index]
            # Reset the cross-attention cache when each chunk uses a different
            # prompt; otherwise the model reuses the previous chunk's k/v and
            # ignores the current conditional_dict.
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False

            # Update RoPE phase offset on shot boundaries.
            is_shot_boundary = self._is_shot_boundary(raw_prompts, chunk_index)
            if is_shot_boundary:
                scene_shot_index += 1
            if self.capsule_runtime is not None:
                self.capsule_runtime.set_runtime_context(
                    chunk_index=chunk_index,
                    shot_index=scene_shot_index,
                    phase="boundary_setup",
                )
                self.capsule_runtime.runtime["num_frames"] = int(current_num_frames)
                self.capsule_runtime.runtime["start_token"] = int(
                    current_start_frame * self.frame_seq_length
                )
                current_prompt = (
                    raw_prompts[chunk_index]
                    if isinstance(raw_prompts, (list, tuple))
                    else str(raw_prompts)
                )
                self.capsule_runtime.set_chunk_prompt(
                    current_prompt, self.text_encoder.tokenizer
                )
                if is_shot_boundary:
                    edit_record = self.capsule_runtime.edit_active_window(
                        self.kv_cache_pos,
                        boundary_shot=scene_shot_index,
                        local_attn_frames=self.local_attn_size,
                        global_sink_frames=self.global_sink_size,
                    )
                    if edit_record is not None:
                        print(f"[capsule] boundary edit: {edit_record}")
            is_shot_end = self._is_shot_end(raw_prompts, chunk_index, len(all_num_frames))
            if is_shot_boundary and phi != 0.0:
                current_shot_index += 1
                self._dit_model.rope_temporal_offset = current_shot_index * phi
                print(f"[inference] multi-shot RoPE: shot_index={current_shot_index}, "
                      f"temporal_offset={self._dit_model.rope_temporal_offset:.4f}")

            # Force-inject the persistent scene anchors on an intra-video shot cut
            # OR on the first chunk of a new perspective (per-perspective path),
            # so the perspective change keeps the established scene.
            is_memory_boundary = self._memory_boundary_active(
                is_shot_boundary,
                force_scene_memory_boundary,
                chunk_index,
                self.scene_memory_injection_schedule,
            )
            self._set_kv_rag_boundary_inject(is_memory_boundary)
            # A new shot drops the per-shot framing memory while the persistent
            # scene anchors survive. Tied only to the intra-video shot cut (not the
            # perspective boundary, which already reset the per-shot partition via
            # _reset_kv_rag_keep_scene) so stale per-shot entries never bleed in.
            if is_shot_boundary:
                self._reset_kv_rag_shot()
            # Only the reference perspective (perspective 0) at shot 0 seeds the
            # persistent anchors; later perspectives retrieve/force-inject but store
            # transient per-shot entries, so generated later views never reseed it.
            store_persistent = self._should_store_persistent(
                self._scene_memory_active,
                seed_scene_memory,
                scene_shot_index,
                scene_memory_rolling=self.scene_memory_rolling,
                is_shot_end=is_shot_end,
            )

            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    **self._kv_rag_call_kwargs(
                        self.kv_rag_pos,
                        retrieve=self.kv_rag_config.retrieve_during_denoise,
                        chunk_index=chunk_index,
                        shot_index=scene_shot_index,
                        phase="denoise",
                    ),
                )
                if use_cfg:
                    flow_pred_uncond, _ = self.generator(
                        noisy_image_or_video=latent_model_input,
                        conditional_dict=unconditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length,
                        **self._kv_rag_call_kwargs(
                            self.kv_rag_neg,
                            retrieve=self.kv_rag_config.retrieve_during_denoise,
                            chunk_index=chunk_index,
                            shot_index=scene_shot_index,
                            phase="denoise_uncond",
                        ),
                    )
                    flow_pred = flow_pred_uncond + self.guidance_scale * (
                        flow_pred_cond - flow_pred_uncond)
                else:
                    flow_pred = flow_pred_cond

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0
                # iter-34: removed per-step debug print of kv_cache scalar
                # tensors (was forcing GPU→CPU sync every sampling step ×
                # 4 steps × 48 chunks = 192 stalls per prompt). Re-enable
                # behind LLV2_DEBUG_KV=1 if needed for debugging.
                if os.environ.get("LLV2_DEBUG_KV", "0") == "1":
                    print(f"kv_cache['local_end_index']: {self.kv_cache_pos[0]['local_end_index']}")
                    print(f"kv_cache['global_end_index']: {self.kv_cache_pos[0]['global_end_index']}")

            # cut-paste donor capture: stash the golden chunk's clean latent
            # as it flows past, for later collage initialization.
            if (
                self.edit_renoise is not None
                and self.edit_renoise["collage"]
                and chunk_index == self.edit_renoise["donor_chunk"]
            ):
                self._edit_donor = latents.detach().clone()
                print(f"[edit_renoise] captured donor latent from chunk {chunk_index}")

            # SDEdit repair pass: masked renoise + re-denoise with the repair
            # prompt. Replaces `latents` so both the decoded output (3.2) and
            # the clean recache (3.3) carry the edited chunk. The active config
            # is the static edit_renoise block (legacy, priority) or a
            # runtime-installed repair plan (agent edit_shot action).
            _edit_cfg = self._active_edit_cfg(chunk_index)
            if _edit_cfg is not None:
                latents = self._apply_edit_renoise(
                    latents,
                    sample_scheduler=sample_scheduler,
                    chunk_index=chunk_index,
                    scene_shot_index=scene_shot_index,
                    current_num_frames=current_num_frames,
                    current_start_frame=current_start_frame,
                    cache_start_frame=cache_start_frame,
                    batch_size=batch_size,
                    cfg=_edit_cfg,
                )

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # The boundary scene-anchor pulse applies only to this chunk's
            # denoising; clear it so the clean recache below never force-injects.
            self._set_kv_rag_boundary_inject(False)

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            is_scene_cut = self._is_scene_cut(raw_prompts, chunk_index)

            if is_scene_cut and self.shot_clean_recache:
                print(f"[inference] Scene cut at chunk {chunk_index}, zeroing KV before recache")
                current_start_tokens = current_start_frame * self.frame_seq_length
                self._zero_kv_data(self.kv_cache_pos, current_start_tokens)
                if use_cfg:
                    self._zero_kv_data(self.kv_cache_neg, current_start_tokens)

            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                **self._kv_rag_call_kwargs(
                    self.kv_rag_pos,
                    retrieve=self.kv_rag_config.retrieve_during_recache,
                    store=self.kv_rag_config.store_after_recache,
                    chunk_index=chunk_index,
                    shot_index=scene_shot_index,
                    phase="clean_recache",
                    persistent=store_persistent,
                ),
            )
            if use_cfg:
                self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    **self._kv_rag_call_kwargs(
                        self.kv_rag_neg,
                        retrieve=self.kv_rag_config.retrieve_during_recache,
                        store=self.kv_rag_config.store_after_recache,
                        chunk_index=chunk_index,
                        shot_index=scene_shot_index,
                        phase="clean_recache_uncond",
                        persistent=store_persistent,
                    ),
                )

            # Step 3.3b: pin the current chunk for multi-shot sink on scene cut.
            if is_scene_cut:
                print(f"[inference] Scene cut at chunk {chunk_index}, pinning chunk as shot-sink")
                pinned_abs_start_tokens = current_start_frame * self.frame_seq_length
                self._pin_current_chunk(
                    self.kv_cache_pos, current_num_frames, pinned_abs_start_tokens
                )
                if use_cfg:
                    self._pin_current_chunk(
                        self.kv_cache_neg, current_num_frames, pinned_abs_start_tokens
                    )

            if streaming_decode:
                if async_vae:
                    diffusion_done = torch.cuda.Event()
                    diffusion_done.record()
                    if prev_vae_done is not None:
                        prev_vae_done.synchronize()
                    with torch.cuda.stream(vae_stream):
                        vae_stream.wait_event(diffusion_done)
                        chunk_bcthw = latents.permute(0, 2, 1, 3, 4).contiguous()
                        decoded_chunk = self._decode_vae_chunk_bcthw(chunk_bcthw, vae_scale)
                        video_chunks.append(decoded_chunk)
                    prev_vae_done = torch.cuda.Event()
                    prev_vae_done.record(vae_stream)
                elif pipeline_vae:
                    latent_on_vae = latents.permute(0, 2, 1, 3, 4).contiguous().to(vae_dev)
                    with vae_queue_lock:
                        vae_work_queue.append(latent_on_vae)
                    vae_work_ready.set()
                else:
                    chunk_bcthw = latents.permute(0, 2, 1, 3, 4).contiguous()
                    decoded_chunk = self._decode_vae_chunk_bcthw(chunk_bcthw, vae_scale)
                    video_chunks.append(decoded_chunk.cpu())
                    del decoded_chunk, chunk_bcthw
                    torch.cuda.empty_cache()

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

            if _LLV2_TIME:
                _ev_e.record()
                _block_events.append((_ev_s, _ev_e))
            if _prof is not None:
                _prof.step()

        if _prof is not None:
            _prof.stop()
            _prof.export_chrome_trace(_prof_trace_path)
            print(f"[LLV2_PROFILE] saved trace -> {_prof_trace_path}", flush=True)

        if _LLV2_TIME and _block_events:
            torch.cuda.synchronize()
            _times = [_s.elapsed_time(_e) for _s, _e in _block_events]
            _sorted = sorted(_times)
            _n = len(_sorted)
            _p10 = _sorted[max(0, _n // 10 - 1)]
            _p50 = statistics.median(_sorted)
            _p90 = _sorted[max(0, _n - _n // 10 - 1)]
            _mean = sum(_times) / _n
            _total = sum(_times)
            print(
                f"[LLV2_TIME] blocks={_n} mean_ms={_mean:.2f} p10_ms={_p10:.2f} "
                f"median_ms={_p50:.2f} p90_ms={_p90:.2f} total_ms={_total:.2f}",
                flush=True,
            )

        if _LLV2_DUMP_LATENT_DIR:
            os.makedirs(_LLV2_DUMP_LATENT_DIR, exist_ok=True)
            _existing = sum(
                1 for _f in os.listdir(_LLV2_DUMP_LATENT_DIR)
                if _f.startswith("latent_") and _f.endswith(".pt")
            )
            _path = os.path.join(_LLV2_DUMP_LATENT_DIR, f"latent_{_existing:04d}.pt")
            torch.save(output.detach().cpu(), _path)
            print(f"[LLV2_DUMP] saved latent {tuple(output.shape)} -> {_path}", flush=True)

        if self.kv_rag_enabled and self.kv_rag_config.verbose:
            if self.kv_rag_pos is not None:
                print(self.kv_rag_pos.format_stats("KV-RAG pos"), flush=True)
            if use_cfg and self.kv_rag_neg is not None:
                print(self.kv_rag_neg.format_stats("KV-RAG neg"), flush=True)

        # Step 4: Decode the output


        if return_latents:
            return output
        elif streaming_decode:
            if async_vae:
                vae_stream.synchronize()
            elif pipeline_vae:
                with vae_queue_lock:
                    vae_work_queue.append(None)
                vae_work_ready.set()
                vae_all_done.wait()
                vae_bg_thread.join()
                if vae_thread_error:
                    raise RuntimeError(
                        f"[pipeline_vae] VAE decode failed: {vae_thread_error[0]}"
                    ) from vae_thread_error[0]
                video_chunks = vae_thread_chunks
            video_bcthw = torch.cat(video_chunks, dim=2)
            video = video_bcthw.permute(0, 2, 1, 3, 4)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            self.vae.model.clear_cache()
            return video
        else:
            video = self.vae.decode_to_pixel(output)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_pos = []
        kv_cache_neg = []
        num_heads = wan_default_config[self.model_name]["num_heads"]
        head_dim = wan_default_config[self.model_name]["head_dim"]
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 3 * self.num_frame_per_block * self.frame_seq_length

        block_token_size = self.num_frame_per_block * self.frame_seq_length
        max_blocks = kv_cache_size // block_token_size

        if self.quantize_kv:
            from utils.quant import clone_quantized_tensor, quantize_to_fp4

            print(
                f"[KV Cache] Quantized (nvfp4): block_token_size={block_token_size}, "
                f"max_blocks={max_blocks}, num_heads={num_heads}, layers={self.num_transformer_blocks}"
            )
            zero_block = torch.zeros(
                [block_token_size * num_heads, head_dim],
                dtype=dtype,
                device=device,
            )
            zero_qt = quantize_to_fp4(zero_block, self.kv_quant_config)

        for _ in range(self.num_transformer_blocks):
            if self.quantize_kv:
                kv_cache_pos.append({
                    "k": [clone_quantized_tensor(zero_qt) for _ in range(max_blocks)],
                    "v": [clone_quantized_tensor(zero_qt) for _ in range(max_blocks)],
                    "quantized": True,
                    "block_token_size": block_token_size,
                    "max_blocks": max_blocks,
                    "num_heads": num_heads,
                    "num_filled_blocks": 0,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
                    # Absolute (un-rolled) token provenance of the pinned chunk,
                    # used by KV-RAG to avoid re-injecting frames already pinned.
                    "pinned_abs_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_abs_len": torch.tensor([0], dtype=torch.long, device=device),
                })
                kv_cache_neg.append({
                    "k": [clone_quantized_tensor(zero_qt) for _ in range(max_blocks)],
                    "v": [clone_quantized_tensor(zero_qt) for _ in range(max_blocks)],
                    "quantized": True,
                    "block_token_size": block_token_size,
                    "max_blocks": max_blocks,
                    "num_heads": num_heads,
                    "num_filled_blocks": 0,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
                    # Absolute (un-rolled) token provenance of the pinned chunk,
                    # used by KV-RAG to avoid re-injecting frames already pinned.
                    "pinned_abs_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_abs_len": torch.tensor([0], dtype=torch.long, device=device),
                })
            else:
                kv_cache_pos.append({
                    "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "quantized": False,
                    "block_token_size": block_token_size,
                    "max_blocks": max_blocks,
                    "num_heads": num_heads,
                    "num_filled_blocks": 0,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
                    # Absolute (un-rolled) token provenance of the pinned chunk,
                    # used by KV-RAG to avoid re-injecting frames already pinned.
                    "pinned_abs_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_abs_len": torch.tensor([0], dtype=torch.long, device=device),
                })
                kv_cache_neg.append({
                    "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "quantized": False,
                    "block_token_size": block_token_size,
                    "max_blocks": max_blocks,
                    "num_heads": num_heads,
                    "num_filled_blocks": 0,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
                    # Absolute (un-rolled) token provenance of the pinned chunk,
                    # used by KV-RAG to avoid re-injecting frames already pinned.
                    "pinned_abs_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_abs_len": torch.tensor([0], dtype=torch.long, device=device),
                })

        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        num_heads = wan_default_config[self.model_name]["num_heads"]
        head_dim = wan_default_config[self.model_name]["head_dim"]
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    def clear_cache(self):
        """
        Explicitly release large KV / cross-attention caches to free GPU memory.
        Safe to call between independent inference calls; caches will be
        re-created on demand by _initialize_kv_cache/_initialize_crossattn_cache.
        """
        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self._reset_kv_rag()

    def _active_edit_cfg(self, chunk_index):
        """Edit config active for this chunk, or None.

        The static ``inference.edit_renoise`` block keeps priority and its
        legacy predicate exactly; only when it does not claim the chunk may a
        runtime-installed plan (agent edit_shot) apply.
        """
        if (
            self.edit_renoise is not None
            and self.edit_renoise["chunk_start"] <= chunk_index <= self.edit_renoise["chunk_end"]
        ):
            return self.edit_renoise
        plan = self._runtime_edit_plan
        if plan is not None and plan["chunk_start"] <= chunk_index <= plan["chunk_end"]:
            return plan
        return None

    def set_runtime_edit_plan(
        self,
        *,
        chunk_start,
        chunk_end,
        masks,
        prompt,
        entry_sigma=1.0,
        mask_threshold=0.35,
        mask_dilate=1,
        noise_seed=4242,
    ):
        """Install a masked-renoise repair plan for the next generation pass.

        Runtime counterpart of the static ``inference.edit_renoise`` config
        block, used by the agent edit_shot action: restore the boundary
        snapshot, install the plan, replay the shot (deterministic pass-1
        denoise, then per edited chunk: renoise inside the subject mask to
        ``entry_sigma``, re-denoise with the repair ``prompt`` while the
        background is pinned back every step), and the clean recache then
        writes the EDITED chunks into KV history automatically because
        ``latents`` is replaced before Step 3.2/3.3 of ``_inference_inner``.

        ``masks`` is either a path to a payload written by
        scripts/prepare_renoise_edit_masks.py / utils.agent_edit.save_mask_payload
        (``{"masks": {chunk: FloatTensor [F, gh, gw]}}``) or that mapping
        directly.  Returns a JSON-compatible summary for the ledger.
        """
        if isinstance(masks, (str, os.PathLike)):
            payload = torch.load(masks, map_location="cpu", weights_only=False)
            mask_dict = {int(k): v for k, v in payload["masks"].items()}
        else:
            mask_dict = {int(k): v for k, v in dict(masks).items()}
        plan = {
            "chunk_start": int(chunk_start),
            "chunk_end": int(chunk_end),
            "entry_index": 0,
            "entry_sigma": float(entry_sigma),
            "mask_path": "",
            "masks": mask_dict,
            "mask_threshold": float(mask_threshold),
            "mask_dilate": int(mask_dilate),
            "prompt": str(prompt),
            "noise_seed": int(noise_seed),
            "collage": False,
            "donor_chunk": -1,
            "donor_frame": 0,
            "donor_mask_path": "",
            "donor_flip": False,
            "paste_pin": False,
        }
        self._runtime_edit_plan = plan
        summary = {
            "chunk_start": plan["chunk_start"],
            "chunk_end": plan["chunk_end"],
            "entry_sigma": plan["entry_sigma"],
            "mask_threshold": plan["mask_threshold"],
            "mask_dilate": plan["mask_dilate"],
            "prompt": plan["prompt"],
            "noise_seed": plan["noise_seed"],
            "mask_chunks": sorted(mask_dict),
            "mask_coverage": {
                int(chunk): float(
                    (soft.float() >= plan["mask_threshold"]).float().mean()
                )
                for chunk, soft in mask_dict.items()
            },
        }
        print(f"[edit_shot] runtime edit plan installed: {summary}")
        return summary

    def clear_runtime_edit_plan(self):
        self._runtime_edit_plan = None

    def _edit_mask_for_chunk(self, chunk_index, device, dtype, cfg=None):
        """Binary subject mask for one chunk, upsampled to latent H x W.

        Mask file: {"masks": {chunk_index: FloatTensor [F, 22, 40] soft}}.
        Returns [1, F, 1, H, W] in {0,1} or None if the chunk has no mask.
        """
        if cfg is None:
            cfg = self.edit_renoise
        if cfg is self.edit_renoise:
            # Legacy static path: lazy-load from mask_path and cache on self.
            if self._edit_masks is None:
                payload = torch.load(
                    self.edit_renoise["mask_path"], map_location="cpu",
                    weights_only=False,
                )
                self._edit_masks = {int(k): v for k, v in payload["masks"].items()}
            masks = self._edit_masks
        else:
            masks = cfg["masks"]
        soft = masks.get(int(chunk_index))
        if soft is None:
            return None
        m = (soft.float() >= cfg["mask_threshold"]).float()
        d = cfg["mask_dilate"]
        if d > 0:
            m = torch.nn.functional.max_pool2d(
                m.unsqueeze(1), kernel_size=2 * d + 1, stride=1, padding=d
            ).squeeze(1)
        m = torch.nn.functional.interpolate(
            m.unsqueeze(1), scale_factor=2, mode="nearest"
        ).squeeze(1)  # [F, H, W]
        return m.unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)

    def _build_collage(self, x0_ref, mask, chunk_index):
        """Paste the donor subject's latent into each target frame's mask bbox.

        Donor patch is cut by the donor frame's CLIPSeg silhouette, optionally
        h-flipped, resized to the target bbox, and blended in only where the
        target mask is on. Rough alignment is fine: the entry-sigma renoise
        harmonizes seams, pose, and lighting.
        """
        cfg = self.edit_renoise
        donor = self._edit_donor[:, cfg["donor_frame"]].float()  # [B, C, H, W]
        dm_payload = torch.load(cfg["donor_mask_path"], map_location="cpu",
                                weights_only=False)
        dsoft = dm_payload["masks"][cfg["donor_chunk"]][cfg["donor_frame"]]
        dm = (dsoft.float() >= 0.35)
        dm = torch.nn.functional.max_pool2d(
            dm.float()[None, None], 3, stride=1, padding=1)[0, 0]
        dm = torch.nn.functional.interpolate(
            dm[None, None], scale_factor=2, mode="nearest")[0, 0]  # [H, W]
        ys, xs = torch.where(dm > 0)
        dy0, dy1 = int(ys.min()), int(ys.max()) + 1
        dx0, dx1 = int(xs.min()), int(xs.max()) + 1
        patch = donor[:, :, dy0:dy1, dx0:dx1]
        sil = dm[dy0:dy1, dx0:dx1][None, None]
        if cfg["donor_flip"]:
            patch = torch.flip(patch, dims=[-1])
            sil = torch.flip(sil, dims=[-1])

        out = x0_ref.clone()
        core = torch.zeros_like(mask)
        n_frames = x0_ref.shape[1]
        for f in range(n_frames):
            mf = mask[0, f, 0]  # [H, W] on device
            ys, xs = torch.where(mf > 0)
            if ys.numel() < 8:
                continue
            ty0, ty1 = int(ys.min()), int(ys.max()) + 1
            tx0, tx1 = int(xs.min()), int(xs.max()) + 1
            th, tw = ty1 - ty0, tx1 - tx0
            patch_t = torch.nn.functional.interpolate(
                patch.to(x0_ref.device), size=(th, tw), mode="bilinear",
                align_corners=False)
            sil_t = torch.nn.functional.interpolate(
                sil.to(x0_ref.device), size=(th, tw), mode="bilinear",
                align_corners=False)
            weight = (sil_t >= 0.5).to(x0_ref.dtype) * mf[ty0:ty1, tx0:tx1]
            region = out[:, f, :, ty0:ty1, tx0:tx1]
            out[:, f, :, ty0:ty1, tx0:tx1] = (
                weight * patch_t.to(x0_ref.dtype) + (1.0 - weight) * region
            )
            # paste core = pasted area eroded by ~2 latent px (ring left free)
            w2 = (weight[0] > 0.5).to(x0_ref.dtype)[None]
            eroded = -torch.nn.functional.max_pool2d(-w2, 5, stride=1, padding=2)
            core[0, f, 0, ty0:ty1, tx0:tx1] = eroded[0]
        print(f"[edit_renoise] chunk {chunk_index}: collage init from donor "
              f"chunk {cfg['donor_chunk']} frame {cfg['donor_frame']} "
              f"(flip={cfg['donor_flip']}, paste_pin={cfg['paste_pin']}, "
              f"core={float(core.mean()):.3f})")
        self._edit_paste_core = core
        return out

    def _apply_edit_renoise(
        self,
        latents,
        *,
        sample_scheduler,
        chunk_index,
        scene_shot_index,
        current_num_frames,
        current_start_frame,
        cache_start_frame,
        batch_size,
        cfg=None,
    ):
        if cfg is None:
            cfg = self.edit_renoise
        mask = self._edit_mask_for_chunk(
            chunk_index, latents.device, latents.dtype, cfg=cfg
        )
        if mask is None:
            print(f"[edit_renoise] chunk {chunk_index}: no mask entry, skipping")
            return latents
        if cfg is self.edit_renoise:
            # Legacy static path: encode once and cache on self.
            if self._edit_conditional_dict is None:
                enc = self.text_encoder(text_prompts=[cfg["prompt"]])
                self._edit_conditional_dict = {
                    "prompt_embeds": enc["prompt_embeds"][0:1]
                }
            edit_conditional_dict = self._edit_conditional_dict
        else:
            # Runtime plan: cache the encoded repair prompt inside the plan.
            edit_conditional_dict = cfg.get("_conditional_dict")
            if edit_conditional_dict is None:
                enc = self.text_encoder(text_prompts=[cfg["prompt"]])
                edit_conditional_dict = {"prompt_embeds": enc["prompt_embeds"][0:1]}
                cfg["_conditional_dict"] = edit_conditional_dict

        sigmas = sample_scheduler.sigmas.detach().float().cpu().tolist()
        timesteps = sample_scheduler.timesteps.detach().float().cpu().tolist()
        pairs = [
            (timesteps[i], sigmas[i],
             sigmas[i + 1] if i + 1 < len(sigmas) else 0.0)
            for i in range(len(timesteps))
        ]
        if cfg["entry_sigma"] is not None:
            entry = float(cfg["entry_sigma"])
            below = [p for p in pairs if p[1] < entry - 1e-6]
            first_next = below[0][1] if below else 0.0
            pairs = [(entry * self.num_train_timesteps, entry, first_next)] + below
        else:
            pairs = pairs[cfg["entry_index"]:]
        if not pairs:
            return latents

        x0_ref = latents.detach().clone()
        gen = torch.Generator(device=latents.device)
        gen.manual_seed(cfg["noise_seed"] + int(chunk_index))
        eps = torch.randn(
            latents.shape, generator=gen, device=latents.device,
            dtype=torch.float32,
        ).to(latents.dtype)

        x0_init = x0_ref
        self._edit_paste_core = None
        if cfg["collage"]:
            if self._edit_donor is None:
                print(f"[edit_renoise] chunk {chunk_index}: no donor captured, "
                      "falling back to plain renoise")
            else:
                x0_init = self._build_collage(x0_ref, mask, chunk_index)

        # x0_pin: what pinned regions are held to at each step. paste_pin mode
        # holds the pasted core to its clean golden value and shrinks the
        # repaint region to the seam ring + non-core mask.
        x0_pin = x0_ref
        if cfg["paste_pin"] and self._edit_paste_core is not None:
            core = self._edit_paste_core
            x0_pin = core * x0_init + (1.0 - core) * x0_ref
            mask = mask * (1.0 - core)

        entry_sigma = pairs[0][1]
        x_start = mask * x0_init + (1.0 - mask) * x0_pin
        x = (1.0 - entry_sigma) * x_start + entry_sigma * eps
        coverage = float(mask.mean().item())
        print(
            f"[edit_renoise] chunk {chunk_index}: entry sigma {entry_sigma:.4f}, "
            f"{len(pairs)} steps, mask coverage {coverage:.3f}"
        )

        # the repair prompt must repopulate the cross-attn cache
        for block_index in range(self.num_transformer_blocks):
            self.crossattn_cache_pos[block_index]["is_init"] = False

        for t, sigma, sigma_next in pairs:
            bg = (1.0 - sigma) * x0_pin + sigma * eps
            x = mask * x + (1.0 - mask) * bg
            timestep = t * torch.ones(
                [batch_size, current_num_frames],
                device=latents.device, dtype=torch.float32,
            )
            flow_pred, _ = self.generator(
                noisy_image_or_video=x,
                conditional_dict=edit_conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                **self._kv_rag_call_kwargs(
                    self.kv_rag_pos,
                    retrieve=self.kv_rag_config.retrieve_during_denoise,
                    chunk_index=chunk_index,
                    shot_index=scene_shot_index,
                    phase="denoise",
                ),
            )
            x = x + flow_pred * (sigma_next - sigma)

        x = mask * x + (1.0 - mask) * x0_pin
        return x.to(latents.dtype)

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler

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

    def _set_all_modules_sink_size(self, sink_size_value: int):
        """
        Override sink_size on all submodules that define it.
        """
        if hasattr(self.generator.model, "sink_size"):
            setattr(self.generator.model, "sink_size", sink_size_value)

        for name, module in self.generator.model.named_modules():
            if hasattr(module, "sink_size"):
                try:
                    setattr(module, "sink_size", sink_size_value)
                except Exception:
                    pass

    def _set_all_modules_global_sink_size(self, value: int):
        """Override global_sink_size on all submodules; create the attribute if missing."""
        setattr(self.generator.model, "global_sink_size", value)
        for _, module in self.generator.model.named_modules():
            try:
                setattr(module, "global_sink_size", value)
            except Exception:
                pass

    def _is_shot_boundary(self, raw_prompts, chunk_index):
        """Return True when *chunk_index* starts a new shot (prompt-based detection).

        Pure prompt check — no dependency on sink config so that Narrative
        RoPE and other shot-aware features can reuse it independently.
        """
        if chunk_index == 0:
            return False
        if not isinstance(raw_prompts, (list, tuple)):
            return False
        if chunk_index >= len(raw_prompts):
            return False
        prompt = raw_prompts[chunk_index]
        return isinstance(prompt, str) and prompt.startswith(self.scene_cut_prefix)

    def _is_shot_end(self, raw_prompts, chunk_index, num_chunks):
        """Return True when *chunk_index* is the last chunk of its current shot."""
        if chunk_index >= int(num_chunks) - 1:
            return True
        return self._is_shot_boundary(raw_prompts, chunk_index + 1)

    def _is_scene_cut(self, raw_prompts, chunk_index):
        """Return True when *chunk_index* is the first chunk of a new scene
        AND multi-shot sink is enabled."""
        if not self.multi_shot_sink:
            return False
        if not self.sink_size or self.sink_size == 0:
            return False
        return self._is_shot_boundary(raw_prompts, chunk_index)

    def _update_sink_for_scene_cut(self, kv_cache, current_num_frames):
        """Legacy copy-to-front sink relocation (used by training pipeline)."""
        global_sink_tokens = self.global_sink_size * self.frame_seq_length
        shot_sink_tokens = self.sink_size * self.frame_seq_length
        chunk_tokens = current_num_frames * self.frame_seq_length
        copy_len = min(shot_sink_tokens, chunk_tokens)

        # iter-38: local_end_index is in lockstep across all blocks (set by
        # _apply_cache_updates with the same value). Read once → 60 syncs to 1.
        local_end = int(kv_cache[0]["local_end_index"].item())
        chunk_start = local_end - chunk_tokens
        dst_start = global_sink_tokens
        src_slice = slice(chunk_start, chunk_start + copy_len)
        dst_slice = slice(dst_start, dst_start + copy_len)

        for block_cache in kv_cache:
            block_cache["k"][:, dst_slice] = block_cache["k"][:, src_slice].clone()
            block_cache["v"][:, dst_slice] = block_cache["v"][:, src_slice].clone()

    def _pin_current_chunk(self, kv_cache, current_num_frames, current_start_tokens=None):
        """Mark the current chunk's buffer position as pinned for multi-shot sink.

        The pinned region REPLACES the original sink on the next rolling event.
        No data is copied here — relocation happens inside the attention layer
        during rolling, ensuring zero duplication.

        ``current_start_tokens`` is the chunk's ABSOLUTE start token (unaffected
        by rolling); recording it lets KV-RAG skip re-injecting frames that are
        still live in the pinned sink.
        """
        chunk_tokens = current_num_frames * self.frame_seq_length
        pin_len = min(self.sink_size * self.frame_seq_length, chunk_tokens)

        # iter-38: local_end_index is in lockstep across all blocks. Read once.
        local_end = int(kv_cache[0]["local_end_index"].item())
        chunk_start = local_end - chunk_tokens

        for block_cache in kv_cache:
            block_cache["pinned_start"].fill_(chunk_start)
            block_cache["pinned_len"].fill_(pin_len)
            if current_start_tokens is not None and "pinned_abs_start" in block_cache:
                block_cache["pinned_abs_start"].fill_(int(current_start_tokens))
                block_cache["pinned_abs_len"].fill_(pin_len)

    def _zero_kv_data(self, kv_cache, current_start_tokens):
        """Reset KV cache for clean recache, preserving global sink."""
        global_sink_tokens = self.global_sink_size * self.frame_seq_length
        for block_cache in kv_cache:
            block_cache["local_end_index"].fill_(global_sink_tokens)
            block_cache["global_end_index"].fill_(current_start_tokens)
            block_cache["pinned_start"].fill_(-1)
            block_cache["pinned_len"].zero_()
            if "pinned_abs_start" in block_cache:
                block_cache["pinned_abs_start"].fill_(-1)
                block_cache["pinned_abs_len"].zero_()
