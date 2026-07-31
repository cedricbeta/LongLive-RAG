"""Measure post-softmax attention mass on LongLive-RAG retrieved memory tokens.

Runs the upstream latentmem pipeline on the first N MovieGenBench prompts
(their benchmark prompt set) and logs, for every probed layer/denoise call,
how much attention mass lands on the retrieved memory frames vs the uniform
baseline. No videos are produced; latents only.
"""
import argparse
import json
import os

import torch
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from attn_probe import AttnMassProbe
from wan.modules.causal_model_latentmem import CausalWanSelfAttention

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs/longlive_latentmem.yaml")
parser.add_argument("--num_prompts", type=int, default=8)
parser.add_argument("--num_output_frames", type=int, default=120)
parser.add_argument("--layers", type=int, nargs="*", default=None, help="probe layers; default all 30")
parser.add_argument("--max_queries", type=int, default=128)
parser.add_argument("--out", type=str, default="attnmass_results.json")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)
config.distributed = False
device = torch.device("cuda")
set_seed(config.seed)
torch.set_grad_enabled(False)

pipeline = CausalInferencePipeline(config, device=device)

state_dict = torch.load(config.generator_ckpt, map_location="cpu")
if "generator" in state_dict or "generator_ema" in state_dict:
    raw = state_dict["generator_ema" if config.use_ema else "generator"]
elif "model" in state_dict:
    raw = state_dict["model"]
else:
    raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
pipeline.generator.load_state_dict(raw)

if getattr(config, "adapter", None):
    from utils.lora_utils import configure_lora_for_model
    import peft

    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model, model_name="generator",
        lora_config=config.adapter, is_main_process=True,
    )
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)

pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.generator.to(device=device)
pipeline.text_encoder.to(device=device)
pipeline.vae.to(device=device)
if pipeline.ae_model is not None:
    pipeline.ae_model.to(device=device, dtype=torch.bfloat16)

probe = AttnMassProbe(layers=set(args.layers) if args.layers else None,
                      max_queries=args.max_queries)
attn_layers = [m for m in pipeline.generator.model.modules()
               if isinstance(m, CausalWanSelfAttention)]
print(f"Attaching probe to {len(attn_layers)} self-attention layers")
for i, m in enumerate(attn_layers):
    m._probe_layer = i
    m._attn_probe = probe

orig_forward = pipeline.generator.forward
def tagged_forward(*a, **kw):
    t = kw.get("timestep")
    probe.meta = {
        "timestep": int(t.flatten()[0].item()) if t is not None else -1,
        "frame_start": int(kw.get("current_start", 0)) // pipeline.frame_seq_length,
    }
    return orig_forward(*a, **kw)
pipeline.generator.forward = tagged_forward

with open(config.data_path) as f:
    prompts = [l.strip() for l in f if l.strip()][: args.num_prompts]

all_results = []
for pi, prompt in enumerate(prompts):
    probe.records = []
    pipeline.latent_descriptors = []
    set_seed(config.seed)
    noise = torch.randn(
        [1, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )
    probe.meta = {}
    _, _ = pipeline.inference(
        noise=noise, text_prompts=[prompt],
        return_latents=True, skip_vae_decode=True,
    )
    n_rec = len(probe.records)
    if n_rec:
        mm = sum(r["mass_mem"] for r in probe.records) / n_rec
        um = sum(r["uniform_mem"] for r in probe.records) / n_rec
        print(f"[prompt {pi}] records={n_rec} mean mass_mem={mm:.4f} uniform={um:.4f} ratio={mm/um:.3f}")
    all_results.append({
        "prompt_index": pi,
        "prompt": prompt,
        "records": probe.records,
        "memory_log": [
            {k: v for k, v in e.items() if k != "selected_similarities"}
            for e in pipeline.memory_indices_log
        ] if hasattr(pipeline, "memory_indices_log") else [],
    })
    with open(args.out, "w") as f:
        json.dump(all_results, f)
    print(f"[prompt {pi}] saved -> {args.out}")

print("done")
