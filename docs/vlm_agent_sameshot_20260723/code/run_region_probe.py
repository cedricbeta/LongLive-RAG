"""Region-split attention probe: where does attention land INSIDE injected
memory frames — on the subject tokens or the background?

Runs the vlm_guided arm with the attention probe attached. The VLM judge
call additionally returns a subject bbox per block (no extra VLM calls);
the pipeline maps bboxes to the 30x52 token grid of each injected frame.
"""
import argparse
import json
import os

import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from attn_probe import AttnMassProbe
from vlm_guide import VLMGuide
from wan.modules.causal_model_latentmem import CausalWanSelfAttention

parser = argparse.ArgumentParser()
parser.add_argument("--prompt_indices", type=str, default="0,2,6")
parser.add_argument("--num_output_frames", type=int, default=120)
parser.add_argument("--max_queries", type=int, default=128)
parser.add_argument("--out_root", type=str, default="region_probe_out")
args = parser.parse_args()

config = OmegaConf.load("configs/longlive_latentmem.yaml")
config.distributed = False
device = torch.device("cuda")
set_seed(config.seed)
torch.set_grad_enabled(False)

pipeline = CausalInferencePipeline(config, device=device)
state_dict = torch.load(config.generator_ckpt, map_location="cpu")
raw = state_dict["generator_ema" if config.use_ema else "generator"] \
    if ("generator" in state_dict or "generator_ema" in state_dict) else state_dict["model"]
pipeline.generator.load_state_dict(raw)

from utils.lora_utils import configure_lora_for_model
import peft
pipeline.generator.model = configure_lora_for_model(
    pipeline.generator.model, model_name="generator",
    lora_config=config.adapter, is_main_process=True)
lora_checkpoint = torch.load(config.lora_ckpt, map_location="cpu")
sd = lora_checkpoint.get("generator_lora", lora_checkpoint)
peft.set_peft_model_state_dict(pipeline.generator.model, sd)

pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.generator.to(device=device)
pipeline.text_encoder.to(device=device)
pipeline.vae.to(device=device)
if pipeline.ae_model is not None:
    pipeline.ae_model.to(device=device, dtype=torch.bfloat16)

pipeline.vlm_guide = VLMGuide()
pipeline.vlm_mode = "guide"
pipeline.vlm_golden_at_frame = 12
pipeline.vlm_retry = False

with open("checkpoints/moviegenbench_128_refined.txt") as f:
    all_prompts = [l.strip() for l in f if l.strip()]

os.makedirs(args.out_root, exist_ok=True)

for pi in [int(i) for i in args.prompt_indices.split(",")]:
    prompt = all_prompts[pi]
    probe = AttnMassProbe(max_queries=args.max_queries)
    attn_layers = [m for m in pipeline.generator.model.modules()
                   if isinstance(m, CausalWanSelfAttention)]
    for i, m in enumerate(attn_layers):
        m._probe_layer = i
        m._attn_probe = probe
    pipeline._attn_probe = probe

    gen = torch.Generator(device=device).manual_seed(1234 + pi)
    noise = torch.randn([1, args.num_output_frames, 16, 60, 104],
                        generator=gen, device=device, dtype=torch.bfloat16)
    set_seed(config.seed)
    pipeline.inference(noise=noise, text_prompts=[prompt],
                       return_latents=True, skip_vae_decode=True)

    out = {
        "prompt_index": pi,
        "prompt": prompt,
        "golden_frames": pipeline.vlm_golden_frames,
        "frame_bboxes": pipeline.vlm_frame_bboxes,
        "records": probe.records,
        "frame_col_mean": {int(f): (probe.frame_col_sum[f] / probe.frame_col_count[f]).tolist()
                           for f in probe.frame_col_sum},
        "frame_col_count": {int(f): probe.frame_col_count[f] for f in probe.frame_col_count},
    }
    with open(os.path.join(args.out_root, f"prompt{pi:03d}_probe.json"), "w") as fjson:
        json.dump(out, fjson)

    tdir = os.path.join(args.out_root, f"prompt{pi:03d}_thumbs")
    os.makedirs(tdir, exist_ok=True)
    for fi, th in pipeline._block_thumbs:
        save_image(th, os.path.join(tdir, f"frame{fi:03d}.jpg"))

    recs = [r for r in probe.records if r.get("mass_mem_subj") is not None]
    if recs:
        ms = sum(r["mass_mem_subj"] for r in recs) / len(recs)
        mb = sum(r["mass_mem_bg"] for r in recs) / len(recs)
        ns = sum(r["n_subj_tokens"] for r in recs) / len(recs)
        nk = sum(r["n_known_tokens"] for r in recs) / len(recs)
        ds = ms / max(ns, 1)
        db = mb / max(nk - ns, 1)
        print(f"[prompt {pi}] n={len(recs)} subj_mass={ms:.4f} bg_mass={mb:.4f} "
              f"subj_tokens={ns:.0f}/{nk:.0f} per-token subj/bg density ratio={ds/db:.2f}")
print("done")
