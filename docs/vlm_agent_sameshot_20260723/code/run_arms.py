"""Three-arm same-shot comparison on MovieGenBench prompts.

Arms:
  native      LongLive sliding-window baseline (no retrieval)
  latentmem   LongLive-RAG as released (+ observe-only VLM labeling for
              pollution diagnostics; labels never influence generation)
  vlm_guided  LongLive-RAG + VLM admission gate + golden-frame anchors

All arms share the same per-prompt noise and RNG so differences are the
intervention only.
"""
import argparse
import json
import os

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video
from torchvision.utils import save_image

from pipeline import CausalInferencePipeline
from utils.misc import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=["native", "latentmem", "vlm_guided", "vlm_agent"], required=True)
parser.add_argument("--prompt_indices", type=str, default="0,1,2,3")
parser.add_argument("--num_output_frames", type=int, default=120)
parser.add_argument("--golden_at", type=int, default=12)
parser.add_argument("--out_root", type=str, default="arms_out")
parser.add_argument("--block0_budget", type=int, default=1,
                    help="reroll budget for block 0 (vlm_agent only)")
parser.add_argument("--prompt_surgery", type=int, default=0,
                    help="max VLM prompt-rewrite restarts after block-0 adherence failure")
args = parser.parse_args()

cfg_path = ("configs/longlive_native.yaml" if args.arm == "native"
            else "configs/longlive_latentmem.yaml")
config = OmegaConf.load(cfg_path)
config.distributed = False
device = torch.device("cuda")
set_seed(config.seed)
torch.set_grad_enabled(False)

pipeline = CausalInferencePipeline(config, device=device)

state_dict = torch.load(config.generator_ckpt, map_location="cpu")
raw = state_dict["generator_ema" if config.use_ema else "generator"] \
    if ("generator" in state_dict or "generator_ema" in state_dict) else state_dict["model"]
pipeline.generator.load_state_dict(raw)

if getattr(config, "adapter", None):
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

if args.arm != "native":
    from vlm_guide import VLMGuide
    pipeline.vlm_guide = VLMGuide()
    pipeline.vlm_mode = "observe" if args.arm == "latentmem" else "guide"
    pipeline.vlm_golden_at_frame = args.golden_at
    pipeline.vlm_retry = (args.arm == "vlm_agent")
    pipeline.vlm_retry_budget_block0 = args.block0_budget
    pipeline.vlm_abort_on_block0_fail = (args.arm == "vlm_agent" and args.prompt_surgery > 0)

with open("checkpoints/moviegenbench_128_refined.txt") as f:
    all_prompts = [l.strip() for l in f if l.strip()]
indices = [int(i) for i in args.prompt_indices.split(",")]

out_dir = os.path.join(args.out_root, args.arm)
os.makedirs(out_dir, exist_ok=True)

from pipeline.causal_inference import Block0AdherenceFailure

for pi in indices:
    prompt = all_prompts[pi]
    gen = torch.Generator(device=device).manual_seed(1234 + pi)
    noise = torch.randn(
        [1, args.num_output_frames, 16, 60, 104],
        generator=gen, device=device, dtype=torch.bfloat16)

    cur_prompt = prompt
    surgery_rounds = []
    for round_idx in range(args.prompt_surgery + 1):
        set_seed(config.seed)
        try:
            video, _ = pipeline.inference(
                noise=noise, text_prompts=[cur_prompt], return_latents=True)
            break
        except Block0AdherenceFailure as e:
            if round_idx == args.prompt_surgery:
                # out of surgery rounds: generate the full video anyway so we
                # still have an output to inspect
                pipeline.vlm_abort_on_block0_fail = False
                set_seed(config.seed)
                video, _ = pipeline.inference(
                    noise=noise, text_prompts=[cur_prompt], return_latents=True)
                pipeline.vlm_abort_on_block0_fail = True
                surgery_rounds.append({"round": round_idx, "gave_up": True})
                break
            rewrite = pipeline.vlm_guide.rewrite_prompt(prompt, e.reasons, e.thumbs)
            surgery_rounds.append({
                "round": round_idx,
                "reject_reasons": e.reasons,
                "rewritten_prompt": rewrite["rewritten_prompt"],
                "rewrite_reason": rewrite.get("reason", ""),
            })
            cur_prompt = rewrite["rewritten_prompt"]
            print(f"[surgery round {round_idx}] new prompt: {cur_prompt[:160]}")
    vid = (rearrange(video[0].float().cpu(), "f c h w -> f h w c") * 255).to(torch.uint8)
    mp4 = os.path.join(out_dir, f"prompt{pi:03d}.mp4")
    write_video(mp4, vid, fps=16)

    log = {
        "arm": args.arm,
        "prompt_index": pi,
        "prompt": prompt,
        "final_prompt": cur_prompt,
        "surgery_rounds": surgery_rounds,
        "block0_budget": args.block0_budget,
        "memory_log": getattr(pipeline, "memory_indices_log", []),
        "vlm_log": getattr(pipeline, "vlm_block_log", []),
        "golden_frames": getattr(pipeline, "vlm_golden_frames", []),
        "rejected_frames": sorted(getattr(pipeline, "vlm_rejected_frames", set())),
        "vlm_calls": pipeline.vlm_guide.calls if getattr(pipeline, "vlm_guide", None) else 0,
    }
    with open(os.path.join(out_dir, f"prompt{pi:03d}_log.json"), "w") as f:
        json.dump(log, f, indent=1)

    thumbs = getattr(pipeline, "_block_thumbs", [])
    if thumbs:
        tdir = os.path.join(out_dir, f"prompt{pi:03d}_thumbs")
        os.makedirs(tdir, exist_ok=True)
        for fi, th in thumbs:
            save_image(th, os.path.join(tdir, f"frame{fi:03d}.jpg"))
    print(f"[{args.arm}] prompt {pi} done -> {mp4}")

print("all done")
