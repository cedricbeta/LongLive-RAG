#!/usr/bin/env python3
"""Load real MemFlow base/LoRA on CPU; perform no generation or CUDA calls."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkout', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not a.output.exists()
checkout, output = a.checkout.resolve(), a.output.resolve()
os.chdir(checkout)
sys.path.insert(0,str(checkout))
started = time.monotonic()
record = {'status':'running','started_unix':time.time(),'pid':os.getpid(),
          'checkpoint_inference_calls':0,'new_videos':0,'gpu_visible':False}
output.write_text(json.dumps(record,indent=2)+'\n')
try:
    import torch
    import peft
    from omegaconf import OmegaConf
    from utils.wan_wrapper import WanDiffusionWrapper
    from utils.lora_utils import configure_lora_for_model
    from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    config = OmegaConf.load('configs/interactive_inference.yaml')
    generator = WanDiffusionWrapper(is_causal=True,**OmegaConf.to_container(config.model_kwargs))
    base = torch.load(config.generator_ckpt,map_location='cpu',weights_only=True,mmap=True)['generator']
    loaded = generator.load_state_dict(base,strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    base_count = len(base)
    del base
    generator.model = configure_lora_for_model(generator.model,'generator',config.adapter,is_main_process=False)
    payload = torch.load(config.lora_ckpt,map_location='cpu',weights_only=True,mmap=True)
    lora = payload['generator_lora'] if 'generator_lora' in payload else payload
    load_result = peft.set_peft_model_state_dict(generator.model,lora)
    actual = peft.get_peft_model_state_dict(generator.model)
    assert actual.keys() == lora.keys()
    mismatches = [name for name in actual if not torch.equal(actual[name],lora[name].to(dtype=actual[name].dtype))]
    assert not mismatches,mismatches[:5]
    assert not load_result.unexpected_keys
    assert not [name for name in load_result.missing_keys if 'lora_' in name]
    assert all(x.device.type=='cpu' for x in generator.parameters())
    assert all(torch.isfinite(x).all().item() for x in generator.parameters())
    assert not torch.cuda.is_initialized()
    record.update(status='passed',base_state_keys=base_count,lora_state_keys=len(actual),
        lora_values_equal_after_loader_dtype_conversion=True,all_parameters_finite=True,
        all_parameters_on_cpu=True,cuda_initialized=False,
        parameter_count=sum(x.numel() for x in generator.parameters()),
        missing_adapter_keys=0,unexpected_adapter_keys=len(load_result.unexpected_keys),
        loader_missing_frozen_base_keys=len(load_result.missing_keys),
        note='PEFT reports frozen base keys absent from the adapter-only state; base was loaded strictly beforehand.',
        model_kwargs=OmegaConf.to_container(config.model_kwargs),adapter=OmegaConf.to_container(config.adapter),
        official_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'])).hexdigest(),
        versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','diffusers','transformers','peft','flash-attn','omegaconf','av']},
        limitations=['CPU loading is not GPU generation, timing correctness, memory-mechanism validation or video-quality evidence.',
                    'Published default SMA is false; the paper full-mechanism setting remains a separately disclosed configuration decision.',
                    'The compatible existing environment differs from the upstream torch 2.8/diffusers 0.31 installation recipe.'])
except Exception as error:
    import traceback
    record.update(status='failed',exception_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
finally:
    record['wall_seconds']=time.monotonic()-started
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
if record['status']!='passed':
    raise SystemExit(1)
