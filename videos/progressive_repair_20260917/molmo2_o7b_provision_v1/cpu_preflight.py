#!/usr/bin/env python3
import hashlib,importlib.metadata,json,time,traceback
from pathlib import Path
root=Path(__file__).resolve().parent;started=time.monotonic()
snapshot=Path.home()/'.cache/huggingface/hub/models--allenai--Molmo2-O-7B/snapshots/784410650d12be9bc086118fdefa32d2c3bced86'
result={'status':'running','model_loads':0,'inference_calls':0,'purpose':'CPU config/custom-class/processor import; no model weights or semantic evaluation.'}
try:
 from transformers import AutoConfig,AutoProcessor
 from transformers.dynamic_module_utils import get_class_from_dynamic_module
 config=AutoConfig.from_pretrained(snapshot,trust_remote_code=True,local_files_only=True)
 processor=AutoProcessor.from_pretrained(snapshot,trust_remote_code=True,local_files_only=True)
 cls=get_class_from_dynamic_module('modeling_molmo2.Molmo2ForConditionalGeneration',str(snapshot),local_files_only=True)
 result.update(status='passed',config_class=type(config).__name__,processor_class=type(processor).__name__,model_class=cls.__name__,
  packages={name:importlib.metadata.version(name) for name in ['torch','torchvision','transformers','numpy','Pillow','einops','accelerate']})
except BaseException as error:result.update(status='failed',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
result.update(wall_seconds=time.monotonic()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(root/'cpu_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
