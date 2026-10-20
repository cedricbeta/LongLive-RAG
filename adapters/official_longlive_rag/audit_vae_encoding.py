#!/usr/bin/env python3
"""Reconstruct two pilot exports and separate raw VAE output from codec effects."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkout',type=Path,required=True)
p.add_argument('--controls',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--gpu',type=int,required=True)
args=p.parse_args()
checkout=args.checkout.resolve();root=args.controls.resolve();out=args.output.resolve()
out.mkdir(parents=True,exist_ok=False)
lock=open('/tmp/longlive_feedback_render_gpu'+str(args.gpu)+'.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
uuid=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
if any(line.split(',')[0].strip()==uuid for line in apps.splitlines()):
    raise RuntimeError('Selected GPU is occupied')
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
sys.path.insert(0,str(checkout));os.chdir(checkout)
import torch
from torchvision.io import write_video
from utils.wan_wrapper import WanVAEWrapper
torch.set_grad_enabled(False)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.use_deterministic_algorithms(True,warn_only=True)
torch.set_num_threads(4)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
tensor_sha=lambda t:hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
started=time.monotonic()
status={'status':'running','pid':os.getpid(),'gpu':args.gpu,'started_unix':time.time()}
(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
try:
    vae=WanVAEWrapper().to(dtype=torch.bfloat16,device='cuda')
    prefix=None;rows=[]
    for mode in ['off_gpu2','on_gpu2']:
        latents=torch.load(root/mode/'diagnostics/latents_0000.pt',map_location='cuda',weights_only=True)
        video=(vae.decode_to_pixel_chunk(latents,use_cache=False)*0.5+0.5).clamp(0,1)
        video=video.permute(0,1,3,4,2).cpu()*255.0
        current=video[0,:237].clone()
        row={'mode':mode,'raw_float_prefix_sha256':tensor_sha(current),
             'raw_uint8_prefix_sha256':tensor_sha(current.to(torch.uint8)),
             'raw_float_prefix_identical':True if prefix is None else torch.equal(prefix,current)}
        if prefix is None:prefix=current
        file=out/(mode+'.mp4');write_video(str(file),video[0],fps=16)
        vae.model.clear_cache()
        original=root/mode/'videos/rank0-0-0_lora.mp4'
        row.update(original_mp4_sha256=sha(original),reconstructed_mp4_sha256=sha(file))
        row['mp4_exactly_reconstructed']=row['original_mp4_sha256']==row['reconstructed_mp4_sha256']
        rows.append(row);del video,latents
    result={'phase':'vae_and_encoding_engineering_audit','quality_claim':False,'rows':rows,
            'raw_prefix_frames_checked':237,'raw_prefix_identical':rows[1]['raw_float_prefix_identical'],
            'both_original_mp4_exactly_reconstructed':all(r['mp4_exactly_reconstructed'] for r in rows),
            'source_sha256':{'utils/wan_wrapper.py':sha(checkout/'utils/wan_wrapper.py'),
                             'audit_vae_encoding.py':sha(Path(__file__).resolve())},
            'wall_seconds':time.monotonic()-started,
            'interpretation':'Raw-prefix equality plus exact reconstruction of both MP4s isolates any decoded-prefix difference to export encoding. No visual quality claim.'}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    status.update(status='completed',wall_seconds=time.monotonic()-started)
    print(json.dumps(result),flush=True)
except Exception as error:
    status.update(status='execution_failed',error=f'{type(error).__name__}: {error}',wall_seconds=time.monotonic()-started)
    raise
finally:
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
