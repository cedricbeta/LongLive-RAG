import fcntl,hashlib,json,os,sys,time,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
m=json.loads((root/'manifest.json').read_text())
lock=open('/tmp/longlive_feedback_render_gpu0.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert not any(l.split(',')[0].strip()==uuid for l in apps.splitlines())
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
sys.path.insert(0,m['checkout']);os.chdir(m['checkout'])
import torch
from utils.wan_wrapper import WanVAEWrapper
from timing_adapter import decode_continuous
torch.set_grad_enabled(False);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True,warn_only=True)
start=time.monotonic();status={'status':'running','pid':os.getpid(),'gpu':0,'started_unix':time.time()};(root/'status.json').write_text(json.dumps(status))
try:
 vae=WanVAEWrapper().to(device='cuda',dtype=torch.bfloat16)
 lat=torch.load(m['latent_path'],map_location='cpu',weights_only=True)[:,:18].to('cuda')
 full=vae.decode_to_pixel(lat,use_cache=False).cpu()
 split=decode_continuous(vae,lat,chunk_size=9)
 h=lambda t:hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 result={'phase':'continuous_vae_engineering','latent_frames':18,'expected_frames':69,'shape':list(full.shape),'split_shape':list(split.shape),'full_sha256':h(full),'split_sha256':h(split),'bitwise_identical':torch.equal(full,split),'max_absolute_error':float((full-split).abs().max()),'mean_absolute_error':float((full-split).abs().mean()),'wall_seconds':time.monotonic()-start,'quality_claim':False}
 (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');status.update(status='completed',wall_seconds=time.monotonic()-start)
except Exception as e:
 status.update(status='execution_failed',error=f'{type(e).__name__}: {e}',wall_seconds=time.monotonic()-start);raise
finally:(root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
