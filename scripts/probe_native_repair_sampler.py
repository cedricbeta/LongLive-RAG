#!/usr/bin/env python3
"""Exercise Euler using the pinned native CUDA model/wrapper on tiny tensors."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def main(args):
    began = time.monotonic()
    assert args.gpu in (0, 1, 2)
    lock = open('/tmp/longlive_feedback_render_gpu' + str(args.gpu) + '.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert not any(line.split(',')[0].strip() == uuid for line in processes.splitlines()), 'GPU occupied'
    sys.path.insert(0, str(args.checkout.resolve()))
    import torch
    from repair_sampler import sample_euler
    from utils.wan_wrapper import WanDiffusionWrapper
    from utils.scheduler import FlowMatchScheduler
    from wan.modules.causal_model_latentmem import CausalWanModel

    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    model = CausalWanModel(dim=64, ffn_dim=128, freq_dim=64, text_dim=64, num_heads=2,
        num_layers=1, local_attn_size=12, sink_size=1, memory_size=6)
    # Bypass unused Diffusers offload hooks, as native wrapper-level device transfer does.
    torch.nn.Module.cuda(model)
    model.bfloat16().eval()
    torch.nn.init.normal_(model.head.head.weight, std=.01)
    wrapper = WanDiffusionWrapper.__new__(WanDiffusionWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.scheduler = FlowMatchScheduler(shift=5., sigma_min=0., extra_one_step=True)
    wrapper.scheduler.set_timesteps(1000, training=True)
    wrapper.uniform_timestep = False
    wrapper.seq_len = 12
    steps = torch.cat([wrapper.scheduler.timesteps.cpu(), torch.zeros(1)])[torch.tensor([0, 250, 500, 750])]
    rows = []
    with torch.inference_mode():
        for frames in (3, 1):
            shape = (1, frames, 16, 4, 4)
            parent = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
            eps = torch.randn(shape, generator=torch.Generator(device='cuda').manual_seed(4243), device='cuda', dtype=torch.bfloat16)
            mask = torch.zeros((1, frames, 1, 4, 4), device='cuda', dtype=torch.bool)
            mask[..., 1:3, 1:3] = True
            kv = [{'k': torch.zeros(1, 48, 2, 32, device='cuda', dtype=torch.bfloat16),
                   'v': torch.zeros(1, 48, 2, 32, device='cuda', dtype=torch.bfloat16),
                   'global_end_index': torch.tensor(0, device='cuda'),
                   'local_end_index': torch.tensor(0, device='cuda'), 'cpu_k_frames': [], 'cpu_v_frames': []}]
            cross = [{'k': torch.zeros(1, 512, 2, 32, device='cuda', dtype=torch.bfloat16),
                      'v': torch.zeros(1, 512, 2, 32, device='cuda', dtype=torch.bfloat16), 'is_init': False}]
            conditional = {'prompt_embeds': torch.randn(1, 512, 64, device='cuda', dtype=torch.bfloat16)}
            inputs, flows = [], []
            def capture(**kwargs):
                inputs.append(kwargs['noisy_image_or_video'].clone())
                result = wrapper(**kwargs)
                flows.append(result[0].clone())
                return result
            p = SimpleNamespace(denoising_step_list=steps, scheduler=wrapper.scheduler, generator=capture,
                                kv_cache1=kv, crossattn_cache=cross, frame_seq_length=4)
            before = torch.cuda.get_rng_state().clone()
            trace = []
            result = sample_euler(p, parent, mask, eps, conditional, 0, None, trace)
            assert len(inputs) == len(flows) == len(trace) == 4
            assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()
            outside = (~mask).expand_as(result)
            assert torch.equal(result[outside], parent[outside])
            assert torch.equal(before, torch.cuda.get_rng_state())
            assert all(f.dtype == torch.bfloat16 and torch.isfinite(f).all() for f in flows)
            assert any(torch.count_nonzero(f) for f in flows)
            for index, x in enumerate(inputs):
                t = torch.full((frames,), float(steps[index]), device='cuda')
                bg = wrapper.scheduler.add_noise(parent.flatten(0, 1), eps.flatten(0, 1), t).unflatten(0, parent.shape[:2])
                assert torch.equal(x[outside], bg[outside])
            rows.append({'frames': frames, 'actual_native_forwards': len(flows), 'flow_dtype': str(flows[0].dtype),
                         'nonzero_flow': True, 'outside_mask_exact': True, 'background_uses_fixed_noise': True,
                         'global_cuda_rng_exact': True, 'trace': trace})
    names = ['repair_sampler.py', 'utils/wan_wrapper.py', 'utils/scheduler.py', 'wan/modules/causal_model_latentmem.py']
    result = {'status': 'passed', 'cases': rows, 'gpu': args.gpu, 'gpu_uuid': uuid,
        'wall_seconds': time.monotonic() - began, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_sources': {n: hashlib.sha256((args.checkout / n).read_bytes()).hexdigest() for n in names},
        'torch': torch.__version__, 'new_video_generations': 0,
        'limits': ['Random tiny native-model weights, one layer and four tokens/frame; no pretrained visual quality claim.',
                   'This exercises the real native wrapper, shifted schedule and CUDA attention kernels.']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'actual_native_forwards': 8, 'seconds': result['wall_seconds']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    began = time.monotonic()
    try:
        main(args)
    except Exception as error:
        args.output.write_text(json.dumps({'status': 'failed', 'error': repr(error),
            'wall_seconds': time.monotonic() - began}, indent=2) + '\n')
        raise
