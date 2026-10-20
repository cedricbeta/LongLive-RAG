#!/usr/bin/env python3
"""Load official metric models on CPU and use synthetic tensors, never videos."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback


def probe(checkout, cache, output):
    if output.exists():
        raise FileExistsError(output)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicitly hide CUDA devices for this CPU preflight')
    sys.path.insert(0, str(checkout))
    import torch
    import clip
    from dreamsim import dreamsim
    from easydict import EasyDict
    from pyiqa.archs.musiq_arch import MUSIQ
    from vbench.aesthetic_quality import get_aesthetic_model
    from vbench.dynamic_degree import DynamicDegree
    from vbench.motion_smoothness import MotionSmoothness
    torch.set_num_threads(4)
    rows = []
    for name in ['dino', 'dinov2', 'clip_b32', 'clip_l14', 'aesthetic', 'musiq', 'raft', 'amt', 'dreamsim']:
        started = time.monotonic()
        try:
            image = torch.zeros(1, 3, 224, 224)
            if name in ('dino', 'dinov2'):
                model = torch.hub.load('facebookresearch/' + name + (':main' if name == 'dino' else ''),
                                       'dino_vitb16' if name == 'dino' else 'dinov2_vitb14')
                model.eval()
                with torch.no_grad(): value = model(image)
            elif name.startswith('clip_'):
                model, _ = clip.load(str(cache / 'clip' / ('ViT-B-32.pt' if name == 'clip_b32' else 'ViT-L-14.pt')), device='cpu')
                with torch.no_grad(): value = model.encode_image(image)
            elif name == 'aesthetic':
                model = get_aesthetic_model(str(cache / 'vbench/aesthetic_model/emb_reader'))
                with torch.no_grad(): value = model(torch.zeros(1, 768))
            elif name == 'musiq':
                model = MUSIQ(pretrained_model_path=str(cache / 'vbench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'))
                model.eval()
                with torch.no_grad(): value = model(image)
            elif name == 'raft':
                args = EasyDict(model=str(cache / 'vbench/raft_model/models/raft-things.pth'), small=False,
                                mixed_precision=False, alternate_corr=False)
                model = DynamicDegree(args, 'cpu').model
                with torch.no_grad(): _, value = model(image[:, :, :128, :128], image[:, :, :128, :128], iters=1, test_mode=True)
            elif name == 'amt':
                model = MotionSmoothness(str(checkout / 'vbench/third_party/amt/cfgs/AMT-S.yaml'),
                                         str(cache / 'vbench/amt_model/amt-s.pth'), 'cpu').model
                with torch.no_grad():
                    value = model(image[:, :, :128, :128], image[:, :, :128, :128], torch.full((1, 1, 1, 1), .5), eval=True)['imgt_pred']
            else:
                model, _ = dreamsim(pretrained=True, device='cpu', cache_dir=str(cache))
                with torch.no_grad(): value = model.embed(image)
            assert bool(torch.isfinite(value).all()), name
            row = {'model': name, 'status': 'passed', 'output_shape': list(value.shape), 'finite': True}
            del model, value
            gc.collect()
        except Exception as error:
            traceback.print_exc()
            row = {'model': name, 'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
        row['wall_seconds'] = time.monotonic() - started
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {'phase': 'official_metric_model_cpu_preflight', 'rows': rows,
        'all_passed': all(r['status'] == 'passed' for r in rows), 'gpu_used': False,
        'actual_video_quality_evaluations': 0, 'input': 'Constructed zero image/feature tensors only',
        'versions': {n: importlib.metadata.version(n) for n in ['torch', 'torchvision', 'pyiqa', 'dreamsim', 'peft']},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    probe(args.checkout.resolve(), args.cache.resolve(), args.output.resolve())
