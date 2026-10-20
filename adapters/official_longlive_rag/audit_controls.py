#!/usr/bin/env python3
"""Audit actual pinned-checkpoint controls. No visual quality decisions."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_comparison(parent, child):
    caps = [cv2.VideoCapture(str(p)) for p in [parent, child]]
    if not all(c.isOpened() for c in caps):
        raise ValueError('Cannot open comparison videos')
    fps = [c.get(cv2.CAP_PROP_FPS) for c in caps]
    digests = [hashlib.sha256(), hashlib.sha256()]
    count, changed, frame_mae = 0, [], []
    try:
        while True:
            (ok_a,a),(ok_b,b) = [c.read() for c in caps]
            if ok_a != ok_b:
                raise ValueError('Frame count mismatch')
            if not ok_a:
                break
            if a.shape != b.shape:
                raise ValueError('Frame dimension mismatch')
            for h,frame in zip(digests,[a,b]): h.update(frame.tobytes())
            if not np.array_equal(a,b):
                changed.append(count)
                frame_mae.append(float(np.abs(a.astype(np.int16)-b.astype(np.int16)).mean()))
            else:
                frame_mae.append(0.0)
            count += 1
    finally:
        for cap in caps: cap.release()
    if not count:
        raise ValueError('Empty video')
    return {'frames':count,'fps':fps,'all_frames_identical':not changed,'changed_frames':changed,
        'per_frame_mean_absolute_error_uint8_scale':frame_mae,
        'parent_sha256':sha(parent),'child_sha256':sha(child),
        'decoded_stream_sha256':[h.hexdigest() for h in digests]}


def only(folder, pattern):
    paths = list(folder.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f'Expected one {pattern} in {folder}, got {len(paths)}')
    return paths[0]


def audit(root, reference):
    statuses = {p.parent.name:json.loads(p.read_text()) for p in root.glob('*/status.json')}
    if any(statuses.get(m,{}).get('status') != 'completed' for m in ['off_gpu2','on_gpu2']):
        raise ValueError('Both GPU2 controls must complete before auditing')
    off, on = root/'off_gpu2', root/'on_gpu2'
    for folder in [off,on]:
        m = json.loads((folder/'manifest.json').read_text())
        if sha(folder/'config.yaml') != m['config_sha256']:
            raise ValueError('Control configuration changed')
        if not all(sha(Path(m['cwd'])/p)==h for p,h in m['source_sha256'].items()):
            raise ValueError('Frozen adapter source changed during controls')
    baseline = only(reference/'videos','*.mp4')
    off_video, on_video = [only(folder/'videos','*.mp4') for folder in [off,on]]
    initial = video_comparison(baseline,off_video)
    switched = video_comparison(off_video,on_video)
    before,after = [torch.load(only(folder/'diagnostics','latents_*.pt'),map_location='cpu',weights_only=True) for folder in [off,on]]
    if before.shape != after.shape or before.dtype != after.dtype:
        raise ValueError('Latent layouts differ')
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError('Nonfinite output latents')
    changed = [i for i in range(before.shape[1]) if not torch.equal(before[:,i],after[:,i])]
    switches = json.loads(only(on/'diagnostics','switches_*.json').read_text())
    if len(switches) != 1 or switches[0]['start_latent_frame'] != 60:
        raise ValueError('Expected exactly one prompt boundary at 60')
    switch = switches[0]
    memory = [json.loads(only(folder/'videos','*_memory_log.json').read_text()) for folder in [reference,off,on]]
    pool_unchanged = (switch['history_content_audited'] and
        switch['history_sha256_before'] == switch['history_sha256_after'] and
        switch['cache_positions_before'] == switch['cache_positions_after'] and
        switch['retrieval_pool_counts_before'] == switch['retrieval_pool_counts_after'])
    boundary = switch['native_decoded_prefix_frames']
    return {'phase':'official_prompt_adapter_engineering','quality_claim':False,'statuses':statuses,
        'off_reproduction':initial,'off_retrieval_log_equal_reference':memory[0]==memory[1],
        'off_to_on_video':switched,
        'encoded_prefix_comparison':{'frames':boundary,
            'changed_frame_count':sum(i<boundary for i in switched['changed_frames']),
            'mean_absolute_error_uint8_scale':float(np.mean(switched['per_frame_mean_absolute_error_uint8_scale'][:boundary])),
            'warning':'MP4 decoding is not the raw VAE output. See the separate VAE/encoding audit before assigning a cause.'},
        'off_to_on_latents':{'shape':list(before.shape),'dtype':str(before.dtype),'changed_frames':changed,
            'prefix_60_identical':all(i>=60 for i in changed),'suffix_changed_count':sum(i>=60 for i in changed)},
        'switch_audit':switch,'history_and_cursors_preserved':pool_unchanged,
        'resident_k_changed_layers':[i for i,(a,b) in enumerate(zip(switch['resident_k_sha256_before'],switch['resident_k_sha256_after'])) if a!=b],
        'resident_v_changed_layers':[i for i,(a,b) in enumerate(zip(switch['resident_v_sha256_before'],switch['resident_v_sha256_after'])) if a!=b],
        'memory_query_counts':[len(m) for m in memory],
        'off_on_retrieval_queries_through_switch_equal':
            [r for r in memory[1] if r['query_frame']<=60] == [r for r in memory[2] if r['query_frame']<=60],
        'source_artifacts':{str(p.relative_to(root)):sha(p) for p in root.glob('*/manifest.json')},
        'limits':['One exposed two-prompt development script, not a 60-second task.',
                  'No correction, prompt-compliance or independent visual-quality evaluation.',
                  'Initial GPU0 resource refusal and interrupted GPU1 startup remain recorded in statuses.']}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    result=audit(args.root,args.reference)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['off_retrieval_log_equal_reference','history_and_cursors_preserved','off_to_on_latents']}))
