#!/usr/bin/env python3
"""Inspect saved tensors from a real parent replay; never infer visual safety."""
import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.agent_runtime import file_sha256
from utils.edit_renoise_plans import normalize_edit_renoise_config
from utils.parent_latent_replay import ParentLatentReplay


def audit(config_path, candidate_path):
    config = yaml.safe_load(config_path.read_text())['inference']
    replay = ParentLatentReplay(config['replay_parent'])
    child = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if child.shape != replay.latent.shape or child.dtype != replay.latent.dtype:
        raise ValueError('Candidate and parent raw latent layouts differ')
    plans = config['edit_renoise_plans']
    if len(plans) != 1:
        raise ValueError('This audit expects exactly one progressive edit range')
    plan = normalize_edit_renoise_config(plans[0])
    masks = torch.load(plan['mask_path'], map_location='cpu', weights_only=False)['masks']
    records = []
    for index in range(replay.num_chunks):
        offset = index * replay.frames_per_chunk
        before = replay.latent[:, offset:offset + replay.frames_per_chunk]
        after = child[:, offset:offset + replay.frames_per_chunk]
        different = after != before
        item = {'chunk': index, 'changed_elements': int(different.sum()),
                'mean_absolute_difference': float((after.float()-before.float()).abs().mean())}
        if index < replay.start:
            item['region'] = 'strict_prefix'
        elif index <= replay.end:
            # Mirror the frozen renderer's threshold, dilation and 2x spatial
            # expansion; the audit checks saved outputs, not a mock renderer.
            mask = (masks[index].float() >= plan['mask_threshold']).float()[:, None]
            d = plan['mask_dilate']
            if d:
                mask = torch.nn.functional.max_pool2d(mask, 2*d+1, stride=1, padding=d)
            mask = torch.nn.functional.interpolate(mask, scale_factor=2, mode='nearest')
            mask = mask[None].expand_as(before).bool()
            item.update(region='edited_shot', background_elements=int((~mask).sum()),
                changed_background_elements=int((different & ~mask).sum()),
                changed_masked_elements=int((different & mask).sum()))
        else:
            item['region'] = 'regenerated_suffix'
        records.append(item)
    return {'config_sha256': file_sha256(config_path), 'candidate_latent_sha256': file_sha256(candidate_path),
        'parent': replay.config, 'shape': list(child.shape), 'dtype': str(child.dtype), 'chunks': records,
        'strict_prefix_chunks_checked': replay.start,
        'strict_prefix_identical': all(not r['changed_elements'] for r in records if r['region']=='strict_prefix'),
        'edited_background_identical': all(not r['changed_background_elements'] for r in records if r['region']=='edited_shot'),
        'suffix_changed_chunks': [r['chunk'] for r in records if r['region']=='regenerated_suffix' and r['changed_elements']],
        'limitations': ['Latent equality outside a mask does not guarantee pixel equality or visual safety.',
                       'A changed suffix is not evidence that the change is beneficial.',
                       'A zero-edit control and cache-state audit are still needed to isolate propagation fully.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--candidate-latent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.config, args.candidate_latent)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['strict_prefix_chunks_checked','strict_prefix_identical',
                     'edited_background_identical','suffix_changed_chunks']}))
