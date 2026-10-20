#!/usr/bin/env python3
"""Audit a native candidate against its actual saved parent, without a judge."""
import argparse
import json
from pathlib import Path
import time

import torch
import yaml

from audit_controls import sha, only, video_comparison


def audit(folder, parent, output):
    if output.exists():
        raise FileExistsError('Do not overwrite an earlier audit')
    started = time.monotonic()
    status = json.loads((folder / 'status.json').read_text())
    assert status['status'] == 'completed'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert sha(folder / 'config.yaml') == manifest['config_sha256']
    assert all(sha(Path(manifest['cwd']) / p) == h for p, h in manifest['source_sha256'].items())
    request = yaml.safe_load((folder / 'config.yaml').read_text())['repair_request']
    for key, path in [('parent_latents', parent / 'diagnostics/latents_0000.pt'),
                      ('parent_video', only(parent / 'videos', '*.mp4')),
                      ('parent_config', parent / 'config.yaml'),
                      ('parent_timing', parent / 'diagnostics/timing_0000.json'),
                      ('mask', Path(request['mask_path']))]:
        assert sha(path) == request[key + '_sha256']
    timing = json.loads((parent / 'diagnostics/timing_0000.json').read_text())
    assert timing == json.loads((folder / 'diagnostics/timing_0000.json').read_text())
    original = torch.load(parent / 'diagnostics/latents_0000.pt', map_location='cpu', weights_only=True)
    result = torch.load(folder / 'diagnostics/latents_0000.pt', map_location='cpu', weights_only=True)
    assert result.shape == original.shape and result.dtype == original.dtype and torch.isfinite(result).all()
    mask = torch.load(request['mask_path'], map_location='cpu', weights_only=True)['mask'].bool()[None,:,None]
    first, last = request['latent_range']
    assert torch.equal(result[:, :first], original[:, :first])
    outside = (~mask).expand_as(result[:, first:last])
    assert torch.equal(result[:, first:last][outside], original[:, first:last][outside])
    changed = [i for i in range(result.shape[1]) if not torch.equal(result[:,i],original[:,i])]
    repairs = json.loads((folder / 'diagnostics/repair_0000.json').read_text())
    assert len(repairs) == len(timing['blocks'])
    cursor = 0
    for row, count in zip(repairs, timing['blocks']):
        assert row['start_latent_frame'] == cursor and row['latent_frames'] == count
        expected = 'prefix' if cursor < first else 'edit' if cursor < last else 'suffix'
        assert row['phase'] == expected
        cursor += count
        assert row['descriptor_count_after'] == cursor
        tokens = original.shape[-2] * original.shape[-1] // 4
        assert len(row['cache_positions_after_clean_recache']) == 30
        assert all(v == [cursor * tokens, min(cursor,12) * tokens, max(0,cursor-12), max(0,cursor-12)]
                   for v in row['cache_positions_after_clean_recache'])
        if row['phase'] == 'edit':
            assert row['entry_sigma'] == 1.0 and row['denoising_steps'] == 4
            assert all(row[k] for k in ['global_rng_preserved','cache_positions_preserved',
                                      'outside_mask_exact','authored_cross_restored_exact'])
    old_switches = json.loads((parent / 'diagnostics/switches_0000.json').read_text())
    switches = json.loads((folder / 'diagnostics/switches_0000.json').read_text())
    assert len(old_switches) == len(switches)
    state_changes = []
    for old, row in zip(old_switches,switches):
        assert row['start_latent_frame'] == old['start_latent_frame']
        assert row['history_sha256_before'] == row['history_sha256_after']
        assert row['cache_positions_before'] == row['cache_positions_after']
        assert row['retrieval_pool_counts_before'] == row['retrieval_pool_counts_after']
        changed_state = {k: row[k] != old[k] for k in
            ['history_sha256_before','resident_k_sha256_before','resident_v_sha256_before']}
        if row['start_latent_frame'] <= first:
            assert not any(changed_state.values())
        state_changes.append({'start_latent_frame':row['start_latent_frame'], 'differs_from_parent':changed_state})
    comparison = video_comparison(only(parent / 'videos','*.mp4'),only(folder / 'videos','*.mp4'))
    assert comparison['frames'] == timing['export_frames'] and comparison['fps']==[16.0,16.0]
    memory = json.loads(only(folder / 'videos','*memory_log.json').read_text())
    original_memory = json.loads(only(parent / 'videos','*memory_log.json').read_text())
    edited = [r for r in repairs if r['phase']=='edit']
    report = {'all_engineering_checks_passed':True, 'audit_source_sha256':sha(Path(__file__)),
        'status':status, 'latent_shape':list(result.shape), 'latent_sha256':sha(folder / 'diagnostics/latents_0000.pt'),
        'video_sha256':sha(only(folder / 'videos','*.mp4')), 'edited_range':[first,last],
        'strict_prefix_latents':first, 'prefix_latents_exact':True, 'outside_mask_latents_exact':True,
        'changed_edited_latents':[i for i in changed if first<=i<last],
        'changed_suffix_latents':[i for i in changed if i>=last], 'switch_state_comparison':state_changes,
        'additional_denoiser_calls':sum(r['denoising_steps'] for r in edited),
        'repair_stage_seconds':sum(r['repair_wall_seconds'] for r in edited),
        'retrieval_queries':len(memory), 'retrieval_records_identical_to_parent':memory==original_memory,
        'video_comparison':comparison,'audit_wall_seconds':time.monotonic()-started,
        'candidate_retained':False,'quality_claim':False,'held_out':False,
        'limits':['No zero-mask replica for this particular parent is assumed.',
                  'Per-block AE descriptor changes are not compared with an unavailable parent log.',
                  'Latent/cursor invariants and switch-state differences do not establish beneficial visual change.',
                  'Raw VAE prefix pixels and current/future semantic guards are separate checks.']}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'all_engineering_checks_passed':True,'changed_edit':len(report['changed_edited_latents']),
                      'changed_suffix':len(report['changed_suffix_latents'])}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['folder','parent','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    audit(args.folder,args.parent,args.output)
