#!/usr/bin/env python3
"""Audit native nonzero replay state propagation without a quality judgment."""
import argparse
import json
from pathlib import Path
import time

import torch
import yaml

from audit_controls import sha, only, video_comparison


def audit(folder, parent, zero):
    started = time.monotonic()
    status = json.loads((folder / 'status.json').read_text())
    if status['status'] != 'completed':
        raise ValueError('Only completed runs can pass the replay audit')
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert sha(folder / 'config.yaml') == manifest['config_sha256']
    assert all(sha(Path(manifest['cwd']) / p) == h for p, h in manifest['source_sha256'].items())
    request = yaml.safe_load((folder / 'config.yaml').read_text())['repair_request']
    assert request['parent_latents_sha256'] == sha(parent / 'diagnostics/latents_0000.pt')
    assert request['parent_video_sha256'] == sha(only(parent / 'videos', '*.mp4'))
    mask_payload = torch.load(request['mask_path'], map_location='cpu', weights_only=True)
    assert sha(request['mask_path']) == request['mask_sha256']
    mask = mask_payload['mask'].bool()[None, :, None]
    first, last = request['latent_range']
    original = torch.load(parent / 'diagnostics/latents_0000.pt', map_location='cpu', weights_only=True)
    result = torch.load(folder / 'diagnostics/latents_0000.pt', map_location='cpu', weights_only=True)
    assert result.shape == original.shape and result.dtype == original.dtype and torch.isfinite(result).all()
    delta = result != original
    outside = (~mask).expand_as(result[:, first:last])
    changed = [i for i in range(result.shape[1]) if bool(delta[:, i].any())]
    prefix_exact = torch.equal(result[:, :first], original[:, :first])
    outside_exact = not bool(delta[:, first:last][outside].any())
    changed_edited = [i for i in changed if first <= i < last]
    changed_suffix = [i for i in changed if i >= last]
    assert prefix_exact and outside_exact and changed_edited and changed_suffix
    repairs = json.loads((folder / 'diagnostics/repair_0000.json').read_text())
    zero_repairs = json.loads((zero / 'diagnostics/repair_0000.json').read_text())
    assert len(repairs) == len(zero_repairs)
    edited = [r for r in repairs if r['phase'] == 'edit']
    assert all(r['entry_sigma'] == 1.0 and r['denoising_steps'] == 4 and
               r['global_rng_preserved'] and r['cache_positions_preserved'] and
               r['outside_mask_exact'] and r['authored_cross_restored_exact'] for r in edited)
    descriptor_changed = []
    for row, inert in zip(repairs, zero_repairs):
        assert row['start_latent_frame'] == inert['start_latent_frame']
        assert row['descriptor_count_after'] == inert['descriptor_count_after']
        assert row['cache_positions_after_clean_recache'] == inert['cache_positions_after_clean_recache']
        unequal = row['descriptor_sha256_after'] != inert['descriptor_sha256_after']
        if row['phase'] == 'prefix':
            assert not unequal
        if unequal:
            descriptor_changed.append(row['start_latent_frame'])
    switches = json.loads((folder / 'diagnostics/switches_0000.json').read_text())
    parent_switches = json.loads((parent / 'diagnostics/switches_0000.json').read_text())
    assert len(switches) == len(parent_switches) == 5
    switch_changes = []
    for row, old in zip(switches, parent_switches):
        assert row['start_latent_frame'] == old['start_latent_frame']
        assert row['history_sha256_before'] == row['history_sha256_after']
        assert row['cache_positions_before'] == row['cache_positions_after']
        assert row['retrieval_pool_counts_before'] == row['retrieval_pool_counts_after']
        switch_changes.append({'start_latent_frame': row['start_latent_frame'],
            'historical_pool_and_descriptors_differ_from_parent': row['history_sha256_before'] != old['history_sha256_before'],
            'resident_k_differ_from_parent': row['resident_k_sha256_before'] != old['resident_k_sha256_before'],
            'resident_v_differ_from_parent': row['resident_v_sha256_before'] != old['resident_v_sha256_before']})
        if row['start_latent_frame'] <= first:
            assert row['history_sha256_before'] == old['history_sha256_before']
    assert descriptor_changed and any(r['historical_pool_and_descriptors_differ_from_parent']
        for r in switch_changes if r['start_latent_frame'] >= last)
    video = only(folder / 'videos', '*.mp4')
    memory = json.loads(only(folder / 'videos', '*memory_log.json').read_text())
    parent_memory = json.loads(only(parent / 'videos', '*memory_log.json').read_text())
    report = {'phase': 'official_nonzero_mask_state_propagation', 'all_engineering_checks_passed': True,
        'audit_source_sha256': sha(Path(__file__)),
        'status': status, 'latent_shape': list(result.shape), 'latent_sha256': sha(folder / 'diagnostics/latents_0000.pt'),
        'video_sha256': sha(video), 'strict_prefix_latents': first, 'prefix_latents_exact': prefix_exact,
        'edited_range': [first, last], 'outside_mask_latents_exact': outside_exact,
        'changed_edited_latents': changed_edited, 'changed_suffix_latents': changed_suffix,
        'changed_descriptor_block_starts': descriptor_changed, 'switch_state_comparison': switch_changes,
        'repair_blocks': len(edited), 'additional_denoiser_calls': sum(r['denoising_steps'] for r in edited),
        'repair_stage_seconds': sum(r['repair_wall_seconds'] for r in edited),
        'retrieval_queries': len(memory), 'retrieval_records_identical_to_parent': memory == parent_memory,
        'video_comparison': video_comparison(only(parent / 'videos', '*.mp4'), video),
        'audit_wall_seconds': time.monotonic() - started, 'held_out': False, 'quality_claim': False,
        'candidate_retained': False, 'limits': [
            'Sparse automatic mask has known localization deficiencies; this is a state-propagation test.',
            'Latent pinning does not establish pixel pinning or preservation of identity, actions or quality.',
            'No checker accepted this candidate and no future quality guard was run.',
            'Same checkpoint, timing, prompts and global seed as parent; extra private repair calls have full replay cost.',
            'Compressed prefix may differ because of VAE temporal context or encoding; raw prefix decoding is not audited here.']}
    (folder / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'video_comparison'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['folder', 'parent', 'zero']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    audit(args.folder, args.parent, args.zero)
