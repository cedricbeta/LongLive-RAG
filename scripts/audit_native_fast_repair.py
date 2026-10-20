#!/usr/bin/env python3
"""Focused checks for native repairs whose optional cache byte audit is off.

Uses actual parent/mask tensors and repair logs. It deliberately leaves
unmeasured history/resident contents unknown and makes no quality decision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import yaml

adapter_dir = next(p / 'adapters/official_longlive_rag' for p in Path(__file__).resolve().parents
                   if (p / 'adapters/official_longlive_rag/audit_controls.py').is_file())
sys.path.insert(0, str(adapter_dir))
from audit_controls import only, video_comparison


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(folder, parent):
    began = time.monotonic()
    torch.set_num_threads(4)
    read = lambda p: json.loads(p.read_text())
    assert not (folder / 'audit.json').exists()
    status, manifest = [read(folder / n) for n in ('status.json', 'manifest.json')]
    assert status['status'] == 'completed'
    assert sha(folder / 'config.yaml') == manifest['config_sha256']
    assert all(sha(Path(manifest['cwd']) / p) == h for p, h in manifest['source_sha256'].items())
    assert sha(parent / 'manifest.json') == manifest['parent_manifest_sha256']
    config = yaml.safe_load((folder / 'config.yaml').read_text())
    assert config['audit_prompt_switch'] is False
    request = config['repair_request']
    for key, path in [('parent_latents', parent / 'diagnostics/latents_0000.pt'),
                      ('parent_video', only(parent / 'videos', '*.mp4')),
                      ('parent_config', parent / 'config.yaml'),
                      ('parent_timing', parent / 'diagnostics/timing_0000.json'),
                      ('mask', Path(request['mask_path']))]:
        assert sha(path) == request[key + '_sha256']
    timing = read(parent / 'diagnostics/timing_0000.json')
    assert timing == read(folder / 'diagnostics/timing_0000.json')
    original, result = [torch.load(p / 'diagnostics/latents_0000.pt', map_location='cpu', weights_only=True)
                        for p in (parent, folder)]
    assert result.shape == original.shape and result.dtype == original.dtype and torch.isfinite(result).all()
    payload = torch.load(request['mask_path'], map_location='cpu', weights_only=True)
    assert payload['latent_range'] == request['latent_range']
    assert ((payload['mask'] == 0) | (payload['mask'] == 1)).all()
    mask = payload['mask'].bool()[None, :, None]
    first, last = request['latent_range']
    assert torch.equal(result[:, :first], original[:, :first])
    outside = (~mask).expand_as(result[:, first:last])
    assert torch.equal(result[:, first:last][outside], original[:, first:last][outside])
    changed = [i for i in range(result.shape[1]) if not torch.equal(result[:, i], original[:, i])]
    repairs = read(folder / 'diagnostics/repair_0000.json')
    assert len(repairs) == len(timing['blocks'])
    cursor = 0
    for row, count in zip(repairs, timing['blocks']):
        assert row['start_latent_frame'] == cursor and row['latent_frames'] == count
        assert row['phase'] == ('prefix' if cursor < first else 'edit' if cursor < last else 'suffix')
        cursor += count
        assert row['descriptor_count_after'] == cursor
        tokens = original.shape[-2] * original.shape[-1] // 4
        assert len(row['cache_positions_after_clean_recache']) == 30
        assert all(v == [cursor * tokens, min(cursor, 12) * tokens, max(0, cursor - 12), max(0, cursor - 12)]
                   for v in row['cache_positions_after_clean_recache'])
        if row['phase'] == 'edit':
            assert row['entry_sigma'] == 1.0 and row['denoising_steps'] == 4
            assert all(row[k] for k in ('global_rng_preserved', 'cache_positions_preserved',
                                       'outside_mask_exact', 'authored_cross_restored_exact'))
    before, switches = [read(p / 'diagnostics/switches_0000.json') for p in (parent, folder)]
    assert len(before) == len(switches) == 5
    for old, row in zip(before, switches):
        assert row['start_latent_frame'] == old['start_latent_frame']
        assert row['cache_positions_before'] == row['cache_positions_after']
        assert row['retrieval_pool_counts_before'] == row['retrieval_pool_counts_after']
        assert all(row['cross_attention_initialized']) and not row['history_content_audited']
        for key in ('history_sha256_before', 'history_sha256_after', 'resident_k_sha256_before',
                    'resident_k_sha256_after', 'resident_v_sha256_before', 'resident_v_sha256_after'):
            assert row[key] is None and old[key] is None
    comparison = video_comparison(only(parent / 'videos', '*.mp4'), only(folder / 'videos', '*.mp4'))
    assert comparison['frames'] == timing['export_frames'] == 960 and comparison['fps'] == [16., 16.]
    memory = read(only(folder / 'videos', '*memory_log.json'))
    old_memory = read(only(parent / 'videos', '*memory_log.json'))
    edited = [r for r in repairs if r['phase'] == 'edit']
    record = {'phase': 'fixed_target_native_development_probe', 'measured_engineering_checks_passed': True,
        'audit_source_sha256': sha(Path(__file__)), 'video_comparison_source_sha256': sha(Path(video_comparison.__code__.co_filename)),
        'status': status, 'latent_shape': list(result.shape), 'latent_sha256': sha(folder / 'diagnostics/latents_0000.pt'),
        'video_sha256': sha(only(folder / 'videos', '*.mp4')), 'edited_range': [first, last],
        'strict_prefix_latents': first, 'nonempty_prefix_tested': first > 0,
        'prefix_latents_exact': True if first > 0 else None, 'outside_mask_latents_exact': True,
        'changed_edited_latents': [i for i in changed if first <= i < last],
        'changed_suffix_latents': [i for i in changed if i >= last],
        'history_content_hashes_audited': False, 'resident_content_hashes_audited': False,
        'switch_state_content_comparison': None, 'prompt_switch_cursor_and_pool_lengths_preserved': True,
        'additional_denoiser_calls': sum(r['denoising_steps'] for r in edited),
        'repair_stage_seconds_nested_in_generation': sum(r['repair_wall_seconds'] for r in edited),
        'retrieval_queries': len(memory), 'retrieval_records_identical_to_parent': memory == old_memory,
        'video_comparison': comparison, 'audit_wall_seconds': time.monotonic() - began,
        'candidate_retained': False, 'quality_claim': False, 'held_out': False, 'human_labels': 0,
        'limits': ['Fixed target chosen during development; this is not automatic controller acceptance.',
                   'History/resident content hashes were not measured; null equality is not preservation evidence.',
                   ('The nonempty prefix latent values are checked; this alone does not establish decoded pixel equality or semantic safety.'
                    if first > 0 else 'Editing starts at latent zero, so this run has no nonempty prefix preservation test.'),
                   'Outside-mask latent equality does not establish pixel preservation or semantic safety.',
                   'Pixel and retrieval differences are not quality scores.']}
    (folder / 'audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'checked': True, 'edited_changed': len(record['changed_edited_latents']),
                      'suffix_changed': len(record['changed_suffix_latents']), 'seconds': record['audit_wall_seconds']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    args = p.parse_args()
    audit(args.folder.resolve(), args.parent.resolve())
