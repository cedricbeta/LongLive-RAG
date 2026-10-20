#!/usr/bin/env python3
"""Audit actual OFF/zero-mask replay against the saved official timed parent."""
import argparse
import json
from pathlib import Path

import torch

from audit_controls import sha, only, video_comparison


def inspect_run(folder, parent):
    status = json.loads((folder / 'status.json').read_text())
    if status['status'] != 'completed':
        raise ValueError('Run is not completed: ' + str(folder))
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert sha(folder / 'config.yaml') == manifest['config_sha256']
    assert all(sha(Path(manifest['cwd']) / name) == digest for name, digest in manifest['source_sha256'].items())
    reference = torch.load(parent / 'diagnostics/latents_0000.pt', map_location='cpu', weights_only=True)
    output = torch.load(folder / 'diagnostics/latents_0000.pt', map_location='cpu', weights_only=True)
    assert reference.shape == output.shape and reference.dtype == output.dtype
    assert torch.isfinite(output).all()
    changed = [i for i in range(output.shape[1]) if not torch.equal(output[:, i], reference[:, i])]
    video = video_comparison(only(parent / 'videos', '*.mp4'), only(folder / 'videos', '*.mp4'))
    memory = json.loads(only(folder / 'videos', '*memory_log.json').read_text())
    reference_memory = json.loads(only(parent / 'videos', '*memory_log.json').read_text())
    switches = json.loads((folder / 'diagnostics/switches_0000.json').read_text())
    reference_switches = json.loads((parent / 'diagnostics/switches_0000.json').read_text())
    # Wall time is deliberately not compared: audit and resource costs can differ.
    state_keys = ['start_latent_frame', 'resident_source_frames', 'cache_positions_before', 'cache_positions_after',
                  'retrieval_pool_counts_before', 'retrieval_pool_counts_after', 'descriptor_count',
                  'history_sha256_before', 'history_sha256_after', 'resident_k_sha256_before', 'resident_k_sha256_after',
                  'resident_v_sha256_before', 'resident_v_sha256_after', 'cross_attention_initialized',
                  'prompt_sha256', 'export_prefix_frames']
    state_equal = len(switches) == len(reference_switches) and all(
        {k: a[k] for k in state_keys} == {k: b[k] for k in state_keys} for a, b in zip(switches, reference_switches))
    repair = json.loads((folder / 'diagnostics/repair_0000.json').read_text())
    if manifest['repair_enabled']:
        assert len(repair) == 84
        cursor = 0
        for row in repair:
            assert row['start_latent_frame'] == cursor
            cursor += row['latent_frames']
            assert row['descriptor_count_after'] == cursor
            assert len(row['cache_positions_after_clean_recache']) == 30
            assert all(position == [cursor * 1560, min(cursor, 12) * 1560,
                                    max(0, cursor - 12), max(0, cursor - 12)]
                       for position in row['cache_positions_after_clean_recache'])
        edited = [row for row in repair if row['phase'] == 'edit']
        prefix = [row for row in repair if row['phase'] == 'prefix']
        suffix = [row for row in repair if row['phase'] == 'suffix']
        assert sum(row['latent_frames'] for row in prefix) == 41
        assert sum(row['latent_frames'] for row in edited) == 40
        assert sum(row['latent_frames'] for row in suffix) == 160
        assert len(edited) == 14
        assert all(row['mask_fraction'] == 0 and row['entry_sigma'] == 1.0 and row['denoising_steps'] == 4
                   and row['global_rng_preserved'] and row['cache_positions_preserved']
                   and row['outside_mask_exact'] and row['authored_cross_restored_exact'] for row in edited)
    else:
        assert not repair
        edited = []
    passed = not changed and video['all_frames_identical'] and memory == reference_memory and state_equal
    return {'status': status, 'all_checks_passed': passed, 'latent_shape': list(output.shape),
            'changed_latent_frames': changed, 'video_comparison': video,
            'retrieval_records_identical': memory == reference_memory, 'retrieval_queries': len(memory),
            'all_five_switch_state_records_identical_except_cost': state_equal,
            'edited_blocks': len(edited), 'additional_repair_denoiser_calls': sum(row['denoising_steps'] for row in edited),
            'strict_prefix_latents': 41 if edited else None, 'regenerated_suffix_latents': 160 if edited else None,
            'repair_stage_seconds': sum(row['repair_wall_seconds'] for row in edited),
            'allocated_gpu_seconds': status['wall_seconds'],
            'artifact_sha256': {str(p.relative_to(folder)): sha(p) for p in [folder / 'manifest.json', folder / 'status.json',
                folder / 'config.yaml', folder / 'diagnostics/latents_0000.pt', folder / 'diagnostics/repair_0000.json',
                folder / 'diagnostics/switches_0000.json']}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = {name: inspect_run(args.root / name, args.parent) for name in ['off', 'zero']}
    report = {'phase': 'official_masked_parent_replay_engineering', 'runs': rows,
              'all_checks_passed': all(row['all_checks_passed'] for row in rows.values()),
              'quality_claim': False, 'held_out': False, 'branch_promoted': False,
              'limits': ['One exposed script with zero edited area; no repair effectiveness evidence.',
                         'Caches are reconstructed through full replay, not loaded from a complete state snapshot.',
                         'Actual wall times include full-history hashes, loading and export; allocated GPU time is not kernel time.',
                         'The controller, automatic acceptance and official nonzero-mask repair are separate validation steps.']}
    (args.root / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: {k: row[k] for k in ['all_checks_passed', 'changed_latent_frames', 'additional_repair_denoiser_calls',
        'all_five_switch_state_records_identical_except_cost']} for name, row in rows.items()}))
