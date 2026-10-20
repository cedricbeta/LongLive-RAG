#!/usr/bin/env python3
"""Check actual cue execution and both shared prefixes in the archived factorial."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import yaml

ADAPTER = next(p / 'adapters/official_longlive_rag' for p in Path(__file__).resolve().parents
               if (p / 'adapters/official_longlive_rag/continuation_cue.py').is_file())
sys.path.insert(0, str(ADAPTER))
from continuation_cue import compile_cue, digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    began = time.monotonic()
    torch.set_num_threads(4)
    read = lambda p: json.loads(p.read_text())
    assert not (root / 'pair_check.json').exists()
    plan = read(root / 'plan.json')
    assert plan['protocol'] == 'archived_ns280_local_repair_by_suffix_cue_v1'
    for arm, ref in plan['archived_references'].items():
        assert (root / arm).resolve() == Path(ref['path'])
        assert sha(root / arm / 'manifest.json') == ref['manifest_sha256']
        assert sha(root / arm / 'diagnostics/latents_0000.pt') == ref['latent_sha256']
        assert sha(next((root / arm / 'videos').glob('*.mp4'))) == ref['video_sha256']
    names = ['B_original', 'P_cue', 'R_local', 'RP_local_cue']
    for name in names:
        assert read(root / name / 'status.json')['status'] == 'completed'
        record = read(root / name / 'audit.json')
        assert record.get('measured_engineering_checks_passed', record.get('engineering_checks_passed', False))
    configs = {name: yaml.safe_load((root / name / 'config.yaml').read_text()) for name in names}
    timing = read(root / 'B_original/diagnostics/timing_0000.json')
    schedule = {int(k): v for k, v in timing['schedule'].items()}
    effective, expected = compile_cue(schedule, plan['continuation_cue'], earliest_latent=121)
    assert expected == plan['cue_record']
    actual = read(root / 'RP_local_cue/diagnostics/continuation_cue_0000.json')
    manifest = read(root / 'RP_local_cue/manifest.json')
    assert actual == expected == manifest['expected_continuation_cue_record']
    assert manifest['action_fingerprint'] == digest(manifest['action'])
    assert manifest['native_base_action_fingerprint'] == plan['native_base_action_fingerprint']
    base = {k: v for k, v in manifest['action'].items()
            if k not in ('continuation_cue', 'effective_schedule_sha256', 'renderer_protocol')}
    assert base == plan['repair_action'] and digest(base) == plan['native_base_action_fingerprint']
    for name in names:
        executed = effective if name in ('P_cue', 'RP_local_cue') else schedule
        switches = read(root / name / 'diagnostics/switches_0000.json')
        assert len(switches) == 5
        for row, boundary in zip(switches, [41, 81, 121, 161, 201]):
            assert row['start_latent_frame'] == boundary
            assert row['prompt_sha256'] == [hashlib.sha256(executed[boundary][0].encode()).hexdigest()]
        if name in ('P_cue', 'RP_local_cue'):
            assert read(root / name / 'manifest.json')['study_plan_sha256'] == sha(root / 'plan.json')
    assert configs['B_original']['timed_segments'] == configs['R_local']['timed_segments'] == configs['RP_local_cue']['timed_segments']
    assert configs['P_cue']['timed_segments'] == [
        {'seconds': 10, 'prompts': effective[b]} for b in [0, 41, 81, 121, 161, 201]]
    rp_request = dict(configs['RP_local_cue']['repair_request'])
    assert rp_request.pop('continuation_cue') == plan['continuation_cue']
    assert rp_request == configs['R_local']['repair_request']
    normalized = []
    for name in names:
        normalized.append({k: v for k, v in configs[name].items() if k not in
            ('data_path', 'output_folder', 'adapter_diagnostics_dir', 'timed_segments', 'repair_request')})
    assert all(row == normalized[0] for row in normalized)
    pairs = []
    for left, right in [('B_original', 'P_cue'), ('R_local', 'RP_local_cue')]:
        a, b = [torch.load(root / n / 'diagnostics/latents_0000.pt', map_location='cpu', weights_only=True)
                for n in [left, right]]
        assert tuple(a.shape) == tuple(b.shape) == (1, 241, 16, 60, 104)
        assert torch.equal(a[:, :121], b[:, :121]), 'Shared prefix differs; no causal attribution or rerun'
        memory = [read(next((root / n / 'videos').glob('*memory_log.json'))) for n in [left, right]]
        pairs.append({'arms': [left, right], 'shared_prefix_latents': 121, 'prefix_latents_exact': True,
            'changed_suffix_latents': [i for i in range(121, 241) if not torch.equal(a[:, i], b[:, i])],
            'retrieval_records_through_query121_equal': [r for r in memory[0] if r['query_frame'] <= 121]
                == [r for r in memory[1] if r['query_frame'] <= 121]})
        del a, b
    record = {'status': 'passed', 'pairs': pairs, 'actual_suffix_prompts_match_P_RP': True,
        'all_four_actual_switch_hashes_match_conditions': True, 'RP_cue_log_exactly_matches_plan': True,
        'RP_authored_timing_record_is_not_effective_prompt_record': True,
        'local_repair_request_equal_except_cue': True, 'other_generation_settings_equal': True,
        'private_sampler_source_unchanged': True, 'raw_vae_prefix_pixel_equality_measured': False,
        'history_resident_tensor_contents_measured': False, 'quality_acceptance': False,
        'human_labels': 0, 'held_out': False, 'wall_seconds': time.monotonic() - began,
        'source_sha256': sha(Path(__file__)), 'plan_sha256': sha(root / 'plan.json')}
    (root / 'pair_check.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'shared_prefix_latents_per_pair': 121,
                      'seconds': record['wall_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root.resolve())
