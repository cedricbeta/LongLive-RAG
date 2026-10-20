#!/usr/bin/env python3
"""Audit context-only action bindings, native state guards and OFF conformance."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
import yaml


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit(args):
    began = time.monotonic()
    torch.set_num_threads(4)
    folder = args.folder.resolve()
    plan = read(folder / 'plan.json')
    case = next(c for c in plan['cases'] if c['id'] == args.case)
    output = folder / args.case
    reference = Path(plan['reference'])
    if (output / 'context_audit.json').exists():
        raise FileExistsError('Audit already exists')
    manifest = read(output / 'manifest.json')
    assert manifest['context_control_plan_sha256'] == sha(folder / 'plan.json')
    assert manifest['config_sha256'] == sha(output / 'config.yaml')
    assert sha(reference / 'manifest.json') == plan['reference_manifest_sha256']
    assert read(output / 'status.json')['status'] == 'completed'
    assert sha(reference / 'diagnostics/latents_0000.pt') == plan['reference_latents_sha256']
    assert sha(next((reference / 'videos').glob('*.mp4'))) == plan['reference_video_sha256']
    config = yaml.safe_load((output / 'config.yaml').read_text())
    assert config['repair_request']['repair_context_mode'] == case['mode']
    expected_action = {**read(reference / 'manifest.json')['action'],
        'repair_context_mode': case['mode'], 'renderer_protocol': plan['protocol']}
    assert manifest['action'] == expected_action
    assert manifest['action_fingerprint'] == hashlib.sha256(json.dumps(expected_action, sort_keys=True).encode()).hexdigest()
    for name, digest in plan['native_source_sha256'].items():
        assert sha(Path(plan['checkout']) / name) == sha(folder / 'native_source' / name) == digest
    # Reuse the same actual-parent/latent/mask/native-state audit as older candidates.
    sys.path.insert(0, str(folder / 'source'))
    spec = importlib.util.spec_from_file_location('context_native_audit', folder / 'source/audit_candidate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not (output / 'audit.json').exists():
        module.audit(output, Path(plan['parent']), output / 'audit.json')
    native_audit = read(output / 'audit.json')
    assert native_audit['all_engineering_checks_passed']
    repairs = read(output / 'diagnostics/repair_0000.json')
    traces = []
    for block in repairs:
        if block['phase'] != 'edit':
            assert 'context_trace' not in block
            continue
        assert len(block['context_trace']) == 4
        for step in block['context_trace']:
            assert step['mode'] == case['mode'] and step['callable_restored'] and step['layers_expected'] == 30
            if args.case == 'off':
                assert step['intercepted_calls'] == []
            else:
                assert len(step['intercepted_calls']) == 30
                for layer in step['intercepted_calls']:
                    assert layer['query_tokens'] == layer['used_key_tokens'] == block['latent_frames'] * 1560
                    assert layer['native_key_tokens'] == 12 * 1560
            traces.append(step)
    assert len(traces) == 56
    old = torch.load(reference / 'diagnostics/latents_0000.pt', map_location='cpu', weights_only=True)
    new = torch.load(output / 'diagnostics/latents_0000.pt', map_location='cpu', weights_only=True)
    equality = [torch.equal(old[:, i], new[:, i]) for i in range(old.shape[1])]
    video_equal = sha(next((output / 'videos').glob('*.mp4'))) == plan['reference_video_sha256']
    memory_equal = read(next((output / 'videos').glob('*memory_log.json'))) == read(next((reference / 'videos').glob('*memory_log.json')))
    def exclude(rows, keys):
        return [{k: v for k, v in row.items() if k not in keys} for row in rows]
    repairs_equal = exclude(repairs, {'context_trace', 'repair_wall_seconds'}) == exclude(
        read(reference / 'diagnostics/repair_0000.json'), {'repair_wall_seconds'})
    switches_equal = exclude(read(output / 'diagnostics/switches_0000.json'), {'wall_seconds_including_optional_audit'}) == exclude(
        read(reference / 'diagnostics/switches_0000.json'), {'wall_seconds_including_optional_audit'})
    off_pass = args.case == 'off' and all(equality) and video_equal and memory_equal and repairs_equal and switches_equal
    record = {'status': 'completed', 'case': args.case, 'mode': case['mode'],
        'plan_sha256': sha(folder / 'plan.json'), 'audit_source_sha256': sha(Path(__file__)),
        'action_fingerprint': manifest['action_fingerprint'], 'native_engineering_audit_sha256': sha(output / 'audit.json'),
        'native_engineering_checks_passed': True, 'private_forwards': len(traces),
        'intercepted_self_attention_calls': sum(len(t['intercepted_calls']) for t in traces),
        'context_only_private_calls': True, 'context_callable_restored_each_step': True,
        'equal_reference_latent_indices': [i for i, equal in enumerate(equality) if equal],
        'different_reference_latent_indices': [i for i, equal in enumerate(equality) if not equal],
        'reference_mp4_bytes_exact': video_equal, 'reference_retrieval_records_exact': memory_equal,
        'reference_repair_records_exact_excluding_time_and_context_trace': repairs_equal,
        'reference_switch_records_exact_excluding_time': switches_equal,
        'off_conformance_passed': off_pass if args.case == 'off' else None,
        'audit_wall_seconds': time.monotonic() - began,
        'candidate_retained': False, 'quality_claim': False,
        'limits': ['Recorded private scopes plus the pinned installation source establish interception boundaries; ordinary attention calls are not separately instrumented.',
                   'Engineering invariants and differences do not establish improved clothing, identity, action or later-shot quality.',
                   'MP4 equality is checked against the archived C candidate; compressed prefix differences against the original parent are a separate codec effect.']}
    (output / 'context_audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('case', 'private_forwards', 'intercepted_self_attention_calls', 'off_conformance_passed')}))
    if args.case == 'off' and not off_pass:
        raise RuntimeError('OFF conformance failed: do not run ON')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--case', choices=('off', 'on'), required=True)
    audit(p.parse_args())
