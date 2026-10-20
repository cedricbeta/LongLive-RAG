#!/usr/bin/env python3
"""Verify published policy/bridge and paired diagnostics without model execution."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def verify(root):
    import torch
    import yaml
    sys.path.insert(0, str(root / 'implementation'))
    from utils.agent_authored_expression import authored_garment_prompt
    from utils.agent_runtime import fingerprint
    contracts = root / 'results/authored_expression_policy_contracts'
    tests = read(contracts / 'exported_tests.json')
    assert tests['returncode'] == 0
    assert sha(contracts / 'exported_tests.txt') == tests['test_output_sha256']
    for path, expected in tests['source_sha256'].items():
        assert sha(root / path) == expected, path
    history = read(contracts / 'historical_source_bindings.json')
    for path, item in history.items():
        assert sha(root / item['historical_path']) == item['sha256']
        assert sha(root / path) != item['sha256']
    feedback = root / 'results/native_iris_mixed_evidence_v1/run'
    ledger = [json.loads(line) for line in (feedback / 'ledger.jsonl').read_text().splitlines()]
    first = next(row for row in ledger if row['event'] == 'bounded_action_reserved')
    initial = next(row for row in ledger if row['event'] == 'verifier' and row['attempt'] == 0)
    requirements = [row['constraint'] for row in initial['metrics']['constraint_assessments']
                    if row['constraint']['id'] in first['targets']]
    compiled = read(feedback / 'compiled_script.json')
    prompt = authored_garment_prompt(first['actual']['prompt'], requirements, compiled, first['shot_index'])
    reference = root / 'results/native_iris_natural_expression_v1/candidate'
    manifest = read(reference / 'manifest.json')
    mask_path = root / 'reproduction/native_expression_control/automatic_mask.tensor'
    mask = torch.load(mask_path, map_location='cpu', weights_only=True)
    assert mask['parent_video_sha256'] == first['actual']['parent_video_sha256']
    assert mask['parent_latents_sha256'] == first['actual']['parent_latents_sha256']
    coordinates = {'format': 'native', 'latent_range': mask['latent_range'],
                   'grid': mask['grid'], 'timing': mask['parent_timing_sha256']}
    digest = hashlib.sha256(json.dumps(coordinates, sort_keys=True).encode())
    digest.update(json.dumps([mask['latent_range'][0], list(mask['mask'].shape)]).encode())
    digest.update(mask['mask'].float().contiguous().numpy().tobytes())
    input_paths = {'feedback_ledger': feedback / 'ledger.jsonl', 'compiled_script': feedback / 'compiled_script.json',
        'reference_candidate_manifest': reference / 'manifest.json', 'reference_candidate_config': reference / 'config.yaml',
        'mask': mask_path}
    bridges = {}
    for version in (1, 2, 3):
        folder = root / 'results' / f'native_authored_expression_bridge_v{version}'
        native = read(folder / 'prepared/manifest.json')
        assert sha(folder / 'prepared/config.yaml') == native['config_sha256']
        config = yaml.safe_load((folder / 'prepared/config.yaml').read_text())
        assert config['repair_request']['prompts'] == [prompt]
        for field in ('action', 'action_fingerprint', 'source_sha256', 'paired_assets'):
            assert native[field] == manifest[field], field
        assert hashlib.sha256(prompt.encode()).hexdigest() == native['action']['prompt_sha256']
        assert hashlib.sha256(json.dumps(native['action'], sort_keys=True).encode()).hexdigest() == native['action_fingerprint']
        if version == 1:
            failure = read(folder / 'failure.json')
            assert failure['returncode'] == 1 and failure['whole_process_wall_seconds'] is None
            assert 'KeyError' in (folder / 'native_authored_expression_bridge_v1.log.txt').read_text()
            bridges[str(version)] = failure
            continue
        result = read(folder / 'result.json')
        assert result['target_requirements'] == requirements and result['first_actual_action_unchanged']
        assert result['new_model_calls'] == result['new_generated_videos'] == 0 and not result['candidate_retained']
        assert result['selected_action']['params']['prompt'] == prompt
        assert result['native_action_fingerprint'] == manifest['action_fingerprint']
        for key, path in input_paths.items():
            assert sha(path) == result['input_sha256'][key], key
        for path, expected in result['source_sha256'].items():
            assert sha(folder / 'source' / path) == expected, path
            if version == 3:
                target = root / path if path.startswith('adapters/') else root / 'implementation' / path
                assert sha(target) == expected, path
        if version == 3:
            assert read(folder / 'plan.json')['source_sha256'] == result['source_sha256']
        actual = result['policy_actual']
        assert actual['prompt'] == prompt and actual['coordinates'] == coordinates
        assert actual['mask_tensor_sha256'] == digest.hexdigest()
        assert actual['parent_latents_sha256'] == mask['parent_latents_sha256']
        assert actual['parent_video_sha256'] == mask['parent_video_sha256']
        family = {k: v for k, v in actual.items() if k not in ('family_fingerprint', 'action_fingerprint', 'noise_seed')}
        assert fingerprint(family) == actual['family_fingerprint']
        assert fingerprint({**family, 'noise_seed': actual['noise_seed']}) == actual['action_fingerprint']
        supervisor = read(folder / f'native_authored_expression_bridge_v{version}_supervisor.json')
        assert supervisor['returncode'] == 0
        bridges[str(version)] = {'worker_seconds': result['wall_seconds'], 'supervisor': supervisor,
            'native_action_fingerprint': result['native_action_fingerprint'], 'final_canonical_source_bound': version == 3}
    scripts = root / 'implementation/scripts'
    auditor = module(scripts / 'audit_native_paired_reviews.py', 'paired_export_auditor')
    audits = {}
    for name in ('native_iris_paired_review_v1', 'native_iris_short_pair_descriptions_v1'):
        folder = root / 'results' / name
        result = auditor.audit(root, folder, root / 'results' / (name + '_telemetry'))
        assert json.dumps(result, indent=2) + '\n' == (folder / 'audit.json').read_text()
        audits[name] = {k: result[k] for k in ('actual_calls', 'valid_replies', 'unattempted_case_ids',
            'all_planned_source_frames_reconstructed', 'supervisor_seconds')}
    summary = module(scripts / 'summarize_native_paired_reviews.py', 'paired_export_summary')
    result = summary.summarize(root)
    assert json.dumps(result, indent=2) + '\n' == (contracts / 'paired_summary.json').read_text()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / 'gallery.html'
        summary.gallery(result, output)
        assert sha(output) == sha(root / 'artifacts/native_iris_paired_reviews/index.html')
    return {'current_test_bound_sources': len(tests['source_sha256']), 'historical_source_bindings': len(history),
        'bridge_attempts': bridges, 'paired_probes': audits, 'audits_summary_gallery_byte_exact': True,
        'new_model_calls': 0, 'new_generated_videos': 0, 'raw_parent_latents_rehashed': False,
        'model_weights_rehashed': False, 'candidate_retained': False,
        'scope': 'Verifies recorded native preparation, final code bindings, derived prompt/mask/action fingerprints and all paired evidence. Does not rerun preparation, raw latents, models or semantic acceptance.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
