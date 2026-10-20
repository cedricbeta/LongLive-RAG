#!/usr/bin/env python3
"""Verify exported wording-control evidence without generating or judging video."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import torch
import yaml


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
    folder = root / 'results/native_iris_natural_expression_v1'
    candidate = folder / 'candidate'
    reference = root / 'results/native_iris_fixed_noise_v1/run/backend/candidate_0002_shot02_attempt2'
    obs = root / 'results/native_iris_natural_expression_observations_v1'
    plan, binding = read(folder / 'plan.json'), read(folder / 'binding.json')
    old, new = read(reference / 'manifest.json'), read(candidate / 'manifest.json')
    assert sha(reference / 'manifest.json') == plan['reference_manifest_sha256']
    assert sha(reference / 'config.yaml') == plan['reference_config_sha256']
    assert sha(candidate / 'config.yaml') == new['config_sha256']
    assert binding['plan_sha256'] == sha(folder / 'plan.json') == sha(candidate / 'plan.json')
    assert binding['old_action'] == old['action'] and binding['new_action'] == new['action']
    assert binding['only_action_difference'] == [k for k in old['action'] if old['action'][k] != new['action'][k]] == ['prompt_sha256']
    assert old['paired_assets'] == new['paired_assets']
    assert old['source_sha256'] == new['source_sha256']
    for path, expected in new['source_sha256'].items():
        assert sha(candidate / 'source' / path) == expected
    for path, expected in plan['adapter_source_sha256'].items():
        assert sha(folder / 'adapter_source' / path) == expected
    assert sha(folder / 'prepare_source.py') == plan['prepare_source_sha256']
    assert sha(folder / 'compiled_script.json') == plan['compiled_script_sha256']
    assert sha(folder / 'observation_reference_plan.json') == plan['observation_reference_plan_sha256']
    oldconfig = yaml.safe_load((reference / 'config.yaml').read_text())
    config = yaml.safe_load((candidate / 'config.yaml').read_text())
    request, oldrequest = config.pop('repair_request'), oldconfig.pop('repair_request')
    # Runtime output paths are relocated; generator configuration is unchanged.
    for key in ('output_folder', 'adapter_diagnostics_dir', 'data_path'):
        config.pop(key); oldconfig.pop(key)
    assert sha(candidate / 'initial_prompt.txt') == sha(reference / 'initial_prompt.txt')
    assert config == oldconfig
    assert set(request) == set(oldrequest)
    assert [k for k in request if request[k] != oldrequest[k]] == ['prompts', 'mask_path']
    assert request['prompts'] == [plan['new_prompt']]
    assert oldrequest['prompts'] == [plan['old_prompt']]
    assert plan['new_prompt'] == plan['old_prompt'] + ' ' + plan['appended_sentence']
    assert sha(candidate / 'repair_prompt.txt') == sha(folder / 'prompt.txt')
    assert hashlib.sha256(plan['new_prompt'].encode()).hexdigest() == new['action']['prompt_sha256']
    compiled = read(folder / 'compiled_script.json')
    assert plan['appended_sentence'] == 'Iris wears a ' + ' '.join(r['expected'] for r in plan['prompt_source_rows']) + '.'
    for row in plan['prompt_source_rows']:
        assert row['source']['quote'] in compiled['prompts'][row['source']['shot']]
        assert row['expected'] in row['source']['quote']
    maskpath = root / 'reproduction/native_expression_control/automatic_mask.tensor'
    assert sha(maskpath) == plan['automatic_mask_file_sha256'] == request['mask_sha256']
    mask = torch.load(maskpath, map_location='cpu', weights_only=True)
    assert list(mask['mask'].shape) == new['action']['mask_shape']
    assert hashlib.sha256(mask['mask'].to(torch.uint8).contiguous().numpy().tobytes()).hexdigest() == new['action']['mask_tensor_sha256']
    audit, status = read(candidate / 'audit.json'), read(candidate / 'status.json')
    assert audit['status'] == status and status['status'] == 'completed' and status['returncode'] == 0
    assert audit['audit_source_sha256'] == sha(folder / 'adapter_source/audit_candidate.py')
    assert audit['all_engineering_checks_passed'] and not audit['candidate_retained']
    video = root / 'artifacts/native_iris_natural_expression_v1/candidate.mp4'
    parent = root / 'artifacts/official_iris_60s_baseline_prepared/videos/rank0-0-0_lora.mp4'
    assert sha(video) == audit['video_sha256']
    assert sha(parent) == request['parent_video_sha256']
    comparison = module(root / 'adapters/official_longlive_rag/audit_controls.py', 'controls').video_comparison(parent, video)
    assert comparison == audit['video_comparison']
    display = video.with_name('four_way_comparison.mp4')
    display_record = read(display.with_suffix('.json'))
    assert sha(display) == display_record['output_sha256']
    assert sha(root / 'implementation/scripts/plot_native_expression_control.py') == display_record['source_sha256']
    for path, expected in display_record['inputs_sha256'].items():
        assert sha(root / path) == expected
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,duration,nb_read_frames', '-of', 'json', str(display)], text=True))['streams'][0]
    assert probe == display_record['output_probe']
    telemetry = root / 'results/native_iris_natural_expression_v1_telemetry'
    tp, ts = read(telemetry / 'plan.json'), read(telemetry / 'summary.json')
    assert tp['target_plan_sha256'] == sha(candidate / 'plan.json')
    assert tp['target_manifest_sha256'] == sha(candidate / 'manifest.json')
    assert tp['source_sha256'] == sha(telemetry / 'monitor_source.py')
    assert read(telemetry / 'target_terminal_status.json') == status
    assert ts['target_status_sha256'] == sha(telemetry / 'target_terminal_status.json')
    assert ts['samples_sha256'] == sha(telemetry / 'samples.jsonl')
    samples = [json.loads(s) for s in (telemetry / 'samples.jsonl').read_text().splitlines()]
    assert module(telemetry / 'monitor_source.py', 'telemetry').summarize(samples, status) == {
        k: v for k, v in ts.items() if k not in ('samples_sha256', 'target_status_sha256')}
    op = read(obs / 'plan.json')
    assert sha(obs / 'render_plan.json') == op['render_plan_sha256'] == sha(folder / 'plan.json')
    assert sha(obs / 'render_audit.json') == op['render_audit_sha256'] == sha(candidate / 'audit.json')
    assert sha(obs / 'reference_plan.json') == op['observation_reference_plan_sha256'] == plan['observation_reference_plan_sha256']
    previous = read(obs / 'reference_plan.json')
    for key in ('prompt', 'generation', 'seed_per_call', 'dtype', 'attention', 'package_versions',
                'model', 'model_revision', 'source_sha256', 'model_manifest_sha256'):
        assert op[key] == previous[key]
    assert sorted(c['frame'] for c in op['cases'] if c['control'] is None) == plan['review_frames']
    return {'phase': 'offline_natural_expression_export_verification', 'action_difference': ['prompt_sha256'],
        'same_parent_mask_seed_sigma_assets_and_native_sources': True,
        'actual_mask_file_and_tensor_checked': True, 'decoded_parent_candidate_frames_compared': comparison['frames'],
        'full_video_pixel_comparison_byte_exact': True, 'comparison_video_frames': int(probe['nb_read_frames']),
        'telemetry_recomputed_exactly': True, 'followup_protocol_inheritance_verified': True,
        'raw_latents_rechecked': False, 'weights_reloaded': False, 'new_model_calls': 0,
        'candidate_adopted': False, 'quality_claim': False,
        'scope': 'Published bytes, actual decoded video, mask, settings, sources and telemetry verified. Latent invariants remain bound to the recorded runtime audit; raw latents are not exported. No semantic acceptance check is executed.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    result = verify(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
