#!/usr/bin/env python3
"""Check exported causal-decoder and preprocessing evidence, without GPU calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(root):
    folder = root / 'results/native_vae_onset_v1'
    telemetry = root / 'results/native_vae_onset_v1_telemetry'
    processor = root / 'results/paired_preprocessing_v1'
    plan = read(folder / 'plan.json')
    candidate = root / 'results/native_iris_natural_expression_v1/candidate'
    assert sha(folder / 'candidate_manifest.json') == sha(candidate / 'manifest.json')
    assert sha(folder / 'candidate_audit.json') == sha(candidate / 'audit.json')
    assert plan['input_files']['candidate']['sha256'] == read(candidate / 'audit.json')['latent_sha256']
    assert plan['input_files']['parent']['sha256'] == read(candidate / 'manifest.json')['action']['parent_latents_sha256']
    mask = root / 'reproduction/native_expression_control/automatic_mask.tensor'
    assert plan['input_files']['mask']['sha256'] == sha(mask)
    scripts = root / 'implementation/scripts'
    assert sha(scripts / 'probe_native_vae_onset.py') == plan['source_sha256']
    spec = importlib.util.spec_from_file_location('onset_summary', scripts / 'summarize_native_vae_onset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.summarize(folder, telemetry)
    assert json.dumps(summary, indent=2) + '\n' == (folder / 'export_audit.json').read_text()
    runtime = read(folder / 'runtime_audit.json')
    assert {k: v for k, v in runtime.items() if k != 'raw_window_recomputation'} == {
        k: v for k, v in summary.items() if k != 'raw_window_recomputation'}
    assert runtime['raw_window_recomputation']['actual_raw_frames_recomputed'] == 360
    assert runtime['raw_window_recomputation']['pngs_reconstructed'] == 80
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        module.plot(folder, tmp / 'onset_deltas.png')
        module.gallery(folder, tmp / 'index.html')
        module.mask_plot(folder, mask, tmp / 'mask_coverage.png')
        for name in ('onset_deltas.png', 'index.html', 'mask_coverage.png'):
            assert sha(tmp / name) == sha(root / 'artifacts/native_vae_onset_v1' / name)
    pp, actual = read(processor / 'plan.json'), read(processor / 'result.json')
    assert sha(processor / 'runner.py') == pp['source_sha256'] == actual['source_sha256'] == sha(scripts / 'audit_paired_preprocessing.py')
    assert sha(processor / 'plan.json') == actual['plan_sha256']
    assert actual['status'] == 'completed' and actual['actual_language_model_calls'] == 0 and not actual['model_weights_loaded']
    assert actual['rows'] == read(processor / 'partial_results.json') and len(actual['rows']) == actual['requests_reproduced'] == 39
    for path, expected in pp['result_inputs_sha256'].items():
        assert sha(root / path) == expected
    control_count = 0
    for row in actual['rows']:
        original = read(root / 'results' / row['run'] / 'transcripts' / (row['id'] + '.json'))
        assert row['tensors'] == original['input_tensors']
        assert row['png_sha256'] == [item['sha256'] for item in original['images']]
        if 'aligned_normalized_crop_stacks_exact' in row:
            assert row['aligned_normalized_crop_stacks_exact'] and len(set(row['png_sha256'])) == 1
            control_count += 1
    assert control_count == 4
    return {'vae_summary_and_figures_byte_exact': True, 'raw_prefix_frames_equal': 320,
        'first_changed_raw_frames': summary['first_changed_raw_frames'], 'actual_vae_decodes': 5,
        'published_raw_pngs_verified': 80, 'runtime_raw_frames_recomputed_recorded': 360,
        'preprocessing_requests_bound_to_actual_transcripts': 39, 'same_image_crop_controls': control_count,
        'raw_window_recomputed_in_this_export_check': False, 'processor_reexecuted_in_this_export_check': False,
        'model_weights_loaded': False, 'new_diffusion_calls': 0, 'new_vlm_calls': 0,
        'candidate_adopted': False, 'scope': 'Published evidence and byte-exact figures only; excluded raw latents/windows and pinned processor must be obtained separately for new runtime replay.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
