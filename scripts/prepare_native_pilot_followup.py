#!/usr/bin/env python3
"""Prepare two further originals from the frozen cohort after a completed batch."""
import argparse, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
import yaml

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def prepare(args):
    started = time.monotonic()
    root, cohort, previous, reference = [p.resolve() for p in [args.root, args.cohort, args.previous_batch, args.reference]]
    read = lambda p: json.loads(p.read_text())
    write = lambda p, value: p.write_text(json.dumps(value, indent=2) + '\n')
    prior, batch = read(cohort / 'plan.json'), read(previous / 'plan.json')
    summary = read(previous / 'summary.json')
    assert prior['protocol'] == 'twelve_exposed_scripts_native_development_pilot_v1'
    assert read(previous / 'queue_status.json')['status'] == 'completed'
    assert batch['prior_cohort_plan_sha256'] == sha(cohort / 'plan.json')
    assert summary['selection_plan_sha256'] == sha(previous / 'plan.json')
    assert batch['cases'] == prior['cases']
    completed = summary['completed_original_indices']
    assert len(completed) == len(set(completed)) == summary['total_completed_originals']
    assert set(batch['batch_indices']).issubset(completed)
    for index in batch['batch_indices']:
        folder = previous / f'ns{index:03d}' / 'baseline'
        assert read(folder / 'status.json')['status'] == 'completed'
        assert read(folder / 'audit.json')['engineering_checks_passed']
    chosen = [case for case in prior['cases'] if case['index'] not in completed][:2]
    assert len(chosen) == 2
    manifest = read(reference / 'manifest.json')
    assert read(reference / 'status.json')['status'] == 'completed'
    checkout = Path(manifest['cwd'])
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == manifest['official_revision']
    assert sha(reference / 'config.yaml') == manifest['config_sha256']
    assert manifest['source_sha256'] == prior['generation_source_sha256']
    for name, digest in manifest['source_sha256'].items():
        assert sha(checkout / name) == digest, name
    assert manifest['paired_assets'] == prior['paired_assets']
    for row in prior['paired_assets']:
        assert sha(checkout / row['path']) == row['sha256'], row['path']
    original = yaml.safe_load((reference / 'config.yaml').read_text())
    assert original['seed'] == 0 and 'repair_request' not in original and original['audit_prompt_switch'] is False
    assert shutil.disk_usage(root.parent).free > 8 * 2**30
    root.mkdir(parents=True, exist_ok=False)
    count = len(completed) + len(chosen)
    plan = {
        'protocol': 'twelve_script_pilot_followup_original_batch_v1', 'frozen_unix': time.time(),
        'prior_cohort_plan_sha256': sha(cohort / 'plan.json'), 'prior_cohort_plan': str(cohort / 'plan.json'),
        'previous_batch_plan_sha256': sha(previous / 'plan.json'), 'previous_batch_summary_sha256': sha(previous / 'summary.json'),
        'selection': 'Next two unrendered cases in the original frozen scene order. No output-based replacement or hidden appearance requirements.',
        'cases': prior['cases'], 'batch_indices': [case['index'] for case in chosen],
        'previously_completed_original_indices': completed, 'planned_script_count': 12,
        'new_original_attempt_ceiling': 2, 'new_repair_or_control_attempt_ceiling': 0,
        'expected_completed_original_count_after_this_batch': count, 'remaining_originals_after_success': 12 - count,
        'seed': 0, 'seconds': 60, 'fps': 16, 'resolution': [832, 480], 'per_case_timeout_seconds': 1800,
        'generation_source_sha256': manifest['source_sha256'], 'paired_assets': prior['paired_assets'],
        'reference_manifest_sha256': sha(reference / 'manifest.json'), 'preparer_sha256': sha(Path(__file__)),
        'review_frames': prior['review_frames'], 'benchmark_sha256': prior['benchmark_sha256'],
        'failure_policy': 'One attempt per original. Preserve failures; no restart, seed search, replacement, extra appearance or repair in this batch.',
        'comparative_pilot_status': 'Original baseline inputs only. Full multi-strategy action/budget/evaluator protocol remains pending; no efficacy claim.',
        'hidden_cast_metadata_to_generator': False, 'held_out': False, 'automatic_acceptance': False,
        'quality_claim': False, 'human_labels': 0}
    write(root / 'plan.json', plan)
    shutil.copyfile(Path(__file__), root / 'prepare_source.py')
    for case, gpu in zip(chosen, [0, 2]):
        folder = root / case['id'] / 'baseline'
        folder.mkdir(parents=True)
        config = {**original, 'data_path': str(folder / 'initial_prompt.txt'), 'output_folder': str(folder / 'videos'),
            'adapter_diagnostics_dir': str(folder / 'diagnostics'),
            'timed_segments': [{'seconds': 10, 'prompts': [text]} for text in case['prompts']]}
        (folder / 'initial_prompt.txt').write_text(case['prompts'][0] + '\n')
        (folder / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        child = {**manifest, 'phase': plan['protocol'], 'gpu': gpu, 'source_benchmark_index': case['index'],
            'command': [sys.executable, '-u', 'inference.py', '--config_path', str(folder / 'config.yaml')],
            'timeout_seconds': 1800, 'config_sha256': sha(folder / 'config.yaml'), 'source_case_plan_sha256': sha(root / 'plan.json'),
            'purpose': 'Next two original videos from the frozen twelve-script development cohort; zero repair/control budget.'}
        write(folder / 'manifest.json', child)
        shutil.copyfile(root / 'plan.json', folder / 'plan.json')
        shutil.copyfile(args.supervisor, folder / 'supervisor.py')
    write(root / 'preparation.json', {'status': 'prepared_unlaunched', 'wall_seconds': time.monotonic() - started,
        'assets_verified': len(prior['paired_assets']), 'new_original_indices': plan['batch_indices']})
    print(json.dumps(read(root / 'preparation.json')))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'cohort', 'previous-batch', 'reference', 'supervisor']:
        p.add_argument('--' + name, type=Path, required=True)
    prepare(p.parse_args())
