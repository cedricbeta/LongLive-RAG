from pathlib import Path
import hashlib, json, subprocess, sys, tempfile, time

source = Path('/home/chendong/video-gen/LongLive-RAG')
runtime = source / 'videos/progressive_repair_20260917'
local = source.parent / 'ICLR27-Longlive-agent'
remote = runtime / 'remote_verified_checkout'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
git = lambda p, *args: subprocess.check_output(['git', *args], cwd=p, text=True).strip()
commit = git(local, 'rev-parse', 'HEAD')
assert git(remote, 'rev-parse', 'HEAD') == commit
paths = lambda root: set(filter(None, subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')))
files = paths(local)
assert paths(remote) == files
hashes = {}
for name in sorted(files):
    hashes[name] = sha(remote / name)
    assert sha(local / name) == hashes[name], name
assert not any('private_assignment_key' in name for name in files)
tests = {}
for name in ['exported_tests', 'fixed_control_tests']:
    record = read(remote / f'results/mixed_evidence_contracts/{name}.json')
    assert record['returncode'] == 0
    for path, expected in record['source_sha256'].items():
        assert sha(remote / path) == expected, path
    tests[name] = len(record['source_sha256'])
plans = []
for name in ['native_iris_mixed_evidence_v1', 'native_iris_fixed_noise_v1']:
    record = read(remote / 'results' / name / 'plan.json')
    for path, expected in record['source_sha256'].items():
        target = remote / path if path.startswith('adapters/') else remote / 'implementation' / path
        assert sha(target) == expected, path
    plans.append({'name': name, 'source_hashes': len(record['source_sha256']),
                  'max_worker_seconds': record['max_seconds'], 'max_renders': record['max_edits_per_shot']})
with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp) / 'replay.json'
    subprocess.run([sys.executable, str(remote / 'implementation/scripts/audit_progressive_evidence.py'),
        '--folder', str(remote / 'results/native_iris_names_v6_window'), '--output', str(output)],
        cwd=tmp, check=True, capture_output=True)
    assert output.read_bytes() == (remote / 'results/mixed_evidence_contracts/archived_routing_replay.json').read_bytes()
native_exports = {}
with tempfile.TemporaryDirectory() as tmp:
    for name in ['native_iris_mixed_evidence_v1', 'native_iris_fixed_noise_v1']:
        output = Path(tmp) / (name + '.json')
        subprocess.run([sys.executable, str(remote / 'implementation/scripts/verify_native_trial_export.py'),
            '--root', str(remote), '--name', name, '--output', str(output)],
            cwd=tmp, check=True, capture_output=True)
        native_exports[name] = read(output)
    pair = read(remote / 'results/native_iris_budget_pair_v1/provenance.json')
    for field in ['source_sha256', 'outputs_sha256']:
        for path, expected in pair[field].items():
            assert sha(remote / path) == expected, path
    commands = [
        ['summarize_native_pair.py', '--root', str(remote), '--output', str(Path(tmp) / 'summary.json')],
        ['audit_native_pair_bindings.py', '--feedback', str(remote / 'results/native_iris_mixed_evidence_v1'),
         '--control', str(remote / 'results/native_iris_fixed_noise_v1'), '--output', str(Path(tmp) / 'action_bindings.json')],
        ['plot_native_pair.py', '--root', str(remote), '--output', str(Path(tmp) / 'review_frames.png')],
        ['verify_native_single_prompt_export.py', '--root', str(remote), '--output', str(Path(tmp) / 'long-check.json')],
        ['plot_native_single_prompt.py', '--root', str(remote), '--output', str(Path(tmp) / 'fixed_samples.png')]]
    for command in commands:
        subprocess.run([sys.executable, str(remote / 'implementation/scripts' / command[0]), *command[1:]],
                       cwd=tmp, check=True, capture_output=True)
    for path in pair['outputs_sha256']:
        assert sha(remote / path) == sha(Path(tmp) / Path(path).name), path
    for suffix in ['.png', '.json']:
        assert sha(remote / ('artifacts/official_moviegen_000_120s_v1/fixed_samples' + suffix)) == sha(Path(tmp) / ('fixed_samples' + suffix))
    long_export = read(Path(tmp) / 'long-check.json')
telemetry_tests = read(remote / 'results/native_gpu_telemetry_contracts/exported_tests.json')
assert telemetry_tests['returncode'] == 0
for path, expected in telemetry_tests['source_sha256'].items():
    assert sha(remote / path) == expected, path
assert sha(remote / 'results/native_gpu_telemetry_contracts/exported_tests.txt') == telemetry_tests['test_output_sha256']
tests['telemetry_tests'] = len(telemetry_tests['source_sha256'])
long_folder = remote / 'results/official_moviegen_000_120s_v1'
long_plan = read(long_folder / 'plan.json')
for name, expected in long_plan['source_sha256'].items():
    canonical = remote / name if name.startswith('adapters/') else remote / 'implementation' / name
    assert sha(canonical) == sha(long_folder / 'source' / name) == expected, name
for name, expected in long_plan['runtime_files_sha256'].items():
    assert sha(long_folder / name) == expected, name
assert sha(long_folder / 'queue_source.py') == long_plan['queue_source_sha256']
timing = read(long_folder / 'timing_preflight.json')['timing']
assert timing['latent_frames'] == 481 and timing['export_frames'] == 1920 and len(timing['schedule']) == 1
assert len(timing['blocks']) == 161 and timing['blocks'][-1] == 1
paper = read(remote / 'paper/tables/provenance.json')
for path, expected in paper.items():
    assert sha(remote / path) == expected, path
result = {'repository': 'https://github.com/cedricbeta/ICLR27-Longlive-agent',
    'branch': 'research/progressive-repair-20260917', 'commit': commit, 'verified_unix': time.time(),
    'tracked_files_verified': len(files), 'test_source_hashes': tests, 'frozen_plans': plans,
    'archived_routing_replay_byte_exact': True, 'paper_inputs': len(paper),
    'paper_sha256': sha(remote / 'paper/build/main.pdf'), 'paper_pages': int(__import__('re').search(r'Pages:\s+(\d+)', subprocess.check_output(['pdfinfo', str(remote / 'paper/build/main.pdf')], text=True)).group(1)),
    'private_human_key_absent': True, 'submission_ready': False, 'trial_outcomes': {'feedback': 'completed_zero_retained', 'fixed_noise': 'completed_zero_retained', 'single_prompt_120s': 'completed_engineering_audit'}, 'native_exports': native_exports, 'paired_summary_bindings_and_figure_byte_exact': True, 'single_prompt_export': long_export, 'single_prompt_figure_byte_exact': True, 'single_prompt_preflight_verified': True,
    'method': 'Independent SSH fetch and all tracked hashes; test-bound sources, both terminal native exports with actual model images and 480 mask slices, paired summaries/bindings/figures, actual 120-second decoded video/sample frames and recomputed telemetry, paper input hashes verified. Raw latents are runtime-only and were not regenerated.',
    'files_sha256': hashes}
for path in [runtime / 'remote_delivery_check_latest.json', runtime / f'remote_delivery_check_{commit[:7]}.json']:
    path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'files_sha256'}, indent=2))
