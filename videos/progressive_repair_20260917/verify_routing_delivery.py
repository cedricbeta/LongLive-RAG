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
paper = read(remote / 'paper/tables/provenance.json')
for path, expected in paper.items():
    assert sha(remote / path) == expected, path
result = {'repository': 'https://github.com/cedricbeta/ICLR27-Longlive-agent',
    'branch': 'research/progressive-repair-20260917', 'commit': commit, 'verified_unix': time.time(),
    'tracked_files_verified': len(files), 'test_source_hashes': tests, 'frozen_plans': plans,
    'archived_routing_replay_byte_exact': True, 'paper_inputs': len(paper),
    'paper_sha256': sha(remote / 'paper/build/main.pdf'), 'paper_pages': 14,
    'private_human_key_absent': True, 'submission_ready': False, 'trial_outcomes': 'pending',
    'method': 'Independent SSH fetch, all tracked file hashes matched; new test-bound sources, both frozen plan sources, paper inputs and byte-exact offline routing replay verified.',
    'files_sha256': hashes}
for path in [runtime / 'remote_delivery_check_latest.json', runtime / f'remote_delivery_check_{commit[:7]}.json']:
    path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'files_sha256'}, indent=2))
