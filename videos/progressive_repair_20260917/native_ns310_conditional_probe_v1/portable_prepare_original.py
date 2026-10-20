#!/usr/bin/env python3
"""Rebind the published ns310 original preparation without launching generation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
for name in ['repo', 'output', 'checkout', 'python', 'audit-python', 'observer-python']:
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--archive', type=Path, help='Defaults to the published ns310 result directory.')
a = p.parse_args()
repo, output = a.repo.resolve(), a.output.resolve()
archive = (a.archive or repo / 'results/native_ns310_conditional_probe_v1').resolve()
reference = repo / 'results/native_development_pilot12_batch6_v1/ns180/baseline'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not output.exists()
output.mkdir(parents=True)
shutil.copyfile(archive / 'prepare_original.py', output / 'prepare_original.py')
manifest = json.loads((reference / 'manifest.json').read_text())
manifest['cwd'] = str(a.checkout.resolve())
with tempfile.TemporaryDirectory(prefix='native-ns310-reference-') as tmp:
    tmp = Path(tmp)
    (tmp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for name in ['config.yaml', 'status.json', 'audit.json']:
        shutil.copyfile(reference / name, tmp / name)
    subprocess.run([str(a.python.absolute()), str(output / 'prepare_original.py'),
        '--output', str(output), '--reference', str(tmp),
        '--benchmark', str(repo / 'results/native_development_pilot12_v1/benchmark.jsonl'),
        '--supervisor', str(repo / 'adapters/official_longlive_rag/run_prepared.py')], check=True)
    shutil.copyfile(tmp / 'manifest.json', output / 'rebound_reference_manifest.json')
queue = (archive / 'run_original.py').read_text()
changes = {
    'workspace = root.parents[2]': 'workspace = Path(' + repr(str(repo / 'implementation')) + ')',
    "python = root.parent / 'env_longlive_rag/bin/python'": 'python = Path(' + repr(str(a.python.absolute())) + ')',
    "observer_python = '/home/chendong/miniconda3/envs/sglang/bin/python'": 'observer_python = ' + repr(str(a.observer_python.absolute())),
    "str(workspace/'venv/bin/python')": repr(str(a.audit_python.absolute()))}
for before, after in changes.items():
    assert queue.count(before) == 1, before
    queue = queue.replace(before, after)
(output / 'run_original.py').write_text(queue)
(output / 'portable_binding.json').write_text(json.dumps({
    'published_reference_manifest_sha256': sha(reference / 'manifest.json'),
    'rebound_reference_manifest_sha256': sha(output / 'rebound_reference_manifest.json'),
    'archived_prepare_sha256': sha(archive / 'prepare_original.py'),
    'archived_queue_sha256': sha(archive / 'run_original.py'),
    'rebound_queue_sha256': sha(output / 'run_original.py'),
    'baseline_attempts_started': 0, 'followup_attempts_started': 0}, indent=2) + '\n')
print('Prepared the single original only; no GPU generation or conditional repair launched.')
