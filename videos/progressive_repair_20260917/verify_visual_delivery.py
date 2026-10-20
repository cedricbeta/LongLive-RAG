from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, tempfile, time

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--compare', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = a.repo.resolve()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_text())
git = lambda folder, *args: subprocess.check_output(['git', *args], cwd=folder, text=True).strip()
exports = []
with tempfile.TemporaryDirectory(prefix='visual-delivery-') as temporary:
    tmp = Path(temporary)
    for name, figure in [('visual_schema_context_v1', 'schema_text'), ('visual_evidence_first_v1', 'evidence_first')]:
        folder = root / 'results' / name
        exported = read(folder / 'export_manifest.json')['files_sha256']
        for relative, expected in exported.items():
            assert sha(folder / relative) == expected, relative
        manifest = read(folder / 'run/manifest.json')
        assert sha(folder / 'plan.json') == manifest['plan_sha256']
        frozen = folder / 'run/source'
        for relative, expected in manifest['source_sha256'].items():
            assert sha(frozen / relative) == expected, relative
        for relative, expected in read(folder / 'plan.json')['source_sha256'].items():
            assert sha(frozen / relative) == expected, relative
        code = """import pathlib, runpy, sys
source = pathlib.Path(sys.argv[1])
sys.argv = [str(source / 'scripts/probe_visual_schema_context.py'), '--help']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as error:
    assert error.code == 0
for name in ['agent_attribute_diagnosis', 'agent_runtime', 'agent_schema_context', 'agent_vlm']:
    assert pathlib.Path(sys.modules['utils.' + name].__file__).is_relative_to(source)
"""
        subprocess.run([sys.executable, '-c', code, str(frozen)], cwd=tmp,
                       env={**os.environ, 'PYTHONPATH': str(frozen)}, check=True, capture_output=True, text=True)
        rebuilt = tmp / (name + '.json')
        subprocess.run([sys.executable, str(root / 'implementation/scripts/summarize_visual_schema_context.py'),
                        '--folder', str(folder), '--output', str(rebuilt)], cwd=tmp, check=True, capture_output=True, text=True)
        assert rebuilt.read_bytes() == (folder / 'summary.json').read_bytes()
        summary = read(rebuilt)
        assert summary['actual_http_requests'] == 14 and not summary['failures']
        assert summary['adopted_arm'] is None and not summary['actual_human_labels']
        assert len(summary['request_audit']) == 14 and all(row['request_bytes_verified'] for row in summary['request_audit'])
        image = tmp / (figure + '.png')
        subprocess.run([sys.executable, str(root / 'implementation/scripts/plot_visual_protocol_probe.py'),
                        '--folder', str(folder), '--output', str(image)], cwd=tmp, check=True, capture_output=True, text=True)
        artifact = root / 'artifacts/visual_checker_protocols' / (figure + '.png')
        assert image.read_bytes() == artifact.read_bytes()
        assert image.with_suffix('.json').read_bytes() == artifact.with_suffix('.json').read_bytes()
        binding = read(artifact.with_suffix('.json'))
        assert sha(folder / 'summary.json') == binding['summary_sha256']
        assert sha(artifact) == binding['figure_sha256']
        assert sha(root / 'implementation/scripts/plot_visual_protocol_probe.py') == binding['source_script_sha256']
        exports.append({'run': name, 'export_files': len(exported), 'frozen_sources': len(manifest['source_sha256']),
                        'summary_byte_exact': True, 'figure_byte_exact': True, 'actual_http_requests_verified': 14,
                        'frozen_imports_verified_in_clean_cwd': True})

record = read(root / 'results/name_candidate_contracts/exported_tests.json')
for relative, expected in record['source_sha256'].items():
    assert sha(root / relative) == expected, relative
paper = read(root / 'paper/tables/provenance.json')
for relative, expected in paper.items():
    assert sha(root / relative) == expected, relative
result = {'repository': 'https://github.com/cedricbeta/ICLR27-Longlive-agent',
          'branch': 'research/progressive-repair-20260917', 'commit': git(root, 'rev-parse', 'HEAD'),
          'verified_unix': time.time(), 'exports': exports, 'test_source_hashes': len(record['source_sha256']),
          'paper_inputs': len(paper), 'paper_sha256': sha(root / 'paper/build/main.pdf'),
          'submission_ready': False, 'new_model_calls': 0}
if a.compare:
    compare = a.compare.resolve()
    assert git(compare, 'rev-parse', 'HEAD') == result['commit']
    files = lambda folder: set(filter(None, subprocess.check_output(['git', 'ls-files', '-z'], cwd=folder).decode().split('\0')))
    tracked = files(root)
    assert files(compare) == tracked
    hashes = {}
    for relative in sorted(tracked):
        hashes[relative] = sha(root / relative)
        assert sha(compare / relative) == hashes[relative], relative
    assert not any('private_assignment_key' in path for path in tracked)
    for entry in exports:
        folder = root / 'results' / entry['run']
        for relative in read(folder / 'export_manifest.json')['files_sha256']:
            assert str((folder / relative).relative_to(root)) in tracked
    result.update(tracked_files_verified=len(hashes), private_human_key_absent=True, files_sha256=hashes,
                  method='Independent SSH fetch, all tracked file SHA256 compared to delivery; frozen imports, exact summaries and figures reproduced offline; 28 actual HTTP payload hashes reconstructed.')
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'files_sha256'}, indent=2))
