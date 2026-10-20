#!/usr/bin/env python3
"""Select twelve exposed scripts before new outputs; prepare only the first two originals."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    started = time.monotonic()
    root, reference = args.root.resolve(), args.reference.resolve()
    original = read(reference / 'manifest.json')
    checkout = Path(original['cwd'])
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == original['official_revision']
    for name, digest in original['source_sha256'].items():
        assert sha(checkout / name) == digest
    assets = read(args.assets_from)['paired_assets']
    assert len(assets) == 12
    for row in assets:
        assert sha(checkout / row['path']) == row['sha256'], row['path']
    exposure, structure = read(args.exposure), read(args.structure)
    assert sha(args.benchmark) == exposure['corpus_sha256'] == structure['corpus_sha256']
    scripts = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    assert len(scripts) == 324
    characters = {row['index']: row['characters_in_first_mention_order'] for row in structure['rows']}
    excluded = {0, 90, 190, 220, 280, 310}
    groups = {}
    for row in exposure['rows']:
        if not row['direct_exposure'] or row['index'] in excluded:
            continue
        assert len(row['context_groups']) == 1
        groups.setdefault(row['context_groups'][0], []).append(row['index'])
    assert len(groups) == 12
    cases = []
    for scene, indices in sorted(groups.items()):
        index = min(indices, key=lambda i: (len(characters[i]), i))
        cases.append({'id': f'ns{index:03d}', 'index': index, 'scene': scene,
            'characters_for_sampling_only': characters[index], 'prompts': scripts[index]['prompts']})
    first = sorted([row for row in cases if len(row['characters_for_sampling_only']) == 2
                    and not set(row['characters_for_sampling_only']) & {'Iris', 'Kai', 'Noah'}], key=lambda row: row['index'])
    assert [row['index'] for row in first] == [40, 130]
    root.mkdir(parents=True, exist_ok=False)
    for path, name in [(args.benchmark, 'benchmark.jsonl'), (args.exposure, 'exposure.json'),
                       (args.structure, 'structure.json'), (Path(__file__), 'prepare_source.py')]:
        shutil.copyfile(path, root / name)
    (root / 'generation_source').mkdir()
    for name in original['source_sha256']:
        dest = root / 'generation_source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkout / name, dest)
    plan = {'protocol': 'twelve_exposed_scripts_native_development_pilot_v1', 'frozen_unix': time.time(),
        'selection': 'One script per scene from the historical exposed 33: minimum named-cast size, then minimum index; exclude six closed native case studies. No new output or defect score used.',
        'excluded_closed_native_indices': sorted(excluded), 'cases': cases,
        'selection_metadata_scope': 'Historical cast registry is used only for sampling, never passed as hidden entity/appearance requirements to generation or repair.',
        'first_batch_selection': 'Two-person selected scripts excluding Iris/Kai/Noah to move away from repeatedly diagnosed actors: ns040 and ns130, ascending index.',
        'first_batch_indices': [40, 130], 'prepared_baseline_indices': [40, 130],
        'planned_script_count': 12, 'baseline_attempts_maximum_this_batch': 2,
        'new_repair_or_control_attempts_authorized_by_this_plan': 0,
        'comparative_pilot_status': 'Input selection only; full multi-strategy action/budget/evaluator protocol is not yet frozen or executed. These are exact original baseline inputs for that later comparison.',
        'planned_strategies': ['native', 'fixed local retry', 'Best-of-N', 'prompt-only feedback', 'full progressive repair'],
        'seed': 0, 'seconds': 60, 'fps': 16, 'resolution': [832, 480], 'per_case_timeout_seconds': 1800,
        'benchmark_sha256': sha(args.benchmark), 'exposure_sha256': sha(args.exposure), 'structure_sha256': sha(args.structure),
        'preparer_sha256': sha(Path(__file__)), 'reference_manifest_sha256': sha(reference / 'manifest.json'),
        'generation_source_sha256': original['source_sha256'], 'paired_assets': assets,
        'review_frames': [frame for shot in range(6) for frame in (shot * 160, shot * 160 + 80, shot * 160 + 159)],
        'failure_policy': 'Keep failed/missing original slots. No restart, replacement scene, seed search, extra appearance prompt or defect-conditioned sample removal.',
        'audit_prompt_switch': False, 'history_resident_byte_hashes_measured': False,
        'held_out': False, 'automatic_acceptance': False, 'human_labels': 0,
        'quality_claim': False, 'submission_ready': False}
    write(root / 'plan.json', plan)
    for row, gpu in zip(first, [0, 2]):
        folder = root / row['id'] / 'baseline'
        folder.mkdir(parents=True)
        config = yaml.safe_load((reference / 'config.yaml').read_text())
        assert config['seed'] == 0 and 'repair_request' not in config
        config.update(data_path=str(folder / 'initial_prompt.txt'), output_folder=str(folder / 'videos'),
            adapter_diagnostics_dir=str(folder / 'diagnostics'), audit_prompt_switch=False,
            timed_segments=[{'seconds': 10, 'prompts': [prompt]} for prompt in row['prompts']])
        (folder / 'initial_prompt.txt').write_text(row['prompts'][0] + '\n')
        (folder / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        manifest = {**original, 'phase': plan['protocol'], 'gpu': gpu, 'source_benchmark_index': row['index'],
            'command': [sys.executable, '-u', 'inference.py', '--config_path', str(folder / 'config.yaml')],
            'config_sha256': sha(folder / 'config.yaml'), 'paired_assets': assets,
            'source_case_plan_sha256': sha(root / 'plan.json'), 'purpose': 'First two exact original videos for the preselected twelve-script development pilot; no repair benefit assumed.'}
        write(folder / 'manifest.json', manifest)
        shutil.copyfile(args.supervisor, folder / 'supervisor.py')
        shutil.copyfile(root / 'plan.json', folder / 'plan.json')
    write(root / 'preparation.json', {'status': 'prepared_only', 'wall_seconds': time.monotonic() - started,
        'selected_indices': [row['index'] for row in cases], 'prepared_indices': [40, 130], 'generation_attempts_started': 0})
    print(json.dumps(read(root / 'preparation.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'reference', 'assets-from', 'benchmark', 'exposure', 'structure', 'supervisor']:
        parser.add_argument('--' + name, type=Path, required=True)
    prepare(parser.parse_args())
