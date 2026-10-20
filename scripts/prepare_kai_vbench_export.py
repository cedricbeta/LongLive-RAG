#!/usr/bin/env python3
"""Prepare fresh metric attempts from portable published inputs and pinned assets."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time
from prepare_kai_vbench_comparison import sha, read, write


def prepare(args):
    started = time.monotonic()
    repo, output, checkout = (p.resolve() for p in (args.repo, args.output, args.checkout))
    published = repo / 'results/native_kai_vbench_visual_v1'
    manifest = read(published / 'export_manifest.json')
    old_root = Path(manifest['runtime_root'])
    old_checkout = old_root.parent / 'official_vbench'
    assets = read(repo / 'results/official_vbench_provision_v1/model_assets_manifest.json')
    old_cache, cache = Path(assets['cache_root']), Path.home() / '.cache'
    policy = copy.deepcopy(read(published / 'plan.json'))
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == policy['evaluator_revision']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout, text=True).strip()
    for arm in policy['inputs']:
        archived = read(published / arm / 'plan.json')
        for name, version in archived['package_versions'].items():
            assert importlib.metadata.version(name) == version, name
        key = arm + '/input/moviegen000-0.mp4'
        entry = manifest['files'][key]
        source = repo / entry['repository_path']
        assert sha(source) == entry['sha256'] == policy['inputs'][arm]['sha256']
        policy['inputs'][arm]['source'] = str(source)
    policy.update(frozen_unix=time.time(), replay_of_published_plan_sha256=sha(published / 'plan.json'),
                  preparer_sha256=sha(Path(__file__)))
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'plan.json', policy)
    for name in ('source_plan.json', 'source_summary.json', 'source_pair_check.json'):
        shutil.copyfile(published / name, output / name)
    for arm, item in policy['inputs'].items():
        folder = output / arm
        (folder / 'input').mkdir(parents=True)
        shutil.copyfile(item['source'], folder / 'input/moviegen000-0.mp4')
        shutil.copyfile(published / arm / 'supervisor.py', folder / 'supervisor.py')
        plan = copy.deepcopy(read(published / arm / 'plan.json'))
        checks = {}
        for raw, expected in plan['source_and_asset_sha256'].items():
            path = Path(raw)
            if path.is_relative_to(old_checkout):
                new_path = checkout / path.relative_to(old_checkout)
            elif path.is_relative_to(old_cache):
                new_path = cache / path.relative_to(old_cache)
            elif path.is_relative_to(old_root):
                relative = path.relative_to(old_root)
                new_path = output / relative
                if relative == Path('plan.json'):
                    expected = sha(new_path)
            else:
                raise ValueError('Unmapped frozen input: ' + raw)
            assert sha(new_path) == expected, new_path
            checks[str(new_path)] = expected
        command = plan['command']
        command[0] = str(args.python.absolute())
        command[2] = str(checkout / 'vbench2_beta_long/eval_long.py')
        command[command.index('--videos_path') + 1] = str(folder / 'input')
        command[command.index('--output_path') + 1] = str(folder / 'outputs')
        plan['environment']['PYTHONPATH'] = str(checkout)
        plan.update(frozen_unix=policy['frozen_unix'], checkout=str(checkout), source_video=item['source'],
                    shared_comparison_plan_sha256=sha(output / 'plan.json'), source_and_asset_sha256=checks,
                    replay_of_published_arm_plan_sha256=sha(published / arm / 'plan.json'))
        write(folder / 'plan.json', plan)
        shutil.copyfile(published / arm / 'manifest.json', folder / 'manifest.json')
    queue = (published / 'run_queue.py').read_text()
    replacements = {
        'workspace = root.parents[2]': 'workspace = Path(' + repr(str(repo / 'implementation')) + ')',
        "eval_python = root.parent / 'env_vbench_long/bin/python'": 'eval_python = Path(' + repr(str(args.python.absolute())) + ')',
        "observer_python = '/home/chendong/miniconda3/envs/sglang/bin/python'": 'observer_python = ' + repr(str(args.observer_python.absolute())),
    }
    for original, replacement in replacements.items():
        assert queue.count(original) == 1
        queue = queue.replace(original, replacement)
    (output / 'run_queue.py').write_text(queue)
    write(output / 'preparation.json', {'status': 'prepared_only', 'wall_seconds': time.monotonic() - started,
        'actual_evaluations_started': 0, 'published_manifest_sha256': sha(published / 'export_manifest.json'),
        'path_rebinding_only': True, 'inputs': policy['inputs']})
    print(json.dumps({'status': 'prepared_only', 'root': str(output), 'evaluations_started': 0,
                      'seconds': time.monotonic() - started}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--observer-python', type=Path, required=True)
    prepare(parser.parse_args())
