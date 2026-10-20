#!/usr/bin/env python3
"""Freeze three one-shot visual evaluations of the closed Kai timing control."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    started = time.monotonic()
    root, reference, study = (p.resolve() for p in (args.root, args.reference, args.study))
    old = read(reference / 'plan.json')
    assert read(reference / 'status.json')['status'] == 'completed'
    assert len(read(reference / 'audit.json')['rows']) == 6
    assert read(study / 'summary.json')['remaining_generation_budget'] == 0
    assert read(study / 'pair_check.json')['status'] == 'passed'
    assert old['package_versions']['av'] == '12.0.0'
    for package, version in old['package_versions'].items():
        assert importlib.metadata.version(package) == version, package
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=old['checkout'], text=True).strip() == old['revision']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=old['checkout'], text=True).strip()
    checks = {}
    for path, expected in old['source_and_asset_sha256'].items():
        assert sha(Path(path)) == expected, path
        if not Path(path).is_relative_to(reference):
            checks[path] = expected
    arms = {
        'B_original': ('59aa1d38f032a1c23c66bb0e4ea367952fab201cbe3e6d5123b8f2865df948ce', 0),
        'RP_local_cue': ('30850c2d845ee9ab91fd8c54756be45b1a6684114d4074c2f15333759821230f', 2),
        'P20_prompt': ('78a0a4959ac529ab6449e2078b1b5bf80b7f8ee35d03ab4288fe06fa68fcd907', 0),
    }
    inputs = {}
    for arm, (expected, gpu) in arms.items():
        source = study / arm / 'videos/rank0-0-0_lora.mp4'
        assert read(study / arm / 'status.json')['status'] == 'completed'
        assert sha(source) == expected, arm
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries', 'stream=width,height,avg_frame_rate,duration,nb_read_frames',
            '-of', 'json', str(source)], text=True))['streams'][0]
        assert probe == dict(width=832, height=480, avg_frame_rate='16/1', duration='60.000000', nb_read_frames='960'), probe
        inputs[arm] = {'source': str(source), 'sha256': expected, 'gpu': gpu, 'probe': probe,
                      'generation_seconds_previously_incurred': read(study / arm / 'status.json')['wall_seconds']}
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, root / 'preparer.py')
    for name in ('plan.json', 'pair_check.json', 'summary.json'):
        shutil.copyfile(study / name, root / ('source_' + name))
    policy = {
        'protocol': 'closed_ns280_same_boundary_vbench_visual_comparison_v1', 'frozen_unix': time.time(),
        'inputs': inputs, 'dimensions': old['dimensions'], 'evaluator_revision': old['revision'],
        'reference_evaluation_plan_sha256': sha(reference / 'plan.json'),
        'source_study_plan_sha256': sha(study / 'plan.json'),
        'source_study_pair_check_sha256': sha(study / 'pair_check.json'),
        'evaluation_ceiling': 3, 'attempts_per_arm': 1, 'timeout_seconds_per_arm': 900,
        'new_video_generation_budget': 0, 'new_semantic_checker_budget': 0,
        'execution': 'B on GPU 0 and RP on GPU 2 concurrently; P20 on GPU 0 after both terminal.',
        'comparisons': ['RP_local_cue minus P20_prompt (primary visual comparison)',
                        'P20_prompt minus B_original', 'RP_local_cue minus B_original'],
        'aggregation': 'Report all six official scalars and paired raw-scale differences. No synthetic overall score, confidence interval, threshold tuning, or acceptance decision.',
        'scope': 'One previously exposed development script, seed 0, three frozen full 60-second outputs. Thirty correlated two-second clips per video are not independent samples. Visual metrics do not verify hair color, entity identity, action compliance, or repair correctness.',
        'failure_policy': 'Preserve every attempt and partial output. No retry, new checkpoint, or new video based on scores.',
        'legacy_filename': 'moviegen000-0.mp4 is solely the inherited supervisor input identifier. All three inputs are NarraStream ns280, not MovieGen.',
        'automatic_acceptance': False, 'human_labels': 0, 'held_out': False,
        'preparer_sha256': sha(Path(__file__)),
    }
    write(root / 'plan.json', policy)
    for arm, item in inputs.items():
        folder = root / arm
        (folder / 'input').mkdir(parents=True)
        video = folder / 'input/moviegen000-0.mp4'
        shutil.copyfile(item['source'], video)
        shutil.copyfile(reference / 'supervisor.py', folder / 'supervisor.py')
        plan = copy.deepcopy(old)
        command = plan['command']
        command[command.index('--videos_path') + 1] = str(folder / 'input')
        command[command.index('--output_path') + 1] = str(folder / 'outputs')
        plan.update(protocol=policy['protocol'], frozen_unix=policy['frozen_unix'], arm=arm,
                    gpu=item['gpu'], timeout_seconds=900, source_video=item['source'],
                    video_sha256=item['sha256'], prompt=None, source_benchmark='NarraStream ns280',
                    shared_comparison_plan_sha256=sha(root / 'plan.json'),
                    quality_scope=policy['scope'], aggregation=policy['aggregation'],
                    failure_policy=policy['failure_policy'])
        plan['environment']['CUDA_VISIBLE_DEVICES'] = str(item['gpu'])
        plan['expected_preprocessing']['clip_count'] = 30
        plan['source_and_asset_sha256'] = {**checks, str(video): sha(video),
            str(folder / 'supervisor.py'): sha(folder / 'supervisor.py'),
            str(root / 'plan.json'): sha(root / 'plan.json')}
        write(folder / 'plan.json', plan)
        write(folder / 'manifest.json', {'gpu': item['gpu'], 'phase': policy['protocol'], 'video_sha256': item['sha256']})
    write(root / 'preparation.json', {'status': 'prepared', 'wall_seconds': time.monotonic() - started,
        'source_and_asset_hashes_per_arm': len(plan['source_and_asset_sha256']), 'inputs': inputs})
    print(json.dumps({'status': 'prepared', 'root': str(root), 'seconds': time.monotonic() - started}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    prepare(parser.parse_args())
