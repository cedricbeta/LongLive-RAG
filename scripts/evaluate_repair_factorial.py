#!/usr/bin/env python3
"""Evaluate every factorial first shot using the frozen historical judge.

Development evidence only: same Qwen family as the planner, no independent
identity/action audit. No cell is selected or dropped. Judge scope is shot 0.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.experiment / 'source'))
    from utils.agent_entity_judge import EntityDriftVerifier, Sam2Tracker, load_entity_specs
    from utils.agent_role_assignment import ContinuityRoleLocalizer
    from utils.agent_appearance_color import ColorCheckedAppearanceComparer, set_part_device
    from utils.agent_runtime import NormativeGraph, ShotObservation, TemporalInput, file_sha256, fingerprint
    from utils.agent_vlm import FingerprintedVLMClient
    manifest = json.loads((args.experiment / 'manifest.json').read_text())
    scene = manifest['scene']
    old = json.loads((args.archive / 'manifest.json').read_text())
    original = next(c for c in old['cases'] if c['scene'] == scene)
    population = [{'id': 'baseline', 'video': original['baseline_video'], 'sha256': original['baseline_sha256']}]
    for cell in manifest['cells']:
        status = json.loads((args.experiment / cell['id'] / 'status.json').read_text())
        if status['status'] == 'running':
            raise RuntimeError('Cell still running: ' + cell['id'])
        if status['status'] != 'completed':
            population.append({'id': cell['id'], 'generation_status': status['status'], 'video': None})
        else:
            video, = status['outputs']
            population.append({'id': cell['id'], 'video': str(args.experiment / video['path']), 'sha256': video['sha256']})
    result = {'phase': 'development_diagnostic', 'quality_claim': False, 'independent_model': False,
              'checked_shots_0based': [0], 'population': population, 'cases': [],
              'limitations': ['Historical automatic judge, no independent human/model assessment.',
                              'Only first-shot evidence; no future/whole-film quality guarantee.']}
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    specs = load_entity_specs(args.archive / 'frozen/entity_specs.json', scene)
    graph = NormativeGraph.compile_scene(args.archive / 'frozen/data' / scene, allow_missing_global=True)
    client = FingerprintedVLMClient(base_url=args.base_url, model='Qwen/Qwen3-VL-8B-Instruct',
                                    cache_dir=args.output / 'vlm_cache', max_tokens=1024, seed=0)
    result['sampling'] = client.sampling_config
    tracker = Sam2Tracker(device=args.device)
    set_part_device(args.device)
    for item in population:
        started = time.monotonic()
        row = {'id': item['id']}
        try:
            if not item['video']:
                row['status'] = 'generation_failed'
            else:
                video = Path(item['video'])
                if file_sha256(video) != item['sha256']:
                    raise ValueError('Video bytes changed')
                verifier = EntityDriftVerifier(
                    specs, localizer=ContinuityRoleLocalizer(client),
                    comparer=ColorCheckedAppearanceComparer(client), tracker=tracker,
                    work_dir=args.output / item['id'], place_absent='never', uncertain_policy='accept_unknown',
                    attribute_masks=True, part_device=args.device, judge_chunks='sparse',
                )
                observation = ShotObservation(shot_index=0, attempt=0, frames=[], crops=[],
                    telemetry={'shot_video': str(video), 'chunk_start': 0, 'chunk_end': 7},
                    inputs=(TemporalInput(kind='candidate_video', identifier=str(video), sha256=item['sha256'], available_after_shot=0),),
                    input_fingerprint=fingerprint({'video_sha256': item['sha256'], 'shot': 0, 'source': 'frozen_factorial'}))
                verdict = verifier.judge(observation, graph.view(0))
                row.update(status='completed', accept=verdict.accept, reason=verdict.reason, metrics=verdict.metrics)
        except Exception as error:
            row.update(status='evaluation_failed', error=f'{type(error).__name__}: {error}')
        row['wall_seconds'] = time.monotonic() - started
        result['cases'].append(row)
        (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: row[k] for k in ('id', 'status', 'wall_seconds')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
