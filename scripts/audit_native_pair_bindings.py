#!/usr/bin/env python3
"""Audit actual paired actions and completed first-render bindings."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(feedback, control):
    inputs = {}
    def read(folder, name, lines=False):
        path = folder / name
        inputs[folder.name + '/' + name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return [json.loads(x) for x in path.read_text().splitlines()] if lines else json.loads(path.read_text())
    actions, first_renders, second_statuses, states = [], [], [], []
    for folder in (feedback, control):
        events = read(folder, 'run/ledger.jsonl', True)
        reserved = [r for r in events if r['event'] == 'bounded_action_reserved']
        assert len(reserved) == 2 and [r['attempt'] for r in reserved] == [1, 2]
        actions.append(reserved)
        first_status = read(folder, 'run/backend/candidate_0001_shot02_attempt1/status.json')
        assert first_status['status'] == 'completed'
        first_renders.append(read(folder, 'run/backend/candidate_0001_shot02_attempt1/audit.json'))
        second_statuses.append(read(folder, 'run/backend/candidate_0002_shot02_attempt2/status.json')['status'])
        states.append(read(folder, 'run/status.json')['status'])
    a, b = actions
    second_differences = [key for key in sorted(set(a[1]['actual']) | set(b[1]['actual']))
                          if a[1]['actual'].get(key) != b[1]['actual'].get(key)]
    first_equal = a[0]['actual'] == b[0]['actual']
    unchanged = {key: a[1]['actual'][key] == b[1]['actual'][key] for key in
        ('coordinates', 'entry_sigma', 'mask_tensor_sha256', 'noise_seed',
         'parent_latents_sha256', 'parent_video_sha256', 'shot')}
    control_change = [key for key in sorted(b[0]['actual'])
                      if b[0]['actual'][key] != b[1]['actual'][key]]
    assert first_equal and all(unchanged.values())
    assert second_differences == ['action_fingerprint', 'family_fingerprint', 'prompt']
    assert control_change == ['action_fingerprint', 'noise_seed']
    assert b[1]['actual']['noise_seed'] == a[1]['actual']['noise_seed'] == 4244
    assert first_renders[0]['video_sha256'] == first_renders[1]['video_sha256']
    assert first_renders[0]['latent_sha256'] == first_renders[1]['latent_sha256']
    return {'phase': 'paired_actual_action_audit', 'feedback_run': feedback.name, 'control_run': control.name,
        'worker_statuses_at_audit': states, 'second_render_statuses_at_audit': second_statuses,
        'initial_actual_actions_equal': first_equal, 'initial_video_bytes_equal': True,
        'initial_latent_file_bytes_equal': True,
        'initial_video_sha256': first_renders[0]['video_sha256'],
        'initial_latent_sha256': first_renders[0]['latent_sha256'],
        'second_action_shared_inputs': unchanged, 'second_action_differing_fields': second_differences,
        'second_shared_noise_seed': 4244, 'control_only_changes_noise': True,
        'feedback_second_prompt': a[1]['actual']['prompt'], 'control_second_prompt': b[1]['actual']['prompt'],
        'new_model_calls': 0, 'new_generated_videos': 0, 'quality_claim': False,
        'limits': ['This audit proves recorded input equivalence, not identical later outputs or useful repair.',
                   'The feedback policy additionally incurs a planner call; actual total costs differ.',
                   'First-render hashes are compared from completed engineering audits; raw latents are retained in the runtime, not Git.',
                   'A completed second render is not sufficient evidence of a terminal trial verdict.'],
        'inputs_sha256': inputs}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feedback', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.feedback.resolve(), args.control.resolve())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('initial_actual_actions_equal',
        'initial_video_bytes_equal', 'initial_latent_file_bytes_equal', 'second_action_differing_fields',
        'second_render_statuses_at_audit')}))
