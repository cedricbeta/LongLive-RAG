#!/usr/bin/env python3
"""Audit a terminal mixed-evidence trial against its earlier first action."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_native_window import summarize


def main(root, name, prior):
    result = summarize(root, name)
    folder = root / 'results' / name
    earlier = root / 'results' / prior
    def read(path, lines=False):
        result['inputs_sha256'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return [json.loads(line) for line in path.read_text().splitlines()] if lines else json.loads(path.read_text())
    current = read(folder / 'run/ledger.jsonl', True)
    old = read(earlier / 'run/ledger.jsonl', True)
    actions = [r for r in current if r['event'] == 'bounded_action_reserved']
    old_first = next(r for r in old if r['event'] == 'bounded_action_reserved')
    plans = [r for r in current if r['event'] == 'bounded_feedback_plan']
    routing = [r for r in current if r['event'] == 'progressive_evidence_routing']
    result.update(evidence_routing=routing, actual_actions=actions, feedback_plans=plans,
                  historical_first_trial=prior, same_compute_efficacy_comparison=False,
                  first_action_matches_historical=None, first_video_matches_historical=None,
                  changed_action_audits=[])
    if actions:
        first = actions[0]['actual']
        result['first_action_matches_historical'] = first == old_first['actual']
        fields = sorted(set(first) | set(old_first['actual']))
        result['first_action_differing_fields'] = [k for k in fields if first.get(k) != old_first['actual'].get(k)]
        for action in actions[1:]:
            now = action['actual']
            result['changed_action_audits'].append({'attempt': action['attempt'],
                'program_kind': action['kind'], 'same_parent_video': now['parent_video_sha256'] == first['parent_video_sha256'],
                'same_parent_latents': now['parent_latents_sha256'] == first['parent_latents_sha256'],
                'same_mask_tensor': now['mask_tensor_sha256'] == first['mask_tensor_sha256'],
                'same_coordinates': now['coordinates'] == first['coordinates'],
                'original_prompt_preserved_as_prefix': now['prompt'].startswith(first['prompt']),
                'prompt_changed': now['prompt'] != first['prompt'],
                'family_changed': now['family_fingerprint'] != first['family_fingerprint'],
                'strength': now['entry_sigma'], 'noise_seed': now['noise_seed']})
    videos = [(r['folder'], r['engineering_audit']['video_sha256']) for r in result['native_render_jobs']
              if r.get('engineering_audit')]
    old_audit = read(earlier / 'run/backend/candidate_0001_shot02_attempt1/audit.json')
    if videos:
        result['first_video_matches_historical'] = videos[0][1] == old_audit['video_sha256']
    result['limits'] += [
        'This development trial permits up to two total candidates; the historical trial allowed one. It is not a budget-matched efficacy comparison.',
        'Routing categorizes recorded model labels, not semantic truth or actual visibility.',
        'An eligible next action is not evidence that the candidate was retained or independently improved.',
        'No output-protocol probe variant is adopted; the original checker remains in use.']
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', default='native_iris_mixed_evidence_v1')
    p.add_argument('--prior', default='native_iris_names_v6_window')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = main(args.root.resolve(), args.name, args.prior)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ['recorded_model_calls', 'completed_native_videos',
        'first_action_matches_historical', 'first_video_matches_historical', 'selected_parent_changed']}))
