#!/usr/bin/env python3
"""Replay evidence routing on an archived native ledger; never relabel a video."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from utils.agent_constraints import attach_constraint_assessments
from utils.agent_evidence_routing import mixed_target_revision
from utils.agent_progressive_repair import improvement_decision
from utils.agent_runtime import Verdict


def audit(folder):
    hashes = {}
    def read(path):
        hashes[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    manifest = read(folder / 'run/manifest.json')
    origin = Path(manifest['arguments']['output'])
    ledger = folder / 'run/ledger.jsonl'
    hashes[str(ledger.relative_to(folder))] = hashlib.sha256(ledger.read_bytes()).hexdigest()
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows = [r for r in events if r['event'] == 'verifier']
    assert len(rows) == 2 and [r['attempt'] for r in rows] == [0, 1]
    targets = next(r['targets'] for r in events if r['event'] == 'bounded_action_reserved')
    reviews, preserved = [], []
    for row in rows:
        metrics = attach_constraint_assessments(row['metrics'], shot_index=row['shot_index'])
        old = {r['constraint']['id']: r for r in row['metrics']['constraint_assessments']}
        for assessment in metrics['constraint_assessments']:
            historical = old[assessment['constraint']['id']]
            for key in ['constraint', 'status', 'score', 'violations']:
                assert assessment[key] == historical[key], (assessment['constraint']['id'], key)
            for evidence in assessment['evidence']:
                meta = evidence['metadata']
                if 'frame_path' in meta:
                    path = folder / 'run' / Path(meta['frame_path']).relative_to(origin)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    assert digest == meta['frame_sha256']
                    hashes[str(path.relative_to(folder))] = digest
        preserved.append({'attempt': row['attempt'], 'constraints': len(old),
                          'all_definitions_statuses_scores_violations_unchanged': True})
        reviews.append(Verdict(row['accept'], row['reason'], metrics=metrics,
                               input_fingerprint=row['input_fingerprint']))
    decision = improvement_decision(reviews[0], reviews[1], reviews[0], targets)
    historical = next(r['decision'] for r in events if r['event'] == 'progressive_current_guard')
    assert not decision['allow'] and not historical['allow']
    for key in decision:
        assert (sorted(decision[key]) == sorted(historical[key]) if key == 'reasons'
                else decision[key] == historical[key]), key
    return {'scope': 'Offline routing replay of one already exposed, rejected native candidate.',
            'acceptance_unchanged': True, 'preserved_verdicts': preserved,
            'original_termination': next(r['reason'] for r in events if r['event'] == 'repair_finished'),
            'routing': mixed_target_revision(reviews[0], reviews[1], targets),
            'new_model_calls': 0, 'new_video_generations': 0, 'actual_human_labels': 0,
            'candidate_kept': False, 'quality_claim': False, 'inputs_sha256': hashes}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.folder.resolve())
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: result[key] for key in ['acceptance_unchanged', 'routing', 'new_model_calls']}))
