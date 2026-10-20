#!/usr/bin/env python3
"""Recompute temporal decisions from frozen frame observations, with no model call."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.agent_attribute_diagnosis import aggregate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    started = time.monotonic()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = json.loads((args.probe / 'result.json').read_text())
    manifest = json.loads((args.probe / 'manifest.json').read_text())
    original = before['result']['attribute_assessments']
    requirements = manifest['frozen_requirements']
    frame_map = {}
    for attribute in original:
        for item in attribute['evidence']:
            frame = frame_map.setdefault(item['sha256'], {'path': item['path'], 'sha256': item['sha256'], 'assessments': {}})
            assert attribute['id'] not in frame['assessments']
            frame['assessments'][attribute['id']] = {k: item[k] for k in ['status','observed','reason']}
    assert all(set(f['assessments']) == {r['id'] for r in requirements} for f in frame_map.values())
    result = aggregate(requirements, list(frame_map.values()))
    changes = [{'id': old['id'], 'part': old['part'], 'aspect': old['aspect'],
                'old_initial_status': old.get('initial_status',old['status']), 'old_final_status': old['status'],
                'new_aggregate_status': new['status'], 'new_reason':new['reason'],
                'old_joint_review_eligible_under_new_aggregate': new['status']=='unknown'}
               for old,new in zip(original,result)]
    assert [r['id'] for r in original] == [r['id'] for r in result]
    source_hashes = {}
    for name in ['scripts/reaggregate_attribute_evidence.py','utils/agent_attribute_diagnosis.py']:
        dest=out/'source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
        source_hashes[name]=sha(dest)
    report = {'phase':'single_factor_saved_observation_reaggregation', 'input_result_sha256':sha(args.probe/'result.json'),
        'input_manifest_sha256':sha(args.probe/'manifest.json'), 'source_sha256':source_hashes,
        'model_calls':0, 'new_video_generations':0, 'source_frame_reviews':list(frame_map.values()),
        'attribute_assessments':result,'changes':changes, 'candidate_promoted':False,
        'wall_seconds':time.monotonic()-started,
        'limits':['Post-hoc fix on exposed development evidence, not held-out validation.',
                  'Exact same distinct-frame observations; model, image selection and predicate definitions are unchanged.',
                  'The old joint review is retained in its original record; it is not rerun or used to erase supported failures.',
                  'VLM observations can still be wrong; this fixes evidence handling, not model accuracy.']}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(changes,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
