#!/usr/bin/env python3
"""One frozen diagnostic call on an archived candidate; never promote a branch."""
import argparse
import dataclasses
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.agent_progressive_repair import PairedVisualProtection
from utils.agent_runtime import NormativeGraph, ShotObservation, TemporalInput, file_sha256
from utils.agent_vlm import FingerprintedVLMClient


def run(archive, output, base_url, model):
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'; source.mkdir()
    frozen = {}
    for rel in ('scripts/probe_paired_protection.py', 'utils/agent_progressive_repair.py', 'utils/agent_vlm.py'):
        dest = source / Path(rel).name
        shutil.copy2(ROOT / rel, dest)
        frozen[rel] = file_sha256(dest)
    scene = 'ns090_iris_stands_at_the'
    run_root = archive / 'run' / scene
    rows = [json.loads(line) for line in (run_root / 'ledger.jsonl').read_text().splitlines()]
    reviews = {r['attempt']: r for r in rows if r['event'] == 'verifier' and r['shot_index'] == 0}
    prior = next(r for r in rows if r['event'] == 'progressive_current_guard' and r['attempt'] == 1)
    action = next(r for r in rows if r['event'] == 'progressive_action' and r['attempt'] == 1)
    def observation(attempt, branch):
        paths = sorted((run_root / 'attempts').glob(f'{branch}_shot00_attempt{attempt}*.png'))
        if len(paths) != 3:
            raise ValueError('Expected exactly three archived frames')
        return ShotObservation(shot_index=0, attempt=attempt, frames=tuple(str(p) for p in paths), crops=(),
            telemetry={}, inputs=tuple(TemporalInput(kind='archived_protection_frame', identifier=str(p),
                sha256=file_sha256(p), available_after_shot=0) for p in paths),
            input_fingerprint=reviews[attempt]['input_fingerprint'])
    parent, child = observation(0, 'branch-0000'), observation(1, 'branch-0001')
    original = dataclasses.replace(parent, input_fingerprint=prior['protection']['original_fingerprint'])
    targets = action['targets']
    constraints = [row['constraint'] for row in reviews[0]['metrics']['constraint_assessments']]
    manifest = {'phase': 'development_guard_diagnosis', 'live_branch_promoted': False,
        'independent_model': False, 'protocol': 'v2', 'model': model, 'source_sha256': frozen,
        'archived_ledger_sha256': file_sha256(run_root / 'ledger.jsonl'),
        'frames': [{'path': p, 'sha256': file_sha256(p)} for p in (*parent.frames, *child.frames)],
        'comparison': 'Archived v1 contradicted its own target-color statement and marked absent extra actors unknown. '
                      'V2 makes domain scopes explicit and deduplicates an identical original anchor. '
                      'Output budget also changes from 1024 to 2048 tokens. This is not a one-factor effectiveness experiment.',
        'max_visual_calls': 1, 'created_unix': time.time(), 'sampling': {'temperature': 0, 'seed': 0, 'max_tokens': 2048}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    client = FingerprintedVLMClient(base_url=base_url, model=model,
        cache_dir=output / 'vlm_cache', max_tokens=2048, seed=0)
    guard = PairedVisualProtection(client, protocol='v2')
    normative = NormativeGraph.compile_scene(archive / 'data' / scene, allow_missing_global=True).view(0)
    result = guard.review(parent, child, original, normative,
        target_requirements=[row for row in constraints if row['id'] in targets],
        protected_requirements=[row for row in constraints if row['id'] not in targets],
        entity_specs=reviews[0]['metrics'].get('repair_specs', {}))
    (output / 'result.json').write_text(json.dumps({'archived_v1': prior['protection'], 'v2': result,
        'outcome_scope': 'One development judge comparison, not ground truth or candidate acceptance.'}, indent=2) + '\n')
    print(json.dumps({'allow': result['allow'], 'checks': result['checks']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()
    run(args.archive.resolve(), args.output.resolve(), args.base_url, args.model)
