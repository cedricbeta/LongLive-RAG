#!/usr/bin/env python3
"""Reproduce compiler observations and the v5 refactor replay without model calls."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent if BASE.name == 'implementation' else BASE
sys.path.insert(0, str(BASE))
from utils.agent_schema_context import SchemaContextClient
from utils.agent_script_candidates import compile_candidate_entities


def audit(root):
    hashes = {}

    def read(path):
        path = root / path
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    folder = Path('results/typed_script_compiler_v6_development')
    plan = read(folder / 'plan.json')
    status = read(folder / 'run/status.json')
    rows = read(folder / 'run/results.json')
    assert status['status'] == 'completed' and not plan['held_out']
    assert [r['source_benchmark_index'] for r in rows] == [90, 310, 50]
    calls = [read(p.relative_to(root)) for p in sorted((root / folder / 'run/vlm_cache').rglob('*.json'))]
    assert all(not r['parse_attempts'] and not r['ordered_image_paths'] for r in calls)
    compiled = {}
    for row in rows:
        if 'compiled_path' in row:
            item = read(folder / 'run' / row['compiled_path'])
            assert item['compiler_protocol'] == 'typed_name_candidates_v6'
            assert not item['semantically_validated']
            compiled[str(row['source_benchmark_index'])] = {
                'entities': item['entities'], 'unresolved': item['unresolved'],
                'ready_for_development': item['ready_for_development'],
            }

    old = Path('results/typed_script_compiler_v5_schema_context')
    old_plan = read(old / 'plan.json')
    old_source = root / old / 'run/source/utils/agent_script_candidates.py'
    spec = importlib.util.spec_from_file_location('archived_v5_compiler', old_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_calls = [read(p.relative_to(root)) for p in sorted((root / old / 'run/vlm_cache').rglob('*.json'))]

    class Replay:
        def __init__(self):
            self.kinds = []

        def call(self, **kwargs):
            matches = [r for r in old_calls if r['call_kind'] == kwargs['call_kind']
                       and r['schema'] == kwargs['schema']
                       and r['request']['messages'][0]['content'] == [{'type': 'text', 'text': kwargs['prompt']}]]
            assert len(matches) == 1, 'Fresh archived request text/schema must match exactly'
            record = matches[0]
            # The cache writer sorts keys. Restore the original model response
            # order to reproduce the original fresh downstream prompt bytes.
            parsed = json.loads(record['response']['choices'][0]['message']['content'])
            assert parsed == record['parsed']
            self.kinds.append(record['call_kind'])
            return {**record, 'parsed': parsed}

    replay_rows = []
    for case in old_plan['cases']:
        a, b = Replay(), Replay()
        before = module.compile_candidate_entities(SchemaContextClient(a), case['raw_prompts'])
        after = compile_candidate_entities(SchemaContextClient(b), case['raw_prompts'])
        assert before == after and a.kinds == b.kinds
        replay_rows.append({'source_benchmark_index': case['source_benchmark_index'],
                            'actual_archived_request_text_and_schema_exact': True,
                            'compiled_output_exact': True, 'replayed_call_kinds': a.kinds})
    for path in [old_source, BASE / 'utils/agent_script_candidates.py', Path(__file__)]:
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'cases': rows, 'compiled': compiled, 'worker_wall_seconds': status['wall_seconds'],
            'recorded_model_calls': len(calls), 'call_kinds': dict(Counter(r['call_kind'] for r in calls)),
            'recorded_call_wall_seconds': sum(r['wall_seconds'] for r in calls),
            'program_ready_cases': [r['source_benchmark_index'] for r in rows if r['status'] == 'program_ready'],
            'v5_refactor_replay': replay_rows, 'audit_new_model_calls': 0,
            'new_video_generations': 0, 'actual_human_labels': 0, 'held_out': False,
            'quality_claim': False, 'inputs_sha256': hashes,
            'limits': ['Sequential exposed development, not compiler semantic accuracy.',
                       'ns050 now fails a noncharacter/presence consistency check; no outcome omitted.',
                       'Replay restores actual response key order; sorted cache dictionaries can change subsequent unsorted JSON prompt bytes.',
                       'Worker and call times overlap; server startup is separate.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ['recorded_model_calls', 'program_ready_cases', 'worker_wall_seconds']}))
