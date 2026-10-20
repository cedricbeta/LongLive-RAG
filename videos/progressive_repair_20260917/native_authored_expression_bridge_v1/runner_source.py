#!/usr/bin/env python3
"""Bind the opt-in policy to a previously generated real native action, on CPU."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import torch
import yaml

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent if BASE.name == 'implementation' else BASE
sys.path[:0] = [str(BASE), str(ROOT / 'adapters/official_longlive_rag')]
from utils.agent_bounded_feedback import BoundedFeedbackPolicy
from utils.agent_runtime import Action, file_sha256
from feedback_bridge import prepare_selected


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    read = lambda p: json.loads(p.read_text())
    rows = [json.loads(line) for line in (args.feedback / 'ledger.jsonl').read_text().splitlines()]
    reserved = next(row for row in rows if row['event'] == 'bounded_action_reserved')
    initial = next(row for row in rows if row['event'] == 'verifier' and row['attempt'] == 0)
    routed = next(row for row in rows if row['event'] == 'progressive_evidence_routing' and row['attempt'] == 1)
    assert routed['routing']['allow_revision'] and not routed['routing']['acceptance_changed']
    requirements = [row['constraint'] for row in initial['metrics']['constraint_assessments']
                    if row['constraint']['id'] in reserved['targets']]
    assert sorted(row['id'] for row in requirements) == reserved['targets']
    compiled = read(args.feedback / 'compiled_script.json')
    reference = read(args.candidate / 'manifest.json')
    request = yaml.safe_load((args.candidate / 'config.yaml').read_text())['repair_request']
    action = Action('edit_shot', {'shot': reserved['shot_index'], 'entry_sigma': 1.0,
        'prompt': reserved['actual']['prompt'], 'mask_path': str(args.mask.resolve())})
    video, = (args.parent / 'videos').glob('*.mp4')
    common = {'first': action, 'options': [], 'requirements': requirements,
              'parent_video': video, 'parent_latents': args.parent / 'diagnostics/latents_0000.pt'}
    policy = BoundedFeedbackPolicy(allow_mixed_evidence_revision=True,
        expression_protocol='authored_garment_v1', compiled_script=compiled)
    first, = policy.menu(**common, seed=reserved['actual']['noise_seed'], history=[], failure=None)[0]
    assert first['actual'] == reserved['actual']
    history = [{'actual': first['actual'], 'targets': first['targets']}]
    choices, reason = policy.menu(**common, seed=request['noise_seed'], history=history, failure='target_evidence_mixed')
    chosen, = choices
    assert chosen['purpose'] == 'target_expression' and chosen['kind'] == 'revise'
    assert chosen['action'].params['prompt'] == request['prompts'][0]
    prepared = prepare_selected(chosen, parent=args.parent, parent_branch=request['parent_branch_id'],
        checkout=args.checkout, output=out / 'prepared', assets_manifest=args.assets_manifest, gpu=args.gpu)
    assert prepared['action_fingerprint'] == reference['action_fingerprint']
    assert prepared['action'] == reference['action']
    assert prepared['source_sha256'] == reference['source_sha256']
    assert prepared['paired_assets'] == reference['paired_assets']
    source_paths = [Path(__file__), BASE / 'utils/agent_authored_expression.py',
        BASE / 'utils/agent_bounded_feedback.py', BASE / 'utils/agent_typed_script.py',
        ROOT / 'adapters/official_longlive_rag/feedback_bridge.py',
        ROOT / 'adapters/official_longlive_rag/prepare_repair.py']
    hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = file_sha256(target)
    result = {'phase': 'authored_expression_real_file_bridge', 'first_actual_action_unchanged': True,
        'target_requirements': requirements, 'archived_routing': routed['routing'],
        'menu_reason': reason, 'eligible_actions': len(choices),
        'selected_action': chosen['action'].to_dict(), 'policy_actual': chosen['actual'],
        'native_action_fingerprint': prepared['action_fingerprint'],
        'matches_previously_generated_natural_candidate': True,
        'previous_candidate_video_sha256': read(args.candidate / 'audit.json')['video_sha256'],
        'same_native_assets_and_source': True, 'source_sha256': hashes,
        'input_sha256': {'feedback_ledger': file_sha256(args.feedback / 'ledger.jsonl'),
            'compiled_script': file_sha256(args.feedback / 'compiled_script.json'),
            'reference_candidate_manifest': file_sha256(args.candidate / 'manifest.json'),
            'reference_candidate_config': file_sha256(args.candidate / 'config.yaml'),
            'mask': file_sha256(args.mask)},
        'new_model_calls': 0, 'new_generated_videos': 0, 'candidate_retained': False,
        'wall_seconds': time.monotonic() - started,
        'scope': 'Programmatically selects the sole legal expression for this bridge check and invokes the real CPU native preparer. No visual planner, generator, acceptance or future guard is executed. It reconstructs an existing action, not a new end-to-end feedback result.'}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['first_actual_action_unchanged', 'native_action_fingerprint',
        'matches_previously_generated_natural_candidate', 'wall_seconds', 'new_generated_videos']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('feedback', 'candidate', 'parent', 'mask', 'checkout', 'assets-manifest', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--gpu', type=int, choices=[0, 1, 2], default=0)
    torch.set_num_threads(4)
    run(p.parse_args())
