#!/usr/bin/env python3
"""Audit bounded actions on existing native artifacts; do not generate video."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = (ROOT.parent if ROOT.name == 'implementation' else ROOT) / 'adapters/official_longlive_rag'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ADAPTERS))
from utils.agent_bounded_feedback import BoundedFeedbackPolicy
from utils.agent_runtime import Action, file_sha256
from feedback_bridge import prepare_selected


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(), 'device': 'cpu'}
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    try:
        previous = json.loads((args.previous / 'manifest.json').read_text())
        assert json.loads((args.previous / 'status.json').read_text())['status'] == 'completed'
        assert previous['parent_manifest_sha256'] == file_sha256(args.parent / 'manifest.json')
        summary = json.loads(args.summary.read_text())
        assert summary['candidate_retained'] is False
        assert summary['actual_action_fingerprint'] == previous['action_fingerprint']
        assert all(row['diagnostic_decision']['status'] == 'unknown' for row in summary['face_pair_diagnostic'])
        request = yaml.safe_load((args.previous / 'config.yaml').read_text())['repair_request']
        plan = json.loads((args.previous / 'plan.json').read_text())
        config = yaml.safe_load((args.parent / 'config.yaml').read_text())
        raw_script = ' '.join(row['prompts'][0] for row in config['timed_segments'][:plan['segment'] + 1])
        assert args.entity in raw_script and plan['target']['expected'] in raw_script
        requirements = [{'id': plan['target']['id'], 'subjects': [args.entity], 'expected': plan['target']['expected'],
            'parameters': {'part': plan['target']['part'], 'aspect': plan['target']['aspect'], 'basis': 'fixed_script_quote'}}]
        fixed = Action('edit_shot', {'shot': plan['segment'], 'entry_sigma': 1.0,
            'prompt': request['prompts'][0], 'mask_path': request['mask_path']})
        video, = (args.parent / 'videos').glob('*.mp4')
        assert summary['selected_video_sha256'] == file_sha256(video), 'Feedback refers to another parent'
        binding = {'parent_video': video, 'parent_latents': args.parent / 'diagnostics/latents_0000.pt'}
        regions = json.loads((args.regions / 'result.json').read_text())
        original_mask = torch.load(request['mask_path'], weights_only=True, map_location='cpu')['mask']
        recreated_mask = torch.load(args.regions / 'mask.pt', weights_only=True, map_location='cpu')['mask']
        assert torch.equal(original_mask, recreated_mask), 'Default recipe mask changed'
        options = []
        for row in regions['program_region_alternatives']:
            path = args.regions / row['path']
            assert file_sha256(path) == row['sha256']
            options.append({'id': 'part_threshold:' + str(row['part_threshold']),
                'targets': [plan['target']['id']], 'mask_sha256': row['sha256'],
                'action': Action('edit_shot', {**fixed.params, 'mask_path': str(path)}).to_dict()})
        policy = BoundedFeedbackPolicy()
        common = {'first': fixed, 'options': options, 'requirements': requirements, **binding}
        first_menu, _ = policy.menu(**common, seed=request['noise_seed'], history=[], failure=None)
        chosen, selection = policy.choose(first_menu, feedback={})
        prepared = prepare_selected(chosen, parent=args.parent, parent_branch='audited_original_parent',
            checkout=args.checkout, output=out / 'prepared_first', assets_manifest=args.assets_manifest, gpu=args.gpu)
        assert prepared['action_fingerprint'] == previous['action_fingerprint'], 'First native recipe changed'
        history = [{'actual': chosen['actual'], 'targets': chosen['targets']}]
        next_menu, actual_stop = policy.menu(**common, seed=request['noise_seed'] + 1,
            history=history, failure='insufficient_evidence')
        assert not next_menu and actual_stop == 'insufficient_evidence'
        counterfactual = {}
        for failure in ['target_not_improved', 'localization_miss', 'protection_regression', 'future_regression']:
            rows, reason = policy.menu(**common, seed=request['noise_seed'] + 1, history=history, failure=failure)
            counterfactual[failure] = {'reason': reason, 'options': [
                {key: value for key, value in row.items() if key != 'action'} for row in rows]}
        sources = {}
        for path in [Path(__file__), ROOT / 'utils/agent_bounded_feedback.py', ADAPTERS / 'feedback_bridge.py',
                     ADAPTERS / 'prepare_repair.py', ADAPTERS / 'repair_adapter.py', ADAPTERS / 'prepare_mask.py']:
            dest = out / 'source' / path.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(path, dest)
            sources[path.name] = file_sha256(dest)
        result = {'phase': 'existing_native_artifact_action_bridge_audit', 'new_video_generations': 0,
            'new_model_calls': 0, 'candidate_retained': False, 'default_mask_tensor_exact': True,
            'first_recipe': prepared, 'first_selection': selection,
            'actual_archived_evidence_stop': actual_stop, 'actual_archived_evidence_menu_size': len(next_menu),
            'counterfactual_feedback_routing': counterfactual, 'source_sha256': sources,
            'input_sha256': {'parent_manifest': file_sha256(args.parent / 'manifest.json'),
                'previous_candidate_manifest': file_sha256(args.previous / 'manifest.json'),
                'previous_summary': file_sha256(args.summary), 'region_result': file_sha256(args.regions / 'result.json')},
            'wall_seconds': time.monotonic() - started,
            'limits': ['Counterfactual categories test routing on real geometry; they are not observed repair outcomes.',
                'The native bridge prepares an executable request; it is not a full native progressive controller.',
                'No generation, new judge call, human label, retention or effectiveness result is produced.',
                'The actual archived trial remains unresolved; missing identity evidence cannot justify resampling.']}
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        status.update(status='completed', wall_seconds=time.monotonic() - started)
        print(json.dumps({'native_first_action_fingerprint': prepared['action_fingerprint'],
                          'actual_stop': actual_stop, 'new_video_generations': 0}), flush=True)
    except Exception as error:
        status.update(status='execution_failed', error=f'{type(error).__name__}: {error}',
                      wall_seconds=time.monotonic() - started)
        raise
    finally:
        (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent', 'previous', 'summary', 'regions', 'checkout', 'assets-manifest', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--entity', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    main(args)
