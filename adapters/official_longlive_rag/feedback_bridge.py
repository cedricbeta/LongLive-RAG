"""Bind bounded-policy choices to the genuine native preparer, without judging.

This is an action bridge, not a complete native progressive session. The caller
must maintain accepted branches and supply actual, auditable feedback. Preparing
an action neither runs a generator nor accepts a candidate.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = ROOT / 'implementation' if (ROOT / 'implementation').is_dir() else ROOT
sys.path.insert(0, str(IMPLEMENTATION))

from utils.agent_bounded_feedback import actual_action
from utils.agent_runtime import file_sha256
from prepare_repair import prepare
from repair_adapter import tensor_hash


def prepare_selected(choice, *, parent, parent_branch, checkout, output, assets_manifest, gpu,
                     timeout=1800):
    if choice is None:
        raise ValueError('A stop or empty menu cannot prepare a render')
    parent, output = Path(parent).resolve(), Path(output).resolve()
    video, = (parent / 'videos').glob('*.mp4')
    latent = parent / 'diagnostics/latents_0000.pt'
    action, expected = choice['action'], choice['actual']
    current = actual_action(action, parent_video=video, parent_latents=latent, seed=expected['noise_seed'])
    if current != expected:
        raise ValueError('Selected native action changed after planning')
    if output.exists():
        raise FileExistsError('Use a fresh native candidate directory')
    # A separate immutable input file avoids creating a candidate directory
    # before the native preparer has finished its paired-asset validation.
    prompt_file = output.with_name(output.name + '_selected_prompt.txt')
    with prompt_file.open('x') as stream:
        stream.write(expected['prompt'] + '\n')
    result = prepare(SimpleNamespace(parent=parent, parent_branch=parent_branch, checkout=Path(checkout),
        output=output, assets_manifest=Path(assets_manifest), mask=Path(action.params['mask_path']),
        prompt=prompt_file, noise_seed=expected['noise_seed'], gpu=gpu, timeout=timeout))
    config = yaml.safe_load((output / 'config.yaml').read_text())
    request = config['repair_request']
    payload = torch.load(request['mask_path'], map_location='cpu', weights_only=True)
    if (request['parent_video_sha256'] != expected['parent_video_sha256']
            or request['parent_latents_sha256'] != expected['parent_latents_sha256']
            or request['prompts'] != [expected['prompt']] or request['noise_seed'] != expected['noise_seed']
            or request['entry_sigma'] != 1.0 or request['latent_range'] != expected['coordinates']['latent_range']):
        raise ValueError('Native preparation does not execute the selected policy action')
    return {**result, 'policy_action_fingerprint': expected['action_fingerprint'],
            'native_mask_uint8_sha256': tensor_hash(payload['mask'].to(torch.uint8)),
            'prepared_config_sha256': file_sha256(output / 'config.yaml'),
            'generation_started': False, 'candidate_retained': False}


if __name__ == '__main__':
    import argparse
    import json
    from utils.agent_runtime import Action
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    raw = request.pop('choice')
    raw['action'] = Action(raw['action']['tool'], raw['action']['params'],
                           reason=raw['action'].get('reason') or 'selected_native_action')
    torch.set_num_threads(4)
    result = prepare_selected(raw, **request)
    with args.result.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
