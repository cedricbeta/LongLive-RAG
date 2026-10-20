#!/usr/bin/env python3
"""Install the opt-in offline repair adapter in a fresh pinned checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from install import install, once


def install_repair(root):
    helper = root / 'repair_adapter.py'
    if helper.exists():
        raise FileExistsError(helper)
    base = install(root, timed=True)
    path = root / 'pipeline/causal_inference.py'
    source = path.read_text()
    source = once(source, '        timed_segments=None,',
                  '        timed_segments=None,\n        repair_request=None,')
    source = once(source, '        conditional_dict = self.text_encoder(',
        '        self.repair_log = []\n'
        '        repair = None\n'
        '        if repair_request is not None:\n'
        '            from repair_adapter import MaskedParentRepair\n'
        '            repair = MaskedParentRepair(repair_request, noise, self.timing_record, self.args)\n'
        '            self.repair_log = repair.log\n\n'
        '        conditional_dict = self.text_encoder(')
    source = once(source, '            # Step 2.2: record the model\'s output',
        '            if repair is not None:\n'
        '                denoised_pred = repair.apply(self, denoised_pred, current_start_frame, memory_indices)\n\n'
        '            # Step 2.2: record the model\'s output')
    source = once(source, '            if profile:\n                block_end.record()',
        '            if repair is not None:\n'
        '                repair.finish_block(self, denoised_pred, current_start_frame)\n\n'
        '            if profile:\n                block_end.record()')
    runner_path = root / 'inference.py'
    runner = runner_path.read_text()
    runner = once(runner, '        profile=False,',
        '        profile=False,\n'
        '        repair_request=OmegaConf.to_container(config.repair_request, resolve=True) if getattr(config, "repair_request", None) else None,')
    runner = once(runner, '        torch.save(latents.cpu(), diagnostic_dir / f"latents_{idx:04d}.pt")',
        '        torch.save(latents.cpu(), diagnostic_dir / f"latents_{idx:04d}.pt")\n'
        '        (diagnostic_dir / f"repair_{idx:04d}.json").write_text(json.dumps(pipeline.repair_log, indent=2) + "\\n")')
    compile(source, str(path), 'exec')
    compile(runner, str(runner_path), 'exec')
    path.write_text(source)
    runner_path.write_text(runner)
    shutil.copy2(Path(__file__).with_name('repair_adapter.py'), helper)
    names = ['pipeline/causal_inference.py', 'inference.py', 'prompt_schedule_adapter.py',
             'timing_adapter.py', 'repair_adapter.py']
    record = {'official_revision': base['official_revision'],
              'timed_adapter_before_repair': base['adapted_sha256'],
              'adapted_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
              'status': 'Experimental offline parent replay and masked native-sampler repair; no acceptance policy',
              'limits': ['Caches reconstructed through ordinary replay, not serialized snapshots',
                         'Strength fixed at 1.0; exactly one authored segment per transaction',
                         'Ordinary denoising still runs before prefix pinning/editing to preserve native RNG']}
    (root / 'repair_adapter_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()
    print(json.dumps(install_repair(args.checkout.resolve()), indent=2))
