#!/usr/bin/env python3
"""Install the disclosed suffix-cue intervention in a separate fresh checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from install import once
from install_repair import install_repair


def patch_pipeline(source):
    return once(source, '        conditional_dict = self.text_encoder(\n',
        '        self.continuation_cue_record = {"enabled": False, "changes": []}\n'
        '        if repair is not None:\n'
        '            from continuation_cue import compile_cue\n'
        '            schedule, self.continuation_cue_record = compile_cue(\n'
        '                schedule, repair_request.get("continuation_cue"), earliest_latent=repair.end)\n\n'
        '        conditional_dict = self.text_encoder(\n')


def patch_runner(source):
    return once(source,
        '        (diagnostic_dir / f"repair_{idx:04d}.json").write_text(json.dumps(pipeline.repair_log, indent=2) + "\\n")',
        '        (diagnostic_dir / f"repair_{idx:04d}.json").write_text(json.dumps(pipeline.repair_log, indent=2) + "\\n")\n'
        '        (diagnostic_dir / f"continuation_cue_{idx:04d}.json").write_text(json.dumps(pipeline.continuation_cue_record, indent=2) + "\\n")')


def install_cue(root):
    if (root / 'continuation_cue.py').exists():
        raise FileExistsError('Cue helper already exists')
    base = install_repair(root)
    paths = {'pipeline/causal_inference.py': patch_pipeline, 'inference.py': patch_runner}
    transformed = {name: fn((root / name).read_text()) for name, fn in paths.items()}
    for name, source in transformed.items():
        compile(source, str(root / name), 'exec')
    for name, source in transformed.items():
        (root / name).write_text(source)
    shutil.copy2(Path(__file__).with_name('continuation_cue.py'), root / 'continuation_cue.py')
    record = {**base, 'repair_adapter_before_continuation_cue': base['adapted_sha256'],
        'adapted_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in [*base['adapted_sha256'], 'continuation_cue.py']},
        'status': 'Experimental local native repair with optional literal suffix text cue; no acceptance policy',
        'timing_record_semantics': 'timing_0000.json retains authored timing/prompts; continuation_cue_0000.json records changed effective prompts. Actual switch hashes must match the effective schedule.',
        'limits': base['limits'] + ['Only explicitly selected existing post-repair boundaries can receive an authored hair-color cue.',
            'Cue-only control is necessary; reminder efficacy is not evidence of corrective-feedback benefit.',
            'Disabled cue changes no generation math or text; this does not certify pretrained quality.']}
    (root / 'repair_adapter_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    print(json.dumps(install_cue(parser.parse_args().checkout.resolve()), indent=2))
