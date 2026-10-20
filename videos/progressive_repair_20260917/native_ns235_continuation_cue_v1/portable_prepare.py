#!/usr/bin/env python3
"""Rebind the published ns235 inputs to a verified local timed checkout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main(args):
    repo = args.repository.resolve()
    archive = repo / 'results/native_ns235_continuation_cue_v1/B_original'
    manifest = json.loads((archive / 'manifest.json').read_text())
    manifest['cwd'] = str(args.checkout.resolve())
    # The original preparer checks revision, four runtime sources and all
    # twelve assets before creating any new output; no generation starts here.
    with tempfile.TemporaryDirectory(prefix='ns235_reference_') as tmp:
        reference = Path(tmp)
        (reference / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (reference / 'config.yaml').write_bytes((archive / 'config.yaml').read_bytes())
        subprocess.run([sys.executable,
            str(repo / 'implementation/scripts/prepare_native_cue_study.py'),
            '--root', str(args.output.resolve()), '--reference', str(reference),
            '--benchmark', str(repo / 'reproduction/native_fast_cases/streambench_324.jsonl')],
            check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'checkout', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
