#!/usr/bin/env python3
"""Install the experimental private-context diagnostic into a fresh checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from install import once
from install_repair import install_repair
from repair_context import PINNED_SOURCES


def install_context(root):
    for name, expected in PINNED_SOURCES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Unsupported native source: ' + name)
    helper = root / 'repair_context.py'
    if helper.exists():
        raise FileExistsError(helper)
    base = install_repair(root)
    path = root / 'repair_adapter.py'
    text = path.read_text()
    text = once(text, 'import torch\n', 'import torch\nfrom repair_context import private_forward\n')
    text = once(text, '_, x0 = pipeline.generator(noisy_image_or_video=x,',
                '_, x0 = private_forward(pipeline, self.request, report.setdefault("context_trace", []), noisy_image_or_video=x,')
    compile(text, str(path), 'exec')
    path.write_text(text)
    shutil.copy2(Path(__file__).with_name('repair_context.py'), helper)
    record = dict(base)
    record['repair_adapter_before_context'] = base['adapted_sha256']
    record['adapted_sha256'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in [*base['adapted_sha256'], 'repair_context.py', *PINNED_SOURCES]}
    record['status'] = 'Experimental repair-only historical self-attention context diagnostic; not an automatic controller action'
    record['context_modes'] = ['full_context_v1', 'current_block_only_v1']
    record['limits'] = base['limits'] + ['Combined sink, retrieval and local-past intervention; does not isolate RAG alone',
        'Only private repair calls change; ordinary generation and clean recache retain native full context',
        'Sequential single-process inference only; no concurrent model calls supported']
    (root / 'repair_adapter_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    print(json.dumps(install_context(parser.parse_args().checkout.resolve()), indent=2))
