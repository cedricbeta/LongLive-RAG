#!/usr/bin/env python3
"""Install one opt-in repair-sampler comparison in a fresh pinned checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from install import once
from install_repair import install_repair


def patch_adapter(source):
    source = once(source, 'import torch\n', 'import torch\nfrom repair_sampler import NATIVE, EULER, sample_euler\n')
    source = once(source, '        self.request = dict(request)\n',
        '        self.request = dict(request)\n'
        '        self.sampler_mode = request.get("repair_sampler", NATIVE)\n'
        '        if self.sampler_mode not in (NATIVE, EULER):\n'
        '            raise ValueError("Unknown private repair sampler")\n')
    first = source.index('                x = eps\n')
    last = source.index('            finally:\n', first)
    legacy = source[first:last]
    replacement = ('                if self.sampler_mode == EULER:\n'
        '                    result = sample_euler(pipeline, parent, mask, eps, self.conditional,\n'
        '                        start, memory_indices, report.setdefault("sampler_trace", []))\n'
        '                else:\n' + ''.join('    ' + line if line.strip() else line for line in legacy.splitlines(keepends=True)))
    source = source[:first] + replacement + source[last:]
    source = once(source, '            report.update(mask_fraction=',
                  '            report["repair_sampler"] = self.sampler_mode\n            report.update(mask_fraction=')
    compile(source, 'repair_adapter_with_sampler.py', 'exec')
    return source


def install_sampler(root):
    helper = root / 'repair_sampler.py'
    if helper.exists():
        raise FileExistsError(helper)
    base = install_repair(root)
    path = root / 'repair_adapter.py'
    path.write_text(patch_adapter(path.read_text()))
    shutil.copy2(Path(__file__).with_name('repair_sampler.py'), helper)
    record = dict(base)
    record['repair_adapter_before_sampler'] = base['adapted_sha256']
    record['adapted_sha256'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in [*base['adapted_sha256'], 'repair_sampler.py']}
    record['status'] = 'Experimental native-four-step fixed-noise Euler repair; not automatic controller acceptance'
    record['sampler_modes'] = ['native_x0_renoise_v1', 'fixed_noise_euler_v1']
    record['limits'] = base['limits'] + ['Only private repair integration changes; normal generation/retrieval stays native',
        'Four native timesteps are kept; this does not reproduce the historical 5B checkpoint or schedule']
    (root / 'repair_adapter_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkout', type=Path)
    print(json.dumps(install_sampler(p.parse_args().checkout.resolve()), indent=2))
