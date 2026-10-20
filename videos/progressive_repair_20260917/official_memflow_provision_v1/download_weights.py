#!/usr/bin/env python3
"""Download the public pinned MemFlow LoRA and verify both checkpoint bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--shared-base', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root, output = Path(__file__).resolve().parent, a.output.resolve()
meta = json.loads((root / 'remote_metadata.json').read_text())
assert meta['gated'] is False and meta['private'] is False
assert meta['revision'] == 'c41043fd7df4576e62603d3eb40b25aaa5542022'
assert not (root / 'download_status.json').exists()
assert shutil.disk_usage(output.parent).free > 8 * 2**30
output.mkdir(parents=True, exist_ok=False)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

started = time.monotonic()
status = {'status':'running','started_unix':time.time(),'pid':os.getpid(),
          'repo':meta['repo'],'revision':meta['revision'],'token_used':False,
          'gpu_used':False,'files':[]}
record = root / 'download_status.json'
record.write_text(json.dumps(status,indent=2)+'\n')
try:
    rows = {x['path']:x for x in meta['files']}
    source = a.shared_base.resolve()
    expected = rows['base.pt']
    assert source.stat().st_size == expected['size']
    assert sha(source) == expected['lfs_sha256']
    (output/'base.pt').symlink_to(source)
    status['files'].append({'path':'base.pt','method':'reused_verified_identical_bytes',
        'source':str(source),'bytes':source.stat().st_size,'sha256':expected['lfs_sha256']})
    record.write_text(json.dumps(status,indent=2)+'\n')
    downloaded = Path(hf_hub_download(meta['repo'],'lora.pt',revision=meta['revision'],
                                    local_dir=str(output),token=False))
    expected = rows['lora.pt']
    digest = sha(downloaded)
    assert downloaded.stat().st_size == expected['size'] and digest == expected['lfs_sha256']
    status['files'].append({'path':'lora.pt','method':'public_pinned_download',
        'bytes':downloaded.stat().st_size,'sha256':digest})
    status['status'] = 'completed'
except Exception as error:
    status.update(status='failed',exception_type=type(error).__name__,
                  http_status=getattr(getattr(error,'response',None),'status_code',None))
finally:
    status['wall_seconds'] = time.monotonic()-started
    record.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)
if status['status'] != 'completed':
    raise SystemExit(1)
