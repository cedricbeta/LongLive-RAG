#!/usr/bin/env python3
"""Download the pinned official LongLive-RAG assets and verify content hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url

REPOSITORY = 'qixinhu11/LongLive-RAG'
REVISION = 'aaafe325726e0204ada9d1f019356fde97e08c2e'
FILES = ('checkpoints/longlive_base.pt', 'checkpoints/longlive_lora.pt',
         'checkpoints/ae_latent_mem.pt', 'checkpoints/moviegenbench_128_refined.txt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for filename in FILES:
        started = time.monotonic()
        meta = get_hf_file_metadata(hf_hub_url(REPOSITORY, filename, revision=REVISION))
        # Use shared cache once, with local links; never duplicate multi-GB weights.
        cached = Path(hf_hub_download(REPOSITORY, filename, revision=REVISION))
        h = hashlib.sha256()
        with cached.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                h.update(block)
        digest = h.hexdigest()
        if len(meta.etag) == 64 and digest != meta.etag:
            raise ValueError(f'Official LFS hash mismatch: {filename}')
        if cached.stat().st_size != meta.size:
            raise ValueError(f'Size mismatch: {filename}')
        target = args.output / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.resolve() != cached.resolve():
            raise FileExistsError(target)
        if not target.exists():
            target.symlink_to(cached.resolve())
        row = {'file': filename, 'repository': REPOSITORY, 'revision': REVISION, 'bytes': meta.size,
               'sha256': digest, 'upstream_etag': meta.etag, 'wall_seconds': time.monotonic() - started}
        rows.append(row)
        (args.output / 'assets_manifest.json').write_text(json.dumps({'files': rows}, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
