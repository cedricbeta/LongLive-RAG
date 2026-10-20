"""Pinned public download, hash-verified reuse, bounded retries; no credentials."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parent
REVISION = '784410650d12be9bc086118fdefa32d2c3bced86'
metadata = Path((ROOT / 'metadata_path.txt').read_text().strip())
target = ROOT / 'model'
target.mkdir(exist_ok=True)
info = json.loads((ROOT / 'model_blobs.json').read_text())
assert info['sha'] == REVISION
partials = {r['file']: r for r in json.loads((ROOT / 'partial_weight_audit.json').read_text())}
for p in metadata.iterdir():
    if p.is_file() and p.suffix != '.safetensors' and not (target / p.name).exists():
        (target / p.name).symlink_to(p)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def fetch(item):
    name, digest, size = item['rfilename'], item['lfs']['sha256'], item['size']
    path = target / name
    started = time.monotonic()
    if name in partials and partials[name]['valid']:
        source = Path(partials[name]['path'])
        assert source.stat().st_size == size
        if not path.exists():
            os.link(source, path)
        assert sha(path) == digest
        return {'file': name, 'sha256': digest, 'size': size, 'reused_verified_partial': True}
    errors = []
    url = f'https://huggingface.co/allenai/Molmo2-O-7B/resolve/{REVISION}/{name}'
    temporary = target / (name + '.http_partial')
    for attempt in range(3):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset == size and sha(temporary) == digest:
            temporary.rename(path)
            break
        try:
            with requests.get(url, stream=True, timeout=(20, 40),
                              headers={'Range': f'bytes={offset}-'} if offset else {}) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise ValueError('Server did not honor partial continuation')
                with temporary.open('ab' if offset else 'wb') as stream:
                    for data in response.iter_content(chunk_size=1024 * 1024):
                        stream.write(data)
            if temporary.stat().st_size != size or sha(temporary) != digest:
                raise ValueError('Downloaded size or content hash differs from pinned LFS metadata')
            temporary.rename(path)
            break
        except Exception as error:
            errors.append({'attempt': attempt, 'type': type(error).__name__})
            if attempt == 2:
                raise RuntimeError(f'Bounded download failed for {name}: {errors}') from None
    return {'file': name, 'sha256': digest, 'size': size, 'reused_verified_partial': False,
            'wall_seconds': time.monotonic() - started, 'errors': errors}


started = time.monotonic()
status = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time()}
(ROOT / 'http_status.json').write_text(json.dumps(status, indent=2) + '\n')
rows = []
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, item) for item in info['siblings'] if item['rfilename'].endswith('.safetensors')]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            (ROOT / 'http_partial_results.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(rows[-1], flush=True)
    files = {p.name: {'sha256': sha(p), 'size': p.stat().st_size} for p in target.iterdir() if p.is_file()}
    (ROOT / 'verified_model.json').write_text(json.dumps({
        'repo': 'allenai/Molmo2-O-7B', 'revision': REVISION, 'files': files,
        'weight_lfs_verified': True, 'rows': rows}, indent=2) + '\n')
    (ROOT / 'model_path.txt').write_text(str(target) + '\n')
    status.update(status='completed', wall_seconds=time.monotonic() - started)
except Exception as error:
    status.update(status='failed', error=str(error), wall_seconds=time.monotonic() - started)
    raise
finally:
    (ROOT / 'http_status.json').write_text(json.dumps(status, indent=2) + '\n')
