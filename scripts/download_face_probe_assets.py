#!/usr/bin/env python3
"""Fetch pinned OpenCV Zoo face diagnostics with Git-LFS content verification."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import requests

REVISION = '47534e27c9851bb1128ccc0102f1145e27f23f98'
FILES = ['models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
         'models/face_recognition_sface/face_recognition_sface_2021dec.onnx']


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = []
    raw = f'https://raw.githubusercontent.com/opencv/opencv_zoo/{REVISION}/'
    for name in FILES:
        response = requests.get(raw + name, timeout=30)
        response.raise_for_status()
        pointer = response.text
        expected = re.search(r'^oid sha256:([a-f0-9]{64})$', pointer, re.M).group(1)
        size = int(re.search(r'^size ([0-9]+)$', pointer, re.M).group(1))
        dest = output / Path(name).name
        dest.with_suffix('.lfs.txt').write_text(pointer)
        url = f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/{name}'
        with requests.get(url, timeout=(20, 60), stream=True) as download:
            download.raise_for_status()
            with dest.with_suffix('.partial').open('xb') as file:
                for chunk in download.iter_content(1024 * 1024):
                    file.write(chunk)
        temporary = dest.with_suffix('.partial')
        actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if actual != expected or temporary.stat().st_size != size:
            raise ValueError('Downloaded model fails the pinned LFS hash/size: ' + name)
        temporary.rename(dest)
        rows.append({'repository_path': name, 'file': dest.name, 'url': url,
                     'sha256': actual, 'size': size, 'lfs_verified': True})
        print(dest.name, size, actual, flush=True)
        parent = Path(name).parent
        for supporting in ['LICENSE', 'README.md', 'sface.py' if 'sface' in name else 'yunet.py']:
            response = requests.get(raw + str(parent / supporting), timeout=30)
            response.raise_for_status()
            artifact = output / parent.name / supporting
            artifact.parent.mkdir(exist_ok=True)
            artifact.write_bytes(response.content)
    manifest = {'repository': 'https://github.com/opencv/opencv_zoo', 'revision': REVISION,
                'files': rows, 'download_wall_seconds': time.monotonic() - started,
                'purpose': 'CPU development diagnostics on generated character images; not a calibrated acceptance policy',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
