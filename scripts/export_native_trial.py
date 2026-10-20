#!/usr/bin/env python3
"""Export a terminal native trial, keeping every actual VLM input image."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source, destination):
    status = json.loads((source / 'run/status.json').read_text())
    if status['status'] not in ('completed', 'execution_failed', 'timeout'):
        raise ValueError('Refuse to export live results as terminal')
    if not (source / 'server/EXITED').exists():
        raise ValueError('Server logs are not terminal yet')
    manifest = json.loads((source / 'run/manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        assert sha(source / 'run/source' / name) == expected, name
    requested = set()
    for transcript in (source / 'run/vlm_cache').rglob('*.json'):
        record = json.loads(transcript.read_text())
        assert len(record.get('ordered_image_paths', [])) == len(record.get('ordered_image_hashes', []))
        for path, expected in zip(record.get('ordered_image_paths', []), record.get('ordered_image_hashes', [])):
            image = Path(path).resolve()
            assert image.is_relative_to(source), 'External request image needs explicit export binding: ' + str(image)
            assert sha(image) == expected, str(image)
            requested.add(image)
    destination.mkdir(parents=True, exist_ok=True)
    copied, excluded = {}, []
    for path in sorted(source.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(source)
        name = str(relative)
        include = (path.suffix in ('.json', '.jsonl', '.yaml', '.yml', '.py', '.txt', '.log', '.npz')
                   or path.name in ('EXITED', 'STOP', 'server.pid'))
        if path.suffix in ('.png', '.jpg', '.jpeg'):
            include = path.resolve() in requested or 'frames' not in relative.parts or 'backend' in relative.parts
        if '.call_locks' in relative.parts or relative.parts[0] == 'comparison':
            include = False
        if not include:
            excluded.append({'path': name, 'bytes': path.stat().st_size})
            continue
        target_relative = name + '.txt' if path.suffix == '.log' else name
        target = destination / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != sha(path):
            raise ValueError('Would overwrite different exported evidence: ' + target_relative)
        shutil.copy2(path, target)
        copied[target_relative] = sha(target)
    for path in requested:
        assert str(path.relative_to(source)) in copied
    report = {'source_runtime_folder': source.name, 'files_sha256': copied,
              'exclusions': excluded, 'actual_request_images_verified': len(requested),
              'image_scope': 'Every actual VLM request image plus backend evidence and actor crops; unused bulk frame dumps omitted.',
              'tensor_scope': 'Raw tensors excluded; audited portable masks live in audit/*.npz. Videos are published separately under artifacts.',
              'quality_claim': False}
    (destination / 'export_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.source.resolve(), args.destination.resolve())
    print(json.dumps({'files_exported': len(result['files_sha256']),
                      'request_images': result['actual_request_images_verified'],
                      'excluded_files': len(result['exclusions'])}))
