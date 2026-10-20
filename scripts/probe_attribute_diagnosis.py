#!/usr/bin/env python3
"""Development-only appearance probe on archived Iris parent/candidate crops.

Automatic historical crops, not human labels. This diagnoses the judge interface
and cannot establish identity/action preservation or final repair effectiveness.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.agent_attribute_diagnosis import AttributeEvidenceComparer
from utils.agent_entity_judge import load_entity_specs
from utils.agent_runtime import TemporalInput, file_sha256
from utils.agent_vlm import FingerprintedVLMClient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = 'ns090_iris_stands_at_the'
    archive = args.archive_root
    specs = archive / 'frozen/entity_specs.json'
    entity = next(e for e in load_entity_specs(specs, scene) if e.name == 'Iris')
    client = FingerprintedVLMClient(base_url=args.base_url, model='Qwen/Qwen3-VL-8B-Instruct',
                                    cache_dir=args.output / 'vlm_cache', seed=0)
    comparer = AttributeEvidenceComparer(client)
    rows = []
    work = archive / 'runs/feedback' / scene / scene / 'entity_judge'
    inputs = [TemporalInput(kind='fixed_entity_script', identifier=str(specs), sha256=file_sha256(specs), available_after_shot=-1)]
    packet = []
    for attempt in (0, 1):
        directory, = work.glob(f'shot00_attempt{attempt}_*')
        crops = sorted(directory.glob('Iris_frame*.png'))
        assert len(crops) == 3, 'Expected three historical distinct frame crops'
        packet.append({'attempt': attempt, 'crops': [{'path': str(c), 'sha256': file_sha256(c)} for c in crops]})
    provenance = {'schema_version': 1, 'quality_claim': False, 'phase': 'development_diagnostic',
                  'entity': entity.__dict__, 'packet': packet, 'sampling': client.sampling_config,
                  'limitations': ['Archived automatic localization; not independent annotation.',
                                  'Same judge family as historical runs; not independent model validation.',
                                  'Appearance crops do not verify full-shot identity/action/quality.']}
    (args.output / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for item in packet:
        started = time.monotonic()
        crops = [Path(c['path']) for c in item['crops']]
        temporal = tuple(inputs + [TemporalInput(kind='historical_crop', identifier=str(c), sha256=file_sha256(c), available_after_shot=0) for c in crops])
        row = {'attempt': item['attempt']}
        try:
            row.update(status='completed', result=comparer.compare(entity, None, crops, shot_index=0, inputs=temporal))
        except Exception as error:
            row.update(status='execution_failed', error=f'{type(error).__name__}: {error}')
        row['wall_seconds'] = time.monotonic() - started
        rows.append(row)
        (args.output / 'results.json').write_text(json.dumps({'quality_claim': False, 'cases': rows}, indent=2) + '\n')
        print(json.dumps({'attempt': item['attempt'], 'status': row['status'],
                          'wall_seconds': row['wall_seconds'], 'error': row.get('error')}), flush=True)


if __name__ == '__main__':
    main()
