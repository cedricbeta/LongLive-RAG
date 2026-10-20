#!/usr/bin/env python3
"""Frozen paired replay of exposed visual packets, with schema text as one factor."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent if BASE.name == 'implementation' else BASE
sys.path.insert(0, str(BASE))
from utils.agent_attribute_diagnosis import assessment_schema
from utils.agent_runtime import TemporalInput, file_sha256
from utils.agent_schema_context import SchemaContextClient
from utils.agent_vlm import FingerprintedVLMClient


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def packet_inputs(plan_root, packet):
    archive = plan_root / packet['archived_call']
    if file_sha256(archive) != packet['archived_call_sha256']:
        raise ValueError('Archived request changed')
    record = json.loads(archive.read_text())
    images = [plan_root / p for p in packet['images']]
    if [file_sha256(p) for p in images] != record['ordered_image_hashes']:
        raise ValueError('Frozen evidence image changed')
    prompt = record['request']['messages'][0]['content'][0]['text']
    requirements = json.loads(prompt)['requirements']
    # Restore the producer's original schema property order; archive JSON sorts
    # keys. The historical raw HTTP byte stream itself was not retained.
    schema = assessment_schema(requirements)
    if schema != record['schema']:
        raise ValueError('Schema producer differs from archived structure')
    request = record['request']
    if (request['model'] != 'Qwen/Qwen3-VL-8B-Instruct'
            or (request['seed'], request['temperature'], request['max_tokens']) != (0, 0, 2048)):
        raise ValueError('Archived sampling differs from frozen protocol')
    return record, dict(prompt=prompt, schema=schema, image_paths=images,
        call_kind=record['call_kind'], shot_index=record['shot_index'],
        temporal_inputs=[TemporalInput(**r) for r in record['temporal_inputs']])


def evidence_first_schema(schema):
    """Keep the same fields/constraints; request observation before decision."""
    result = deepcopy(schema)
    order = ['observed', 'reason', 'status']
    for rule in result['properties'].values():
        if set(rule['properties']) != set(order) or set(rule['required']) != set(order):
            raise ValueError('Expected the frozen three-field attribute schema')
        rule['properties'] = {key: rule['properties'][key] for key in order}
        # Required-array order has no logical effect on the allowed fields, but
        # remains part of the client's fingerprint, separating these requests.
        rule['required'] = list(order)
    return result


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'status': 'running', 'pid': os.getpid(), 'pgid': os.getpgrp(), 'started_unix': time.time()}
    write(out / 'status.json', status)
    rows, http_calls = [], []
    try:
        plan = json.loads(args.plan.read_text())
        assert plan['max_requests'] == 14 and len(plan['packets']) == 7
        variant = plan.get('comparison_arm', 'schema_in_message')
        if variant not in ('schema_in_message', 'evidence_first'):
            raise ValueError('Unsupported frozen comparison arm')
        arms = ['sampling_only', variant]
        assert len(plan['schedule']) == 14
        assert {(r['packet'], r['arm']) for r in plan['schedule']} == {
            (p['id'], arm) for p in plan['packets'] for arm in arms}
        sources = {}
        for p in [Path(__file__), *(BASE / 'utils').glob('agent_*.py')]:
            relative = p.relative_to(ROOT)
            target = out / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            sources[str(relative)] = file_sha256(target)
        for relative, expected in plan['source_sha256'].items():
            if file_sha256(BASE / relative) != expected:
                raise ValueError('Code changed since plan freeze: ' + relative)
        write(out / 'manifest.json', {'plan_sha256': file_sha256(args.plan), 'source_sha256': sources,
            'held_out': False, 'new_video_generations': 0, 'quality_claim': False,
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
        uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()

        def resources():
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
            for line in raw.splitlines():
                device, pid = (s.strip() for s in line.split(','))
                try:
                    group = os.getpgid(int(pid))
                except ProcessLookupError:
                    continue
                if device == uuid and group != args.judge_pgid:
                    raise RuntimeError('resource_interrupted: foreign compute process')

        class MeasuredClient(FingerprintedVLMClient):
            TRANSPORT_ATTEMPTS = 1

            def _post(self, payload):
                resources()
                if 'response_format' not in payload:
                    raise RuntimeError('Schema fallback is disallowed in this controlled probe')
                if len(http_calls) >= plan['max_requests']:
                    raise RuntimeError('Frozen HTTP request budget exhausted')
                row = {'index': len(http_calls), 'arm': self.arm, 'packet': self.packet,
                       'started_unix': time.time(),
                       'request_bytes_sha256': hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
                http_calls.append(row)
                write(out / 'http_attempts.json', http_calls)
                before = time.monotonic()
                try:
                    result = super()._post(payload)
                    row.update(status='completed', response_id=result.get('id'), usage=result.get('usage'))
                    return result
                except Exception as error:
                    row.update(status='failed', error=f'{type(error).__name__}: {error}')
                    raise
                finally:
                    row['wall_seconds'] = time.monotonic() - before
                    write(out / 'http_attempts.json', http_calls)

        clients = {arm: MeasuredClient(base_url=args.base_url, model=plan['model'],
            cache_dir=out / arm / 'vlm_cache', seed=0, max_tokens=2048, max_parse_retries=0)
            for arm in arms}
        packets = {p['id']: p for p in plan['packets']}
        # Validate every archived input before the first new HTTP request.
        fixed = {key: packet_inputs(args.plan.parent, p) for key, p in packets.items()}
        for entry in plan['schedule']:
            if time.monotonic() - started >= plan['max_worker_seconds']:
                status['status'] = 'timeout'
                break
            packet, arm = entry['packet'], entry['arm']
            old, inputs = fixed[packet]
            client = clients[arm]
            client.arm, client.packet = arm, packet
            row = {'packet': packet, 'arm': arm, 'archived_call': packets[packet]['archived_call']}
            before = time.monotonic()
            try:
                # Recheck image bindings immediately before every arm; no old
                # answer, planner explanation or preceding response is sent.
                packet_inputs(args.plan.parent, packets[packet])
                current = {**inputs, 'schema': evidence_first_schema(inputs['schema'])} if arm == 'evidence_first' else inputs
                response = (SchemaContextClient(client) if arm == 'schema_in_message' else client).call(**current)
                assert not response['cached'], 'Both arms require fresh responses'
                row.update(status='completed', parsed=response['parsed'], call_id=response['call_id'],
                    transcript=str(Path(response['transcript']).relative_to(out)),
                    parsed_equals_archive=response['parsed'] == old['parsed'])
            except Exception as error:
                row.update(status='execution_failed', error=f'{type(error).__name__}: {error}')
            row['wall_seconds'] = time.monotonic() - before
            rows.append(row)
            write(out / 'results.json', rows)
        else:
            status['status'] = 'completed'
    except Exception as error:
        status.update(status='execution_failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status.update(wall_seconds=time.monotonic() - started, finished_unix=time.time(),
            scheduled_results=len(rows), actual_http_requests=len(http_calls), new_video_generations=0,
            production_protocol_changed=False, adopted_arm=None, quality_claim=False)
        write(out / 'status.json', status)
        (args.server_dir / 'STOP').touch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['plan', 'output', 'server-dir']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--judge-pgid', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    run(parser.parse_args())
