#!/usr/bin/env python3
"""Observe one native supervisor with NVML; never launch or alter its job."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time


def summarize(samples, status):
    start = status['started_unix']
    end = start + status['wall_seconds']
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError('Invalid supervisor interval')
    ordered = sorted(samples, key=lambda r: r['query_started_unix'])
    valid = [r for r in ordered if 'error' not in r]
    inner = [r for r in valid if start <= r['query_started_unix'] <= r['query_finished_unix'] <= end]
    before = [r for r in valid if r['query_finished_unix'] <= start]
    after = [r for r in valid if r['query_started_unix'] >= end]
    relevant = (before[-1:] + inner + after[:1])
    deltas = [b['query_started_unix'] - a['query_started_unix'] for a, b in zip(relevant, relevant[1:])]
    clock_jumps = [i for i, (a, b) in enumerate(zip(relevant, relevant[1:]))
                   if abs((b['query_started_unix'] - a['query_started_unix']) -
                          (b['query_started_monotonic'] - a['query_started_monotonic'])) > .5]
    counters = [r.get('energy_millijoules') for r in relevant]
    available = bool(counters) and all(v is not None for v in counters)
    monotonic = available and all(b >= a for a, b in zip(counters, counters[1:]))
    bounds = None
    if before and after and len(inner) >= 2 and monotonic and not clock_jumps:
        bounds = {'lower_joules': (inner[-1]['energy_millijoules'] - inner[0]['energy_millijoules']) / 1000,
                  'upper_joules': (after[0]['energy_millijoules'] - before[-1]['energy_millijoules']) / 1000,
                  'start_bracket_seconds': inner[0]['query_finished_unix'] - before[-1]['query_started_unix'],
                  'end_bracket_seconds': after[0]['query_finished_unix'] - inner[-1]['query_started_unix']}
    utilization_seconds = sum((b['query_started_monotonic'] - a['query_started_monotonic']) *
                              (a['gpu_utilization_percent'] + b['gpu_utilization_percent']) / 200
                              for a, b in zip(inner, inner[1:])) if not clock_jumps else None
    owned_pid = status.get('pid')
    foreign = sorted({pid for r in inner for pid in r['compute_pids'] if pid != owned_pid})
    return {'supervisor_status': status['status'], 'supervisor_seconds': status['wall_seconds'],
            'gpu': status['gpu'], 'gpu_uuid': status.get('gpu_uuid'), 'owned_generator_pid': owned_pid,
            'samples_strictly_inside': len(inner), 'before_start_sample': bool(before),
            'after_end_sample': bool(after), 'query_errors': sum('error' in r for r in ordered),
            'max_observed_gap_seconds': max(deltas, default=None), 'clock_jump_intervals': clock_jumps,
            'energy_counter_available': available, 'energy_counter_monotonic': monotonic,
            'device_energy_bounds': bounds, 'observed_foreign_compute_pids_inside': foreign,
            'integrated_sampled_utilization_seconds': utilization_seconds,
            'utilization_covered_seconds': (inner[-1]['query_started_monotonic'] - inner[0]['query_started_monotonic']) if len(inner) >= 2 else 0,
            'peak_observed_memory_used_bytes': max((r['memory_used_bytes'] for r in inner), default=None),
            'limits': ['Device energy includes idle power; it is not incremental application energy.',
                       'Energy bounds use measured monotonic counters outside/inside the supervisor interval, without interpolating boundary readings.',
                       'Integrated sampled utilization is a device-activity proxy, not kernel execution time.',
                       'Discrete observations cannot exclude short unobserved foreign processes or utilization changes.',
                       'Wall-clock discontinuities or decreasing energy counters invalidate interval energy bounds.',
                       'This monitor measures one GPU and cannot reconstruct unobserved earlier trials.']}


def observe(args):
    import pynvml
    folder, output = args.folder.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((folder / 'manifest.json').read_text())
    gpu = manifest['gpu']
    if gpu not in (0, 1, 2):
        raise ValueError('GPU 3 is reserved')
    if (folder / 'status.json').exists():
        raise ValueError('Start this observer before the native attempt starts')
    if args.interval <= 0 or args.max_wait <= 0:
        raise ValueError('Positive observation interval and wait budget required')
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + '\n')
    plan = {'protocol': 'native_nvml_counter_bounds_v1', 'frozen_unix': time.time(),
            'target_runtime_folder': folder.name, 'gpu': gpu, 'interval_seconds': args.interval,
            'maximum_wait_seconds': args.max_wait, 'pid': os.getpid(), 'pgid': os.getpgrp(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'target_manifest_sha256': hashlib.sha256((folder / 'manifest.json').read_bytes()).hexdigest(),
            'target_plan_sha256': hashlib.sha256((folder / 'plan.json').read_bytes()).hexdigest(),
            'python': sys.version, 'nvml_python_distribution': importlib.metadata.version('nvidia-ml-py'),
            'role': 'Read-only CPU observer; no model execution or alteration of frozen generation inputs.'}
    (output / 'monitor_source.py').write_bytes(Path(__file__).read_bytes())
    write('plan.json', plan)
    state = {'status': 'waiting_supervisor', 'pid': os.getpid(), 'started_unix': time.time()}
    write('status.json', state)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    write('device.json', {'name': pynvml.nvmlDeviceGetName(handle), 'uuid': pynvml.nvmlDeviceGetUUID(handle),
                          'driver': pynvml.nvmlSystemGetDriverVersion()})
    started = time.monotonic()
    samples = []
    terminal = None
    try:
        with (output / 'samples.jsonl').open('x') as log:
            while True:
                sample = {'query_started_unix': time.time(), 'query_started_monotonic': time.monotonic()}
                try:
                    try:
                        sample['energy_millijoules'] = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                    except pynvml.NVMLError_NotSupported:
                        sample['energy_millijoules'] = None
                    sample.update(power_milliwatts=pynvml.nvmlDeviceGetPowerUsage(handle),
                        gpu_utilization_percent=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                        memory_used_bytes=pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                        compute_pids=[p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)])
                except pynvml.NVMLError as error:
                    sample['error'] = f'{type(error).__name__}: {error}'
                sample['query_finished_unix'] = time.time()
                samples.append(sample)
                log.write(json.dumps(sample) + '\n'); log.flush()
                status_path = folder / 'status.json'
                if status_path.exists():
                    try:
                        status = json.loads(status_path.read_text())
                        if status['status'] not in ('running', 'preflight') and 'wall_seconds' in status:
                            terminal = status
                        elif state['status'] == 'waiting_supervisor':
                            state['status'] = 'observing'; write('status.json', state)
                    except (OSError, ValueError):
                        state['status_read_failures'] = state.get('status_read_failures', 0) + 1
                if terminal and sample['query_started_unix'] >= terminal['started_unix'] + terminal['wall_seconds']:
                    write('target_terminal_status.json', terminal)
                    report = summarize(samples, terminal)
                    report['samples_sha256'] = hashlib.sha256((output / 'samples.jsonl').read_bytes()).hexdigest()
                    report['target_status_sha256'] = hashlib.sha256((output / 'target_terminal_status.json').read_bytes()).hexdigest()
                    write('summary.json', report)
                    state['status'] = 'completed'
                    break
                if time.monotonic() - started > args.max_wait:
                    raise TimeoutError('Observer wait budget exhausted; target is not restarted or terminated')
                time.sleep(args.interval)
    except Exception as error:
        state.update(status='observer_failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        pynvml.nvmlShutdown()
        state.update(finished_unix=time.time(), wall_seconds=time.monotonic() - started)
        write('status.json', state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interval', type=float, default=2)
    p.add_argument('--max-wait', type=float, default=14400)
    observe(p.parse_args())
