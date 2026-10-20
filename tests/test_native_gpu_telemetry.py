import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('monitor_native_gpu',
    Path(__file__).resolve().parents[1] / 'scripts/monitor_native_gpu.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def measurements():
    return [{'query_started_unix': float(i), 'query_finished_unix': i + .1,
             'query_started_monotonic': 100.0 + i, 'energy_millijoules': 100 * i,
             'gpu_utilization_percent': 50, 'memory_used_bytes': 20,
             'compute_pids': [42]} for i in range(7)]


def status():
    return {'started_unix': 1.5, 'wall_seconds': 3, 'pid': 42, 'gpu': 0,
            'gpu_uuid': 'example', 'status': 'completed'}


def test_energy_boundaries_do_not_fabricate_interpolation():
    report = module.summarize(measurements(), status())
    assert report['device_energy_bounds']['lower_joules'] == .2
    assert report['device_energy_bounds']['upper_joules'] == .4
    assert report['samples_strictly_inside'] == 3
    assert report['integrated_sampled_utilization_seconds'] == 1
    assert report['utilization_covered_seconds'] == 2


def test_late_observer_cannot_claim_bounded_whole_run_energy():
    report = module.summarize(measurements()[2:], status())
    assert not report['before_start_sample']
    assert report['device_energy_bounds'] is None


def test_counter_reset_invalidates_energy_but_preserves_observations():
    rows = measurements()
    rows[3]['energy_millijoules'] = 1
    report = module.summarize(rows, status())
    assert not report['energy_counter_monotonic']
    assert report['device_energy_bounds'] is None
    assert report['samples_strictly_inside'] == 3


def test_clock_discontinuity_invalidates_interval_estimates():
    rows = measurements()
    rows[3]['query_started_monotonic'] += 2
    report = module.summarize(rows, status())
    assert report['clock_jump_intervals']
    assert report['device_energy_bounds'] is None
    assert report['integrated_sampled_utilization_seconds'] is None


def test_missing_energy_and_foreign_compute_are_explicit():
    rows = measurements()
    rows[3]['energy_millijoules'] = None
    rows[3]['compute_pids'].append(99)
    report = module.summarize(rows, status())
    assert not report['energy_counter_available']
    assert report['device_energy_bounds'] is None
    assert report['observed_foreign_compute_pids_inside'] == [99]
