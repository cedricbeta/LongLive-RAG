import pytest
from utils.agent_gpu_ownership import diagnostic_owners, require_owned_compute


def owners(segment=2, render=0, judge=2, shared=True, **kwargs):
    return diagnostic_owners(segment, render, judge, worker_group=101, judge_group=202,
        uuid_for_gpu=kwargs.pop('uuid_for_gpu', lambda gpu: f'GPU-{gpu}'),
        share_diagnostic_judge=shared, **kwargs)


def test_shared_gpu_accepts_only_the_two_owned_groups():
    allowed = owners()
    assert allowed == {'GPU-2': {101, 202}}
    require_owned_compute('GPU-2, 11\nGPU-2, 12\nGPU-3, 99', allowed,
        getpgid=lambda pid: {11: 101, 12: 202}[pid])
    with pytest.raises(RuntimeError, match='foreign compute'):
        require_owned_compute('GPU-2, 13', allowed, getpgid=lambda pid: 303)


def test_isolated_gpu_does_not_admit_the_other_owned_group():
    allowed = owners(segment=1, shared=False)
    require_owned_compute('GPU-1, 11\nGPU-2, 12', allowed,
        getpgid=lambda pid: {11: 101, 12: 202}[pid])
    with pytest.raises(RuntimeError, match='foreign compute'):
        require_owned_compute('GPU-1, 12', allowed, getpgid=lambda pid: 202)


@pytest.mark.parametrize('segment,render,judge,shared', [
    (2, 0, 2, False), (1, 0, 2, True), (2, 2, 2, True),
    (1, 2, 2, False), (3, 0, 3, True), (True, 0, 2, False)])
def test_invalid_or_reserved_placement_is_rejected(segment, render, judge, shared):
    with pytest.raises(ValueError):
        owners(segment, render, judge, shared)


def test_disappeared_process_is_allowed_but_permission_error_is_not():
    def gone(pid):
        raise ProcessLookupError(pid)
    def inaccessible(pid):
        raise PermissionError(pid)
    require_owned_compute('GPU-2, 12', owners(), getpgid=gone)
    with pytest.raises(PermissionError):
        require_owned_compute('GPU-2, 12', owners(), getpgid=inaccessible)


def test_different_indices_cannot_alias_the_same_device():
    with pytest.raises(ValueError, match='same UUID'):
        owners(segment=1, shared=False, uuid_for_gpu=lambda gpu: 'GPU-2')
