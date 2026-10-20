"""Explicit ownership for isolated or colocated native diagnostic processes."""
import os


def diagnostic_owners(segment_gpu, render_gpu, judge_gpu, *, worker_group,
                      judge_group, uuid_for_gpu, share_diagnostic_judge=False):
    values = (segment_gpu, render_gpu, judge_gpu)
    if any(isinstance(g, bool) or not isinstance(g, int) or g not in (0, 1, 2) for g in values):
        raise ValueError('Only authorized GPUs 0/1/2 are supported; GPU 3 is reserved')
    if render_gpu in (segment_gpu, judge_gpu):
        raise ValueError('Rendering requires its own GPU')
    if (segment_gpu == judge_gpu) != bool(share_diagnostic_judge):
        raise ValueError('Shared diagnosis/judge placement requires the explicit sharing flag')
    if any(isinstance(group, bool) or not isinstance(group, int) or group <= 0 for group in (worker_group, judge_group)):
        raise ValueError('Positive owned process groups are required')
    owners = {}
    for gpu, group in ((segment_gpu, worker_group), (judge_gpu, judge_group)):
        uuid = uuid_for_gpu(gpu)
        if not isinstance(uuid, str) or not uuid:
            raise ValueError('Missing GPU UUID')
        owners.setdefault(uuid, set()).add(group)
    if segment_gpu != judge_gpu and len(owners) != 2:
        raise ValueError('Different GPU indices resolved to the same UUID')
    return owners


def require_owned_compute(raw, owners, *, getpgid=os.getpgid):
    for line in raw.splitlines():
        if not line.strip():
            continue
        uuid, pid = (part.strip() for part in line.split(','))
        pid = int(pid)
        if uuid not in owners:
            continue
        try:
            group = getpgid(pid)
        except ProcessLookupError:
            continue
        if group not in owners[uuid]:
            raise RuntimeError('resource_interrupted: foreign compute process on diagnosis/judge GPU')
