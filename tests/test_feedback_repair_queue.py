import argparse
import json
from pathlib import Path

import pytest

from scripts import queue_feedback_repair_v2 as queue

POLICY = {"render_min_free_mib": 75000, "judge_min_free_mib": 70000,
          "max_utilization": 5, "stable_observations": 2, "poll_seconds": 30}


def gpu(index, free=80000, occupied=False, utilization=0):
    return {"index": index, "free_mib": free, "occupied": occupied, "utilization": utilization}


def test_scheduler_requires_two_distinct_idle_gpus_and_headroom():
    assert queue.choose_pair([gpu(0)], POLICY) is None
    assert queue.choose_pair([gpu(0, 58000), gpu(1, 50000)], POLICY) is None
    assert queue.choose_pair([gpu(0, occupied=True), gpu(1)], POLICY) is None
    assert queue.choose_pair([gpu(0, utilization=90), gpu(1)], POLICY) is None
    assert queue.choose_pair([gpu(0), gpu(1, 72000)], POLICY) == (0, 1)
    assert queue.choose_pair([gpu(0, 72000), gpu(1)], POLICY) == (1, 0)


def test_waits_for_stability_and_rechecks_after_lock(tmp_path, monkeypatch):
    busy = [gpu(0, occupied=True), gpu(1)]
    free = [gpu(0), gpu(1)]
    samples = iter([busy, free, free, busy, free, free, free])
    monkeypatch.setattr(queue, "gpu_inventory", lambda: next(samples))
    monkeypatch.setattr(queue.time, "sleep", lambda _: None)
    handles = []
    def lock(indices):
        h = (tmp_path / f"lock{len(handles)}").open("w")
        handles.append(h)
        return [h]
    monkeypatch.setattr(queue, "acquire_gpu_locks", lock)
    worker = queue.Worker(tmp_path, {"gpu_policy": POLICY})
    assert worker.wait_pair("iris") == (0, 1)
    assert len(handles) == 2
    assert handles[0].closed
    worker.cleanup()
    assert handles[1].closed


def test_stop_file_ends_wait_without_launching(tmp_path):
    (tmp_path / "STOP").touch()
    with pytest.raises(queue.StopRequested):
        queue.Worker(tmp_path, {"gpu_policy": POLICY}).wait_pair("iris")


def test_prepare_reuses_frozen_inputs_and_checks_archived_results(tmp_path, monkeypatch):
    reference, output = tmp_path / "old", tmp_path / "new"
    data = reference / "frozen/data/iris"
    data.mkdir(parents=True)
    (data / "0.json").write_text('{"caption": "Iris"}')
    for name in ("base.yaml", "entity_specs.json"):
        (reference / "frozen" / name).write_text("frozen input")
    baseline = reference / "baseline.mp4"
    baseline.write_bytes(b"baseline video")
    for arm in ("original", "feedback"):
        root = reference / "runs" / arm / "iris"
        (root / "iris").mkdir(parents=True)
        video = root / "final.mp4"
        video.write_bytes(arm.encode())
        queue.write_json(root / "status.json", {"status": "completed", "wall_seconds": 2})
        queue.write_json(root / "iris/summary.json", {"scenes": {"iris": {
            "run_completed": True, "final_delivery_video": str(video),
            "accepted_repairs": [], "edit_attempts": {}, "unresolved_shots": [],
        }}})
    queue.write_json(reference / "manifest.json", {"runtime_python": "python", "model": "judge",
        "source_hashes": {str(baseline): queue.digest(baseline)}, "budget": {"max_edits_per_shot": 3},
        "cases": [{"scene": "iris", "baseline_video": str(baseline), "reason_zh": "hair"}]})
    def snapshot(target):
        target.mkdir()
        (target / "test.py").write_text("frozen source")
        return {"test.py": queue.digest(target / "test.py")}
    monkeypatch.setattr(queue, "snapshot_source", snapshot)
    args = argparse.Namespace(output=output, reference=reference, scenes=["iris"])
    queue.prepare(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert (output / "frozen/data/iris/0.json").read_bytes() == (data / "0.json").read_bytes()
    assert manifest["budget"]["max_edits_per_shot"] == 3
    assert manifest["cases"][0]["baseline_video"] == str(baseline)
    queue.verify(manifest)
    old_video = reference / "runs/feedback/iris/final.mp4"
    old_video.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="Frozen sources"):
        queue.verify(manifest)
