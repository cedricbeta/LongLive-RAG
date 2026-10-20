import pytest
from scripts.run_ready_scene_evaluation import aggregate_rows, ready_scenes, remap_ids


def test_ready_does_not_require_all_scenes_and_never_counts_failure():
    state = {"scenes": ["a", "b", "c", "d"], "results": [
        {"scene": "a", "status": "driver_completed"}, {"scene": "b", "status": "driver_failed"},
        {"scene": "c", "status": "running"}, {"scene": "d", "status": "driver_completed"}]}
    assert ready_scenes(state, set()) == ["a", "d"]
    assert ready_scenes(state, {"a"}) == ["d"]


def test_ids_survive_missing_scenes():
    assert remap_ids([{"sample_id": 0, "scene": "c"}], ["a", "b", "c"])[0]["sample_id"] == 2


def test_aggregation_keeps_missing_metrics_null():
    result = aggregate_rows([{"subject_consistency": 0.8}, {"subject_consistency": None}])
    assert result["subject_consistency"] == pytest.approx(0.8)
    assert result["metric_coverage"]["subject_consistency"] == {"valid_count": 1, "total_count": 2}
    assert result["total_score"] is None
    assert aggregate_rows([])["subject_consistency"] is None
