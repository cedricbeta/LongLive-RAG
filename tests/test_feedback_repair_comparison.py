import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("feedback_comparison", Path(__file__).parents[1] / "scripts/run_feedback_repair_comparison.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_both_arms_share_inputs_and_budget_but_execute_separate_sources(tmp_path):
    manifest = {"runtime_python": "python", "model": "judge", "budget": {
        "max_edits_per_shot": 3, "commit_lookahead_shots": 2, "max_repair_seconds": 1800}}
    old_source, old_root, old = runner.command(tmp_path, manifest, "scene", "original", 0, "http://local/v1")
    new_source, new_root, new = runner.command(tmp_path, manifest, "scene", "feedback", 0, "http://local/v1")
    for flag in ("--base_config", "--data_root", "--entity_specs", "--max_edits_per_shot", "--commit_lookahead_shots", "--judge_seed", "--absent_action"):
        assert old[old.index(flag)+1] == new[new.index(flag)+1]
    assert old_source != new_source and old_root != new_root
    assert "--repair_policy" not in old
    assert new[new.index("--repair_policy")+1] == "feedback"


def test_frozen_source_change_stops_comparison(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original")
    manifest = {"source_hashes": {str(source): runner.digest(source)}}
    runner.verify(manifest)
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="Frozen sources"):
        runner.verify(manifest)
