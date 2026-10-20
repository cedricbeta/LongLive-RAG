import hashlib
import json

import pytest

from scripts.run_frozen_entity_benchmark import scene_command, sha256, verify_hashes, write_json


def test_policy_is_explicit(tmp_path):
    command = scene_command(tmp_path, "scene", "http://127.0.0.1:9999/v1", 2)
    for flag, value in {"--place_absent": "never", "--max_edits_per_shot": "3",
                        "--commit_lookahead_shots": "2", "--judge_seed": "0",
                        "--clipseg_device": "cuda:2", "--scenes": "scene"}.items():
        assert command[command.index(flag) + 1] == value


def test_frozen_hashes_reject_mutations(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("original")
    hashes = {str(path): sha256(path)}
    assert hashes[str(path)] == hashlib.sha256(b"original").hexdigest()
    verify_hashes(hashes)
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="Frozen inputs/code changed"):
        verify_hashes(hashes)
    path.unlink()
    with pytest.raises(RuntimeError):
        verify_hashes(hashes)


def test_progress_json_replaces_atomically(tmp_path):
    path = tmp_path / "run.json"
    write_json(path, {"evaluation_complete": False, "results": []})
    write_json(path, {"evaluation_complete": False, "results": [{"status": "driver_failed"}]})
    assert json.loads(path.read_text())["results"][0]["status"] == "driver_failed"
    assert not path.with_suffix(".tmp").exists()
