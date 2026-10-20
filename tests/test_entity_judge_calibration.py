"""Calibration evidence integrity and gates must fail closed."""
import copy
import json
from pathlib import Path

import pytest

from scripts.calibrate_entity_judge import gate_checks, score_cases, validate_packet
from utils.agent_appearance_judge import ObservedAppearanceComparer
from utils.agent_appearance_evidence import EvidenceAppearanceComparer
from utils.agent_entity_judge import EntitySpec
from utils.agent_runtime import file_sha256
from utils.agent_role_assignment import ContinuityRoleLocalizer, resolve_role_assignments
from utils.agent_vlm import FingerprintedVLMClient, _atomic_write_text


class Client:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return {"parsed": next(self.answers)}


def test_blind_observation_cannot_see_desired_description_or_role_name():
    actual = dict(subject_type="person", hair_or_surface="short white hair",
                  clothing_or_material="green coat", face_or_shape="visible", visibility="visible")
    client = Client([actual, dict(matches_reference=False, differences="white not red hair", target_phrase="red hair")])
    result = ObservedAppearanceComparer(client)._compare_crop(
        EntitySpec("Iris", "curly red hair and green coat"), None, Path("current.png"), shot_index=0, inputs=())
    assert len(client.calls) == 2
    blind, comparison = client.calls
    assert "Iris" not in blind["prompt"] and "red hair" not in blind["prompt"]
    assert blind["image_paths"] == [Path("current.png")]
    assert json.loads(comparison["prompt"])["current_blind_observation"] == actual
    assert result["matches_reference"] is False


def test_reference_is_observed_separately_and_comparison_keeps_original_images():
    obs = dict(subject_type="person", hair_or_surface="black", clothing_or_material="tan",
               face_or_shape="round", visibility="visible")
    client = Client([obs, obs, dict(matches_reference=True, differences="", target_phrase="")])
    ObservedAppearanceComparer(client)._compare_crop(EntitySpec("Nora", "woman"), Path("ref.png"), Path("now.png"), shot_index=1, inputs=())
    assert [c["image_paths"] for c in client.calls] == [[Path("now.png")], [Path("ref.png")], [Path("ref.png"), Path("now.png")]]


def test_abstention_does_not_disappear_from_assignment_denominator():
    labels = {"cases": {"case": {"Nora": {"candidate_ids": [0, -1, None], "appearance": [False, None, None]}}}}
    assignments = [{"Nora": {"status": "uncertain", "subject_index": -1}} for _ in range(3)]
    result = dict(id="case", assignments=assignments, comparisons={"Nora": dict(
        sample_indices=[0], crop_votes=[True], has_reference=False, reference_registration_eligible=True)})
    score = score_cases([result], labels)
    assert score["assignment_total"] == 2
    assert score["assignment_excluded"] == 1
    assert score["assignment_accuracy"] == 0
    assert score["appearance_accuracy"] == 0
    assert score["false_reference_registrations"] == 1


def test_gate_requires_safety_even_with_high_average_accuracy():
    summary = dict(assignment_accuracy=1, appearance_accuracy=1, drift_recall=1, normal_specificity=1,
                   false_reference_registrations=1, false_present_on_reviewed_absence=0)
    policy = dict(holdout_assignment_accuracy_min=.95, holdout_appearance_accuracy_min=.9,
                  holdout_drift_recall_min=.9, holdout_normal_specificity_min=.9,
                  false_reference_registrations_max=0, false_present_on_reviewed_absence_max=0)
    assert not gate_checks(summary, policy, "holdout")["passed"]
    summary["false_reference_registrations"] = 0
    assert gate_checks(summary, policy, "holdout")["passed"]
    summary["drift_recall"] = None
    assert not gate_checks(summary, policy, "holdout")["passed"]


def test_frozen_packet_and_labels_are_valid():
    root = Path(__file__).resolve().parents[1]
    path = root / "videos/entity_judge_calibration_20260905/review/packet.json"
    if not path.exists():
        pytest.skip("local archived calibration assets not distributed")
    packet = json.loads(path.read_text())
    labels = json.loads((root / "docs/entity_judge_calibration_20260905/labels.json").read_text())
    validate_packet(packet, labels, path)
    bad = copy.deepcopy(labels)
    bad["packet_sha256"] = "wrong"
    with pytest.raises(ValueError, match="changed"):
        validate_packet(packet, bad, path)
    bad = copy.deepcopy(labels)
    bad["cases"].pop(next(iter(bad["cases"])))
    with pytest.raises(ValueError, match="every case"):
        validate_packet(packet, bad, path)
    assert file_sha256(path) == labels["packet_sha256"]


def role(index, status="assigned"):
    return dict(status=status, subject_index=index, reason="model evidence")


def test_candidate_exhaustion_separates_missing_role_from_owned_subjects():
    entities = [EntitySpec(n, "person") for n in ("Nora", "Ethan", "Clara")]
    boxes = [(0, 0, 20, 40), (30, 0, 50, 40)]
    result = resolve_role_assignments(entities, boxes,
        {"Nora": role(0), "Ethan": role(1), "Clara": role(-1, "uncertain")}, {})
    assert result["Clara"]["status"] == "absent"
    assert result["Clara"]["resolution"] == "candidate_exhaustion"


def test_unique_same_shot_overlap_can_resolve_vlm_abstention():
    entities = [EntitySpec(n, "person") for n in ("Nora", "Ethan", "Clara")]
    boxes = [(0, 0, 20, 40), (30, 0, 50, 40)]
    result = resolve_role_assignments(entities, boxes, {e.name: role(-1, "uncertain") for e in entities},
                                      {"Nora": (1, 0, 21, 40), "Ethan": (31, 0, 51, 40)})
    assert result["Nora"]["subject_index"] == 0
    assert result["Ethan"]["subject_index"] == 1
    assert result["Clara"]["status"] == "absent"


def test_crossing_conflicts_and_large_motion_remain_uncertain():
    entities = [EntitySpec(n, "person") for n in ("Nora", "Clara")]
    raw = {e.name: role(-1, "uncertain") for e in entities}
    same = {e.name: (0, 0, 20, 40) for e in entities}
    result = resolve_role_assignments(entities, [(0, 0, 20, 40)], raw, same)
    assert all(r["status"] == "uncertain" for r in result.values())
    result = resolve_role_assignments(entities, [(100, 0, 120, 40)], raw, same)
    assert all(r["status"] == "uncertain" for r in result.values())


def test_invalid_claim_is_not_repaired_by_continuity_or_exhaustion():
    entities = [EntitySpec(n, "person") for n in ("Nora", "Clara")]
    result = resolve_role_assignments(entities, [(0, 0, 20, 40)],
        {"Nora": role(0), "Clara": role(999)}, {"Clara": (0, 0, 20, 40)})
    assert result["Clara"]["status"] == "uncertain"


def test_declared_absence_is_not_overridden_by_overlap():
    entity = EntitySpec("Nora", "person")
    result = resolve_role_assignments([entity], [(0, 0, 20, 40)],
        {"Nora": role(-1, "absent")}, {"Nora": (0, 0, 20, 40)})
    assert result["Nora"]["status"] == "absent"


def test_index_protocol_cannot_return_assigned_with_negative_index(tmp_path):
    from PIL import Image
    frame = tmp_path / "frame.png"
    Image.new("RGB", (60, 40)).save(frame)
    client = Client([{"Nora": {"subject_index": 0, "reason": "left actor"},
                      "Clara": {"subject_index": -1, "reason": "no additional actor"}}])
    client.cache_dir = tmp_path / "cache"
    result = ContinuityRoleLocalizer(client).assign_subjects(
        [EntitySpec("Nora", "person"), EntitySpec("Clara", "person")], {}, frame, [(1,1,20,39)],
        caption="Nora works and Clara joins", previous_boxes={}, shot_index=0, inputs=())
    assert result["Nora"]["status"] == "assigned"
    assert result["Clara"]["status"] == "absent"
    assert client.calls[0]["call_kind"] == "entity_assign_index_v3"
    assert "status" not in client.calls[0]["schema"]["properties"]["Clara"]["properties"]


def test_protocol_uncertainty_keeps_reason_and_public_negative_sentinel(tmp_path):
    from PIL import Image
    frame = tmp_path / "frame.png"
    Image.new("RGB", (60,40)).save(frame)
    client = Client([{"Nora": {"subject_index": -2, "reason": "two plausible actors"}}])
    client.cache_dir = tmp_path / "cache"
    result = ContinuityRoleLocalizer(client).assign_subjects([EntitySpec("Nora", "person")], {}, frame,
        [(1,1,20,39), (30,1,50,39)], caption="Nora works", previous_boxes={}, shot_index=0, inputs=())
    assert result["Nora"] == dict(status="uncertain", subject_index=-1, reason="two plausible actors")


@pytest.mark.parametrize("compatible,unknown,contradictions,expected", [
    (["tan outerwear"], ["coat hem outside crop"], [], True),
    (["green coat"], [], ["white hair versus required red"], False),
    ([], ["subject fully obscured"], [], False),
])
def test_only_grounded_contradictions_reject_otherwise_supported_appearance(compatible, unknown, contradictions, expected):
    observation = dict(subject_type="person", hair_or_surface="unknown", clothing_or_material="coat",
                       face_or_shape="visible", visibility="partial")
    evidence = dict(compatible_visible_requirements=compatible,
                    unobservable_or_unspecified=unknown, supported_contradictions=contradictions)
    client = Client([observation, evidence])
    result = EvidenceAppearanceComparer(client)._compare_crop(
        EntitySpec("Clara", "long tan coat"), None, Path("crop.png"), shot_index=0, inputs=())
    assert result["matches_reference"] is expected
    assert result["differences"] == "; ".join(contradictions)
    assert result["target_phrase"] == ""


def test_identical_concurrent_requests_share_one_complete_transcript(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    class SlowClient(FingerprintedVLMClient):
        calls = 0
        def _post(self, payload):
            SlowClient.calls += 1
            time.sleep(.02)
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}
    frame = tmp_path / "frame.png"
    Image.new("RGB", (20,20)).save(frame)
    clients = [SlowClient(base_url="http://unused", model="test", cache_dir=tmp_path / "cache") for _ in range(2)]
    kwargs = dict(prompt="test", image_paths=[frame], schema={"type":"object", "properties":{"ok":{"type":"boolean"}}, "required":["ok"]},
                  call_kind="test", shot_index=0, temporal_inputs=())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: clients[i % 2].call(**kwargs), range(4)))
    assert SlowClient.calls == 1
    assert sum(not r["cached"] for r in results) == 1
    assert all(r["parsed"] == {"ok": True} for r in results)
    assert len({r["call_id"] for r in results}) == 1


def test_atomic_transcript_failure_preserves_prior_complete_file(tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    target.write_text('{"prior":true}\n')
    def fail_replace(*args):
        raise OSError("interrupted publication")
    monkeypatch.setattr("utils.agent_vlm.os.replace", fail_replace)
    with pytest.raises(OSError):
        _atomic_write_text(target, '{"new":true}\n')
    assert json.loads(target.read_text()) == {"prior": True}
    assert list(tmp_path.iterdir()) == [target]


def test_vlm_client_retries_transport_errors_then_raises(monkeypatch, tmp_path):
    """Timeouts / 5xx are retried with identical requests; 4xx and the final failure raise."""
    import io, json, socket, urllib.error, urllib.request
    from utils.agent_vlm import FingerprintedVLMClient

    client = FingerprintedVLMClient(base_url="http://127.0.0.1:1/v1", model="m", cache_dir=tmp_path, max_tokens=8)
    client.TRANSPORT_BACKOFF_SECONDS = 0.0
    calls = []

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise socket.timeout("timed out")
        if len(calls) == 2:
            raise urllib.error.HTTPError(request.full_url, 500, "Internal Server Error", {}, None)
        return _Resp(json.dumps({"choices": []}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert client._post({"x": 1}) == {"choices": []}
    assert len(calls) == 3 and all(t == client.timeout for t in calls)

    calls.clear()
    def always_timeout(request, timeout):
        calls.append(1)
        raise socket.timeout("timed out")
    monkeypatch.setattr(urllib.request, "urlopen", always_timeout)
    import pytest as _pytest
    with _pytest.raises(socket.timeout):
        client._post({"x": 1})
    assert len(calls) == client.TRANSPORT_ATTEMPTS

    calls.clear()
    def bad_request(request, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", bad_request)
    with _pytest.raises(urllib.error.HTTPError):
        client._post({"x": 1})
    assert len(calls) == 1
