"""Role ownership, vote-based registration and authored presence exclusions."""
from pathlib import Path

import pytest

from tests.test_agent_entity_judge import _BoxTracker, _StubComparer, _StubLocalizer, _observation, _tiny_video
from utils.agent_entity_judge import EntityDriftVerifier, EntitySpec, VLMComparer, VLMLocalizer, validate_role_assignments
from utils.agent_runtime import NormativeGraph
from utils.agent_vlm import FingerprintedVLMClient


class Client:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return {"parsed": next(self.answers)}


def row(index, status="assigned"):
    return {"status": status, "subject_index": index, "reason": "role continuity"}


def test_joint_role_assignment_only_returns_candidate_indices(tmp_path):
    from PIL import Image
    frame = tmp_path / "frame.png"
    Image.new("RGB", (100, 80)).save(frame)
    entities = [EntitySpec("Nora", "beige sweater"), EntitySpec("Clara", "tan coat")]
    client = Client([{"Nora": row(1), "Clara": row(0)}])
    client.cache_dir = tmp_path / "cache"
    localizer = VLMLocalizer(client)
    result = localizer.assign_subjects(
        entities, {}, frame, [(1, 2, 40, 70), (50, 2, 90, 70)],
        caption="Clara stands to Nora's left", previous_boxes={}, shot_index=1, inputs=(),
    )
    assert result["Clara"]["subject_index"] == 0
    assert len(client.calls) == 1
    assert client.calls[0]["schema"]["required"] == ["Nora", "Clara"]
    assert "bbox_2d" not in str(client.calls[0]["schema"])
    assert client.calls[0]["image_paths"][0] != frame
    assert client.calls[0]["image_paths"][0].is_file()
    assert "script_description" not in client.calls[0]["prompt"]


@pytest.mark.parametrize("raw", [
    {"Nora": row(0), "Clara": row(0)},
    {"Nora": row(9), "Clara": row(-1, "uncertain")},
    {},
])
def test_invalid_or_conflicting_roles_abstain(raw):
    entities = [EntitySpec("Nora", "woman"), EntitySpec("Clara", "woman")]
    result = validate_role_assignments(entities, [(0, 0, 10, 20)], raw)
    assert all(r["status"] == "uncertain" for r in result.values())


def test_duplicate_geometry_cannot_claim_same_person_twice():
    entities = [EntitySpec("Nora", "woman"), EntitySpec("Clara", "woman")]
    result = validate_role_assignments(entities, [(0, 0, 20, 40), (0, 0, 20, 40)], {"Nora": row(0), "Clara": row(1)})
    assert all(r["status"] == "uncertain" for r in result.values())


@pytest.mark.parametrize("votes,eligible,agreement", [
    ([True, True, True], True, 1.0),
    ([True, True, False], False, 2 / 3),
    ([False, False, False], False, 1.0),
    ([True, True], False, 1.0),
])
def test_registration_requires_three_separate_positive_crop_votes(votes, eligible, agreement):
    client = Client([{"matches_reference": vote, "differences": "" if vote else "wrong hair", "target_phrase": ""} for vote in votes])
    crops = [Path(f"crop{i}.png") for i in range(len(votes))]
    result = VLMComparer(client).compare(EntitySpec("Clara", "auburn hair"), None, crops, shot_index=1, inputs=())
    assert result["reference_registration_eligible"] is eligible
    assert result["confidence"] == pytest.approx(agreement)
    assert [c["image_paths"] for c in client.calls] == [[crop] for crop in crops]
    assert all("confidence" not in c["schema"]["properties"] for c in client.calls)


def setup_judge(tmp_path, entities, loc, comparer, tracker=None, **judge_kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = _tiny_video(tmp_path / "shot.mp4")
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "0.json").write_text('{"caption":"Clara stands to Nora\'s left."}')
    graph = NormativeGraph.compile_scene(scene, allow_missing_global=True)
    judge = EntityDriftVerifier(entities, localizer=loc, comparer=comparer,
                               tracker=tracker or _BoxTracker(), work_dir=tmp_path / "judge", grid=(4, 8), place_absent="never",
                               **judge_kwargs)
    return judge, _observation(video, 0), graph.view(0)


def test_wrong_appearance_stays_in_assigned_role_and_masks_that_person(tmp_path):
    boxes = {("Clara", c): (2, 2, 28, 30) for c in range(2)}
    boxes.update({("Nora", c): (40, 2, 60, 30) for c in range(2)})
    judge, observation, normative = setup_judge(
        tmp_path, [EntitySpec("Clara", "auburn hair"), EntitySpec("Nora", "black curls")],
        _StubLocalizer(boxes, {}), _StubComparer({"Clara": {"matches_reference": False}, "Nora": {"matches_reference": True}}),
    )
    verdict = judge.judge(observation, normative)
    assert verdict.reason == "reject_entity_drift"
    assert verdict.metrics["entities"]["Clara"]["located_chunks"] == [0, 1]
    assert verdict.metrics["mask_keys"] == ["Clara"]
    assert verdict.metrics["unassigned_subjects"] == []


def test_thin_sam_track_does_not_turn_claimed_nora_into_impostor(tmp_path):
    class ThinTracker(_BoxTracker):
        def tracks(self, frames_dir, prompts, num_frames, hw):
            result = super().tracks(frames_dir, prompts, num_frames, hw)
            for track in result.values():
                track[:, :, :55] = False  # fills too little of Nora's box for old overlap assignment
            return result
    judge, observation, normative = setup_judge(
        tmp_path, [EntitySpec("Nora", "woman"), EntitySpec("Clara", "woman")],
        _StubLocalizer({("Nora", c): (40, 2, 60, 30) for c in range(2)}, {}),
        _StubComparer({"Nora": {"matches_reference": True}}), ThinTracker(),
    )
    verdict = judge.judge(observation, normative)
    assert verdict.reason == "reject_entity_absent"
    assert not verdict.directed_actions
    assert verdict.metrics["unassigned_subjects"] == []


def test_ambiguous_presence_neither_scores_nor_registers_nor_repaints(tmp_path):
    entity = EntitySpec.from_value({"name": "Clara", "description": "woman", "shots": [0], "presence": {"0": "ambiguous"}})
    judge, observation, normative = setup_judge(tmp_path, [entity], _StubLocalizer({("Clara", 0): (2, 2, 30, 30)}, {}), _StubComparer({}))
    verdict = judge.judge(observation, normative)
    assert verdict.accept  # allows delivery; explicitly not an evaluation success
    assert verdict.reason == "exclude_ambiguous_presence"
    assert verdict.metrics["repair_score"] is None
    assert verdict.metrics["excluded_ambiguous_entities"] == ["Clara"]
    assert not judge.reference_crops
    assert not verdict.directed_actions
    assert not any("Clara" in r["constraint"]["subjects"] for r in verdict.metrics["constraint_assessments"])
    assert all(r["status"] == "unknown" for r in verdict.metrics["constraint_assessments"])


def test_role_conflict_is_unknown_and_never_repainted_as_absence(tmp_path):
    class ConflictLocalizer(_StubLocalizer):
        def assign_subjects(self, entities, reference_crops, frame, boxes, **kwargs):
            return {e.name: row(0) for e in entities}
    judge, observation, normative = setup_judge(tmp_path, [EntitySpec("Nora", "woman"), EntitySpec("Clara", "woman")],
        ConflictLocalizer({}, {0: [(2, 2, 30, 30)], 1: [(2, 2, 30, 30)]}), _StubComparer({}))
    verdict = judge.judge(observation, normative)
    assert verdict.reason == "reject_entity_assignment_uncertain"
    assert not verdict.directed_actions
    assert all(r["status"] == "unknown" for r in verdict.metrics["constraint_assessments"] if r["constraint"]["kind"] == "presence")


def test_attribute_masks_narrow_the_repaint_to_vetoed_parts(tmp_path):
    import torch
    boxes = {("Clara", c): (2, 2, 28, 30) for c in range(2)}
    comparer = _StubComparer({"Clara": {"matches_reference": False, "differences": "hair measured white vs script red",
                                        "vetoed_parts": ["hair"], "crop_votes": [False, False, False]}})

    def fake_parts(video, chunks, phrase):
        # "hair" = top quarter of the frame at the 4x8 latent grid; no face anywhere
        mask = torch.zeros(8, 4, 8)
        if phrase == "hair":
            mask[:, :1, :] = 1.0
        return {c: mask for c in chunks}

    judge, observation, normative = setup_judge(tmp_path, [EntitySpec("Clara", "a woman with red hair")],
        _StubLocalizer(boxes, {}), comparer, attribute_masks=True, part_mask_fn=fake_parts, part_min_coverage=0.0)
    verdict = judge.judge(observation, normative)
    assert verdict.reason == "reject_entity_drift" and verdict.directed_actions
    info = verdict.metrics["attribute_masks"]["Clara"]
    assert info["parts"] == ["hair"] and info["phrases"] == ["hair"]
    assert all(v <= p for v, p in zip(info["coverage"].values(), info["person_coverage"].values()))
    assert verdict.metrics["mask_coverage_floor"] == 0.002   # narrowed masks are not "unmaskable"
    masks = torch.load(dict(verdict.directed_actions[0].params)["mask_path"], weights_only=False)["masks"][0]
    assert float(masks[:, 1:, :].max()) == 0.0          # nothing below the hair band is repainted
    assert float(masks[:, :1, :].max()) > 0.0

    # without the flag the whole person track is the repaint region
    judge2, observation2, normative2 = setup_judge(tmp_path / "whole", [EntitySpec("Clara", "a woman with red hair")],
        _StubLocalizer(boxes, {}), comparer)
    verdict2 = judge2.judge(observation2, normative2)
    masks2 = torch.load(dict(verdict2.directed_actions[0].params)["mask_path"], weights_only=False)["masks"][0]
    assert float(masks2[:, 1:, :].max()) > 0.0 and verdict2.metrics["attribute_masks"] == {}


def test_sparse_judging_calls_the_localizer_on_first_middle_last_chunks_only(tmp_path):
    boxes = {("Clara", c): (2, 2, 28, 30) for c in range(2)}
    loc = _StubLocalizer(boxes, {})
    judge, observation, normative = setup_judge(tmp_path, [EntitySpec("Clara", "woman")], loc,
                                                _StubComparer({"Clara": {"matches_reference": True}}), judge_chunks="sparse")
    verdict = judge.judge(observation, normative)
    assert verdict.accept
    judged = sorted({c for _, c, _ in loc.calls})
    assert judged == [0, 1]                      # two-chunk fixture: first and last
    with pytest.raises(ValueError):
        setup_judge(tmp_path / "bad", [EntitySpec("Clara", "woman")], loc, _StubComparer({}), judge_chunks="every")


def test_accept_unknown_policy_delivers_clean_shot_without_registering_the_unresolved_role(tmp_path):
    class ConflictLocalizer(_StubLocalizer):
        def assign_subjects(self, entities, reference_crops, frame, boxes, **kwargs):
            return {e.name: row(0) for e in entities}
    judge, observation, normative = setup_judge(tmp_path, [EntitySpec("Nora", "woman"), EntitySpec("Clara", "woman")],
        ConflictLocalizer({}, {0: [(2, 2, 30, 30)], 1: [(2, 2, 30, 30)]}), _StubComparer({}), uncertain_policy="accept_unknown")
    verdict = judge.judge(observation, normative)
    assert verdict.accept and verdict.reason == "accept_with_unknown_assignment"
    assert verdict.metrics["accepted_with_unknown_assignment"] == ["Clara", "Nora"]
    assert judge.reference_crops == {}
    assert all(r["status"] == "unknown" for r in verdict.metrics["constraint_assessments"] if r["constraint"]["kind"] == "presence")
    with pytest.raises(ValueError):
        setup_judge(tmp_path / "bad", [EntitySpec("Nora", "woman")], ConflictLocalizer({}, {}), _StubComparer({}), uncertain_policy="maybe")


def test_ambiguous_role_does_not_score_spatial_relations_or_affect_other_roles(tmp_path):
    entities = [EntitySpec("Nora", "woman"), EntitySpec("Clara", "woman", presence={0: "ambiguous"})]
    boxes = {("Nora", c): (40, 2, 60, 30) for c in range(2)}
    boxes.update({("Clara", c): (2, 2, 30, 30) for c in range(2)})
    judge, observation, normative = setup_judge(tmp_path, entities, _StubLocalizer(boxes, {}), _StubComparer({"Nora": {"matches_reference": True}}))
    verdict = judge.judge(observation, normative)
    assert verdict.accept and verdict.metrics["repair_score"] == 1.0
    assert set(judge.reference_crops) == {"Nora"}
    assert verdict.metrics["spatial_relations"][0]["excluded_ambiguous_presence"]
    assert not any("Clara" in r["constraint"]["subjects"] for r in verdict.metrics["constraint_assessments"])


def test_unanimity_cannot_be_bypassed_by_old_confidence_threshold(tmp_path):
    client = Client([{"matches_reference": v, "differences": "", "target_phrase": ""} for v in [True, False, True]])
    judge, observation, normative = setup_judge(tmp_path, [EntitySpec("Nora", "woman")],
        _StubLocalizer({("Nora", c): (2, 2, 30, 30) for c in range(2)}, {}), VLMComparer(client))
    judge.reference_registration_min_confidence = 0.0
    verdict = judge.judge(observation, normative)
    assert not verdict.accept
    assert not judge.reference_crops
    assert len({str(c["image_paths"][-1]) for c in client.calls}) == 3


def test_sampling_and_image_content_are_part_of_cache_identity(tmp_path):
    class FakeClient(FingerprintedVLMClient):
        requests = []
        def _post(self, payload):
            self.requests.append(payload)
            return {"choices": [{"message": {"content": '{"ok":true}'}}]}
    from PIL import Image
    image = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(image)
    copied = tmp_path / "b.png"
    copied.write_bytes(image.read_bytes())
    client = FakeClient(base_url="http://unused", model="test", cache_dir=tmp_path / "cache", seed=7, max_tokens=123)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    kwargs = dict(prompt="test", image_paths=[image], schema=schema, call_kind="test", shot_index=0, temporal_inputs=())
    first = client.call(**kwargs)
    second = client.call(**{**kwargs, "image_paths": [copied]})
    assert second["cached"] and first["call_id"] == second["call_id"]
    assert len(client.requests) == 1
    assert client.requests[0]["temperature"] == 0 and client.requests[0]["seed"] == 7
    client.seed = 8
    assert client.call(**kwargs)["call_id"] != first["call_id"]
    client.max_tokens = 124
    assert not client.call(**kwargs)["cached"]
