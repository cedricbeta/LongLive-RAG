from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from tests.test_agent_runtime import _Controller, _LookaheadBackend, _LookaheadVerifier, _scene
from utils.agent_constraints import compare_constraint_assessments, entity_judge_metrics_to_assessments
from utils.agent_repair_planner import OfflineFeedbackShotSession, VLMRepairPlanner
from utils.agent_runtime import AgentRuntimeConfig, Verdict, file_sha256


def test_missing_person_reappearing_does_not_redefine_appearance_requirement():
    missing = {"entities": {"Iris": {"expected": True, "located_chunks": [], "matches": None,
                                     "comparison_basis": None}}, "reference_available": {"Iris": True}}
    present = copy.deepcopy(missing)
    present["entities"]["Iris"].update(located_chunks=[0], matches=True, confidence=1,
                                       comparison_basis="reference_crop")
    before = entity_judge_metrics_to_assessments(missing, shot_index=1)
    after = entity_judge_metrics_to_assessments(present, shot_index=1)
    result = compare_constraint_assessments(before, after)
    assert result.allow_commit
    assert not result.regressions
    # Losing a previously established person still blocks a repair.
    reverse = compare_constraint_assessments(after, before)
    assert not reverse.allow_commit


class Client:
    def __init__(self, plans=None):
        self.requests = []
        self.plans = list(plans or [])

    def call(self, **request):
        self.requests.append(request)
        packet = json.loads(request["prompt"])["packet"]
        plan = self.plans.pop(0) if self.plans else {
            "option_id": packet["options"][0]["id"],
            "prompt": ["Iris sits at the table wearing a red coat.",
                       "Iris sits at the table wearing a bright red coat.",
                       "Iris sits at the table wearing a vivid red coat."][min(packet.get("attempt", 1) - 1, 2)],
            "entry_sigma": 1.0, "decision": "revise",
            "reason": "Restore the missing person while keeping the scene.",
            "change_from_previous": "Describe both the person and the ongoing action.",
            "expected_observation": "The same Iris is visible and the following shot retains her.",
        }
        return {"parsed": plan, "call_id": str(len(self.requests)), "cached": False, "wall_seconds": 0.01}


class Backend(_LookaheadBackend):
    def __init__(self, root, *, preview_error=False):
        super().__init__()
        self.root = root
        self.edits = []
        self.preview_error = preview_error

    def edit(self, shot_index, attempt, action, verdict):
        self.edits.append(attempt)
        return super().edit(shot_index, attempt, action, verdict)

    def observe(self, generated):
        observation = super().observe(generated)
        path = self.root / f"{generated.shot_index}_{generated.attempt}_{generated.output.get('branch', 'parent')}.png"
        path.write_bytes(b"fixed stub image")
        return dataclasses.replace(observation, frames=[str(path)],
                                   telemetry={**observation.telemetry, "test_output": generated.output})

    def preview(self, *args):
        if self.preview_error:
            raise RuntimeError("test evaluator unavailable")
        return super().preview(*args)


class Verifier(_LookaheadVerifier):
    def __init__(self, success_attempt=2, clean=False):
        super().__init__({1: True, 2: True}, {1: False, 2: True})
        self.success_attempt = success_attempt
        self.clean = clean

    def judge(self, observation, normative):
        output = observation.telemetry["test_output"]
        if self.clean:
            return Verdict(True, "already correct", input_fingerprint=observation.input_fingerprint)
        result = super().judge(dataclasses.replace(observation, frames=output), normative)
        if "preview" in output and output.get("branch") == "candidate" and observation.attempt >= self.success_attempt:
            return dataclasses.replace(result, accept=True, metrics={"repair_score": 1.0})
        if result.directed_actions:
            result = dataclasses.replace(result, metrics={
                **result.metrics, "repair_specs": {"Iris": "woman in a red coat"},
                "repair_options": [{"id": "detected_regions", "targets": ["Iris"],
                                    "action": result.directed_actions[0].to_dict(),
                                    "allowed_sigmas": [0.83, 1.0]}],
            })
        return result


def session(root, *, success_attempt=2, clean=False, preview_error=False):
    config = AgentRuntimeConfig(enabled=True, max_edits_per_shot=3, commit_lookahead_shots=2)
    backend = Backend(root, preview_error=preview_error)
    client = Client()
    verifier = Verifier(success_attempt, clean)
    instance = OfflineFeedbackShotSession(
        config, _scene(root), backend, _Controller(), verifier,
        planner=VLMRepairPlanner(client), ledger_path=root / "ledger.jsonl", acquire_lock=False,
    )
    return instance, backend, verifier, client


def test_future_failure_reaches_next_plan_and_only_final_candidate_is_delivered(tmp_path):
    instance, backend, verifier, client = session(tmp_path)
    result = instance.run_shot(0)
    assert result.accepted
    assert backend.edits == [1, 2]
    assert len(backend.commits) == 1 and not backend.commits[0]["quarantined"]
    assert verifier.reference == "accepted-current-reference"
    request = client.requests[1]
    packet = json.loads(request["prompt"])["packet"]
    assert packet["target_shot"] == 0
    assert packet["observed_through_shot"] == request["shot_index"] == 1
    assert packet["history"][0]["failure_kind"] == "future_regression"
    assert {r["role"] for r in packet["images"]} >= {"future_parent", "future_candidate", "latest_candidate"}
    assert request["image_paths"]
    assert max(t.available_after_shot for t in request["temporal_inputs"]) == 1
    assert instance.memory.partition_counts() == {"candidate": 0, "persistent": 1, "quarantine": 2}


def test_exhausted_budget_delivers_parent_once_and_discards_candidate_identity(tmp_path):
    instance, backend, verifier, client = session(tmp_path, success_attempt=99)
    result = instance.run_shot(0)
    assert not result.accepted and result.quarantined
    assert backend.edits == [1, 2, 3]
    assert len(backend.commits) == 1 and backend.commits[0]["output"].get("branch") != "candidate"
    assert verifier.reference is None
    assert instance.memory.partition_counts() == {"candidate": 0, "persistent": 0, "quarantine": 4}


def test_evaluation_error_is_not_sent_as_a_visual_failure_for_another_render(tmp_path):
    instance, backend, verifier, client = session(tmp_path, preview_error=True)
    result = instance.run_shot(0)
    assert not result.accepted
    assert backend.edits == [1] and len(client.requests) == 1
    assert verifier.reference is None
    assert any(r.get("reason") == "evaluation_error" for r in instance.ledger.records())


def test_already_correct_control_never_calls_planner_or_renderer(tmp_path):
    instance, backend, _, client = session(tmp_path, clean=True)
    assert instance.run_shot(0).accepted
    assert backend.edits == [] and client.requests == []


def test_planner_cannot_change_mask_or_use_unsupported_regeneration_strength(tmp_path):
    mask = tmp_path / "mask.pt"
    mask.write_bytes(b"parent mask")
    action = {"tool": "edit_shot", "params": {"shot": 0, "mask_path": str(mask), "prompt": "Iris", "entry_sigma": 1.0}, "reason": "repair"}
    packet = {"target_shot": 0, "observed_through_shot": 0, "caption": "Iris sits.",
              "required_characters": {"Iris": "woman"},
              "options": [{"id": "regenerate_shot", "action": action, "allowed_sigmas": [1.0], "mask_sha256": file_sha256(mask)}]}
    wrong = {"option_id": "regenerate_shot", "prompt": "Iris sits.", "entry_sigma": 0.83, "decision": "revise",
             "reason": "test", "change_from_previous": "test", "expected_observation": "test"}
    client = Client([wrong, {**wrong, "entry_sigma": 1.0}])
    result, details = VLMRepairPlanner(client).plan(packet, image_paths=[], inputs=())
    assert result.params["mask_path"] == str(mask) and result.params["entry_sigma"] == 1
    assert len(details["rejected_plans"]) == 1
    mask.write_bytes(b"modified after offer")
    with pytest.raises(RuntimeError, match="mask changed"):
        VLMRepairPlanner(Client([{**wrong, "entry_sigma": 1.0}])).plan(packet, image_paths=[], inputs=())


def repeat_packet(tmp_path):
    options = []
    for name, content in (("base", b"base"), ("alias", b"base"), ("expanded", b"expanded")):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(content)
        action = {"tool": "edit_shot", "params": {"shot": 0, "mask_path": str(path),
                  "prompt": "Iris sits with red hair.", "entry_sigma": 1.0}, "reason": "default answer"}
        options.append({"id": name, "action": action, "mask_sha256": file_sha256(path), "allowed_sigmas": [1.0]})
    packet = {"target_shot": 0, "observed_through_shot": 0, "caption": "Iris sits.",
              "required_characters": {"Iris": "woman with red hair"}, "options": options,
              "history": [{"action": options[0]["action"]}]}
    plan = {"option_id": "alias", "prompt": "Iris sits with RED hair!", "entry_sigma": 1.0,
            "decision": "revise", "reason": "Hair still dark", "change_from_previous": "I changed it",
            "expected_observation": "Red hair"}
    return packet, plan


def test_claimed_change_and_alias_mask_cannot_bypass_duplicate_check(tmp_path):
    packet, plan = repeat_packet(tmp_path)
    client = Client([plan, {**plan, "option_id": "expanded"}])
    action, details = VLMRepairPlanner(client).plan(packet, image_paths=[], inputs=())
    assert action.params["mask_path"].endswith("expanded.pt")
    assert details["actual_changes"] == ["mask"]
    assert "unchanged" in details["rejected_plans"][0]["error"]
    sent = json.loads(client.requests[0]["prompt"])["packet"]
    assert all("action" not in option for option in sent["options"])
    assert "default answer" not in json.dumps(sent["options"])


def test_only_one_explicit_resample_allowed_per_action(tmp_path):
    packet, plan = repeat_packet(tmp_path)
    plan["decision"] = "resample"
    action, details = VLMRepairPlanner(Client([plan])).plan(packet, image_paths=[], inputs=())
    assert details["actual_changes"] == []
    packet["history"].append({"action": action.to_dict(), "planning": details})
    with pytest.raises(ValueError, match="already rendered twice"):
        VLMRepairPlanner(Client([plan, plan])).plan(packet, image_paths=[], inputs=())


def test_uncertain_feedback_does_not_trigger_another_render(tmp_path):
    instance, backend, verifier, client = session(tmp_path)
    judge = verifier.judge
    def uncertain(observation, normative):
        verdict = judge(observation, normative)
        if observation.attempt:
            return dataclasses.replace(verdict, accept=False, reason="reject_entity_assignment_uncertain")
        return verdict
    verifier.judge = uncertain
    result = instance.run_shot(0)
    assert not result.accepted
    assert backend.edits == [1]
    assert len(client.requests) == 1
    assert list(instance.ledger.records())[-1]["reason"] == "insufficient_evidence"


def test_packet_attaches_region_previews_and_audits_the_parent_source(tmp_path):
    import torch
    from PIL import Image
    from tests.test_agent_entity_judge import _tiny_video, _observation
    from utils.agent_edit import save_mask_payload
    instance, _, _, _ = session(tmp_path)
    context = instance.pre_shot(0)
    video = _tiny_video(tmp_path / "parent.mp4")
    frame = tmp_path / "frame.png"
    Image.new("RGB", (64, 32)).save(frame)
    source = dataclasses.replace(_observation(video, 0), frames=[str(frame)], inputs=context.inputs)
    instance.observations["fp"] = source
    mask = save_mask_payload({0: torch.ones(8, 4, 8)}, tmp_path / "mask.pt", phrase="Iris", video=video, grid=(4, 8))
    action = {"tool": "edit_shot", "params": {"shot": 0, "mask_path": str(mask), "prompt": "Iris sits.", "entry_sigma": 1.0}}
    original = Verdict(False, "reject_entity_drift", input_fingerprint="fp", metrics={
        "repair_options": [{"id": "detected_regions", "targets": ["Iris"], "action": action,
                            "mask_sha256": file_sha256(mask), "allowed_sigmas": [1.0]}],
    })
    packet, images, audited = instance._planner_packet(context, original, original, [], 1)
    index = packet["options"][0]["preview_images"][0]
    assert packet["images"][index - 1]["role"] == "region_preview"
    assert images[index - 1].endswith(".jpg")
    assert {item.kind for item in audited} >= {"repair_region_preview", "repair_region_mask", "repair_region_source"}
    assert all(item.available_after_shot <= 0 for item in audited)
