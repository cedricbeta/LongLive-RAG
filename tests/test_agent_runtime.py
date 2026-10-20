from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.agent_constraints import (
    AssessmentStatus,
    Constraint,
    ConstraintAssessment,
    ConstraintEvidence,
    ConstraintKind,
    Violation,
)
from utils.agent_runtime import (
    Action,
    ActionProgram,
    AgentRuntimeConfig,
    CausalShotSession,
    GeneratedShot,
    MemoryPayload,
    NormativeGraph,
    ShotContext,
    ShotObservation,
    TemporalInput,
    TrustPartitionedMemory,
    Verdict,
)
from utils.agent_snapshot import PipelineBoundaryAdapter, REQUIRED_INVENTORY
from utils.agent_vlm import (
    ACTION_PROGRAM_SCHEMA,
    VERIFIER_SCHEMA,
    BoundedVLMController,
    _validate_json,
)
from utils.kv_rag import KVRAGConfig, KVRAGEntry, KVRAGMemory


def _scene(tmp_path: Path, shots: int = 3) -> NormativeGraph:
    root = tmp_path / "scene"
    root.mkdir()
    (root / "global.json").write_text(
        json.dumps({"caption": "same person", "invariant_caption": "red coat", "invariants": ["red coat"]}),
        encoding="utf-8",
    )
    for index in range(shots):
        (root / f"{index}.json").write_text(
            json.dumps({"caption": f"shot {index}"}), encoding="utf-8"
        )
    return NormativeGraph.compile_scene(root)


class _Backend:
    def __init__(self, *, future_leak: bool = False):
        self.state = 0
        self.applied = []
        self.commits = []
        self.generations = []
        self.restores = 0
        self.future_leak = future_leak

    def telemetry(self, shot_index):
        return {"committed_state": self.state, "shot": shot_index}

    def apply(self, shot_index, program, *, retry):
        self.applied.append((shot_index, retry, [a.tool for a in program.actions]))
        return [{"metrics": {"applied": True}} for _ in program.actions]

    def snapshot(self, shot_index):
        return {"shot_index": shot_index, "state": self.state}

    def restore(self, snapshot):
        self.state = snapshot["state"]
        self.restores += 1

    def generate(self, shot_index, attempt):
        self.state += 1
        self.generations.append((shot_index, attempt, self.state))
        ref = TemporalInput.from_value(
            kind="generated_frame",
            identifier=f"shot{shot_index}:attempt{attempt}",
            value={"state": self.state},
            available_after_shot=shot_index,
        )
        return GeneratedShot(
            shot_index=shot_index,
            attempt=attempt,
            output={"state": self.state},
            visual_candidate={"crop": f"crop-{shot_index}-{attempt}"},
            model_native_candidate={"kv": f"kv-{shot_index}-{attempt}"},
            input_refs=(ref,),
        )

    def observe(self, generated):
        available = generated.shot_index + 1 if self.future_leak else generated.shot_index
        ref = TemporalInput.from_value(
            kind="observation_frame",
            identifier=f"shot{generated.shot_index}:attempt{generated.attempt}",
            value=generated.output,
            available_after_shot=available,
        )
        return ShotObservation(
            shot_index=generated.shot_index,
            attempt=generated.attempt,
            frames=generated.output,
            crops=generated.visual_candidate,
            telemetry={"state": self.state},
            inputs=(ref,),
            input_fingerprint="",
        )

    def commit(self, generated, *, quarantined):
        value = {"output": generated.output, "quarantined": quarantined}
        self.commits.append(value)
        return value


class _Controller:
    def decide(self, context):
        return ActionProgram.build(
            [Action("abstain", reason="test no-op")],
            diagnosis="test",
            input_fingerprint=context.input_fingerprint,
            shot_index=context.shot_index,
        )

    def repair(self, context, verdict):
        return ActionProgram.build(
            [Action("logit_bias", {"lambda": 1}, "directed retry")],
            diagnosis="repair",
            input_fingerprint=verdict.input_fingerprint,
            shot_index=context.shot_index,
        )


class _Verifier:
    def __init__(self, accepts):
        self.accepts = list(accepts)
        self.calls = 0

    def judge(self, observation, normative):
        accept = self.accepts[min(self.calls, len(self.accepts) - 1)]
        self.calls += 1
        return Verdict(
            accept=accept,
            reason="pass" if accept else "forced failure",
            metrics={"guard": int(accept)},
            input_fingerprint=observation.input_fingerprint,
        )


def _config(tmp_path: Path) -> AgentRuntimeConfig:
    return AgentRuntimeConfig.from_config(
        {
            "enabled": True,
            "ledger_dir": str(tmp_path),
            "lock_path": str(tmp_path / "runtime.lock"),
        }
    )


def test_agent_runtime_defaults_off_and_v1_retry_is_fixed():
    assert AgentRuntimeConfig.from_config(None).enabled is False
    assert AgentRuntimeConfig.from_config({"enabled": "false"}).enabled is False
    with pytest.raises(ValueError, match="exactly one"):
        AgentRuntimeConfig.from_config({"enabled": True, "max_retries_per_shot": 2})


def test_normative_graph_hides_future_shot_text(tmp_path):
    graph = _scene(tmp_path)
    view = graph.view(0)
    assert len(view.shot_state) == 1
    assert "1.json" not in view.source_fingerprints
    assert "2.json" not in view.source_fingerprints
    assert graph.view(2).shot_state[-1]["caption"] == "shot 2"
    graph.assert_unchanged()
    with pytest.raises(TypeError):
        graph.view(0).global_state["x"] = "mutation"
    with pytest.raises(TypeError):
        graph.view(0).global_state["invariants"][0] = "mutation"


@pytest.mark.parametrize(
    "action,match",
    [
        (Action("steer_v", {"entity": "x", "layers": [1], "alpha": 1}, "r"), "layers"),
        (Action("steer_v", {"entity": "x", "layers": [7], "alpha": 0.3}, "r"), "alpha"),
        (Action("inject_entity_kv", {"entity": "x", "anchors": [1, 2, 3, 4, 5], "region_mask": "m"}, "r"), "1..4"),
        (Action("patch_prompt", {"invariant": "x", "review_approved": False}, "r"), "review"),
        (Action("logit_bias", {"lambda": 3}, "r"), "lambda"),
    ],
)
def test_bounded_action_domains_reject_out_of_domain(action, match):
    with pytest.raises(ValueError, match=match):
        action.normalized(shot_index=1)


def test_bounded_action_domains_accept_valid_actions():
    actions = [
        Action("patch_prompt", {"invariant": "same red coat", "review_approved": True}, "authored"),
        Action("inject_entity_kv", {"entity": "person", "boundary": 2, "anchors": [0, 1], "region_mask": "mask.pt"}, "missing"),
        Action("steer_v", {"entity": "person", "layers": [7, 21], "alpha": 0.5}, "drift"),
        Action("inject_scene_anchor_kv", {"boundary": 2}, "layout"),
        Action("rope_remap", {"entries": [0, 2]}, "old tokens"),
        Action("logit_bias", {"lambda": 2}, "old tokens"),
        Action("quarantine", {"shot": 2}, "untrusted"),
    ]
    assert all(action.normalized(shot_index=2).forward_only for action in actions)


def test_candidate_transitions_and_quarantine_is_invisible():
    memory = TrustPartitionedMemory()
    trusted = MemoryPayload(0, 0, {"crop": 0}, {"kv": 0})
    rejected = MemoryPayload(1, 0, {"crop": 1}, {"kv": 1})
    memory.stage(trusted)
    memory.admit(0, 0)
    memory.stage(rejected)
    memory.quarantine(1, 0)
    assert [item.shot_index for item in memory.trusted(as_of_shot=2)] == [0]
    assert memory.partition_counts() == {"candidate": 0, "persistent": 1, "quarantine": 1}
    assert memory.audit_invisible() == []


def test_accept_commits_visual_and_model_native_together(tmp_path):
    backend = _Backend()
    session = CausalShotSession(
        _config(tmp_path), _scene(tmp_path), backend, _Controller(), _Verifier([True]),
        ledger_path=tmp_path / "accepted.jsonl", acquire_lock=False,
    )
    result = session.run_shot(0)
    assert result.accepted and not result.quarantined and result.attempts == 1
    payload = session.memory.trusted(as_of_shot=1)[0]
    assert payload.visual == {"crop": "crop-0-0"}
    assert payload.model_native == {"kv": "kv-0-0"}


def test_forced_fail_retries_exactly_once_then_accepts_quarantined(tmp_path):
    backend = _Backend()
    verifier = _Verifier([False, False])
    ledger = tmp_path / "forced_fail.jsonl"
    session = CausalShotSession(
        _config(tmp_path), _scene(tmp_path), backend, _Controller(), verifier,
        ledger_path=ledger, acquire_lock=False,
    )
    result = session.run_shot(0)
    assert result.attempts == 2
    assert not result.accepted and result.quarantined
    assert backend.restores == 1
    assert [attempt for _, attempt, _ in backend.generations] == [0, 1]
    assert session.retry_counts == {0: 1}
    assert session.memory.trusted(as_of_shot=1) == ()
    assert session.memory.partition_counts() == {"candidate": 0, "persistent": 0, "quarantine": 2}
    records = session.ledger.records()
    assert sum(row["event"] == "restore" for row in records) == 1
    assert records[-1]["event"] == "commit" and records[-1]["quarantined"] is True


def test_temporal_audit_fails_closed_on_future_frame(tmp_path):
    session = CausalShotSession(
        _config(tmp_path), _scene(tmp_path), _Backend(future_leak=True), _Controller(), _Verifier([True]),
        ledger_path=tmp_path / "future.jsonl", acquire_lock=False,
    )
    with pytest.raises(RuntimeError, match="future information leak"):
        session.run_shot(0)
    audits = [row for row in session.ledger.records() if row["event"] == "temporal_audit"]
    assert audits[-1]["passed"] is False


class _LookaheadBackend(_Backend):
    """Tiny full-render backend: previews expose either branch read-only."""

    def edit(self, shot_index, attempt, action, verdict):
        self.state += 100
        ref = TemporalInput.from_value(
            kind="edited_frame",
            identifier=f"shot{shot_index}:attempt{attempt}",
            value={"state": self.state},
            available_after_shot=shot_index,
        )
        generated = GeneratedShot(
            shot_index=shot_index,
            attempt=attempt,
            output={"state": self.state, "branch": "candidate"},
            visual_candidate={"crop": f"edited-{shot_index}"},
            model_native_candidate={"kv": f"edited-kv-{shot_index}"},
            telemetry={"edited": True},
            input_refs=(ref,),
        )
        return generated, {"metrics": {"edited": True}}

    def preview(self, generated, shot_index):
        branch = generated.output.get("branch", "parent")
        ref = TemporalInput.from_value(
            kind="lookahead_frame",
            identifier=f"{branch}:{generated.shot_index}:future{shot_index}",
            value={"branch": branch, "shot": shot_index},
            available_after_shot=shot_index,
        )
        return GeneratedShot(
            shot_index=shot_index,
            attempt=generated.attempt,
            output={"state": self.state, "branch": branch, "preview": shot_index},
            visual_candidate={"crop": f"preview-{branch}-{shot_index}"},
            model_native_candidate={"kv": f"preview-kv-{branch}-{shot_index}"},
            telemetry={"edited": branch == "candidate", "preview": True},
            input_refs=(ref,),
        )


class _LookaheadVerifier:
    def __init__(self, parent_future, candidate_future, *, repair_dependencies=None):
        self.parent_future = dict(parent_future)
        self.candidate_future = dict(candidate_future)
        self.repair_dependencies = repair_dependencies
        self.reference = None
        self.preview_start_states = []

    def snapshot_state(self):
        return {"reference": self.reference}

    def restore_state(self, state):
        self.reference = state["reference"]

    def judge(self, observation, normative):
        output = observation.frames
        if "preview" not in output and output.get("branch") != "candidate":
            action = Action(
                "edit_shot",
                {
                    "shot": observation.shot_index,
                    "prompt": "same person in a red coat",
                    "mask_phrase": "person",
                    "entry_sigma": 1.0,
                },
                "repair identity",
            )
            return Verdict(
                accept=False,
                reason="repair",
                metrics=(
                    {"repair_dependencies": self.repair_dependencies}
                    if self.repair_dependencies is not None
                    else {}
                ),
                directed_actions=(action,),
                input_fingerprint=observation.input_fingerprint,
            )
        if "preview" not in output:
            self.reference = "accepted-current-reference"
            return Verdict(
                accept=True,
                reason="pass current repair",
                metrics={"repair_score": 1.0},
                input_fingerprint=observation.input_fingerprint,
            )

        branch = output["branch"]
        self.preview_start_states.append(
            (branch, observation.shot_index, self.reference)
        )
        # Deliberately mutate state during preview; the session must restore
        # the same frozen state before every parent/candidate judgment.
        self.reference = f"speculative-{branch}-shot-{observation.shot_index}"
        raw = (
            self.parent_future if branch == "parent" else self.candidate_future
        )[observation.shot_index]
        if isinstance(raw, dict):
            accept = bool(raw["accept"])
            metrics = dict(raw.get("metrics", {}))
        elif isinstance(raw, tuple):
            accept, score = raw
            metrics = {"repair_score": score}
        else:
            accept, score = bool(raw), float(bool(raw))
            metrics = {"repair_score": score}
        return Verdict(
            accept=accept,
            reason=f"{branch} pass" if accept else f"{branch} future defect",
            metrics=metrics,
            input_fingerprint=observation.input_fingerprint,
        )


class _FailedEditVerifier:
    """Requests one edit, then rejects both the edit and normal retry."""

    def __init__(self):
        self.calls = 0
        self.reference = None

    def snapshot_state(self):
        return {"reference": self.reference}

    def restore_state(self, state):
        self.reference = state["reference"]

    def judge(self, observation, normative):
        self.calls += 1
        actions = ()
        if self.calls == 1:
            actions = (
                Action(
                    "edit_shot",
                    {
                        "shot": observation.shot_index,
                        "prompt": "same person in a red coat",
                        "mask_phrase": "person",
                        "entry_sigma": 1.0,
                    },
                    "repair identity",
                ),
            )
        if observation.telemetry.get("edited"):
            self.reference = "must-not-survive-rejected-edit"
        else:
            # Baseline and ordinary retry judgments are untrusted candidates too.
            # Their verifier-side learning must be restored after rejection.
            self.reference = "must-not-survive-rejected-ordinary-candidate"
        return Verdict(
            accept=False,
            reason="forced failure",
            metrics={"repair_score": 1.0 if observation.telemetry.get("edited") else 0.0},
            directed_actions=actions,
            input_fingerprint=observation.input_fingerprint,
        )


def _lookahead_config(tmp_path: Path) -> AgentRuntimeConfig:
    return AgentRuntimeConfig.from_config(
        {
            "enabled": True,
            "ledger_dir": str(tmp_path),
            "lock_path": str(tmp_path / "lookahead.lock"),
            "max_edits_per_shot": 1,
            "commit_lookahead_shots": 2,
        }
    )


def test_edited_candidate_promotes_only_after_two_read_only_previews(tmp_path):
    backend = _LookaheadBackend()
    verifier = _LookaheadVerifier(
        {1: (True, 1.0), 2: (True, 1.0)},
        {1: (True, 1.0), 2: (True, 1.0)},
    )
    session = CausalShotSession(
        _lookahead_config(tmp_path),
        _scene(tmp_path),
        backend,
        _Controller(),
        verifier,
        ledger_path=tmp_path / "lookahead-pass.jsonl",
        acquire_lock=False,
    )
    result = session.run_shot(0)
    assert result.accepted and not result.quarantined
    assert len(backend.commits) == 1 and not backend.commits[0]["quarantined"]
    assert session.memory.partition_counts() == {
        "candidate": 0,
        "persistent": 1,
        "quarantine": 1,
    }
    assert verifier.reference == "accepted-current-reference"
    # Both suffixes are judged from the state that preceded the current
    # candidate's judgment (no reference yet), never from the reference the
    # accepted repair has just registered.
    assert verifier.preview_start_states == [
        ("parent", 1, None),
        ("candidate", 1, None),
        ("parent", 2, None),
        ("candidate", 2, None),
    ]
    records = session.ledger.records()
    previews = [row for row in records if row["event"] == "lookahead_verifier"]
    assert [row["shot_index"] for row in previews] == [1, 1, 2, 2]
    assert [row["branch_role"] for row in previews] == [
        "parent", "candidate", "parent", "candidate"
    ]
    guards = [row for row in records if row["event"] == "commit_guard"]
    assert guards[0]["lookahead"]["decision"] == "commit_no_regression"
    assert guards[0]["lookahead"]["verifier_basis"] == "pre_current_judgment"
    assert sum(row["event"] == "commit" for row in records) == 1


def test_second_lookahead_failure_rolls_back_to_parent_and_reference(tmp_path):
    backend = _LookaheadBackend()
    verifier = _LookaheadVerifier(
        {1: (True, 1.0), 2: (True, 1.0)},
        {1: (True, 1.0), 2: (False, 0.0)},
    )
    session = CausalShotSession(
        _lookahead_config(tmp_path),
        _scene(tmp_path),
        backend,
        _Controller(),
        verifier,
        ledger_path=tmp_path / "lookahead-fail.jsonl",
        acquire_lock=False,
    )
    result = session.run_shot(0)
    assert not result.accepted and result.quarantined
    # The failed repair is not delivered.  Both the returned output and the
    # backend continuation state come from the saved parent version.
    assert backend.state == 1
    assert result.delivered["output"] == {"state": 1}
    assert len(backend.commits) == 1 and backend.commits[0]["quarantined"]
    assert session.memory.trusted(as_of_shot=1) == ()
    assert session.memory.partition_counts() == {
        "candidate": 0,
        "persistent": 0,
        "quarantine": 2,
    }
    assert verifier.reference is None
    records = session.ledger.records()
    assert any(
        row["event"] == "restore" and row.get("purpose") == "lookahead_rollback"
        for row in records
    )
    commits = [row for row in records if row["event"] == "commit"]
    assert len(commits) == 1 and commits[0]["reason"] == "lookahead_rollback"
    assert commits[0]["lookahead"]["decision"] == "rollback_relative_regression"
    regression = commits[0]["lookahead"]["checked"][-1]
    assert regression["parent"]["accept"] is True
    assert regression["candidate"]["accept"] is False
    assert regression["comparison"]["comparison_mode"] == "legacy_accept_repair_score"
    assert "accept to reject" in regression["comparison"]["regressions"][0]["reason"]
    assert commits[0]["attempt"] == 0
    assert commits[0]["rejected_candidate_attempt"] == 1
    assert commits[0]["payload_fingerprint"] == session.memory.quarantined_payload(
        0, 0
    ).payload_fingerprint


def test_existing_future_failure_does_not_erase_successful_current_repair(tmp_path):
    """ns290 shape: both suffixes fail, but candidate improves 0 -> .333."""
    backend = _LookaheadBackend()
    verifier = _LookaheadVerifier(
        {1: (False, 0.0), 2: (False, 0.0)},
        {1: (False, 0.333), 2: (False, 0.0)},
    )
    session = CausalShotSession(
        _lookahead_config(tmp_path),
        _scene(tmp_path),
        backend,
        _Controller(),
        verifier,
        ledger_path=tmp_path / "lookahead-existing-defect.jsonl",
        acquire_lock=False,
    )

    result = session.run_shot(0)

    assert result.accepted and not result.quarantined
    guard = next(
        row for row in session.ledger.records() if row["event"] == "commit_guard"
    )
    assert guard["passed"] is True
    assert guard["lookahead"]["decision"] == "commit_no_regression"
    first = guard["lookahead"]["checked"][0]
    assert first["parent"]["accept"] is False
    assert first["candidate"]["accept"] is False
    improvement = first["comparison"]["improvements"][0]
    assert improvement["parent_score"] == pytest.approx(0.0)
    assert improvement["candidate_score"] == pytest.approx(0.333)
    assert first["decision"] == "keep_candidate"


def test_worse_future_repair_score_rolls_back_even_when_both_fail(tmp_path):
    backend = _LookaheadBackend()
    verifier = _LookaheadVerifier(
        {1: (False, 0.5), 2: (False, 0.0)},
        {1: (False, 0.25), 2: (False, 1.0)},
    )
    session = CausalShotSession(
        _lookahead_config(tmp_path),
        _scene(tmp_path),
        backend,
        _Controller(),
        verifier,
        ledger_path=tmp_path / "lookahead-score-regression.jsonl",
        acquire_lock=False,
    )

    result = session.run_shot(0)

    assert not result.accepted and result.quarantined
    guard = next(
        row for row in session.ledger.records() if row["event"] == "commit_guard"
    )
    assert guard["passed"] is False
    assert len(guard["lookahead"]["checked"]) == 1
    comparison = guard["lookahead"]["checked"][0]["comparison"]
    assert comparison["comparison_mode"] == "legacy_accept_repair_score"
    regression = comparison["regressions"][0]
    assert regression["parent_score"] == pytest.approx(0.5)
    assert regression["candidate_score"] == pytest.approx(0.25)


def test_constraint_regression_takes_priority_over_aggregate_repair_score(tmp_path):
    constraint = Constraint(
        id="shot:1:prop:laptop:presence",
        kind=ConstraintKind.PRESENCE,
        description="the laptop remains present",
        subjects=("laptop",),
        shot_indices=(1,),
    )

    def assessment(status, score):
        return ConstraintAssessment(
            constraint=constraint,
            status=status,
            evidence=(ConstraintEvidence("stub", confidence=1.0),),
            violations=(
                ()
                if status == AssessmentStatus.SATISFIED
                else (Violation("missing", "laptop disappeared"),)
            ),
            score=score,
        ).to_dict()

    backend = _LookaheadBackend()
    verifier = _LookaheadVerifier(
        {
            1: {
                "accept": False,
                "metrics": {
                    "repair_score": 0.0,
                    "constraint_assessments": [
                        assessment(AssessmentStatus.SATISFIED, 1.0)
                    ],
                },
            },
            2: (False, 0.0),
        },
        {
            1: {
                "accept": False,
                "metrics": {
                    # Aggregate score alone says "better", but a previously
                    # satisfied script constraint was broken.
                    "repair_score": 1.0,
                    "constraint_assessments": [
                        assessment(AssessmentStatus.VIOLATED, 0.0)
                    ],
                },
            },
            2: (False, 0.0),
        },
    )
    session = CausalShotSession(
        _lookahead_config(tmp_path),
        _scene(tmp_path),
        backend,
        _Controller(),
        verifier,
        ledger_path=tmp_path / "lookahead-constraint-regression.jsonl",
        acquire_lock=False,
    )

    result = session.run_shot(0)

    assert not result.accepted and result.quarantined
    guard = next(
        row for row in session.ledger.records() if row["event"] == "commit_guard"
    )
    comparison = guard["lookahead"]["checked"][0]["comparison"]
    assert comparison["source"] == "constraint_assessments"
    assert comparison["comparison_mode"] == "constraints"
    assert comparison["allow_commit"] is False
    assert comparison["regressions"][0]["constraint_id"] == constraint.id


def test_lookahead_prefers_repair_dependency_shots_over_fixed_distance(tmp_path):
    backend = _LookaheadBackend()
    verifier = _LookaheadVerifier(
        {3: (True, 1.0), 4: (True, 1.0)},
        {3: (True, 1.0), 4: (True, 1.0)},
        repair_dependencies={"subjects": ["Iris"], "future_shots": [4, 3, 3]},
    )
    session = CausalShotSession(
        _lookahead_config(tmp_path),
        _scene(tmp_path, shots=5),
        backend,
        _Controller(),
        verifier,
        ledger_path=tmp_path / "lookahead-dependencies.jsonl",
        acquire_lock=False,
    )

    result = session.run_shot(0)

    assert result.accepted and not result.quarantined
    guard = next(
        row for row in session.ledger.records() if row["event"] == "commit_guard"
    )
    assert guard["lookahead"]["selection"] == {
        "source": "repair_dependencies",
        "eligible": [3, 4],
        "limited": False,
    }
    assert [row["shot_index"] for row in guard["lookahead"]["checked"]] == [3, 4]


def test_all_future_dependency_expands_before_lookahead_limit(tmp_path):
    backend = _LookaheadBackend()
    verifier = _LookaheadVerifier(
        {1: (True, 1.0), 2: (True, 1.0)},
        {1: (True, 1.0), 2: (True, 1.0)},
        repair_dependencies={
            "entity_ids": ["Iris"],
            "future_shots": [],
            "all_future": True,
        },
    )
    session = CausalShotSession(
        _lookahead_config(tmp_path),
        _scene(tmp_path, shots=5),
        backend,
        _Controller(),
        verifier,
        ledger_path=tmp_path / "lookahead-all-future.jsonl",
        acquire_lock=False,
    )

    result = session.run_shot(0)

    assert result.accepted and not result.quarantined
    guard = next(
        row for row in session.ledger.records() if row["event"] == "commit_guard"
    )
    assert guard["lookahead"]["selection"] == {
        "source": "repair_dependencies",
        "eligible": [1, 2, 3, 4],
        "limited": True,
    }
    assert [row["shot_index"] for row in guard["lookahead"]["checked"]] == [1, 2]


def test_all_failed_edits_deliver_nearest_stable_parent_not_highest_failed_candidate(tmp_path):
    backend = _LookaheadBackend()
    verifier = _FailedEditVerifier()
    session = CausalShotSession(
        _lookahead_config(tmp_path),
        _scene(tmp_path),
        backend,
        _Controller(),
        verifier,
        ledger_path=tmp_path / "all-edits-fail.jsonl",
        acquire_lock=False,
    )

    result = session.run_shot(0)

    assert not result.accepted and result.quarantined
    assert backend.state == 1
    assert result.delivered["output"] == {"state": 1}
    assert verifier.reference is None
    commits = [row for row in session.ledger.records() if row["event"] == "commit"]
    assert len(commits) == 1
    assert commits[0]["selection"] == "nearest_stable_parent"
    assert commits[0]["attempt"] == 0
    assert commits[0]["payload_fingerprint"] == session.memory.quarantined_payload(
        0, 0
    ).payload_fingerprint


def test_schema_validator_is_fail_closed():
    valid = {"diagnosis": "no_action", "actions": [{"tool": "abstain", "params": {}, "reason": "insufficient_evidence_abstain"}]}
    _validate_json(valid, ACTION_PROGRAM_SCHEMA)
    with pytest.raises(ValueError, match="unsupported keys"):
        _validate_json({**valid, "tensor": [1, 2]}, ACTION_PROGRAM_SCHEMA)
    with pytest.raises(ValueError, match="must be one of"):
        _validate_json(
            {"diagnosis": "unbounded free-form diagnosis", "actions": valid["actions"]},
            ACTION_PROGRAM_SCHEMA,
        )
    _validate_json(
        {"accept": True, "reason": "accept_numeric_and_visual_guards", "metrics": {}, "directed_actions": []},
        VERIFIER_SCHEMA,
    )
    with pytest.raises(ValueError, match="unsupported keys"):
        _validate_json(
            {
                "accept": True,
                "reason": "accept_numeric_and_visual_guards",
                "metrics": {"free_form": "unbounded"},
                "directed_actions": [],
            },
            VERIFIER_SCHEMA,
        )
    with pytest.raises(ValueError, match="too many items"):
        _validate_json(
            {
                "accept": False,
                "reason": "reject_subject_missing",
                "metrics": {},
                "directed_actions": valid["actions"] * 2,
            },
            VERIFIER_SCHEMA,
        )


def test_audited_action_contract_schema_is_exact_and_single_action(tmp_path):
    context = ShotContext(
        shot_index=2,
        normative=_scene(tmp_path).view(2),
        trusted_memory=(),
        telemetry={
            "executed_action_contract": {
                "tool": "steer_v",
                "params": {
                    "entity": "watchmaker",
                    "layers": [0, 7, 14, 21, 29],
                    "alpha": 1.0,
                },
            }
        },
        inputs=(),
        input_fingerprint="input",
    )
    schema = BoundedVLMController._schema(context)
    valid = {
        "diagnosis": "audited_action_replay",
        "actions": [
            {
                "tool": "steer_v",
                "params": {
                    "entity": "watchmaker",
                    "layers": [0, 7, 14, 21, 29],
                    "alpha": 1.0,
                },
                "reason": "bounded_v_steer_scheduled",
            }
        ],
    }
    _validate_json(valid, schema)
    with pytest.raises(ValueError, match="too many items"):
        _validate_json({**valid, "actions": valid["actions"] * 2}, schema)
    changed = json.loads(json.dumps(valid))
    changed["actions"][0]["params"]["alpha"] = 0.5
    with pytest.raises(ValueError, match="must be one of"):
        _validate_json(changed, schema)


def test_pipeline_boundary_snapshot_inventory_and_exact_restore():
    class Pipeline:
        pass

    pipeline = Pipeline()
    pipeline.kv_cache_pos = [{"k": torch.arange(4)}]
    pipeline.kv_cache_neg = [{"k": torch.arange(4) + 10}]
    pipeline.crossattn_cache_pos = [{"is_init": True, "v": torch.ones(2)}]
    pipeline.crossattn_cache_neg = [{"is_init": False, "v": torch.zeros(2)}]
    pipeline.kv_rag_pos = None
    pipeline.kv_rag_neg = None
    pipeline._dit_model = SimpleNamespace(rope_temporal_offset=1.5)
    output = torch.arange(24, dtype=torch.float32).reshape(1, 6, 1, 2, 2)
    noise = torch.arange(24, dtype=torch.float32).reshape(1, 6, 1, 2, 2) + 100
    adapter = PipelineBoundaryAdapter(pipeline, output=output, noise=noise)
    snapshot = adapter.capture(
        shot_index=1,
        committed_end_frame=2,
        shot_noise_start=2,
        shot_noise_end=4,
        current_start_frame=2,
        cache_start_frame=2,
        vae_pending_state={"queue": [torch.ones(1)]},
        vae_output_cursor=2,
        extra_indices={"chunk_index": 2},
    )
    expected = snapshot.state_fingerprint()
    output.add_(1000)
    noise[:, 2:4].zero_()
    pipeline.kv_cache_pos[0]["k"].zero_()
    pipeline._dit_model.rope_temporal_offset = 9
    restored = adapter.restore(snapshot, shot_noise_start=2, shot_noise_end=4)
    assert torch.equal(output[:, :2], snapshot.committed_output_prefix)
    assert torch.count_nonzero(output[:, 2:]) == 0
    assert torch.equal(noise[:, 2:4], snapshot.current_shot_noise)
    assert torch.equal(pipeline.kv_cache_pos[0]["k"], torch.arange(4))
    assert pipeline._dit_model.rope_temporal_offset == 1.5
    assert restored["vae_output_cursor"] == 2
    assert REQUIRED_INVENTORY.issubset(snapshot.inventory)
    assert snapshot.state_fingerprint() == expected


def test_kv_rag_snapshot_restores_transient_persistent_and_indices():
    bank = KVRAGMemory(
        KVRAGConfig(enabled=True, scene_memory_enabled=True, boundary_inject_anchors=1)
    )
    transient = KVRAGEntry(
        layer=7,
        k=torch.ones(1, 2, 1, 2),
        v=torch.ones(1, 2, 1, 2) * 2,
        summary=torch.ones(1, 2),
        start_token=0,
        end_token=2,
    )
    persistent = KVRAGEntry(
        layer=7,
        k=torch.ones(1, 2, 1, 2) * 3,
        v=torch.ones(1, 2, 1, 2) * 4,
        summary=torch.ones(1, 2),
        start_token=2,
        end_token=4,
        persistent=True,
    )
    bank.entries_by_layer = {7: [transient]}
    bank.scene_entries_by_layer = {7: [persistent]}
    bank.set_boundary_inject(True)
    bank.set_runtime_context(chunk_index=3, shot_index=1, phase="denoise")
    bank.stats["retrieval_calls"] = 9
    state = bank.snapshot_state()
    bank.entries_by_layer.clear()
    bank.scene_entries_by_layer[7][0].v.zero_()
    bank.stats["retrieval_calls"] = 99
    bank.restore_state(state)
    assert torch.equal(bank.entries_by_layer[7][0].v, torch.ones(1, 2, 1, 2) * 2)
    assert torch.equal(bank.scene_entries_by_layer[7][0].v, torch.ones(1, 2, 1, 2) * 4)
    assert bank.stats["retrieval_calls"] == 9
    assert bank._runtime_context == {"chunk_index": 3, "shot_index": 1, "phase": "denoise"}
    assert bank._boundary_inject is True
