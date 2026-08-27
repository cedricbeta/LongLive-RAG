"""Smoke tests for the edit_shot masked-renoise repair loop (no GPU, no VLM).

Covers, with a mock backend and scripted verifier:
  (a) drift -> edit_shot -> review pass -> commit of the repaired shot;
  (b) edit review failure -> degradation to the pre-existing restore+retry path;
  (c) edit budget: at most one repair per shot, and max_edits_per_shot=0 or a
      backend without edit support disables the stage entirely;
  (d) ledger completeness: hashed, ordered records from which the decision
      chain (verifier directive -> executed edit action) can be reproduced.

Plus CPU-level checks that the pipeline's runtime edit plan reuses the exact
static edit_renoise machinery and that the static path routing is unchanged.

Run: python -m unittest tests.test_agent_edit_shot -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.agent_edit import build_repair_prompt, save_mask_payload
from utils.agent_runtime import (
    Action,
    ActionProgram,
    AgentRuntimeConfig,
    CausalShotSession,
    GeneratedShot,
    NormativeGraph,
    ShotObservation,
    TemporalInput,
    Verdict,
    fingerprint,
)


def _scene(root: Path, shots: int = 1) -> NormativeGraph:
    scene = root / "scene"
    scene.mkdir()
    (scene / "global.json").write_text(
        json.dumps({"caption": "one red snapper fish", "invariants": ["red snapper"]}),
        encoding="utf-8",
    )
    for index in range(shots):
        (scene / f"{index}.json").write_text(
            json.dumps({"caption": f"shot {index}: the same red snapper"}),
            encoding="utf-8",
        )
    return NormativeGraph.compile_scene(scene)


def _config(root: Path, **overrides) -> AgentRuntimeConfig:
    cfg = {
        "enabled": True,
        "ledger_dir": str(root),
        "lock_path": str(root / "runtime.lock"),
    }
    cfg.update(overrides)
    return AgentRuntimeConfig.from_config(cfg)


class _Backend:
    """Deterministic mock backend with optional edit support."""

    def __init__(self):
        self.state = 0
        self.applied = []
        self.commits = []
        self.generations = []
        self.edit_calls = []
        self.restores = 0

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

    def _shot(self, shot_index, attempt, *, edited):
        ref = TemporalInput.from_value(
            kind="edited_frame" if edited else "generated_frame",
            identifier=f"shot{shot_index}:attempt{attempt}",
            value={"state": self.state},
            available_after_shot=shot_index,
        )
        tag = "edited" if edited else "gen"
        return GeneratedShot(
            shot_index=shot_index,
            attempt=attempt,
            output={"state": self.state, "edited": edited},
            visual_candidate={"crop": f"{tag}-crop-{shot_index}-{attempt}"},
            model_native_candidate={"kv": f"{tag}-kv-{shot_index}-{attempt}"},
            input_refs=(ref,),
        )

    def generate(self, shot_index, attempt):
        self.state += 1
        self.generations.append((shot_index, attempt))
        return self._shot(shot_index, attempt, edited=False)

    def edit(self, shot_index, attempt, action, verdict):
        self.edit_calls.append((shot_index, attempt, dict(action.params)))
        self.state += 100
        generated = self._shot(shot_index, attempt, edited=True)
        result = {
            "metrics": {"mask_coverage": 0.12, "chunks_edited": 6},
            "plan": {"entry_sigma": dict(action.params)["entry_sigma"]},
        }
        return generated, result

    def observe(self, generated):
        ref = TemporalInput.from_value(
            kind="observation_frame",
            identifier=f"shot{generated.shot_index}:attempt{generated.attempt}",
            value=generated.output,
            available_after_shot=generated.shot_index,
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


class _BackendNoEdit(_Backend):
    # Not callable -> the session must skip the edit stage entirely.
    edit = None


class _Controller:
    def decide(self, context):
        return ActionProgram.build(
            [Action("abstain", reason="no directed action")],
            diagnosis="no action",
            input_fingerprint=context.input_fingerprint,
            shot_index=context.shot_index,
        )

    def repair(self, context, verdict):
        return ActionProgram.build(
            [Action("logit_bias", {"lambda": 1}, "directed retry")],
            diagnosis="repair after rejection",
            input_fingerprint=verdict.input_fingerprint,
            shot_index=context.shot_index,
        )


def _edit_action(shot_index):
    return Action(
        "edit_shot",
        {
            "shot": shot_index,
            "prompt": "the same red snapper, red scales and fins",
            "mask_phrase": "fish",
            "entry_sigma": 1.0,
        },
        "masked_renoise_repair_scheduled",
    )


class _ScriptedVerifier:
    """Sequence of 'accept' | 'reject_edit' | 'reject_plain' verdicts."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def judge(self, observation, normative):
        kind = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if kind == "accept":
            return Verdict(
                accept=True,
                reason="pass",
                metrics={"subject_drifted": False},
                input_fingerprint=observation.input_fingerprint,
            )
        directed = ()
        if kind == "reject_edit":
            directed = (_edit_action(observation.shot_index),)
        return Verdict(
            accept=False,
            reason="reject_identity_drift",
            metrics={"subject_drifted": True, "drift_description": "fish turned olive-gray"},
            directed_actions=directed,
            input_fingerprint=observation.input_fingerprint,
        )


def _run_single_shot(root: Path, backend, verifier, **config_overrides):
    session = CausalShotSession(
        _config(root, **config_overrides),
        _scene(root),
        backend,
        _Controller(),
        verifier,
        ledger_path=root / "ledger.jsonl",
        acquire_lock=False,
    )
    result = session.run_shot(0)
    return session, result


class EditShotStateMachineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_drift_edit_review_pass_commits_repaired_shot(self):
        backend = _Backend()
        session, result = _run_single_shot(
            self.root, backend, _ScriptedVerifier(["reject_edit", "accept"])
        )
        self.assertTrue(result.accepted)
        self.assertFalse(result.quarantined)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(backend.edit_calls), 1)
        self.assertEqual(backend.edit_calls[0][:2], (0, 1))
        # The edit ran on the restored boundary state, not forward from the
        # rejected provisional shot.
        self.assertEqual(backend.restores, 1)
        # No retry slot was consumed; the retry budget stays available.
        self.assertEqual(session.retry_counts, {})
        self.assertEqual(session.edit_counts, {0: 1})
        # The REPAIRED candidate is what entered trusted memory; the rejected
        # provisional candidate is quarantined and invisible.
        trusted = session.memory.trusted(as_of_shot=1)
        self.assertEqual(len(trusted), 1)
        self.assertEqual(trusted[0].visual, {"crop": "edited-crop-0-1"})
        self.assertEqual(
            session.memory.partition_counts(),
            {"candidate": 0, "persistent": 1, "quarantine": 1},
        )
        self.assertEqual(session.memory.audit_invisible(), [])
        self.assertEqual(backend.commits[-1]["output"]["edited"], True)

    def test_edit_review_fail_degrades_to_restore_retry(self):
        backend = _Backend()
        session, result = _run_single_shot(
            self.root, backend, _ScriptedVerifier(["reject_edit", "reject_edit", "accept"])
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.attempts, 3)
        # Exactly one edit, then the pre-existing restore+retry path.
        self.assertEqual(len(backend.edit_calls), 1)
        self.assertEqual(backend.restores, 2)
        self.assertEqual(session.retry_counts, {0: 1})
        self.assertEqual(session.edit_counts, {0: 1})
        # The retry program went through apply() with the retry flag.
        retry_applies = [row for row in backend.applied if row[1]]
        self.assertEqual(len(retry_applies), 1)
        self.assertEqual(retry_applies[0][2], ["logit_bias"])
        # Both rejected candidates (provisional + failed repair) are quarantined.
        self.assertEqual(
            session.memory.partition_counts(),
            {"candidate": 0, "persistent": 1, "quarantine": 2},
        )
        self.assertEqual(backend.commits[-1]["output"]["edited"], False)

    def test_edit_budget_one_repair_then_quarantine_path(self):
        backend = _Backend()
        session, result = _run_single_shot(
            self.root, backend, _ScriptedVerifier(["reject_edit"])
        )
        # Every review directs an edit, but the budget admits exactly one; the
        # shot ends accept-with-quarantine through the legacy path.
        self.assertFalse(result.accepted)
        self.assertTrue(result.quarantined)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(len(backend.edit_calls), 1)
        self.assertEqual(session.edit_counts, {0: 1})
        self.assertEqual(session.retry_counts, {0: 1})
        self.assertEqual(
            session.memory.partition_counts(),
            {"candidate": 0, "persistent": 0, "quarantine": 3},
        )
        self.assertEqual(session.memory.trusted(as_of_shot=1), ())

    def test_edit_disabled_by_config_budget_zero(self):
        backend = _Backend()
        session, result = _run_single_shot(
            self.root,
            backend,
            _ScriptedVerifier(["reject_edit", "accept"]),
            max_edits_per_shot=0,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(backend.edit_calls, [])
        self.assertEqual(session.retry_counts, {0: 1})

    def test_backend_without_edit_uses_legacy_path(self):
        backend = _BackendNoEdit()
        session, result = _run_single_shot(
            self.root, backend, _ScriptedVerifier(["reject_edit", "accept"])
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(session.retry_counts, {0: 1})
        self.assertEqual(session.edit_counts, {})

    def test_ledger_records_complete_ordered_and_replayable(self):
        backend = _Backend()
        session, _ = _run_single_shot(
            self.root, backend, _ScriptedVerifier(["reject_edit", "reject_edit", "accept"])
        )
        records = session.ledger.records()
        # Every record hash verifies and sequence numbers are contiguous.
        for index, record in enumerate(records):
            payload = dict(record)
            expected = payload.pop("record_sha256")
            self.assertEqual(expected, fingerprint(payload), f"hash mismatch at {index}")
            self.assertEqual(record["sequence"], index)

        events = [row["event"] for row in records]
        # Decision-chain order: provisional candidate rejected, restore for the
        # edit, the edit action, repaired candidate rejected, restore for the
        # retry, the repair action, final candidate committed.
        def indices(name, **match):
            return [
                i
                for i, row in enumerate(records)
                if row["event"] == name and all(row.get(k) == v for k, v in match.items())
            ]

        verifier_rows = [row for row in records if row["event"] == "verifier"]
        self.assertEqual([row["attempt"] for row in verifier_rows], [0, 1, 2])
        edit_actions = [
            row for row in records if row["event"] == "action" and row.get("tool") == "edit_shot"
        ]
        self.assertEqual(len(edit_actions), 1)
        edit_row = edit_actions[0]
        # Required audit fields (scripts/audit_agent_ledger.py ACTION_REQUIRED).
        for key in (
            "diagnostic_input_fingerprint",
            "tool",
            "params",
            "reason",
            "attention_mass_check",
            "result_metrics",
        ):
            self.assertIn(key, edit_row)
        # The edit is keyed to the exact verifier observation that triggered it.
        self.assertEqual(
            edit_row["diagnostic_input_fingerprint"], verifier_rows[0]["input_fingerprint"]
        )
        # Replayability: the verifier record carries the directed action, and
        # the executed edit action matches it (same decision from same input).
        directed = verifier_rows[0]["directed_actions"]
        self.assertEqual(len(directed), 1)
        self.assertEqual(directed[0]["tool"], "edit_shot")
        self.assertEqual(directed[0]["params"], edit_row["params"])
        self.assertEqual(edit_row["result_metrics"], {"mask_coverage": 0.12, "chunks_edited": 6})
        # Ordering constraints.
        reject0 = indices("reject_candidate", attempt=0)[0]
        restore_edit = [
            i for i, row in enumerate(records)
            if row["event"] == "restore" and row.get("purpose") == "edit_shot"
        ]
        self.assertEqual(len(restore_edit), 1)
        edit_index = records.index(edit_row)
        candidate1 = indices("candidate", attempt=1)[0]
        reject1 = indices("reject_candidate", attempt=1)[0]
        restore_retry = [
            i for i, row in enumerate(records)
            if row["event"] == "restore" and "retry_count" in row
        ]
        self.assertEqual(len(restore_retry), 1)
        commit = indices("commit")[0]
        self.assertLess(reject0, restore_edit[0])
        self.assertLess(restore_edit[0], edit_index)
        self.assertLess(edit_index, candidate1)
        self.assertLess(candidate1, reject1)
        self.assertLess(reject1, restore_retry[0])
        self.assertLess(restore_retry[0], commit)
        self.assertIn("commit", events)
        self.assertEqual(records[commit]["attempt"], 2)
        self.assertFalse(records[commit]["quarantined"])


class EditShotActionBoundsTest(unittest.TestCase):
    def test_edit_action_domains(self):
        good = _edit_action(3).normalized(shot_index=3)
        self.assertFalse(good.forward_only)
        self.assertEqual(dict(good.params)["entry_sigma"], 1.0)
        with self.assertRaisesRegex(ValueError, "entry_sigma"):
            Action(
                "edit_shot",
                {"shot": 3, "prompt": "p", "mask_phrase": "fish", "entry_sigma": 0.5},
                "r",
            ).normalized(shot_index=3)
        with self.assertRaisesRegex(ValueError, "repair prompt"):
            Action(
                "edit_shot", {"shot": 3, "prompt": " ", "mask_phrase": "fish"}, "r"
            ).normalized(shot_index=3)
        with self.assertRaisesRegex(ValueError, "mask_phrase"):
            Action(
                "edit_shot", {"shot": 3, "prompt": "p", "mask_phrase": ""}, "r"
            ).normalized(shot_index=3)
        with self.assertRaisesRegex(ValueError, "current shot"):
            Action(
                "edit_shot", {"shot": 2, "prompt": "p", "mask_phrase": "fish"}, "r"
            ).normalized(shot_index=3)

    def test_edit_shot_rejected_in_apply_and_nested_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = CausalShotSession(
                _config(root),
                _scene(root),
                _Backend(),
                _Controller(),
                _ScriptedVerifier(["accept"]),
                ledger_path=root / "ledger.jsonl",
                acquire_lock=False,
            )
            program = ActionProgram.build(
                [_edit_action(0)],
                diagnosis="identity_drift",
                input_fingerprint="fp",
                shot_index=0,
            )
            with self.assertRaisesRegex(ValueError, "edit stage"):
                session.apply(0, program, retry=False)
        with self.assertRaisesRegex(ValueError, "cannot nest"):
            Action(
                "restore_and_retry",
                {"action_program": [_edit_action(0).to_dict()]},
                "r",
            ).normalized(shot_index=0)

    def test_config_bounds(self):
        with self.assertRaisesRegex(ValueError, "at most one edit_shot"):
            AgentRuntimeConfig.from_config({"enabled": True, "max_edits_per_shot": 2})
        with self.assertRaisesRegex(ValueError, "edit_entry_sigma"):
            AgentRuntimeConfig.from_config({"enabled": True, "edit_entry_sigma": 0.5})
        cfg = AgentRuntimeConfig.from_config(
            {"enabled": True, "edit_entry_sigma": 0.94, "max_edits_per_shot": 0}
        )
        self.assertEqual(cfg.edit_entry_sigma, 0.94)
        self.assertEqual(cfg.max_edits_per_shot, 0)

    def test_build_repair_prompt_template_hook(self):
        self.assertEqual(
            build_repair_prompt("A red snapper.", "turned gray", template="{caption} FIX: {drift}"),
            "A red snapper. FIX: turned gray",
        )
        self.assertEqual(build_repair_prompt("A red snapper.", ""), "A red snapper.")
        with self.assertRaises(ValueError):
            build_repair_prompt("", "")


class _StubVLMClient:
    """Returns a scripted drift-judge JSON without any server."""

    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def call(self, *, prompt, image_paths, schema, call_kind, shot_index, temporal_inputs):
        self.calls.append(
            {
                "prompt": json.loads(prompt),
                "images": [str(p) for p in image_paths],
                "call_kind": call_kind,
                "shot_index": shot_index,
            }
        )
        return {"parsed": dict(self.parsed), "call_id": "stub-call", "cached": False}


class ComparativeDriftVerifierTest(unittest.TestCase):
    def _observation(self, root: Path, *, with_reference: bool):
        reference = []
        if with_reference:
            ref = root / "ref0.png"
            ref.write_bytes(b"png")
            reference = [str(ref)]
        return ShotObservation(
            shot_index=1,
            attempt=0,
            frames=["frame_a", "frame_b"],
            crops=[],
            telemetry={"reference_frames": reference},
            inputs=(),
            input_fingerprint="obs-fp",
        )

    def test_drift_verdict_directs_edit_with_templated_prompt(self):
        from utils.agent_vlm import ComparativeDriftVerifier

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = _scene(root, shots=2)
            client = _StubVLMClient(
                {
                    "subject_drifted": True,
                    "drift_description": "the fish turned olive-gray",
                    "confidence": 0.75,
                }
            )
            verifier = ComparativeDriftVerifier(
                client,
                entity_phrase="fish",
                edit_entry_sigma=1.0,
                prompt_template="{caption} FIX: {drift}",
            )
            verdict = verifier.judge(self._observation(root, with_reference=True), graph.view(1))
            self.assertFalse(verdict.accept)
            self.assertEqual(verdict.reason, "reject_identity_drift")
            self.assertEqual(len(verdict.directed_actions), 1)
            action = verdict.directed_actions[0]
            self.assertEqual(action.tool, "edit_shot")
            params = dict(action.params)
            self.assertEqual(params["mask_phrase"], "fish")
            self.assertEqual(params["entry_sigma"], 1.0)
            self.assertEqual(
                params["prompt"],
                "shot 1: the same red snapper FIX: the fish turned olive-gray",
            )
            # Comparative call: reference images precede the current frames and
            # the prompt states the split.
            call = client.calls[0]
            self.assertEqual(call["call_kind"], "drift_judge")
            self.assertEqual(call["prompt"]["reference_image_count"], 1)
            self.assertTrue(call["images"][0].endswith("ref0.png"))

    def test_no_drift_and_no_reference_accept(self):
        from utils.agent_vlm import ComparativeDriftVerifier

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = _scene(root, shots=2)
            client = _StubVLMClient(
                {"subject_drifted": False, "drift_description": "", "confidence": 1}
            )
            verifier = ComparativeDriftVerifier(client, entity_phrase="fish")
            verdict = verifier.judge(self._observation(root, with_reference=True), graph.view(1))
            self.assertTrue(verdict.accept)
            self.assertEqual(verdict.directed_actions, ())
            # Opening shot without committed reference frames: auto-accept, no call.
            client2 = _StubVLMClient({})
            verifier2 = ComparativeDriftVerifier(client2, entity_phrase="fish")
            verdict2 = verifier2.judge(self._observation(root, with_reference=False), graph.view(1))
            self.assertTrue(verdict2.accept)
            self.assertEqual(client2.calls, [])
            # Low-confidence drift claims do not trigger a repair.
            client3 = _StubVLMClient(
                {"subject_drifted": True, "drift_description": "maybe", "confidence": 0.25}
            )
            verifier3 = ComparativeDriftVerifier(client3, entity_phrase="fish")
            verdict3 = verifier3.judge(self._observation(root, with_reference=True), graph.view(1))
            self.assertTrue(verdict3.accept)


def _dummy_pipeline(grid_h=4, grid_w=5, blocks=2):
    """A CausalDiffusionInferencePipeline shell exercising only the edit path."""
    from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline

    pipe = object.__new__(CausalDiffusionInferencePipeline)
    pipe.edit_renoise = None
    pipe._runtime_edit_plan = None
    pipe._edit_masks = None
    pipe._edit_conditional_dict = None
    pipe._edit_donor = None
    pipe.num_transformer_blocks = blocks
    pipe.num_train_timesteps = 1000
    pipe.frame_seq_length = grid_h * grid_w
    pipe.crossattn_cache_pos = [{"is_init": True} for _ in range(blocks)]
    pipe.kv_cache_pos = None
    pipe.kv_rag_pos = None
    pipe.kv_rag_enabled = False
    pipe.kv_rag_config = SimpleNamespace(retrieve_during_denoise=False)
    pipe.capsule_runtime = None
    pipe.text_encoder = lambda text_prompts: {
        "prompt_embeds": torch.zeros(len(text_prompts), 4, 8)
    }
    pipe.generator = (
        lambda noisy_image_or_video, **kwargs: (torch.zeros_like(noisy_image_or_video), None)
    )
    return pipe


def _scheduler():
    return SimpleNamespace(
        sigmas=torch.tensor([1.0, 0.94, 0.63, 0.0]),
        timesteps=torch.tensor([1000.0, 940.0, 630.0]),
    )


def _soft_mask(frames=2, grid_h=4, grid_w=5):
    soft = torch.zeros(frames, grid_h, grid_w)
    soft[:, :, : grid_w // 2] = 1.0  # left half is the subject
    return soft


class PipelineRuntimeEditPlanTest(unittest.TestCase):
    def test_static_predicate_unchanged_and_plan_is_secondary(self):
        pipe = _dummy_pipeline()
        # Feature-off: no static block, no plan -> never active (legacy no-op).
        for chunk in range(0, 50):
            self.assertIsNone(pipe._active_edit_cfg(chunk))
        # Static block: the new helper reproduces the legacy predicate exactly.
        pipe.edit_renoise = {"chunk_start": 30, "chunk_end": 35}
        for chunk in range(0, 50):
            legacy = (
                pipe.edit_renoise is not None
                and pipe.edit_renoise["chunk_start"] <= chunk <= pipe.edit_renoise["chunk_end"]
            )
            active = pipe._active_edit_cfg(chunk)
            self.assertEqual(active is pipe.edit_renoise, legacy, f"chunk {chunk}")
        # A runtime plan never overrides a chunk claimed by the static block.
        pipe.set_runtime_edit_plan(
            chunk_start=33, chunk_end=40, masks={33: _soft_mask()}, prompt="repair"
        )
        for chunk in range(0, 50):
            active = pipe._active_edit_cfg(chunk)
            if 30 <= chunk <= 35:
                self.assertIs(active, pipe.edit_renoise)
            elif 36 <= chunk <= 40:
                self.assertIs(active, pipe._runtime_edit_plan)
            else:
                self.assertIsNone(active)
        pipe.clear_runtime_edit_plan()
        self.assertIsNone(pipe._active_edit_cfg(38))

    def test_runtime_plan_edits_mask_region_and_pins_background(self):
        pipe = _dummy_pipeline()
        summary = pipe.set_runtime_edit_plan(
            chunk_start=3,
            chunk_end=3,
            masks={3: _soft_mask()},
            prompt="the same red snapper",
            entry_sigma=1.0,
            mask_dilate=0,
            noise_seed=7,
        )
        self.assertEqual(summary["chunk_start"], 3)
        self.assertAlmostEqual(summary["mask_coverage"][3], 0.4, places=5)
        torch.manual_seed(0)
        latents = torch.randn(1, 2, 3, 8, 10)
        edited = pipe._apply_edit_renoise(
            latents.clone(),
            sample_scheduler=_scheduler(),
            chunk_index=3,
            scene_shot_index=1,
            current_num_frames=2,
            current_start_frame=24,
            cache_start_frame=24,
            batch_size=1,
            cfg=pipe._active_edit_cfg(3),
        )
        mask = torch.zeros(1, 2, 1, 8, 10)
        mask[..., :4] = 1.0  # upsampled left half
        # Background pinned bit-exactly, masked region rewritten.
        self.assertTrue(torch.equal(edited * (1 - mask), latents * (1 - mask)))
        self.assertFalse(torch.equal(edited * mask, latents * mask))
        # The repair prompt repopulates cross-attention (cache invalidated).
        self.assertTrue(all(not c["is_init"] for c in pipe.crossattn_cache_pos))
        # A chunk without a mask entry is returned untouched.
        untouched = pipe._apply_edit_renoise(
            latents.clone(),
            sample_scheduler=_scheduler(),
            chunk_index=4,
            scene_shot_index=1,
            current_num_frames=2,
            current_start_frame=32,
            cache_start_frame=32,
            batch_size=1,
            cfg=pipe._runtime_edit_plan,
        )
        self.assertTrue(torch.equal(untouched, latents))
        # Deterministic: replaying the same plan yields identical latents.
        pipe2 = _dummy_pipeline()
        pipe2.set_runtime_edit_plan(
            chunk_start=3,
            chunk_end=3,
            masks={3: _soft_mask()},
            prompt="the same red snapper",
            entry_sigma=1.0,
            mask_dilate=0,
            noise_seed=7,
        )
        edited2 = pipe2._apply_edit_renoise(
            latents.clone(),
            sample_scheduler=_scheduler(),
            chunk_index=3,
            scene_shot_index=1,
            current_num_frames=2,
            current_start_frame=24,
            cache_start_frame=24,
            batch_size=1,
            cfg=pipe2._runtime_edit_plan,
        )
        self.assertTrue(torch.equal(edited, edited2))

    def test_static_path_routing_is_default_and_identical(self):
        """cfg=None resolves to the static block: same object, same output."""
        with tempfile.TemporaryDirectory() as tmp:
            mask_path = save_mask_payload(
                {3: _soft_mask()}, Path(tmp) / "masks.pt", phrase="fish"
            )
            static_cfg = {
                "chunk_start": 3,
                "chunk_end": 3,
                "entry_index": 2,
                "entry_sigma": 1.0,
                "mask_path": str(mask_path),
                "mask_threshold": 0.35,
                "mask_dilate": 0,
                "prompt": "repair prompt",
                "noise_seed": 4242,
                "collage": False,
                "donor_chunk": -1,
                "donor_frame": 0,
                "donor_mask_path": "",
                "donor_flip": False,
                "paste_pin": False,
            }
            torch.manual_seed(1)
            latents = torch.randn(1, 2, 3, 8, 10)
            kwargs = dict(
                sample_scheduler=_scheduler(),
                chunk_index=3,
                scene_shot_index=1,
                current_num_frames=2,
                current_start_frame=24,
                cache_start_frame=24,
                batch_size=1,
            )
            # Default-cfg call (the refactored static call path).
            pipe_default = _dummy_pipeline()
            pipe_default.edit_renoise = dict(static_cfg)
            out_default = pipe_default._apply_edit_renoise(latents.clone(), **kwargs)
            # Explicit-cfg call through the new dispatcher.
            pipe_explicit = _dummy_pipeline()
            pipe_explicit.edit_renoise = dict(static_cfg)
            out_explicit = pipe_explicit._apply_edit_renoise(
                latents.clone(), cfg=pipe_explicit._active_edit_cfg(3), **kwargs
            )
            self.assertTrue(torch.equal(out_default, out_explicit))
            # The static path used the legacy self.* caches, not plan-local ones.
            self.assertIsNotNone(pipe_default._edit_masks)
            self.assertIsNotNone(pipe_default._edit_conditional_dict)


if __name__ == "__main__":
    unittest.main()
