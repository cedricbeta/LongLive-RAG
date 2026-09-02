"""Multi-round edit_shot rounds, best-of-N delivery, and the entity judge's decision logic.

No GPU, no VLM, no SAM2: the judge runs with stub localiser / comparer / tracker
objects on a tiny synthetic video.

Run: python -m unittest tests.test_agent_entity_judge -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.test_agent_edit_shot import _Backend, _Controller, _ScriptedVerifier, _config, _edit_action, _scene
from utils.agent_entity_judge import EntityDriftVerifier, EntitySpec, box_iou, load_entity_specs
from utils.agent_runtime import Action, CausalShotSession, NormativeGraph, ShotObservation, Verdict


class _ScoredVerifier(_ScriptedVerifier):
    """Scripted verdicts carrying repair_score so best-of-N can be exercised."""

    def __init__(self, script, scores):
        super().__init__(script)
        self.scores = list(scores)

    def judge(self, observation, normative):
        verdict = super().judge(observation, normative)
        score = self.scores[min(self.calls - 1, len(self.scores) - 1)]
        return Verdict(
            accept=verdict.accept,
            reason=verdict.reason,
            metrics={**dict(verdict.metrics), "repair_score": score},
            directed_actions=verdict.directed_actions,
            input_fingerprint=verdict.input_fingerprint,
        )


def _session(root, backend, verifier, **overrides):
    return CausalShotSession(
        _config(root, **overrides), _scene(root), backend, _Controller(), verifier,
        ledger_path=root / "ledger.jsonl", acquire_lock=False,
    )


class MultiRoundEditTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_second_round_repair_commits(self):
        backend = _Backend()
        verifier = _ScriptedVerifier(["reject_edit", "reject_edit", "accept"])
        result = _session(self.root, backend, verifier, max_edits_per_shot=3).run_shot(0)
        self.assertTrue(result.accepted)
        self.assertEqual([call[1] for call in backend.edit_calls], [1, 2])
        self.assertEqual(backend.restores, 2)
        self.assertEqual(backend.commits[-1]["output"]["edited"], True)
        self.assertEqual(backend.generations, [(0, 0)])  # no blind reseed was needed

    def test_budget_exhausted_then_retry_then_best_of_n(self):
        backend = _Backend()
        # attempt0 gen (score .2) -> edit1 (.6) -> edit2 (.4) -> retry gen (.1): all rejected
        verifier = _ScoredVerifier(["reject_edit", "reject_edit", "reject_edit", "reject_plain"], [0.2, 0.6, 0.4, 0.1])
        session = _session(self.root, backend, verifier, max_edits_per_shot=2)
        result = session.run_shot(0)
        self.assertFalse(result.accepted)
        self.assertTrue(result.quarantined)
        self.assertEqual(len(backend.edit_calls), 2)
        self.assertEqual(backend.generations, [(0, 0), (0, 3)])
        delivered = backend.commits[-1]
        self.assertTrue(delivered["quarantined"])
        self.assertTrue(delivered["output"]["edited"])  # attempt 1 (score .6) delivered
        rows = [json.loads(line) for line in (self.root / "ledger.jsonl").read_text().splitlines()]
        commit = [r for r in rows if r["event"] == "commit"][-1]
        self.assertEqual(commit["attempt"], 1)
        self.assertEqual(commit["best_of"]["attempts"], [0, 1, 2, 3])
        self.assertEqual(session.memory.partition_counts(), {"candidate": 0, "persistent": 0, "quarantine": 4})

    def test_ties_keep_latest_candidate(self):
        backend = _Backend()
        verifier = _ScriptedVerifier(["reject_edit", "reject_edit", "reject_plain"])
        _session(self.root, backend, verifier, max_edits_per_shot=1).run_shot(0)
        self.assertFalse(backend.commits[-1]["output"]["edited"])  # legacy: last (retry) candidate

    def test_verifier_can_stop_directing_edits_early(self):
        backend = _Backend()
        verifier = _ScriptedVerifier(["reject_edit", "reject_plain", "reject_plain"])
        _session(self.root, backend, verifier, max_edits_per_shot=3).run_shot(0)
        self.assertEqual(len(backend.edit_calls), 1)
        self.assertEqual(backend.generations, [(0, 0), (0, 2)])

    def test_mask_path_action(self):
        act = Action("edit_shot", {"shot": 0, "prompt": "p", "mask_path": "/x/m.pt"}, "r").normalized(shot_index=0)
        self.assertEqual(dict(act.params), {"shot": 0, "prompt": "p", "entry_sigma": 1.0, "mask_path": "/x/m.pt"})
        self.assertNotIn("mask_path", dict(_edit_action(0).normalized(shot_index=0).params))
        with self.assertRaisesRegex(ValueError, "mask_phrase or mask_path"):
            Action("edit_shot", {"shot": 0, "prompt": "p"}, "r").normalized(shot_index=0)


# --------------------------------------------------------------------------
# Entity judge decision logic with stubs


class _StubLocalizer:
    """boxes[(entity_name, chunk)] -> box or None; subjects[chunk] -> list of boxes."""

    def __init__(self, boxes, subjects):
        self.boxes = boxes
        self.subjects = subjects
        self.calls = []

    def locate(self, entity, reference_crop, frame, *, shot_index, inputs):
        chunk = int(Path(frame).stem[len("chunk"):])
        self.calls.append((entity.name, chunk, reference_crop is not None))
        return self.boxes.get((entity.name, chunk))

    def enumerate_subjects(self, frame, *, shot_index, inputs):
        chunk = int(Path(frame).stem[len("chunk"):])
        return list(self.subjects.get(chunk, []))


class _StubComparer:
    def __init__(self, results):
        self.results = results

    def compare(self, entity, reference_crop, crops, *, shot_index, inputs):
        return dict(self.results[entity.name])


class _BoxTracker:
    """Fills each prompted box for the whole shot (no real propagation)."""

    def tracks(self, frames_dir, prompts, num_frames, hw):
        H, W = hw
        out = {}
        for key, per_frame in prompts.items():
            track = np.zeros((num_frames, H, W), dtype=bool)
            for _, (x1, y1, x2, y2) in per_frame.items():
                track[:, y1:y2, x1:x2] = True
            out[key] = track
        return out


def _tiny_video(path: Path, frames: int = 16, size=(64, 32)) -> Path:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, size)
    for i in range(frames):
        img = np.full((size[1], size[0], 3), 40 + i, dtype=np.uint8)
        writer.write(img)
    writer.release()
    return path


def _observation(video: Path, shot: int, attempt: int = 0):
    return ShotObservation(
        shot_index=shot, attempt=attempt, frames=[], crops=[],
        telemetry={"shot_video": str(video), "chunk_start": 0, "chunk_end": 1},
        inputs=(), input_fingerprint="fp",
    )


class EntityJudgeLogicTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.video = _tiny_video(self.root / "shot.mp4")
        scene = self.root / "scene"
        scene.mkdir()
        (scene / "global.json").write_text(json.dumps({"caption": "scene"}))
        for i in range(2):
            (scene / f"{i}.json").write_text(json.dumps({"caption": f"shot {i}: Noah works; Iris enters" if i else "shot 0: Noah alone"}))
        self.graph = NormativeGraph.compile_scene(scene)
        self.entities = [EntitySpec("Noah", "tall man in green shirt"), EntitySpec("Iris", "woman with red hair", shots=(1,))]

    def tearDown(self):
        self._tmp.cleanup()

    def _judge(self, localizer, comparer):
        return EntityDriftVerifier(self.entities, localizer=localizer, comparer=comparer, tracker=_BoxTracker(), work_dir=self.root / "judge", grid=(4, 8))

    def test_all_match_accepts_and_stores_reference(self):
        loc = _StubLocalizer({("Noah", 0): (2, 2, 30, 30), ("Noah", 1): (2, 2, 30, 30)}, {0: [(2, 2, 30, 30)]})
        judge = self._judge(loc, _StubComparer({"Noah": {"matches_reference": True, "differences": "", "target_phrase": ""}}))
        verdict = judge.judge(_observation(self.video, 0), self.graph.view(0))
        self.assertTrue(verdict.accept)
        self.assertIn("Noah", judge.reference_crops)
        self.assertEqual(verdict.metrics["repair_score"], 1.0)
        # Iris is not expected in shot 0: never located
        self.assertFalse(any(name == "Iris" for name, _, _ in loc.calls))

    def test_drift_directs_edit_with_track_mask_and_positive_prompt(self):
        loc = _StubLocalizer({("Noah", 0): (2, 2, 30, 30), ("Noah", 1): (2, 2, 30, 30)}, {})
        judge = self._judge(loc, _StubComparer({"Noah": {"matches_reference": False, "differences": "logo on shirt", "target_phrase": "plain dark green work shirt"}}))
        verdict = judge.judge(_observation(self.video, 0), self.graph.view(0))
        self.assertFalse(verdict.accept)
        self.assertEqual(verdict.reason, "reject_entity_drift")
        action = verdict.directed_actions[0]
        params = dict(action.params)
        self.assertIn("mask_path", params)
        self.assertIn("Noah is tall man in green shirt.", params["prompt"])
        self.assertIn("plain dark green work shirt", params["prompt"])
        self.assertNotIn("logo", params["prompt"])  # drift narration never enters the prompt
        payload = torch.load(params["mask_path"], weights_only=False)
        self.assertEqual(sorted(payload["masks"]), [0, 1])
        self.assertGreater(float((payload["masks"][0] >= 0.35).float().mean()), 0.1)
        self.assertEqual(verdict.metrics["repair_score"], 0.0)

    def test_extra_subject_with_all_entities_matching_is_not_drift(self):
        # a prominent bookend enumerated as a subject must not trigger a repaint
        loc = _StubLocalizer({("Noah", 0): (2, 2, 30, 30), ("Noah", 1): (2, 2, 30, 30)}, {0: [(2, 2, 30, 30), (40, 10, 60, 30)], 1: [(40, 10, 60, 30)]})
        judge = self._judge(loc, _StubComparer({"Noah": {"matches_reference": True, "differences": "", "target_phrase": ""}}))
        verdict = judge.judge(_observation(self.video, 0), self.graph.view(0))
        self.assertTrue(verdict.accept)
        self.assertEqual(verdict.metrics["unassigned_subjects"], [])

    def test_absent_entity_without_impostor_has_no_edit_plan(self):
        loc = _StubLocalizer({}, {})
        judge = self._judge(loc, _StubComparer({}))
        verdict = judge.judge(_observation(self.video, 0), self.graph.view(0))
        self.assertFalse(verdict.accept)
        self.assertEqual(verdict.reason, "reject_entity_absent")
        self.assertEqual(verdict.directed_actions, ())

    def test_impostor_in_absent_entitys_place_is_masked(self):
        # Noah absent, an unassigned subject present -> mask the impostor, prompt names Noah
        loc = _StubLocalizer({("Iris", 0): (40, 2, 60, 30), ("Iris", 1): (40, 2, 60, 30)}, {0: [(2, 2, 30, 30), (40, 2, 60, 30)], 1: [(3, 2, 31, 30)]})
        judge = self._judge(loc, _StubComparer({"Iris": {"matches_reference": True, "differences": "", "target_phrase": ""}}))
        verdict = judge.judge(_observation(self.video, 1), self.graph.view(1))
        self.assertFalse(verdict.accept)
        self.assertEqual(verdict.metrics["unassigned_subjects"], ["unassigned_0"])
        self.assertEqual(verdict.metrics["mask_keys"], ["unassigned_0"])
        params = dict(verdict.directed_actions[0].params)
        self.assertIn("Noah is tall man in green shirt.", params["prompt"])
        self.assertNotIn("Iris is", params["prompt"])  # matching entity is left alone
        masks = torch.load(params["mask_path"], weights_only=False)["masks"][0]
        up = torch.nn.functional.interpolate(masks[:1, None], size=(32, 64), mode="nearest")[0, 0]
        self.assertGreater(float(up[:, :32].mean()), 0.5)   # impostor region masked
        self.assertLess(float(up[:, 40:].mean()), 0.1)      # Iris region untouched
        self.assertEqual(verdict.metrics["repair_score"], 0.0)  # (1 good - 1 unassigned) / 2

    def test_specs_and_iou(self):
        (self.root / "e.json").write_text(json.dumps({"s": [{"name": "Fish", "description": "red snapper"}, {"name": "Kai", "description": "older man", "shots": [1, 2]}]}))
        specs = load_entity_specs(self.root / "e.json", "s")
        self.assertTrue(specs[0].expected_in(7))
        self.assertFalse(specs[1].expected_in(0))
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)


if __name__ == "__main__":
    unittest.main()
