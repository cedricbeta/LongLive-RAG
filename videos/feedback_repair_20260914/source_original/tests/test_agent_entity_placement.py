"""Placement of absent entities from caption relations / history, and relation constraints.

CPU-only: stub localiser / comparer / tracker on a tiny synthetic video (64x32,
two chunks of 8 frames, latent grid 4x8).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_agent_entity_judge import _BoxTracker, _StubComparer, _StubLocalizer, _observation, _tiny_video
from utils.agent_constraints import AssessmentStatus, ConstraintKind, entity_judge_metrics_to_assessments
from utils.agent_entity_judge import (
    EntityDriftVerifier,
    EntitySpec,
    parse_spatial_relations,
    placement_box,
    spatial_relation_holds,
)
from utils.agent_runtime import NormativeGraph

NOAH = (40, 2, 60, 30)
IRIS_LEFT = (5, 2, 25, 30)
MATCH = {"matches_reference": True, "differences": "", "target_phrase": ""}


class RelationParserTest(unittest.TestCase):
    def test_possessive_side(self):
        caption = ("At the restaurant prep counter, Nora and Ethan continue mixing ingredients. Clara, a "
                   "middle-aged woman with auburn hair and a long tan coat, joins them, standing to Nora's left.")
        rels = parse_spatial_relations(caption, ["Nora", "Ethan", "Clara"])
        self.assertEqual([(r["subject"], r["relation"], r["anchor"]) for r in rels], [("Clara", "left", "Nora")])

    def test_pronoun_anchor_and_next_to(self):
        caption = ("In the library reading room, Clara, a middle-aged woman, sits at a reading table while Miles, "
                   "a slim man, stands beside her. Iris takes a seat next to Noah at the lecture hall.")
        rels = parse_spatial_relations(caption, ["Clara", "Miles", "Iris", "Noah"])
        self.assertEqual(
            {(r["subject"], r["relation"], r["anchor"]) for r in rels},
            {("Miles", "beside", "Clara"), ("Iris", "beside", "Noah")},
        )

    def test_no_relation_or_unknown_names(self):
        self.assertEqual(parse_spatial_relations("Nora mixes the batter alone.", ["Nora", "Clara"]), [])
        self.assertEqual(parse_spatial_relations("Bob stands to Alice's left.", ["Nora"]), [])


class GeometryTest(unittest.TestCase):
    def test_placement_box_sides_and_clipping(self):
        hw = (32, 64)
        self.assertEqual(placement_box(NOAH, "left", hw), (17, 2, 37, 30))
        self.assertIsNone(placement_box(NOAH, "right", hw))  # 4 px of room < 0.08 W: rejected
        # anchor near the left edge (ns290 shape): the box is laid flush to the edge,
        # 0.6 of the anchor width, overlapping the anchor's bbox
        near_edge = (12, 2, 32, 30)
        self.assertEqual(placement_box(near_edge, "left", hw), (0, 2, 12, 30))
        self.assertEqual(placement_box(NOAH, "beside", hw), (17, 2, 37, 30))  # more room on the left
        self.assertEqual(placement_box(NOAH, "beside", hw, prefer_side="left"), (17, 2, 37, 30))
        behind = placement_box(NOAH, "behind", hw)
        self.assertLess(behind[2] - behind[0], NOAH[2] - NOAH[0])
        self.assertLess(behind[3], NOAH[3])

    def test_relation_holds(self):
        self.assertTrue(spatial_relation_holds(IRIS_LEFT, NOAH, "left"))
        self.assertFalse(spatial_relation_holds(IRIS_LEFT, NOAH, "right"))
        self.assertTrue(spatial_relation_holds((17, 2, 37, 30), NOAH, "beside"))
        self.assertFalse(spatial_relation_holds((0, 2, 10, 30), NOAH, "beside"))  # 30 px gap > anchor width
        self.assertIsNone(spatial_relation_holds(IRIS_LEFT, NOAH, "behind"))


class PlacementJudgeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.video = _tiny_video(self.root / "shot.mp4")
        scene = self.root / "scene"
        scene.mkdir()
        (scene / "global.json").write_text(json.dumps({"caption": "scene"}))
        captions = [
            "shot 0: Noah alone.",
            "Noah works at the desk. Iris enters and takes a seat to Noah's left.",
            "Noah and Iris keep working.",
        ]
        for i, caption in enumerate(captions):
            (scene / f"{i}.json").write_text(json.dumps({"caption": caption}))
        self.graph = NormativeGraph.compile_scene(scene)
        self.entities = [EntitySpec("Noah", "tall man in green shirt"), EntitySpec("Iris", "woman with red hair", shots=(1, 2))]

    def tearDown(self):
        self._tmp.cleanup()

    def _judge(self, localizer, comparer, **kwargs):
        return EntityDriftVerifier(
            self.entities, localizer=localizer, comparer=comparer, tracker=_BoxTracker(),
            work_dir=self.root / "judge", grid=(4, 8), **kwargs,
        )

    @staticmethod
    def _mask_profile(action):
        masks = torch.load(dict(action.params)["mask_path"], weights_only=False)["masks"][0]
        return torch.nn.functional.interpolate(masks[:1, None], size=(32, 64), mode="nearest")[0, 0]

    def test_absent_entity_is_placed_from_caption_relation(self):
        loc = _StubLocalizer({("Noah", 0): NOAH, ("Noah", 1): NOAH}, {})
        judge = self._judge(loc, _StubComparer({"Noah": MATCH}))
        verdict = judge.judge(_observation(self.video, 1), self.graph.view(1))

        self.assertFalse(verdict.accept)
        self.assertEqual(verdict.reason, "reject_entity_absent_place")
        placement = verdict.metrics["placement"]["Iris"]
        self.assertEqual(placement["source"], "relation")
        self.assertEqual(placement["anchor"], "Noah")
        self.assertEqual(placement["boxes"], {0: [17, 2, 37, 30], 1: [17, 2, 37, 30]})
        self.assertEqual(verdict.metrics["mask_keys"], ["place_Iris"])
        self.assertEqual(len(verdict.directed_actions), 1)
        action = verdict.directed_actions[0]
        self.assertEqual(action.reason, "entity_placement_masked_renoise_repair")
        self.assertIn("Iris is woman with red hair.", dict(action.params)["prompt"])
        up = self._mask_profile(action)
        self.assertGreater(float(up[:, 17:37].mean()), 0.5)  # placement box painted
        self.assertLess(float(up[:, 40:].mean()), 0.1)       # Noah untouched
        relation = verdict.metrics["spatial_relations"][0]
        self.assertEqual((relation["subject"], relation["relation"], relation["anchor"]), ("Iris", "left", "Noah"))
        self.assertEqual(relation["status"], "unknown")
        ids = {row["constraint"]["id"]: row for row in verdict.metrics["constraint_assessments"]}
        self.assertEqual(ids["shot:1:relation:Iris:left:Noah"]["constraint"]["kind"], "relation")
        self.assertIn("shot:1:relation:Iris:left:Noah", verdict.metrics["repair_dependencies"]["constraint_ids"])

    def test_spec_only_placement_prompt(self):
        loc = _StubLocalizer({("Noah", 0): NOAH, ("Noah", 1): NOAH}, {})
        judge = self._judge(loc, _StubComparer({"Noah": MATCH}), placement_prompt="spec_only")
        verdict = judge.judge(_observation(self.video, 1), self.graph.view(1))
        prompt = dict(verdict.directed_actions[0].params)["prompt"]
        self.assertEqual(verdict.metrics["repair_prompt_mode"], "spec_only")
        self.assertIn("Iris, woman with red hair, stands in the scene.", prompt)
        self.assertNotIn("takes a seat", prompt)  # shot caption omitted
        self.assertTrue(prompt.startswith("scene"))  # global caption kept
        self.assertEqual(verdict.metrics["unassigned_boxes"], {})

    def test_anchor_geometry_comes_from_the_tight_track_not_the_loose_box(self):
        """A loose VLM box spanning most of the frame must not hide the room beside the anchor."""
        import numpy as np

        class _TightTracker:
            def tracks(self, frames_dir, prompts, num_frames, hw):
                H, W = hw
                out = {}
                for key, per_frame in prompts.items():
                    track = np.zeros((num_frames, H, W), dtype=bool)
                    track[:, 2:30, 40:60] = True  # the person really stands at x 40..60
                    out[key] = track
                return out

        loose = (10, 2, 63, 30)  # locator box swallowing most of the frame
        loc = _StubLocalizer({("Noah", 0): loose, ("Noah", 1): loose}, {})
        judge = EntityDriftVerifier(
            self.entities, localizer=loc, comparer=_StubComparer({"Noah": MATCH}), tracker=_TightTracker(),
            work_dir=self.root / "judge", grid=(4, 8),
        )
        verdict = judge.judge(_observation(self.video, 1), self.graph.view(1))
        self.assertEqual(verdict.reason, "reject_entity_absent_place")
        self.assertEqual(verdict.metrics["placement"]["Iris"]["boxes"][0], [17, 2, 37, 30])

    def test_policy_never_keeps_nothing_to_mask(self):
        loc = _StubLocalizer({("Noah", 0): NOAH, ("Noah", 1): NOAH}, {})
        judge = self._judge(loc, _StubComparer({"Noah": MATCH}), place_absent="never")
        verdict = judge.judge(_observation(self.video, 1), self.graph.view(1))
        self.assertEqual(verdict.reason, "reject_entity_absent")
        self.assertEqual(verdict.directed_actions, ())
        self.assertEqual(verdict.metrics["placement"], {})

    def test_history_placement_when_caption_names_no_relation(self):
        loc = _StubLocalizer(
            {("Noah", 0): NOAH, ("Noah", 1): NOAH, ("Iris", 0): IRIS_LEFT, ("Iris", 1): IRIS_LEFT}, {}
        )
        judge = self._judge(loc, _StubComparer({"Noah": MATCH, "Iris": MATCH}))
        accepted = judge.judge(_observation(self.video, 1), self.graph.view(1))
        self.assertTrue(accepted.accept)
        self.assertEqual(accepted.metrics["spatial_relations"][0]["status"], "satisfied")
        self.assertEqual(judge.last_tracks["Iris"]["boxes"], {0: list(IRIS_LEFT), 1: list(IRIS_LEFT)})

        snapshot = judge.snapshot_state()
        judge.localizer = _StubLocalizer({("Noah", 0): NOAH, ("Noah", 1): NOAH}, {})
        verdict = judge.judge(_observation(self.video, 2), self.graph.view(2))
        self.assertEqual(verdict.reason, "reject_entity_absent_place")
        placement = verdict.metrics["placement"]["Iris"]
        self.assertEqual(placement["source"], "history")
        self.assertEqual(placement["history_shot"], 1)
        self.assertEqual(placement["boxes"][0], list(IRIS_LEFT))
        self.assertEqual(verdict.metrics["spatial_relations"], [])  # shot 2 caption has no relation

        judge.restore_state({**snapshot, "last_tracks": {}})
        self.assertEqual(judge.last_tracks, {})

    def test_relation_violation_is_recorded_but_does_not_gate_acceptance(self):
        iris_right = (42, 2, 62, 30)
        noah_left = (5, 2, 25, 30)
        loc = _StubLocalizer(
            {("Noah", 0): noah_left, ("Noah", 1): noah_left, ("Iris", 0): iris_right, ("Iris", 1): iris_right}, {}
        )
        judge = self._judge(loc, _StubComparer({"Noah": MATCH, "Iris": MATCH}))
        verdict = judge.judge(_observation(self.video, 1), self.graph.view(1))
        self.assertTrue(verdict.accept)
        relation = verdict.metrics["spatial_relations"][0]
        self.assertEqual(relation["status"], "violated")
        self.assertEqual(relation["score"], 0.0)
        by_id = {item.id: item for item in entity_judge_metrics_to_assessments(verdict.metrics, shot_index=1)}
        item = by_id["shot:1:relation:Iris:left:Noah"]
        self.assertEqual(item.constraint.kind, ConstraintKind.RELATION)
        self.assertEqual(item.status, AssessmentStatus.VIOLATED)
        self.assertEqual(item.violations[0].code, "spatial_relation_violated")


if __name__ == "__main__":
    unittest.main()


class _StubRewriter:
    def __init__(self):
        self.calls = []

    def rewrite(self, **kwargs):
        self.calls.append(kwargs)
        return {"prompt": f"Iris, woman with red hair, stands to Noah's left. v{kwargs['attempt']}",
                "source": "stub", "call_id": "x", "rejected": []}


class RegenerateJudgeTest(PlacementJudgeTest):
    """absent_action=regenerate: whole-shot resample with a rewritten caption instead of a box."""

    def _regenerate_judge(self, loc, comparer, rewriter):
        return self._judge(loc, comparer, absent_action="regenerate", prompt_rewriter=rewriter)

    def test_absent_entity_directs_full_frame_regeneration(self):
        loc = _StubLocalizer({("Noah", 0): NOAH, ("Noah", 1): NOAH}, {})
        rewriter = _StubRewriter()
        judge = self._regenerate_judge(loc, _StubComparer({"Noah": MATCH}), rewriter)
        verdict = judge.judge(_observation(self.video, 1), self.graph.view(1))

        self.assertFalse(verdict.accept)
        self.assertEqual(verdict.reason, "reject_entity_absent_regenerate")
        action = verdict.directed_actions[0]
        params = dict(action.params)
        self.assertEqual(action.reason, "entity_absent_prompt_regenerate")
        self.assertEqual(params["prompt"], "Iris, woman with red hair, stands to Noah's left. v0")
        self.assertEqual(params["entry_sigma"], 1.0)
        masks = torch.load(params["mask_path"], weights_only=False)["masks"]
        self.assertEqual(sorted(masks), [0, 1])
        for mask in masks.values():
            self.assertEqual(tuple(mask.shape), (8, 4, 8))
            self.assertEqual(float(mask.min()), 1.0)
        regen = verdict.metrics["regenerate"]
        self.assertEqual(regen["absent"], ["Iris"])
        self.assertEqual(regen["prompt_source"], "stub")
        self.assertEqual(regen["previous_rewrites"], [])
        self.assertEqual(verdict.metrics["mask_keys"], ["regenerate_shot"])
        self.assertEqual(verdict.metrics["mask_coverage"], 1.0)
        self.assertEqual(verdict.metrics["placement"], {})
        self.assertEqual(verdict.metrics["repair_dependencies"]["entity_ids"], ["Iris", "Noah"])
        call = rewriter.calls[0]
        self.assertEqual(call["absent"], [("Iris", "woman with red hair")])
        self.assertEqual(call["present"], [("Noah", "tall man in green shirt")])
        self.assertEqual([(r["subject"], r["relation"], r["anchor"]) for r in call["relations"]], [("Iris", "left", "Noah")])
        self.assertEqual(call["caption"], "Noah works at the desk. Iris enters and takes a seat to Noah's left.")

        # the next round sees what was already tried
        second = judge.judge(_observation(self.video, 1, attempt=1), self.graph.view(1))
        self.assertEqual(rewriter.calls[1]["previous"], ["Iris, woman with red hair, stands to Noah's left. v0"])
        self.assertEqual(second.metrics["regenerate"]["previous_rewrites"], ["Iris, woman with red hair, stands to Noah's left. v0"])

    def test_regeneration_takes_precedence_over_impostor_and_drift_repaints(self):
        impostor = (5, 2, 25, 30)
        loc = _StubLocalizer({("Noah", 0): NOAH, ("Noah", 1): NOAH}, {0: [impostor], 1: [impostor]})
        drift = {"matches_reference": False, "differences": "shirt is blue", "target_phrase": ""}
        judge = self._regenerate_judge(loc, _StubComparer({"Noah": drift}), _StubRewriter())
        verdict = judge.judge(_observation(self.video, 1), self.graph.view(1))
        self.assertEqual(verdict.reason, "reject_entity_absent_regenerate")
        self.assertEqual(verdict.metrics["regenerate"]["also_drifted"], ["Noah"])
        self.assertEqual(verdict.metrics["regenerate"]["impostors"], ["unassigned_0"])
        self.assertEqual(verdict.directed_actions[0].reason, "entity_absent_prompt_regenerate")

    def test_without_rewriter_the_template_is_used(self):
        loc = _StubLocalizer({("Noah", 0): NOAH, ("Noah", 1): NOAH}, {})
        judge = self._judge(loc, _StubComparer({"Noah": MATCH}), absent_action="regenerate")
        verdict = judge.judge(_observation(self.video, 1), self.graph.view(1))
        prompt = dict(verdict.directed_actions[0].params)["prompt"]
        self.assertTrue(prompt.startswith("Iris, woman with red hair, is clearly visible in the frame, to Noah's left."))
        self.assertEqual(verdict.metrics["regenerate"]["prompt_source"], "template")

    def test_present_entities_keep_the_old_paths(self):
        loc = _StubLocalizer({("Noah", 0): NOAH, ("Noah", 1): NOAH, ("Iris", 0): IRIS_LEFT, ("Iris", 1): IRIS_LEFT}, {})
        judge = self._regenerate_judge(loc, _StubComparer({"Noah": MATCH, "Iris": MATCH}), _StubRewriter())
        verdict = judge.judge(_observation(self.video, 1), self.graph.view(1))
        self.assertTrue(verdict.accept)
        self.assertNotIn("regenerate", verdict.metrics)
