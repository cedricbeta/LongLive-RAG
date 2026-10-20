"""No-GPU checks for the human-targeted pilot's review and one-attempt gates."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_oracle_entity_repair as pilot


class OraclePilotTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case_dir = self.root / "ns090"
        self.case_dir.mkdir()
        self.video = self.root / "baseline.mp4"
        self.video.write_bytes(b"fixture-video")
        self.packet = dict(protocol=copy.deepcopy(pilot.PROTOCOL), code_hashes={}, cases=[dict(
            id="ns090", scene="ns090_test", shot=0, video=str(self.video),
            video_sha256=pilot.file_sha256(self.video), baseline_config=str(self.root / "baseline.yaml"),
            mask_path=str(self.root / "mask.pt"), repair_prompt="Iris is a woman with curly red hair.",
            files={str(self.video): pilot.file_sha256(self.video)})])
        self.packet_path = self.root / "packet.json"
        pilot.write_json(self.packet_path, self.packet)
        self.review = dict(reviewer="human-test-fixture", reviewer_kind="human",
            packet_sha256=pilot.file_sha256(self.packet_path), cases={"ns090": dict(
                target_and_defect_confirmed=True, mask_confirmed=True)})
        self.review_path = self.root / "human_review.json"
        pilot.write_json(self.review_path, self.review)

    def validate(self):
        pilot.validate_review(self.packet, self.review, self.packet_path)

    def test_review_accepts_complete_hash_bound_confirmation(self):
        self.validate()

    def test_assistant_annotation_is_not_human_review(self):
        self.review["reviewer_kind"] = "assistant"
        with self.assertRaisesRegex(ValueError, "actual human"):
            self.validate()

    def test_explicit_machine_protocol_accepts_assistant_not_human(self):
        self.packet["protocol"] = copy.deepcopy(pilot.MACHINE_PROTOCOL)
        self.review["reviewer_kind"] = "assistant"
        self.validate()
        self.review["reviewer_kind"] = "human"
        with self.assertRaisesRegex(ValueError, "assistant-review provenance"):
            self.validate()

    def test_machine_fork_preserves_source_and_does_not_fabricate_human_review(self):
        artifact = self.case_dir / "target_masks.pt"
        artifact.write_bytes(b"fixture-mask")
        self.packet["cases"][0]["files"] = {str(artifact): pilot.file_sha256(artifact)}
        self.packet["cases"][0]["mask_path"] = str(artifact)
        self.packet_path.write_text(json.dumps(self.packet))
        before = self.packet_path.read_bytes()
        self.review.update(reviewer_kind="assistant", user_requested_machine_pilot="yes, no human review",
                           packet_sha256=pilot.file_sha256(self.packet_path))
        self.review_path.write_text(json.dumps(self.review))
        target = self.root / "machine"
        pilot.adopt_machine(self.root, target, self.review_path)
        self.assertEqual(self.packet_path.read_bytes(), before)
        packet = json.loads((target / "packet.json").read_text())
        review = json.loads((target / "assistant_review.json").read_text())
        self.assertEqual(packet["protocol"], pilot.MACHINE_PROTOCOL)
        self.assertEqual(review["reviewer_kind"], "assistant")
        self.assertFalse((target / "human_review.json").exists())
        self.assertEqual(Path(packet["cases"][0]["mask_path"]).read_bytes(), b"fixture-mask")
        pilot.validate_review(packet, review, target / "packet.json")

    def test_machine_fork_requires_explicit_user_request(self):
        self.review["reviewer_kind"] = "assistant"
        self.review_path.write_text(json.dumps(self.review))
        with self.assertRaisesRegex(ValueError, "explicitly requested"):
            pilot.adopt_machine(self.root, self.root / "machine", self.review_path)

    def test_missing_case_is_rejected(self):
        self.review["cases"] = {}
        with self.assertRaisesRegex(ValueError, "all fixed cases"):
            self.validate()

    def test_unconfirmed_mask_is_rejected(self):
        self.review["cases"]["ns090"]["mask_confirmed"] = False
        with self.assertRaisesRegex(ValueError, "not confirmed"):
            self.validate()

    def test_modified_packet_is_rejected(self):
        self.packet_path.write_text(self.packet_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "exact packet"):
            self.validate()

    def test_modified_source_is_rejected(self):
        self.video.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.validate()

    def test_modified_implementation_is_rejected(self):
        self.packet["code_hashes"] = {str(self.video): "wrong hash"}
        with self.assertRaisesRegex(ValueError, "implementation changed"):
            self.validate()

    def test_changed_protocol_is_rejected(self):
        self.packet["protocol"]["attempts_per_case"] = 2
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.validate()

    def test_json_write_never_overwrites(self):
        with self.assertRaises(FileExistsError):
            pilot.write_json(self.packet_path, {})

    def test_control_divergence_stops_before_repair(self):
        control = self.root / "control.mp4"
        control.write_bytes(b"changed-video")
        with patch("scripts.run_agent_edit_demo.build_render_config"), \
             patch("scripts.run_agent_edit_demo.render"), \
             patch("scripts.run_agent_edit_demo.find_render", return_value=control), \
             patch("scripts.audit_narrastream_baselines.pixels", side_effect=["one", "two"]), \
             patch("scripts.run_agent_edit_demo.EditRenderBackend") as backend:
            with self.assertRaisesRegex(RuntimeError, "unedited replay differs"):
                pilot.run_case(self.root, self.review_path, "ns090", 0)
            backend.assert_not_called()
        self.assertTrue((self.case_dir / "failure.json").is_file())

    def test_existing_backend_receives_exactly_one_fixed_mask_edit(self):
        with patch("scripts.run_agent_edit_demo.build_render_config"), \
             patch("scripts.run_agent_edit_demo.render") as render, \
             patch("scripts.run_agent_edit_demo.find_render", return_value=self.video), \
             patch("scripts.run_agent_edit_demo.EditRenderBackend") as backend, \
             patch("utils.agent_entity_judge.VLMLocalizer", side_effect=AssertionError("no VLM")), \
             patch("utils.agent_entity_judge.VLMComparer", side_effect=AssertionError("no judge")):
            backend.return_value.edit.return_value = (None, {"edited_video": "fixture-result"})
            pilot.run_case(self.root, self.review_path, "ns090", 2)
            render.assert_called_once()  # no-edit control; backend owns the one repair
            self.assertEqual(backend.call_args.kwargs["noise_seed"], 4242)
            args = backend.return_value.edit.call_args.args
            self.assertEqual(args[:2], (0, 1))
            self.assertEqual(args[2].params["mask_path"], str(self.root / "mask.pt"))
            self.assertEqual(args[2].params["entry_sigma"], 1.0)
            self.assertNotIn("mask_phrase", args[2].params)
            backend.return_value.edit.assert_called_once()
            backend.return_value.commit.assert_not_called()
            with self.assertRaises(FileExistsError):
                pilot.run_case(self.root, self.review_path, "ns090", 2)
            backend.return_value.edit.assert_called_once()
        result = json.loads((self.case_dir / "result.json").read_text())
        self.assertEqual(result["status"], "rendered_not_human_rated")

    def test_export_cannot_hide_missing_results(self):
        with self.assertRaises(FileNotFoundError):
            pilot.export(self.root)
        self.assertFalse((self.root / "blind_review").exists())

    def test_blind_export_separates_key_and_keeps_absolute_ratings(self):
        self.packet["cases"][0].update(entity="Iris", spec={"description": "curly red hair"},
                                      caption="Iris works.", frame_count=384)
        self.packet_path.write_text(json.dumps(self.packet))
        repaired = self.root / "edited.mp4"
        repaired.write_bytes(b"fixture-edit")
        pilot.write_json(self.case_dir / "result.json", dict(
            packet_sha256=pilot.file_sha256(self.packet_path), result={
                "edited_video": str(repaired), "edited_video_sha256": pilot.file_sha256(repaired)}))
        with patch.object(pilot, "clip") as clip, patch.object(pilot.secrets, "choice", return_value="B"):
            pilot.export(self.root)
        self.assertEqual(clip.call_count, 4)
        self.assertEqual(clip.call_args_list[0].args[0], repaired)
        self.assertEqual(clip.call_args_list[2].args[0], self.video)
        self.assertEqual(clip.call_args_list[1].args[2:], (64, 320))
        public = self.root / "blind_review"
        ratings = json.loads((public / "ratings.json").read_text())
        self.assertEqual(ratings["cases"]["item01"]["A"]["appearance"], "unrated")
        self.assertEqual(ratings["cases"]["item01"]["B"]["appearance"], "unrated")
        self.assertNotIn("ns090", (public / "index.html").read_text())
        self.assertNotIn(str(self.video), (public / "index.html").read_text())
        self.assertFalse((public / "blind_key.private.json").exists())
        key = json.loads((self.root / "blind_key.private.json").read_text())
        self.assertEqual(key["item01"], dict(case="ns090", baseline_side="B"))


if __name__ == "__main__":
    unittest.main()
