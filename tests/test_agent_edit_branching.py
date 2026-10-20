"""CPU-only tests for cumulative edit plans and render-branch rollback."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from utils.edit_renoise_plans import normalize_edit_renoise_plans
from utils.agent_runtime import Action, Verdict

ROOT = Path(__file__).resolve().parents[1]
_DEMO_SPEC = importlib.util.spec_from_file_location(
    "longlive_run_agent_edit_demo", ROOT / "scripts" / "run_agent_edit_demo.py"
)
assert _DEMO_SPEC is not None and _DEMO_SPEC.loader is not None
DEMO = importlib.util.module_from_spec(_DEMO_SPEC)
sys.modules[_DEMO_SPEC.name] = DEMO
_DEMO_SPEC.loader.exec_module(DEMO)
EditRenderBackend = DEMO.EditRenderBackend
build_render_config = DEMO.build_render_config
prefix_identity_report = DEMO.prefix_identity_report


class PrefixIdentityTest(unittest.TestCase):
    """Latent dumps decide prefix identity exactly; chunks after the edit may differ."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        torch.manual_seed(0)
        self.parent = torch.randn(1, 12, 4, 3, 3)  # 4 chunks x 3 latent frames
        self.parent_path = self.root / "parent.pt"
        torch.save(self.parent, self.parent_path)

    def tearDown(self):
        self._tmp.cleanup()

    def _report(self, child, chunk_end):
        child_path = self.root / "child.pt"
        torch.save(child, child_path)
        return prefix_identity_report(
            self.root / "parent.mp4",
            self.root / "child.mp4",
            chunk_end,
            num_chunks=4,
            parent_latents=self.parent_path,
            child_latents=child_path,
        )

    def test_edit_after_prefix_is_identical(self):
        child = self.parent.clone()
        child[:, 6:] += 1.0  # chunks 2-3 edited
        report = self._report(child, chunk_end=2)
        self.assertEqual(report["basis"], "latent")
        self.assertTrue(report["identical"])
        self.assertEqual(report["latent_frames"], 6)

    def test_prefix_divergence_is_detected(self):
        child = self.parent.clone()
        child[:, 4] += 1e-3  # inside chunk 1, before the edit at chunk 2
        report = self._report(child, chunk_end=2)
        self.assertFalse(report["identical"])
        self.assertGreater(report["max_abs_diff"], 0.0)

    def test_digests_replace_raw_dump_and_compare_exactly(self):
        render_dir = self.root / "renders"
        for name, tensor in (("parent", self.parent), ("child", self.parent.clone())):
            folder = DEMO.latent_dir_for(render_dir, name)
            folder.mkdir(parents=True)
            torch.save(tensor.to(torch.bfloat16), folder / "latent_0000.pt")
        child_folder = DEMO.latent_dir_for(render_dir, "child")
        edited = torch.load(child_folder / "latent_0000.pt")
        edited[:, 9:] += 1.0  # chunk 3 edited
        torch.save(edited, child_folder / "latent_0000.pt")

        parent_digests = DEMO.condense_latent_dump(render_dir, "parent", 4)
        child_digests = DEMO.condense_latent_dump(render_dir, "child", 4)
        self.assertFalse((DEMO.latent_dir_for(render_dir, "parent") / "latent_0000.pt").exists())
        self.assertEqual(DEMO.find_latents(render_dir, "parent"), parent_digests)

        report = prefix_identity_report(
            self.root / "p.mp4", self.root / "c.mp4", 3, num_chunks=4,
            parent_latents=parent_digests, child_latents=child_digests,
        )
        self.assertEqual(report["basis"], "latent_digests")
        self.assertTrue(report["identical"])
        whole = prefix_identity_report(
            self.root / "p.mp4", self.root / "c.mp4", 4, num_chunks=4,
            parent_latents=parent_digests, child_latents=child_digests,
        )
        self.assertEqual(whole["mismatched_chunks"], [3])
        self.assertFalse(whole["identical"])

    def test_shot_zero_edit_has_no_prefix(self):
        report = self._report(self.parent.clone(), chunk_end=0)
        self.assertEqual(report["basis"], "none")
        self.assertTrue(report["identical"])


class EditPlanConfigTest(unittest.TestCase):
    def test_multiple_plans_are_sorted_and_overlap_is_rejected(self):
        plans = normalize_edit_renoise_plans(
            [
                {"enabled": True, "chunk_start": 4, "chunk_end": 5},
                {"enabled": True, "chunk_start": 1, "chunk_end": 2},
            ]
        )
        self.assertEqual(
            [(plan["chunk_start"], plan["chunk_end"]) for plan in plans],
            [(1, 2), (4, 5)],
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            normalize_edit_renoise_plans(
                [
                    {"enabled": True, "chunk_start": 1, "chunk_end": 3},
                    {"enabled": True, "chunk_start": 3, "chunk_end": 4},
                ]
            )

    def test_render_config_uses_plan_list_and_clears_legacy_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.yaml"
            base.write_text(
                yaml.safe_dump(
                    {
                        "data": {},
                        "edit_renoise": {
                            "enabled": True,
                            "chunk_start": 98,
                            "chunk_end": 98,
                        },
                        "inference": {
                            "edit_renoise": {
                                "enabled": True,
                                "chunk_start": 99,
                                "chunk_end": 99,
                            }
                        },
                        "logging": {"seed": 0},
                    }
                ),
                encoding="utf-8",
            )
            target = root / "candidate.yaml"
            build_render_config(
                base,
                scene="scene",
                data_dir=root / "data",
                output_folder=root / "renders",
                prefix="candidate",
                edit=None,
                edit_plans=(
                    {"chunk_start": 2, "chunk_end": 2, "prompt": "second"},
                    {"chunk_start": 0, "chunk_end": 0, "prompt": "first"},
                ),
                target=target,
            )
            inference = yaml.safe_load(target.read_text(encoding="utf-8"))["inference"]
            rendered_config = yaml.safe_load(target.read_text(encoding="utf-8"))
            self.assertNotIn("edit_renoise", rendered_config)
            self.assertNotIn("edit_renoise_plans", rendered_config)
            self.assertNotIn("edit_renoise", inference)
            self.assertEqual(
                [plan["chunk_start"] for plan in inference["edit_renoise_plans"]],
                [0, 2],
            )
            self.assertTrue(all(plan["enabled"] for plan in inference["edit_renoise_plans"]))


class EditRenderBranchTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.scene_dir = self.root / "scene"
        self.scene_dir.mkdir()
        (self.scene_dir / "shot_durations.txt").write_text("1 1 1\n", encoding="utf-8")
        self.out_dir = self.root / "out"
        self.out_dir.mkdir()
        self.base_config = self.root / "base.yaml"
        self.base_config.write_text(
            yaml.safe_dump({"data": {}, "inference": {}, "logging": {"seed": 0}}),
            encoding="utf-8",
        )
        self.baseline = self.root / "baseline.mp4"
        self.baseline.write_bytes(b"baseline")
        self.mask_path = self.root / "masks.pt"
        torch.save(
            {"masks": {chunk: torch.ones(2, 4, 5) for chunk in range(3)}},
            self.mask_path,
        )
        self.backend = EditRenderBackend(
            scene="scene",
            scene_dir=self.scene_dir,
            out_dir=self.out_dir,
            base_config=self.base_config,
            baseline_video=self.baseline,
            render_gpu=0,
            clipseg_device="cpu",
        )
        self._patches = [
            patch.object(DEMO, "render", return_value=None),
            patch.object(DEMO, "find_render", side_effect=self._fake_render),
            patch.object(DEMO, "extract_frames", side_effect=self._fake_frames),
            patch.object(
                DEMO,
                "prefix_identity_report",
                return_value={"basis": "stub", "identical": True},
            ),
        ]
        for active_patch in self._patches:
            active_patch.start()

    def tearDown(self):
        for active_patch in reversed(self._patches):
            active_patch.stop()
        self._tmp.cleanup()

    def _fake_render(self, output_folder, prefix, scene):
        path = Path(output_folder) / f"{prefix}-rank0-{scene}_regular.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(prefix.encode("utf-8"))
        return path

    def _fake_frames(self, video, frame_indices, out_dir, stem):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for local, frame_index in enumerate(frame_indices):
            path = out / f"{stem}_{local}_{frame_index}.png"
            path.write_bytes(f"{video}:{frame_index}".encode("utf-8"))
            paths.append(path)
        return paths

    def _action(self, shot_index):
        return Action(
            "edit_shot",
            {
                "shot": shot_index,
                "prompt": f"repair shot {shot_index}",
                "mask_path": str(self.mask_path),
                "entry_sigma": 1.0,
            },
            "identity repair",
        )

    def _edit(self, shot_index, attempt=1):
        return self.backend.edit(
            shot_index,
            attempt,
            self._action(shot_index),
            Verdict(False, "drift"),
        )

    def test_consecutive_accepted_edits_accumulate_parent_plans(self):
        first, first_result = self._edit(1)
        first_id = first.model_native_candidate["branch_id"]
        self.assertEqual(first_result["edit_plan_count"], 1)
        self.assertEqual(first.model_native_candidate["parent"], "branch-0000")
        self.assertTrue(first.telemetry["tentative"])
        self.assertNotIn("tentative", first.model_native_candidate)
        self.assertNotIn("safe", first.model_native_candidate)
        self.assertNotIn("verified", first.model_native_candidate)
        self.backend.commit(first, quarantined=False)

        second, second_result = self._edit(2)
        second_id = second.model_native_candidate["branch_id"]
        config = yaml.safe_load(Path(second_result["config"]).read_text(encoding="utf-8"))
        configured = config["inference"]["edit_renoise_plans"]
        self.assertEqual(
            [(plan["chunk_start"], plan["chunk_end"]) for plan in configured],
            [(1, 1), (2, 2)],
        )
        self.assertEqual(second.model_native_candidate["parent"], first_id)
        self.backend.commit(second, quarantined=False)

        self.assertEqual(self.backend.active_branch_id, second_id)
        self.assertEqual(len(self.backend.active_branch.edit_plans), 2)
        self.assertTrue(self.backend.active_branch.safe)
        self.assertFalse(self.backend.active_branch.tentative)
        self.assertTrue(self.backend.active_branch.verified)

    def test_rejected_candidate_and_snapshot_restore_do_not_pollute_parent(self):
        first, _ = self._edit(1)
        self.backend.commit(first, quarantined=False)
        parent_id = self.backend.active_branch_id
        parent_before = self.backend.active_branch.to_dict()
        snapshot = self.backend.snapshot(2)

        rejected, _ = self._edit(2)
        rejected_id = rejected.model_native_candidate["branch_id"]
        self.assertEqual(self.backend.active_branch_id, parent_id)
        self.backend.restore(snapshot)
        self.assertNotIn((2, 1), self.backend._pending_edit_branch)
        self.backend.commit(rejected, quarantined=True)

        self.assertEqual(self.backend.active_branch_id, parent_id)
        parent_after = self.backend.active_branch.to_dict()
        self.assertEqual(parent_after["edit_plans"], parent_before["edit_plans"])
        self.assertEqual(parent_after["last_accepted_shot"], 1)
        self.assertEqual(parent_after["delivered_through_shot"], 2)
        self.assertEqual(parent_after["quarantined_shots"], [2])
        self.assertTrue(parent_after["safe"])
        self.assertTrue(parent_after["verified"])
        rejected_branch = self.backend.branches[rejected_id]
        self.assertTrue(rejected_branch.quarantined)
        self.assertFalse(rejected_branch.safe)
        self.assertFalse(rejected_branch.tentative)
        self.assertEqual(len(self.backend.active_branch.edit_plans), 1)

    def test_explicit_rollback_restores_safe_ancestor(self):
        first, _ = self._edit(1)
        self.backend.commit(first, quarantined=False)
        first_id = self.backend.active_branch_id
        first_state = self.backend.active_branch.to_dict()
        second, _ = self._edit(2)
        self.backend.commit(second, quarantined=False)
        second_id = self.backend.active_branch_id

        result = self.backend.rollback_to(first_id, reason="lookahead_failed")

        self.assertEqual(result["invalidated_branches"], [second_id])
        self.assertEqual(self.backend.active_branch_id, first_id)
        self.assertEqual(self.backend.active_branch.to_dict(), first_state)
        self.assertTrue(self.backend.branches[second_id].quarantined)
        self.assertFalse(self.backend.branches[second_id].safe)
        self.assertEqual(len(self.backend.active_branch.edit_plans), 1)

    def test_prefix_divergence_retries_the_render_once(self):
        reports = iter([{"basis": "stub", "identical": False}, {"basis": "stub", "identical": True}])
        with patch.object(DEMO, "prefix_identity_report", side_effect=lambda *a, **k: next(reports)), \
             patch.object(DEMO, "render", return_value=None) as render:
            candidate, result = self._edit(1)
        self.assertEqual(render.call_count, 2)
        self.assertTrue(result["prefix_check"]["identical"])
        self.assertEqual(result["prefix_check"]["render_attempt"], 2)
        self.assertEqual(len(result["prefix_check"]["earlier_attempts"]), 1)

    def test_prefix_divergence_twice_is_fatal(self):
        with patch.object(DEMO, "prefix_identity_report", return_value={"basis": "stub", "identical": False}), \
             patch.object(DEMO, "render", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "2 render attempts"):
                self._edit(1)

    def test_preview_reads_tentative_video_without_mutating_active_branch(self):
        snapshot = self.backend.snapshot(1)
        candidate, _ = self._edit(1)

        future = self.backend.preview(candidate, 2)
        parent_future = self.backend.preview(self.backend.generate(1, 0), 2)

        self.assertTrue(future.telemetry["edited"])
        self.assertFalse(parent_future.telemetry["edited"])
        self.assertEqual(future.output["video"], candidate.output["video"])
        self.assertEqual(
            future.model_native_candidate["branch_id"],
            candidate.model_native_candidate["branch_id"],
        )
        self.assertEqual(self.backend.snapshot(1), snapshot)


if __name__ == "__main__":
    unittest.main()
