from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.dataset import MultiTextConcatDataset
except Exception as exc:  # pragma: no cover - optional dataset deps may be absent
    MultiTextConcatDataset = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _write_json(path: Path, caption: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"caption": caption}, f)


@unittest.skipIf(MultiTextConcatDataset is None, f"dataset import unavailable: {IMPORT_ERROR}")
class TestMultiTextConcatGlobalCaption(unittest.TestCase):
    def test_global_caption_is_prepended_to_each_shot(self):
        root = Path(tempfile.mkdtemp())
        scene = root / "frying_egg_same_event"
        scene.mkdir(parents=True)
        _write_json(scene / "global.json", "same intact egg in the same dark pan")
        _write_json(scene / "0.json", "macro view of the egg")
        _write_json(scene / "1.json", "man cooking that egg")
        (scene / "shot_durations.txt").write_text("1 1", encoding="utf-8")

        ds = MultiTextConcatDataset(
            str(root),
            num_blocks=2,
            scene_cut_prefix="CUT ",
            deterministic=True,
        )
        item = ds[0]

        self.assertEqual(item["sample_name"], "frying_egg_same_event")
        self.assertEqual(item["shot_durations"], [1, 1])
        self.assertIn("same intact egg", item["prompts"][0])
        self.assertIn("macro view", item["prompts"][0])
        self.assertTrue(item["prompts"][1].startswith("CUT same intact egg"))
        self.assertIn("man cooking", item["prompts"][1])

    def test_missing_global_caption_keeps_legacy_prompt_shape(self):
        root = Path(tempfile.mkdtemp())
        scene = root / "plain_scene"
        scene.mkdir(parents=True)
        _write_json(scene / "0.json", "shot zero")
        _write_json(scene / "1.json", "shot one")
        (scene / "shot_durations.txt").write_text("1 1", encoding="utf-8")

        ds = MultiTextConcatDataset(str(root), num_blocks=2, scene_cut_prefix="CUT ")
        item = ds[0]

        self.assertEqual(item["prompts"], ["shot zero", "CUT shot one"])
        self.assertNotIn("global_caption", item)


if __name__ == "__main__":
    unittest.main()
