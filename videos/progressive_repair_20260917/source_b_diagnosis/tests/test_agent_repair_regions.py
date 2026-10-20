from pathlib import Path

import pytest
import torch
from PIL import Image

from tests.test_agent_entity_judge import _tiny_video, _StubLocalizer, _StubComparer, _BoxTracker, _observation
from utils.agent_edit import save_mask_payload
from utils.agent_repair_regions import attribute_region_variants, mask_content_sha256, region_preview
from utils.agent_runtime import file_sha256, NormativeGraph
from utils.agent_entity_judge import EntityDriftVerifier, EntitySpec


def test_alternative_regions_cover_missed_parts_and_protect_face_and_neighbour():
    person = torch.zeros(2, 8, 12)
    person[:, 1:7, 1:10] = 1
    base = torch.zeros_like(person)
    base[:, 2, 3] = 1
    part = base.clone()
    part[:, 2, 7] = 0.25  # a disconnected, lower-confidence strand
    protected = torch.zeros_like(base)
    protected[:, 3:5, 2:5] = 1  # face
    protected[:, :, 8:] = 1  # another person overlaps the track
    part[:, 2, 9] = 1
    options = attribute_region_variants(base, part, person, protected)
    assert options["expanded"][:, 2, 4].all()
    assert options["sensitive"][:, 2, 7].all()
    assert not options["detected"][:, 2, 7].any()
    for value in options.values():
        assert not value[protected.bool()].any()
        assert not value[person == 0].any()
    assert len({mask_content_sha256({0: value}) for value in options.values()}) == 3


def test_preview_uses_absolute_source_frames_and_rejects_wrong_parent(tmp_path):
    video = _tiny_video(tmp_path / "parent.mp4")
    masks = {1: torch.zeros(8, 4, 8)}
    masks[1][:, 0, 2] = 1
    mask = save_mask_payload(masks, tmp_path / "mask.pt", phrase="hair", video=video, grid=(4, 8))
    option = {"action": {"params": {"mask_path": str(mask)}}, "mask_sha256": file_sha256(mask)}
    preview = region_preview(option, video=str(video), output_dir=tmp_path / "previews")
    assert preview["frame_indices"] == [8]
    assert preview["mask_content_sha256"] == mask_content_sha256(masks)
    with Image.open(preview["path"]) as image:
        assert image.size == (1000, 290)
    with pytest.raises(ValueError, match="same parent video"):
        region_preview(option, video=str(tmp_path / "other.mp4"), output_dir=tmp_path / "previews")
    mask.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="mask changed"):
        region_preview(option, video=str(video), output_dir=tmp_path / "previews")


def test_judge_exports_distinct_parent_regions_with_previews(tmp_path):
    import json
    video = _tiny_video(tmp_path / "parent.mp4")
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "global.json").write_text(json.dumps({"caption": "Iris sits."}))
    (scene / "0.json").write_text(json.dumps({"caption": "Iris sits."}))
    def parts(video, chunks, phrase):
        result = {}
        for chunk in chunks:
            mask = torch.zeros(8, 8, 16)
            if phrase == "hair":
                mask[:, 1, 3] = 1
                mask[:, 1, 6] = 0.25
            elif phrase == "face":
                mask[:, 2:4, 2:6] = 1
            result[chunk] = mask
        return result
    judge = EntityDriftVerifier(
        [EntitySpec("Iris", "woman with red hair")],
        localizer=_StubLocalizer({("Iris", 0): (2, 0, 34, 32), ("Iris", 1): (2, 0, 34, 32)}, {}),
        comparer=_StubComparer({"Iris": {"matches_reference": False, "differences": "black hair", "vetoed_parts": ["hair"]}}),
        tracker=_BoxTracker(), work_dir=tmp_path / "judge", grid=(8, 16),
        attribute_masks=True, part_mask_fn=parts, part_min_coverage=0.002,
        defer_repair_planning=True, edit_entry_sigma=0.63,
    )
    verdict = judge.judge(_observation(video, 0), NormativeGraph.compile_scene(scene).view(0))
    json.dumps(verdict.metrics)
    options = verdict.metrics["repair_options"]
    assert len(options) == 3
    assert len({o["mask_content_sha256"] for o in options}) == 3
    for option in options:
        assert option["allowed_sigmas"] == [1.0]
        assert option["action"]["params"]["entry_sigma"] == 1.0
        preview = region_preview(option, video=str(video), output_dir=tmp_path / "previews")
        assert Path(preview["path"]).is_file()
        assert preview["mask_content_sha256"] == option["mask_content_sha256"]
        payload = torch.load(option["action"]["params"]["mask_path"], weights_only=False)
        assert payload["grid"] == [8, 16]
        assert not payload["masks"][0][:, 2:4, 2:6].any()
