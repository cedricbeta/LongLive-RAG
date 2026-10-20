"""Mask time coordinates must agree with the actually exported 60-second task."""
import pytest
from prepare_mask import exported_frames, entity_crop_box
import torch


@pytest.mark.parametrize('latent,frames', [(0, [0]), (1, [0,1,2,3]),
    (40, [156,157,158,159]), (41, [160,161,162,163]), (240, [956,957,958,959])])
def test_frame_mapping_at_warmup_shot_boundary_and_end(latent, frames):
    assert exported_frames(latent, 960) == frames


def test_projection_cannot_read_past_export():
    with pytest.raises(ValueError):
        exported_frames(241, 960)


def test_empty_entity_cannot_create_a_crop():
    assert entity_crop_box(torch.zeros(60, 104), (832, 480), 0.35) is None


def test_entity_crop_contains_selected_cells_and_clamps_to_video():
    entity = torch.zeros(60, 104)
    entity[0, 103] = 1
    assert entity_crop_box(entity, (832, 480), 0.35) == (816, 0, 832, 16)
    entity[59, 0] = 1
    assert entity_crop_box(entity, (832, 480), 0.35) == (0, 0, 832, 480)
