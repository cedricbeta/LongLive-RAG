"""Pixel colour veto: spec parsing, family classification and conflict rules (CPU, no CLIPSeg)."""
import numpy as np
import pytest

from utils.agent_appearance_color import (
    FAMILIES, classify_pixels, conflict_share, families_conflict, measure_region, parse_appearance_spec,
    check_crop_against_spec, _rgb_to_hsv,
)


def test_parse_spec_hair_and_garment():
    spec = parse_appearance_spec("a middle-aged woman with curly red hair and a moss green coat")
    assert spec["hair"]["colors"] == ["red"] and spec["hair"]["styles"] == ["curly"]
    assert spec["garment"]["colors"] == ["green"] and spec["garment"]["type"] == "coat"
    spec = parse_appearance_spec("a middle-aged woman with auburn hair and a long tan coat")
    assert spec["hair"]["colors"] == ["red"] and spec["garment"]["colors"] == ["tan"]
    spec = parse_appearance_spec("a slim man with a buzz cut and a charcoal jacket")
    assert "hair" not in spec and spec["garment"]["colors"] == ["black"]
    spec = parse_appearance_spec("a young woman with short black curls and a beige sweater")
    assert spec["garment"]["colors"] == ["tan"]
    assert parse_appearance_spec("a person") == {}


def _solid(rgb, h=60, w=40):
    return np.tile(np.array(rgb, dtype=np.uint8), (h, w, 1))


@pytest.mark.parametrize("rgb,family", [
    ((200, 30, 30), "red"), ((120, 40, 20), "brown"), ((225, 200, 160), "tan"), ((20, 20, 20), "black"),
    ((240, 240, 240), "white"), ((120, 120, 120), "grey"), ((40, 110, 40), "green"), ((30, 40, 120), "blue"),
])
def test_classify_solid_colours(rgb, family):
    hsv = _rgb_to_hsv(_solid(rgb).reshape(-1, 3))
    labels = classify_pixels(hsv)
    assert FAMILIES[int(np.bincount(labels).argmax())] == family


def test_hair_tolerance_is_tighter_than_garment_tolerance():
    assert families_conflict(["red"], "black", "hair")            # ns090: dark hair vs curly red hair
    assert not families_conflict(["red"], "black", "garment")     # burgundy coat photographs dark
    assert not families_conflict(["red"], "brown", "hair")        # auburn reads brown
    assert families_conflict(["red"], "white", "hair")            # ns190: white hair
    assert families_conflict(["tan"], "pink", "garment")          # ns310: wine coat vs tan coat
    assert not families_conflict(["tan"], "white", "garment")     # beige sweater in bright light
    assert not families_conflict(["green"], "black", "garment")   # moss green coat in shadow
    assert not families_conflict(["blue"], "purple", "garment")   # hue neighbours


def test_conflict_share_sums_across_conflicting_families():
    # garment region skips the top 12% of rows; blocks: white (rows 0-59), grey (60-119), brown (120-179)
    image = np.concatenate([_solid((240, 240, 240), h=60), _solid((120, 120, 120), h=60), _solid((60, 30, 20), h=60)], axis=0)
    m = measure_region(image, "garment", masks={"garment": np.ones(image.shape[:2])})
    counted_white, grey, brown = 60 - int(0.12 * 180), 60, 60
    expected = (counted_white + grey) / (counted_white + grey + brown)       # white + grey conflict with red, brown tolerated
    assert conflict_share(m, ["red"], "garment") == pytest.approx(expected, abs=0.02)


def test_check_crop_uses_fixed_bands_without_a_device(tmp_path):
    from PIL import Image
    crop = np.zeros((120, 60, 3), dtype=np.uint8)
    crop[:30] = (235, 235, 235)        # white hair band
    crop[40:95] = (40, 110, 40)        # green garment band
    path = tmp_path / "crop.png"
    Image.fromarray(crop).save(path)
    result = check_crop_against_spec(path, parse_appearance_spec("a woman with curly red hair and a moss green coat"))
    assert result["checks"]["segmentation"] == "fixed_band"
    assert result["checks"]["hair"]["conflict"] and not result["checks"]["garment"]["conflict"]
    assert "hair measured white" in result["contradictions"][0]
