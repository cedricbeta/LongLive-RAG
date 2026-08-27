from pathlib import Path
import sys

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from estimate_capsule_subspace import FrameSample
from estimate_capsule_subspace_g1b import fit_within_scene_fisher


SCENES = (
    "bearded_watchmaker_workshop",
    "tattooed_noodle_chef",
    "brown_bear_river",
    "sandy_beach_driftwood",
)


def _samples(scene_offset: float):
    samples = []
    for scene_index, scene in enumerate(SCENES):
        for seed in (0, 1, 2):
            for shot in (0, 2, 4, 6):
                value = torch.zeros(1, 128, dtype=torch.float64)
                value[0, 0] = float(seed - 1)
                value[0, 1] = float((seed - 1) ** 2)
                value[0, 20 + scene_index] = scene_offset * (scene_index + 1)
                value[0, 30 + shot] = 0.05
                samples.append(
                    FrameSample(
                        scene=scene,
                        seed=seed,
                        shot=shot,
                        frame=shot * 8,
                        value=value,
                    )
                )
    return samples


def test_g1b_scene_offsets_do_not_enter_between_scatter():
    no_offset = fit_within_scene_fisher(_samples(0.0), list(SCENES))["fisher"]
    huge_offset = fit_within_scene_fisher(_samples(10000.0), list(SCENES))["fisher"]
    assert torch.allclose(no_offset["eigvals"], huge_offset["eigvals"], atol=1e-8)
    assert min(huge_offset["positive_rank"]) <= 8
