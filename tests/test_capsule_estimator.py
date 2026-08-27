import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "estimate_capsule_subspace", ROOT / "scripts/estimate_capsule_subspace.py"
)
estimator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = estimator
SPEC.loader.exec_module(estimator)


def _synthetic_samples() -> list:
    generator = torch.Generator().manual_seed(17)
    samples = []
    for scene_index, scene in enumerate(estimator.SCENES):
        for seed in (0, 1, 2):
            for shot in range(8):
                for frame_offset in range(4):
                    value = torch.randn(
                        2, 64, generator=generator, dtype=torch.float64
                    )
                    # The three seed directions are shared across scenes, while
                    # a separate scene direction makes all 12 generated
                    # instances distinguishable. A projector trained without
                    # one scene can therefore generalize identity structure.
                    value[:, seed] += 2.5
                    value[:, 8 + scene_index] += 2.0
                    samples.append(
                        estimator.FrameSample(
                            scene=scene,
                            seed=seed,
                            shot=shot,
                            frame=shot * 8 + frame_offset,
                            value=value,
                        )
                    )
    return samples


def test_stack_treats_same_prompt_cross_seed_as_distinct_identities():
    samples = _synthetic_samples()
    _, labels = estimator.stack(
        sample
        for sample in samples
        if sample.scene == estimator.SCENES[0] and sample.shot == 0
    )
    assert set(labels.tolist()) == {0, 1, 2}


def test_fisher_gate_recovers_low_rank_identity_from_held_out_shots():
    result, fit = estimator.evaluate_layer(
        _synthetic_samples(),
        layer=7,
        random_trials=30,
        bootstrap_repeats=200,
        random_seed_value=20260715,
    )
    assert fit is not None
    assert result["identity_classes"] == 12
    assert result["train_shots"] == [0, 2, 4, 6]
    assert result["held_out_shots"] == [1, 3, 5, 7]
    passing = [rank for rank in result["ranks"] if rank["gate_pass"]]
    assert passing
    selected = max(passing, key=lambda rank: rank["identity_balanced_accuracy"])
    assert selected["identity_balanced_accuracy"] > 0.70
    assert selected["identity_balanced_accuracy"] > selected["random_accuracy_p95"]
    assert selected["identity_balanced_accuracy"] > selected["complement_balanced_accuracy"]
    assert selected["pose_time"]["complement_fraction"] > 0.5
    assert selected["leave_one_scene_out"]["gate_pass"]
    assert selected["leave_one_scene_out"]["folds_passing"] >= 3
