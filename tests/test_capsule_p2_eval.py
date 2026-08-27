import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "capsule_p2_eval", ROOT / "scripts/evaluate_capsule_p2.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


GUARD_KEYS = (
    "dynamic_degree",
    "inter_shot_composition_diversity",
    "motion_profile_agreement",
    "within_shot_motion",
    "prompt_adherence_mean",
    "prompt_adherence_min",
    "invariant_margin_mean",
    "invariant_margin_min",
    "palette_agreement",
)


def test_round18_guard_uses_frozen_thresholds_and_fails_closed():
    baseline = {key: 1.0 for key in GUARD_KEYS}
    passing = dict(baseline)
    passing.update(
        {
            "inter_shot_composition_diversity": 0.90,
            "dynamic_degree": 0.80,
            "prompt_adherence_mean": 0.98,
            "prompt_adherence_min": 0.98,
        }
    )
    assert MODULE.guard_verdict(baseline, passing)["passed"]

    for key, value in (
        ("inter_shot_composition_diversity", 0.899),
        ("dynamic_degree", 0.799),
        ("prompt_adherence_min", 0.979),
        ("invariant_margin_mean", 0.999),
    ):
        modified = dict(passing)
        modified[key] = value
        assert not MODULE.guard_verdict(baseline, modified)["passed"]

    nonfinite = dict(passing)
    nonfinite["palette_agreement"] = np.nan
    verdict = MODULE.guard_verdict(baseline, nonfinite)
    assert not verdict["passed"]
    assert "modified:palette_agreement:nonfinite_or_missing" in verdict["failures"]

    missing = dict(passing)
    missing.pop("prompt_adherence_min")
    verdict = MODULE.guard_verdict(baseline, missing)
    assert not verdict["passed"]
    assert "modified:prompt_adherence_min:nonfinite_or_missing" in verdict["failures"]


def test_bootstrap_ci_is_seeded_and_bounds_constant_effect():
    assert MODULE.bootstrap_ci([0.25] * 8, repeats=100, seed=7) == [0.25] * 3
    first = MODULE.bootstrap_ci([0.1, -0.1, 0.3], repeats=200, seed=9)
    second = MODULE.bootstrap_ci([0.1, -0.1, 0.3], repeats=200, seed=9)
    assert first == second


def test_paired_leakage_requires_positive_bootstrap_advantage():
    capsule = {
        scene: {
            "raft_motion_delta_by_shot": {"3": 0.1},
            "background_preservation_by_shot": {"3": 0.98},
            "inter_shot_diversity_delta_by_shot": {"3": -0.01},
        }
        for scene in MODULE.SCENES
    }
    full = {
        scene: {
            "raft_motion_delta_by_shot": {"3": 0.3},
            "background_preservation_by_shot": {"3": 0.90},
            "inter_shot_diversity_delta_by_shot": {"3": -0.20},
        }
        for scene in MODULE.SCENES
    }
    result = MODULE.paired_leakage_advantage(capsule, full, repeats=200)
    assert all(
        record["significantly_less_leakage"] for record in result.values()
    )
    assert result["motion"]["bootstrap_ci95"] == pytest.approx([0.2] * 3)


def test_p2_uses_g0_selected_probe_off_reference(tmp_path, monkeypatch):
    pilot = tmp_path / "videos/pilot"
    reference = tmp_path / "videos/pilot/g0/bear_probe_off/reference.mp4"
    reference.parent.mkdir(parents=True)
    reference.touch()
    (pilot / "g0").mkdir(exist_ok=True)
    (pilot / "g0/g0_results.json").write_text(
        json.dumps(
            {
                "gate_pass": True,
                "scenes": {
                    "brown_bear_river": {
                        "reference": "videos/pilot/g0/bear_probe_off/reference.mp4"
                    }
                },
            }
        )
    )
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    assert MODULE.g0_reference_video(pilot, "brown_bear_river") == reference
