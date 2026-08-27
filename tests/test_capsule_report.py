import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_capsule_pilot_report", ROOT / "scripts/build_capsule_pilot_report.py"
)
reporter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reporter)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _g0(*, passed: bool) -> dict:
    scenes = {
        scene: {
            "decoded_frames_exact": True,
            "probe_transparency_exact": True,
            "reference_kind": "frozen stage_A probe-off artifact",
            "max_per_shot_abs_delta": 0.0,
            "archive_aggregate_max_abs_delta": 0.0,
            "overhead_fraction": 0.01,
            "dump_integrity": {"gib": 0.1},
        }
        for scene in reporter.SCENES
    }
    return {
        "schema_version": 2,
        "gate_pass": passed,
        "gate_status": "pass" if passed else "blocked",
        "archive_sanity_tolerance": 1e-2,
        "blocked_reasons": [] if passed else ["probe transparency unresolved"],
        "overhead": {"status": "exempt"},
        "scenes": scenes,
    }


def _common_inputs(root: Path, *, g0_passed: bool):
    pilot = root / "videos/pilot"
    _write_json(pilot / "g0/g0_results.json", _g0(passed=g0_passed))
    _write_json(
        pilot / "masks/mask_qc_stats.json",
        {"selected_method": "clipseg_precomputed", "scenes": list(reporter.SCENES)},
    )
    _write_json(
        pilot / "probe_replay_audit_seed0.json",
        {"frozen_inputs_unchanged": True},
    )
    return pilot


def _run(root: Path, monkeypatch):
    monkeypatch.setattr(reporter, "ROOT", root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_capsule_pilot_report.py",
            "--pilot_root",
            "videos/pilot",
            "--report_dir",
            "docs/report",
        ],
    )
    reporter.main()
    return (root / "docs/report/REPORT.md").read_text()


def test_g0_failure_writes_infrastructure_blocked_report(tmp_path, monkeypatch):
    pilot = _common_inputs(tmp_path, g0_passed=False)
    g0 = json.loads((pilot / "g0/g0_results.json").read_text())
    g0["scenes"]["brown_bear_river"]["failure_visual"] = (
        "videos/pilot/g0/brown_bear_river_replay_failure.png"
    )
    _write_json(pilot / "g0/g0_results.json", g0)
    _write_json(
        pilot / "g0/visual_review.json",
        {
            "reviewer": "model-assisted reviewer",
            "scenes": {
                "brown_bear_river": {
                    "rating": "broken",
                    "evidence": "Shot 1 is visibly different.",
                }
            },
        },
    )
    report = _run(tmp_path, monkeypatch)
    assert "Final outcome: **blocked**" in report
    assert "G0 infrastructure is blocked" in report
    assert "brown_bear_river_replay_failure.png" in report
    assert "Shot 1 is visibly different" in report
    assert "not a hypothesis kill" in report
    assert "Only if G0 passes" in report
    assert "## G1" not in report


def test_unsafe_final_refit_writes_g1_blocked_report(tmp_path, monkeypatch):
    pilot = _common_inputs(tmp_path, g0_passed=True)
    _write_json(
        pilot / "g1/g1_results.json",
        {
            "gate_pass": False,
            "selection": None,
            "candidate_selection": {
                "final_refit": {
                    "failure_reason": "within covariance remained ill-conditioned"
                }
            },
        },
    )
    report = _run(tmp_path, monkeypatch)
    assert "Final outcome: **blocked**" in report
    assert "mandatory all-data refit" in report
    assert "ill-conditioned" in report


def test_no_held_out_or_loso_subspace_is_a_g1_kill(tmp_path, monkeypatch):
    pilot = _common_inputs(tmp_path, g0_passed=True)
    _write_json(
        pilot / "g1/g1_results.json",
        {
            "gate_pass": False,
            "selection": None,
            "candidate_selection": None,
        },
    )
    report = _run(tmp_path, monkeypatch)
    assert "Final outcome: **kill**" in report
    assert "held-out-shot and leave-one-scene-out" in report


def test_g2_delivery_failure_is_blocked_without_human_verdict(tmp_path, monkeypatch):
    pilot = _common_inputs(tmp_path, g0_passed=True)
    _write_json(
        pilot / "g1/g1_results.json",
        {"gate_pass": True, "selection": None, "candidate_selection": None},
    )
    _write_json(
        pilot / "p2/evaluation/g2_g3_results.json",
        {"g2": {"gate_pass": False}, "g3": {"gate_pass": False}},
    )
    report = _run(tmp_path, monkeypatch)
    assert "Final outcome: **blocked**" in report
    assert "G2 delivery/stability failed" in report


def test_v3_failed_g1b_effective_channel_closes_linear_capsule(tmp_path, monkeypatch):
    pilot = _common_inputs(tmp_path, g0_passed=True)
    _write_json(
        pilot / "g1/g1_results.json",
        {"gate_pass": False, "selection": None, "candidate_selection": None},
    )
    _write_json(
        pilot / "g1b/g1b_results.json",
        {"gate_pass": False, "selection": None, "layers": []},
    )
    arms = {}
    for arm_id in (
        "sandy_beach_driftwood__layer14",
        "sandy_beach_driftwood__all5",
        "bearded_watchmaker_workshop__layer14",
        "bearded_watchmaker_workshop__all5",
    ):
        arms[arm_id] = {
            "reachability": {"edit_reached_graph": True},
                "max_abs_identity_consistency_delta": 0.02,
                "identity_consistency_delta_by_shot": {"7": 0.02},
                "identity_metric_effect": True,
                "guard_delta": {
                    "dynamic_degree": 1.0,
                    "inter_shot_composition_diversity": -0.01,
                    "palette_agreement": 0.02,
                },
            "human_verdict": {
                "visible_subject_appearance_change": True,
                "evidence": "Shot 7 changes the subject appearance.",
            },
            "channel_effective": True,
            "visuals": {
                "two_up": f"videos/pilot/channel_test/visuals/{arm_id}_2up.png",
                "subject_closeups": None,
            },
        }
    _write_json(
        pilot / "channel_test/channel_results.json",
        {
            "channel_effective": True,
            "human_verdict_complete": True,
            "human_reviewer": "model-assisted reviewer",
            "verdict": "channel_effective",
            "arms": arms,
        },
    )
    failed_exit = pilot / "logs/channel__watchmaker__all5.exit_code.failed.20260716"
    failed_exit.parent.mkdir(parents=True, exist_ok=True)
    failed_exit.write_text("1\n", encoding="utf-8")

    report = _run(tmp_path, monkeypatch)

    assert "Final outcome: **close linear capsule — channel effective**" in report
    assert "## G1 — held-out identity subspace" in report
    assert "## G1b — final within-scene estimator" in report
    assert "## Causal V-channel mini-test" in report
    assert "channel effective × G1b fail" in report
    assert "## Future work (outside this pilot)" in report
    assert "nonlinear or learned capsule" in report
    assert "whole-vector prototype steering" in report
    assert "exit_code.failed.20260716" in report
