import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_capsule_g0", ROOT / "scripts/evaluate_capsule_g0.py"
)
g0 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(g0)


def test_bear_decision_tree_all_equal():
    assert g0.classify_bear_hashes("same", "same", "same") == "ALL_EQUAL"


def test_bear_decision_tree_stale_archive_branch_a():
    assert g0.classify_bear_hashes("old", "current", "current") == "A_STALE_ARCHIVE"


def test_bear_decision_tree_probe_perturbation_branch_b():
    assert g0.classify_bear_hashes("archive", "probe", "archive") == "B_PROBE_PERTURBATION"


def test_bear_decision_tree_nondeterminism_branch_c():
    assert g0.classify_bear_hashes("archive", "probe", "third") == "C_SCENE_NONDETERMINISM"


def test_g0_report_links_bear_mask_alignment_qc(tmp_path):
    scene_record = {
        "reference_kind": "test reference",
        "probe_transparency_exact": True,
        "max_per_shot_abs_delta": 0.0,
        "archive_aggregate_max_abs_delta": 0.001,
        "dump_integrity": {"gib": 0.5},
        "passed": True,
    }
    result = {
        "gate_status": "pass",
        "archive_sanity_tolerance": 0.01,
        "overhead": {"status": "exempt", "reason": "shared host"},
        "scenes": {scene: dict(scene_record) for scene in g0.SCENES},
        "bear_diagnostic": {
            "branch": "A_STALE_ARCHIVE",
            "archive_decoded_sha256": "old",
            "probe_on_decoded_sha256": "current",
            "probe_off_decoded_sha256": "current",
            "resolution": "use current reference",
            "stage_a_mask_on_current_probe_visual": (
                "videos/capsule_pilot_20260715/g0/"
                "brown_bear_current_probe_on_stage_a_mask.png"
            ),
        },
    }

    g0.write_report(tmp_path, result)

    report = (tmp_path / "G0_REPORT.md").read_text(encoding="utf-8")
    assert "Mask alignment QC" in report
    assert "brown_bear_current_probe_on_stage_a_mask.png" in report
