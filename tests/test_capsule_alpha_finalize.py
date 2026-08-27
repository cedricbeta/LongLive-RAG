import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_capsule_alpha_scan", ROOT / "scripts/finalize_capsule_alpha_scan.py"
)
finalizer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(finalizer)


def _write_inputs(root: Path, *, accepted: set[float], selected):
    scan = root / "p2/alpha_scan"
    scan.mkdir(parents=True)
    (root / "p2/alpha_numeric_prefilter.json").write_text(
        json.dumps({"gate_pass": True, "eligible_alphas": [0.25, 0.5, 1.0]})
    )
    (scan / "rendered_metrics.json").write_text(
        json.dumps(
            {
                "scene": "bearded_watchmaker_workshop",
                "candidates": [
                    {"alpha": alpha, "automatic_stability_pass": True}
                    for alpha in (0.25, 0.5, 1.0)
                ],
            }
        )
    )
    (scan / "visual_verdict.json").write_text(
        json.dumps(
            {
                "complete": True,
                "reviewer": "test reviewer",
                "rationale": "all candidates were checked",
                "selected_alpha": selected,
                "candidates": {
                    str(alpha): {
                        "accepted": alpha in accepted,
                        "evidence": f"shot {index} visual check",
                    }
                    for index, alpha in enumerate((0.25, 0.5, 1.0), start=1)
                },
            }
        )
    )


def test_all_visual_candidates_rejected_seals_failed_scan(tmp_path, monkeypatch):
    _write_inputs(tmp_path, accepted=set(), selected=None)
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["finalize_capsule_alpha_scan.py", "--pilot_root", "."]
    )
    with pytest.raises(SystemExit) as exc:
        finalizer.main()
    assert exc.value.code == 1
    result = json.loads((tmp_path / "p2/alpha_scan.json").read_text())
    assert result["gate_pass"] is False
    assert result["selected_alpha"] is None
    assert result["visual_review_complete"] is True


def test_accepted_candidate_requires_an_explicit_selection(tmp_path, monkeypatch):
    _write_inputs(tmp_path, accepted={0.5}, selected=None)
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["finalize_capsule_alpha_scan.py", "--pilot_root", "."]
    )
    with pytest.raises(SystemExit, match="accepted at least one alpha"):
        finalizer.main()
