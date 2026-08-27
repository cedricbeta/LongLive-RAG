import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_capsule_channel_test",
    ROOT / "scripts/finalize_capsule_channel_test.py",
)
finalizer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(finalizer)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_finalizer_preserves_retry_ledger_and_seals_channel(tmp_path, monkeypatch):
    pilot = tmp_path / "videos/pilot"
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    monkeypatch.setattr(finalizer, "PILOT", pilot)
    arm_id = "scene__all5"
    _write(
        pilot / "channel_test/channel_results.json",
        {
            "arms": {
                arm_id: {
                    "alpha": 1.0,
                    "identity_metric_effect": True,
                    "max_abs_identity_consistency_delta": 0.02,
                    "identity_consistency_delta_by_shot": {"7": 0.02},
                    "guard_delta": {
                        "dynamic_degree": 1.0,
                        "inter_shot_composition_diversity": -0.01,
                        "palette_agreement": 0.03,
                    },
                    "runtime_integrity": {"passed": True},
                    "reachability": {"edit_reached_graph": True},
                }
            }
        },
    )
    _write(
        pilot / "channel_test/human_verdict.json",
        {
            "complete": True,
            "reviewer": "reviewer",
            "arms": {
                arm_id: {
                    "visible_subject_appearance_change": False,
                    "only_trivial_perturbation": True,
                    "alpha_fallback_used": False,
                    "evidence": "Shot 7 has only a trivial visual perturbation.",
                }
            },
        },
    )
    config = "videos/pilot/channel_test/configs/channel__scene__all5.yaml"
    _write(
        pilot / "channel_test/channel_run_manifest.json",
        {"runs": [{"arm_id": arm_id, "config": config}]},
    )
    archived = (
        pilot
        / "logs/channel__scene__all5.exit_code.failed.20260716T120000"
    )
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_text("1\n", encoding="utf-8")

    finalizer.main()

    result = json.loads(
        (pilot / "channel_test/channel_results.json").read_text()
    )
    manifest = json.loads(
        (pilot / "channel_test/channel_run_manifest.json").read_text()
    )
    assert result["channel_effective"] is True
    assert result["arms"][arm_id]["channel_effective"] is True
    assert manifest["status"] == "complete"
    assert manifest["render_attempts_total"] == 2
    assert manifest["runs"][0]["render_attempts"] == 2
    assert manifest["runs"][0]["archived_failed_exit_codes"] == [
        str(archived.relative_to(tmp_path))
    ]
    assert manifest["runs"][0]["alpha_fallback_used"] is False
    assert manifest["runs"][0]["delivery_integrity_pass"] is True
    assert (pilot / "channel_test/status").read_text().strip() == (
        "COMPLETE channel_effective"
    )
    assert "shot 7: +0.0200" in (
        pilot / "channel_test/CHANNEL_REPORT.md"
    ).read_text()
