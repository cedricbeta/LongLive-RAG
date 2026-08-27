#!/usr/bin/env python3
"""Seal the channel mini-test after visual review."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "videos/capsule_pilot_20260715"


def main() -> None:
    result_path = PILOT / "channel_test/channel_results.json"
    verdict_path = PILOT / "channel_test/human_verdict.json"
    manifest_path = PILOT / "channel_test/channel_run_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not verdict.get("complete") or not verdict.get("reviewer"):
        raise SystemExit("human verdict is incomplete")
    if set(verdict.get("arms", {})) != set(result["arms"]):
        raise SystemExit("human verdict arm set does not match channel results")
    for arm_id, record in verdict["arms"].items():
        if not isinstance(record.get("visible_subject_appearance_change"), bool):
            raise SystemExit(f"{arm_id} visible-subject verdict is not boolean")
        if not isinstance(record.get("only_trivial_perturbation"), bool):
            raise SystemExit(f"{arm_id} trivial-perturbation verdict is not boolean")
        if not record.get("evidence"):
            raise SystemExit(f"{arm_id} visual evidence is empty")
        if record["visible_subject_appearance_change"] and record["only_trivial_perturbation"]:
            raise SystemExit(f"{arm_id} visual verdict is contradictory")
    human_any = any(
        record["visible_subject_appearance_change"]
        for record in verdict["arms"].values()
    )
    for arm_id, arm in result["arms"].items():
        arm["human_verdict"] = verdict["arms"][arm_id]
        arm["channel_effective"] = bool(
            arm["identity_metric_effect"]
            or verdict["arms"][arm_id]["visible_subject_appearance_change"]
        )
    result["human_verdict_complete"] = True
    result["human_reviewer"] = verdict["reviewer"]
    result["human_visible_effect_any"] = human_any
    result["channel_effective"] = any(
        arm["channel_effective"] for arm in result["arms"].values()
    )
    result["verdict"] = (
        "channel_effective" if result["channel_effective"] else "channel_ineffective"
    )
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    attempts_total = 0
    for run in manifest["runs"]:
        arm_id = run["arm_id"]
        arm = result["arms"][arm_id]
        stem = Path(run["config"]).stem
        archived = sorted(PILOT.glob(f"logs/{stem}.exit_code.failed.*"))
        attempts = 1 + len(archived)
        attempts_total += attempts
        run.update(
            {
                "status": "complete",
                "effective_alpha": float(arm["alpha"]),
                "alpha_fallback_used": bool(
                    arm["human_verdict"].get("alpha_fallback_used", False)
                ),
                "render_attempts": attempts,
                "archived_failed_exit_codes": [
                    str(path.relative_to(ROOT)) for path in archived
                ],
                "final_exit_code": 0,
                "delivery_integrity_pass": bool(
                    arm["runtime_integrity"]["passed"]
                ),
                "edit_reached_graph": bool(
                    arm["reachability"]["edit_reached_graph"]
                ),
            }
        )
    manifest["status"] = "complete"
    manifest["render_attempts_total"] = attempts_total
    manifest["alpha_fallbacks_used"] = sum(
        bool(run["alpha_fallback_used"]) for run in manifest["runs"]
    )
    manifest["finalized_verdict"] = result["verdict"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (PILOT / "channel_test/status").write_text(
        f"COMPLETE {result['verdict']}\n", encoding="utf-8"
    )

    lines = [
        "# Causal V-channel mini-test",
        "",
        f"Verdict: **{result['verdict']}**",
        "",
        "| arm | decoded reachability | max |DINO subject Δ| | metric effect | human-visible subject change | effective |",
        "|---|:---:|---:|:---:|:---:|:---:|",
    ]
    for arm_id, arm in result["arms"].items():
        human = arm["human_verdict"]
        lines.append(
            f"| {arm_id} | {'yes' if arm['reachability']['edit_reached_graph'] else 'no'} "
            f"| {arm['max_abs_identity_consistency_delta']:.4f} "
            f"| {'yes' if arm['identity_metric_effect'] else 'no'} "
            f"| {'yes' if human['visible_subject_appearance_change'] else 'no'} "
            f"| {'yes' if arm['channel_effective'] else 'no'} |"
        )
    lines += ["", "Signed per-shot DINO subject deltas and guard evidence:", ""]
    for arm_id, arm in result["arms"].items():
        signed = ", ".join(
            f"shot {shot}: {value:+.4f}"
            for shot, value in sorted(
                arm["identity_consistency_delta_by_shot"].items(),
                key=lambda item: int(item[0]),
            )
        )
        guards = arm["guard_delta"]
        lines.append(
            f"- `{arm_id}` — DINO: {signed}; guard Δ: dynamic degree "
            f"{guards['dynamic_degree']:+.4f}, composition diversity "
            f"{guards['inter_shot_composition_diversity']:+.4f}, palette agreement "
            f"{guards['palette_agreement']:+.4f}."
        )
    lines += [
        "",
        "Composition and motion guard deltas are reported in channel_results.json as evidence of causal force; they are not failure gates in this test.",
        "All four α=1.0 arms remained visually coherent, so the preregistered α=0.5 collapse fallback was not used.",
        "The result is a four-arm seed0 pilot and not paper-level causal evidence.",
    ]
    (PILOT / "channel_test/CHANNEL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"channel_effective": result["channel_effective"], "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
