#!/usr/bin/env python3
"""Seal the fixed P2 alpha only after the rendered visual verdict is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    scan_root = pilot_root / "p2/alpha_scan"
    numeric_path = pilot_root / "p2/alpha_numeric_prefilter.json"
    metrics_path = scan_root / "rendered_metrics.json"
    verdict_path = scan_root / "visual_verdict.json"
    numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    if not verdict.get("complete"):
        raise SystemExit("visual verdict is incomplete")
    for alpha in (0.25, 0.5, 1.0):
        record = verdict.get("candidates", {}).get(str(alpha), {})
        if record.get("accepted") not in {True, False}:
            raise SystemExit(f"visual verdict for alpha={alpha} is missing")
        evidence = str(record.get("evidence") or "").strip()
        if not evidence or "shot" not in evidence.lower():
            raise SystemExit(
                f"visual evidence for alpha={alpha} must name a decisive shot"
            )
    if not str(verdict.get("reviewer") or "").strip():
        raise SystemExit("visual verdict reviewer is missing")
    if not str(verdict.get("rationale") or "").strip():
        raise SystemExit("visual verdict rationale is missing")
    selected = verdict.get("selected_alpha")
    automatic = {
        float(record["alpha"]): bool(record["automatic_stability_pass"])
        for record in metrics["candidates"]
    }
    accepted = {
        alpha
        for alpha in (0.25, 0.5, 1.0)
        if verdict["candidates"][str(alpha)]["accepted"] is True
    }
    if selected is None:
        if accepted:
            raise SystemExit(
                "visual verdict accepted at least one alpha but did not select one"
            )
        gate_pass = False
    else:
        selected = float(selected)
        if selected not in (0.25, 0.5, 1.0):
            raise SystemExit(f"visual verdict selected unexpected alpha={selected}")
        numeric_ok = selected in {
            float(value) for value in numeric["eligible_alphas"]
        }
        candidate_verdict = verdict.get("candidates", {}).get(str(selected), {})
        visual_ok = candidate_verdict.get("accepted") is True
        gate_pass = bool(
            numeric.get("gate_pass")
            and numeric_ok
            and automatic.get(selected)
            and visual_ok
        )
    output = {
        "schema_version": 2,
        "gate": "P2_alpha_rendered_visual_stability",
        "gate_pass": gate_pass,
        "visual_review_complete": True,
        "scene": metrics["scene"],
        "selected_alpha": selected if gate_pass else None,
        "numeric_prefilter": str(numeric_path.relative_to(ROOT)),
        "rendered_metrics": str(metrics_path.relative_to(ROOT)),
        "visual_verdict": str(verdict_path.relative_to(ROOT)),
        "selection_rule": "selected candidate must pass numeric, rendered automatic, and visual stability gates",
        "reviewer": verdict.get("reviewer"),
        "rationale": verdict.get("rationale"),
    }
    path = pilot_root / "p2/alpha_scan.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_pass": gate_pass, "selected_alpha": output["selected_alpha"]}, indent=2))
    raise SystemExit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
