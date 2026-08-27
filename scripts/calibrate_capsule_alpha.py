#!/usr/bin/env python3
"""Numerically prefilter alpha candidates on real clean-recache V.

This is deliberately not the final alpha decision.  The surviving candidates
must be rendered by ``run_capsule_alpha_scan.sh`` and visually adjudicated
before ``p2/alpha_scan.json`` can exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCENE = "bearded_watchmaker_workshop"
CANDIDATES = (0.25, 0.5, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    g1 = json.loads((pilot_root / "g1/g1_results.json").read_text(encoding="utf-8"))
    g0 = json.loads((pilot_root / "g0/g0_results.json").read_text(encoding="utf-8"))
    if not g1.get("gate_pass") or not g0.get("gate_pass"):
        raise SystemExit("G0 and G1 must pass before alpha calibration")
    layer = int(g1["selection"]["layer"])
    rank = int(g1["selection"]["rank"])
    scene_g0 = g0["scenes"][SCENE]
    stream = scene_g0.get(
        "reference_per_shot", scene_g0.get("baseline_per_shot")
    )["subject_anchor_to_shot0"]
    worst_shot = int(np.argmin(np.asarray(stream[1:])) + 1)
    artifact = torch.load(
        pilot_root / f"g1/artifacts/{SCENE}.pt",
        map_location="cpu",
        weights_only=False,
    )
    values = []
    dump_dir = (
        pilot_root
        / f"probe_replays/seed0/dumps/{SCENE}/seed0/v/layer_{layer:02d}"
    )
    for path in sorted(dump_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["shot_index"]) >= worst_shot:
            values.append(payload["v"].float())
    if not values:
        raise SystemExit("no post-drift watchmaker V rows for alpha calibration")
    target = torch.cat(values)
    donor = artifact["donor"].float()
    delta = donor.unsqueeze(0) - target
    delta_w = torch.einsum("hij,nhj->nhi", artifact["W"].float(), delta)
    coefficient = torch.einsum("hdr,nhd->nhr", artifact["U"].float(), delta_w)
    projected_w = torch.einsum("hdr,nhr->nhd", artifact["U"].float(), coefficient)
    update = torch.einsum(
        "hij,nhj->nhi", artifact["W_inv"].float(), projected_w
    )
    target_norm = target.flatten(1).norm(dim=1).clamp_min(1e-8)
    update_norm = update.flatten(1).norm(dim=1)
    records = []
    for alpha in CANDIDATES:
        edited = target + alpha * update
        ratio = alpha * update_norm / target_norm
        finite = bool(torch.isfinite(edited).all() and torch.isfinite(ratio).all())
        p99 = float(torch.quantile(ratio, 0.99))
        maximum = float(ratio.max())
        stable = bool(finite and p99 <= 0.50 and maximum <= 1.00)
        records.append(
            {
                "alpha": alpha,
                "finite": finite,
                "relative_update_norm_mean": float(ratio.mean()),
                "relative_update_norm_p95": float(torch.quantile(ratio, 0.95)),
                "relative_update_norm_p99": p99,
                "relative_update_norm_max": maximum,
                "stable": stable,
            }
        )
    stable = [record for record in records if record["stable"]]
    all_finite = all(record["finite"] for record in records)
    gate_pass = bool(stable and all_finite)
    output = {
        "schema_version": 1,
        "gate": "P2_alpha_numeric_prefilter",
        "gate_pass": gate_pass,
        "scene": SCENE,
        "layer": layer,
        "rank": rank,
        "worst_drift_shot": worst_shot,
        "rows": int(target.shape[0]),
        "candidates": records,
        "eligible_alphas": [float(record["alpha"]) for record in stable],
        "render_alphas": [float(record["alpha"]) for record in records] if all_finite else [],
        "selected_alpha": None,
        "selection_rule": (
            "numeric eligibility requires finite output, p99 relative update norm "
            "<=0.50, and max <=1.00; no alpha is selected until the dedicated "
            "one-scene rendered visual scan is complete"
        ),
        "next_required_step": "render and review baseline/0.25/0.5/1.0 watchmaker scan",
    }
    output_dir = pilot_root / "p2"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alpha_numeric_prefilter.json").write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"gate_pass": gate_pass, "eligible_alphas": output["eligible_alphas"]}, indent=2))
    raise SystemExit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
