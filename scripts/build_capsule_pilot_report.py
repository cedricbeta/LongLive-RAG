#!/usr/bin/env python3
"""Assemble the final G0-G3 and human-verdict pilot report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENES = (
    "bearded_watchmaker_workshop",
    "tattooed_noodle_chef",
    "brown_bear_river",
    "sandy_beach_driftwood",
)
ARMS = ("capsule", "full_vector", "random_subspace")
HUMAN_CHECKS = (
    "identity_recovery",
    "composition_action_leakage",
    "motion_quality",
    "boundary_artifacts",
    "random_placebo",
)


def load(path: Path, *, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: str) -> str:
    return f"../../{path}"


def validate_human_verdict(human: dict) -> None:
    if not human.get("complete"):
        raise RuntimeError("human verdict is incomplete")
    if not str(human.get("reviewer") or "").strip():
        raise RuntimeError("human verdict reviewer is missing")
    allowed = {"better", "same", "worse", "broken"}
    for scene in SCENES:
        checks = human.get("scenes", {}).get(scene, {})
        for check in HUMAN_CHECKS:
            record = checks.get(check, {})
            if record.get("rating") not in allowed:
                raise RuntimeError(f"invalid human rating for {scene}/{check}")
            evidence = str(record.get("evidence") or "").strip()
            if not evidence or "shot" not in evidence.lower():
                raise RuntimeError(
                    f"human evidence for {scene}/{check} must name a decisive shot"
                )
    if not str(human.get("rationale") or "").strip():
        raise RuntimeError("human verdict rationale is missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    parser.add_argument("--report_dir", default="docs/capsule_pilot_20260715")
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    g0 = load(pilot_root / "g0/g0_results.json")
    g0_visual_review = load(pilot_root / "g0/visual_review.json", required=False)
    g1 = load(pilot_root / "g1/g1_results.json", required=False)
    g1b = load(pilot_root / "g1b/g1b_results.json", required=False)
    channel = load(
        pilot_root / "channel_test/channel_results.json", required=False
    )
    masks = load(pilot_root / "masks/mask_qc_stats.json")
    replay_audit_name = "probe_replay_audit.json"
    replay_audit = load(pilot_root / replay_audit_name, required=False)
    if replay_audit is None:
        replay_audit_name = "probe_replay_audit_seed0.json"
        replay_audit = load(pilot_root / replay_audit_name)
    g23 = load(pilot_root / "p2/evaluation/g2_g3_results.json", required=False)
    p2_audit = load(pilot_root / "p2/p2_run_audit.json", required=False)
    human = load(pilot_root / "qualitative/human_verdict.json", required=False)
    alpha_numeric = load(
        pilot_root / "p2/alpha_numeric_prefilter.json", required=False
    )
    alpha_scan = load(pilot_root / "p2/alpha_scan.json", required=False)
    archived_channel_failures = sorted(
        pilot_root.glob("logs/channel__*.exit_code.failed.*")
    )
    v3_complete = bool(
        g1b is not None
        and channel is not None
        and isinstance(channel.get("channel_effective"), bool)
        and channel.get("human_verdict_complete")
    )
    if v3_complete:
        channel_effective = bool(channel["channel_effective"])
        if channel_effective and g1b.get("gate_pass"):
            if g23 is None or human is None or not human.get("complete"):
                raise RuntimeError(
                    "G1b and channel both passed; the decision matrix requires full P2/P3"
                )
            validate_human_verdict(human)
            verdict = str(human["overall_verdict"])
            verdict_reason = (
                "G1b and the causal V channel both passed, so the pilot entered the "
                "full three-arm P2 path; the final P2/P3 verdict is reported below."
            )
        elif not channel_effective:
            verdict = "close capsule — channel ineffective"
            verdict_reason = (
                "The preregistered causal-channel mini-test found no nontrivial identity "
                "or human-visible subject effect in any arm. The channel result overrides "
                "any contradictory estimator result, and the capsule pilot is closed."
            )
        else:
            verdict = "close linear capsule — channel effective"
            verdict_reason = (
                "G1b still failed against the rank-matched random null, permanently closing "
                "the low-rank linear capsule; at least one P=I intervention changed subject "
                "appearance, so causal V-channel validity is retained as a future-work pivot."
            )
    elif g1b is not None or channel is not None:
        raise RuntimeError("protocol v3 tracks exist but the channel human verdict is incomplete")
    elif not g0["gate_pass"]:
        verdict = "blocked"
        reasons = g0.get("blocked_reasons") or [
            "the probe-on/probe-off transparency contract is unresolved"
        ]
        verdict_reason = (
            "G0 infrastructure is blocked: " + " ".join(str(reason) for reason in reasons)
        )
    elif g1 is None:
        raise RuntimeError("passing G0 requires a G1 result before REPORT")
    elif not g1["gate_pass"]:
        candidate = g1.get("candidate_selection")
        if candidate is not None:
            verdict = "blocked"
            reason = candidate["final_refit"]["failure_reason"]
            verdict_reason = (
                "G1 artifact construction is blocked after held-out and LOSO selection: the mandatory all-data refit could "
                f"not produce a numerically safe edit artifact ({reason})."
            )
        else:
            verdict = "kill"
            verdict_reason = "G1 failed: no linear identity subspace passed both held-out-shot and leave-one-scene-out gates against complement and rank-matched random."
    elif alpha_numeric is not None and not alpha_numeric.get("gate_pass"):
        verdict = "blocked"
        verdict_reason = "The real-V alpha numerical prefilter retained no safe intervention strength. Under protocol v2 this is an intervention-infrastructure block, not evidence against the identity subspace."
    elif (
        alpha_scan is not None
        and alpha_scan.get("visual_review_complete")
        and not alpha_scan.get("gate_pass")
    ):
        verdict = "blocked"
        verdict_reason = "All rendered alpha candidates failed the combined numerical/guard/visual stability gate. Under protocol v2 this is blocked rather than a G1/G3 kill."
    elif p2_audit is not None and not p2_audit.get("passed"):
        verdict = "blocked"
        verdict_reason = "P2 completeness or frozen-input audit failed. Under protocol v2 this is a rendering-infrastructure block, not a G3 hypothesis verdict."
    elif g23 is not None and not g23.get("g2", {}).get("gate_pass"):
        verdict = "blocked"
        verdict_reason = "G2 delivery/stability failed. Under protocol v2 this is an intervention-infrastructure block; G3 and the final human verdict are not admissible."
    else:
        if g23 is None or human is None or not human.get("complete"):
            raise RuntimeError("passing G1 requires complete G2/G3 and human verdict before REPORT")
        validate_human_verdict(human)
        verdict = str(human["overall_verdict"])
        if verdict not in {"promising", "kill"}:
            raise RuntimeError("human overall_verdict must be promising or kill")
        numeric = "promising" if g23["g3"]["gate_pass"] else "kill"
        verdict_reason = str(human.get("rationale") or "human visual verdict")
        human["numeric_human_conflict"] = numeric != verdict
        (pilot_root / "qualitative/human_verdict.json").write_text(
            json.dumps(human, indent=2, sort_keys=True), encoding="utf-8"
        )

    lines = [
        "# Typed Memory Capsule feasibility pilot",
        "",
        f"Final outcome: **{verdict}**",
        "",
        verdict_reason,
        "",
        "> This is an exploratory four-scene probe with at most three replay seeds and a single-seed causal intervention. It does not constitute paper-level evidence.",
        "",
        "## Protocol v2/v3 and supersession",
        "",
        "- The v1 `verdict=kill` is **superseded** and reclassified as `G0-infra-blocked`.",
        "- G0 is a probe-transparency infrastructure gate. A G0 failure can only block and trigger diagnosis; it cannot kill the research hypothesis.",
        "- Under protocol v2, only G1 (no held-out linear identity subspace) or G3 (no causal advantage over random / leakage comparable to full-vector) could produce `verdict=kill`; protocol v3 adds the terminal G1b×channel decision matrix below.",
        "- Decoded-frame SHA-256 equality is the hard transparency check. Round18 metric recomputation is a non-gating sanity check at tolerance `1e-2`.",
        "- Cross-day wall-clock overhead is deleted. A same-GPU adjacent probe-on/off timing pair is optional; shared-host anomalies may be recorded and exempted.",
        "- Protocol v3 preserves the original G1 result but tests its stronger-than-application claim once more with the preregistered within-scene G1b estimator; no further estimator variant is allowed.",
        "- Protocol v3 independently tests causal channel reachability with four P=I full-vector arms. G1b and channel validity are combined only by the frozen decision matrix.",
        "",
        "## Fixed-scope integrity",
        "",
        f"- Frozen prompt/video audit: **{'PASS' if replay_audit['frozen_inputs_unchanged'] else 'FAIL'}**.",
        "- Model edit: clean-recache self-attention V only; K and RoPE untouched; no appended columns.",
        "- Feature flags remain default-off (`capsule_probe.enabled=false`, `capsule_edit.enabled=false`).",
        f"- Method transfer and formula: [METHOD_TRANSFER.md](METHOD_TRANSFER.md).",
        f"- Frozen SHA-256 ledger: [{args.pilot_root}/frozen_inputs_sha256.json]({rel(args.pilot_root + '/frozen_inputs_sha256.json')}).",
        "",
        "## Mask selection and QC",
        "",
        "Cross-attention was tested first and rejected because the watchmaker overlay fragmented onto clocks and bench regions. Partial runs were stopped and preserved under `rejected_mask_v0/`; none entered the estimator. The accepted fallback is CLIPSeg, with explicit manual visibility exclusions for known seed1/2 driftwood false positives.",
        "",
        f"- QC decision: **{masks['selected_method']}**, {len(masks['scenes'])} scene×seed sheets accepted.",
        f"- Audit narrative: [MASK_QC_REPORT.md]({rel(args.pilot_root + '/masks/MASK_QC_REPORT.md')}).",
        f"- Rejected-mask failure frame: [cross_attention_watchmaker_rejection.png]({rel(args.pilot_root + '/rejected_mask_v0/cross_attention_watchmaker_rejection.png')}).",
        "- Pixel/token map: 704×1280 pixels → 22×40 latent cells; one token cell is 32×32 decoded pixels; flattening is `frame*880 + y*40 + x`.",
        "",
        "## G0 v2 — probe transparency",
        "",
        f"Status: **{str(g0.get('gate_status', 'pass' if g0['gate_pass'] else 'blocked')).upper()}**. Round18 sanity tolerance: `{g0.get('archive_sanity_tolerance', 1e-2)}` (non-gating). Overhead: `{g0.get('overhead', {}).get('status', 'exempt pending v2 diagnostic')}`.",
        "",
        "| scene | probe-off reference | probe transparency SHA | max per-shot sanity delta | Round18 sanity delta | dump GiB |",
        "|---|---|:---:|---:|---:|---:|",
    ]
    for scene in SCENES:
        record = g0["scenes"][scene]
        lines.append(
            f"| {scene} | {record.get('reference_kind', 'legacy stage_A')} "
            f"| {'yes' if record.get('probe_transparency_exact', record.get('decoded_frames_exact')) else 'no'} "
            f"| {record['max_per_shot_abs_delta']:.3g} "
            f"| {record['archive_aggregate_max_abs_delta']:.3g} "
            f"| {record['dump_integrity']['gib']:.3f} |"
        )
    lines += [
        "",
        f"Detailed G0 report: [G0_REPORT.md]({rel(args.pilot_root + '/g0/G0_REPORT.md')}).",
    ]
    bear_mask_visual = (g0.get("bear_diagnostic") or {}).get(
        "stage_a_mask_on_current_probe_visual"
    )
    if bear_mask_visual:
        lines.append(
            f"- Bear mask alignment on the current replay: [{Path(bear_mask_visual).name}]({rel(bear_mask_visual)})."
        )
    legacy_g0_json = pilot_root / "g0/g0_results.v1-superseded.json"
    legacy_g0_report = pilot_root / "g0/G0_REPORT.v1-superseded.md"
    if legacy_g0_json.exists() and legacy_g0_report.exists():
        lines += [
            f"- Superseded v1 machine record: [g0_results.v1-superseded.json]({rel(args.pilot_root + '/g0/g0_results.v1-superseded.json')}).",
            f"- Superseded v1 narrative: [G0_REPORT.v1-superseded.md]({rel(args.pilot_root + '/g0/G0_REPORT.v1-superseded.md')}).",
        ]
    failure_visuals = [
        (
            scene,
            g0["scenes"][scene].get("comparison_visual")
            or g0["scenes"][scene].get("failure_visual"),
        )
        for scene in SCENES
        if (
            g0["scenes"][scene].get("comparison_visual")
            or g0["scenes"][scene].get("failure_visual")
        )
    ]
    if failure_visuals:
        lines += ["", "G0 diagnostic comparisons (人工必看):", ""]
        lines.extend(
            f"- `{scene}`: [{Path(path).name}]({rel(path)})"
            for scene, path in failure_visuals
        )
    if g0_visual_review is not None:
        lines += [
            "",
            f"G0 visual reviewer: `{g0_visual_review.get('reviewer')}`.",
            "",
        ]
        for scene, review in g0_visual_review.get("scenes", {}).items():
            lines.append(
                f"- `{scene}` ({review.get('rating')}): {review.get('evidence')}"
            )
    if not g0["gate_pass"]:
        lines += [
            "",
            "## Gate termination",
            "",
            "- P1/G1 was not launched because G0 is infrastructure-blocked; no identity-subspace claim was estimated.",
            "- P2 rendering was forbidden by the gate, so no capsule/full-vector/random-subspace intervention was run.",
            "- P3 four-arm materials and `human_verdict` are not applicable until P2 arms exist; the G0 replay-mismatch visual above is the current human-checkable diagnostic evidence.",
            "- This is an infrastructure diagnosis branch, not a hypothesis kill.",
        ]
    if g1 is not None:
        lines += [
            "",
            "## G1 — held-out identity subspace",
            "",
            f"Verdict: **{'PASS' if g1['gate_pass'] else 'FAIL — stop P2'}**. Labels are 12 scene×seed generated identities; same-prompt cross-seed instances are hard negatives. Train shots are {0,2,4,6}, held-out shots are {1,3,5,7}. LOSO fits the projector on three scenes and requires at least 3/4 held-out scenes to beat complement and random.",
            "",
        ]
    if g1 is not None and g1.get("selection"):
        selection = g1["selection"]
        lines += [
            f"Selected layer/rank: **L{selection['layer']} / r={selection['rank']}**, by the smallest-rank-within-one-SE rule.",
            "",
            f"- [capsule spectrum]({rel(args.pilot_root + '/g1/visuals/capsule_spectrum.png')})",
            f"- [head heatmap]({rel(args.pilot_root + '/g1/visuals/head_heatmap.png')})",
            f"- [token UMAP]({rel(args.pilot_root + '/g1/visuals/token_umap.png')})",
            f"- [subspace cosine]({rel(args.pilot_root + '/g1/visuals/subspace_cosine.png')})",
            f"- [projection norm by shot]({rel(args.pilot_root + '/g1/visuals/projection_norm_by_shot.png')})",
        ]
    if g1 is not None:
        lines += [
            "",
            f"Detailed G1 report: [G1_REPORT.md]({rel(args.pilot_root + '/g1/G1_REPORT.md')}).",
        ]

    if g1b is not None:
        passing_candidates = [
            (layer["layer"], rank["rank"])
            for layer in g1b.get("layers", [])
            for rank in layer.get("ranks", [])
            if rank.get("gate_pass")
        ]
        lines += [
            "",
            "## G1b — final within-scene estimator",
            "",
            f"Verdict: **{'PASS' if g1b['gate_pass'] else 'FAIL — close low-rank linear capsule'}**. Scene offsets are removed before accumulating seed-instance between-scatter; samples are per-frame entity-token means, and significance is cluster-bootstrapped by scene×seed×shot.",
            "",
            f"Passing layer×rank candidates: **{len(passing_candidates)}**. The gate requires ≥3/4 scenes to beat both the orthogonal complement and rank-matched random p95 with both cluster-bootstrap lower bounds above zero. LOSO is diagnostic only.",
            "",
            f"Detailed G1b report: [G1B_REPORT.md]({rel(args.pilot_root + '/g1b/G1B_REPORT.md')}).",
            "",
            "This is the final preregistered estimator iteration; no G1c/v4 is permitted.",
        ]

    if channel is not None:
        lines += [
            "",
            "## Causal V-channel mini-test",
            "",
            f"Verdict: **{channel.get('verdict', 'pending human review')}**. Four seed0 P=I arms cover two scenes × {{layer-14-only, all-five-layer}} edits at α=1.0.",
            "",
            "| arm | reachability | max |DINO subject Δ| | metric effect | human-visible subject change | effective |",
            "|---|:---:|---:|:---:|:---:|:---:|",
        ]
        for arm_id, arm in channel.get("arms", {}).items():
            human_arm = arm.get("human_verdict", {})
            lines.append(
                f"| {arm_id} "
                f"| {'yes' if arm['reachability']['edit_reached_graph'] else 'no'} "
                f"| {arm['max_abs_identity_consistency_delta']:.4f} "
                f"| {'yes' if arm['identity_metric_effect'] else 'no'} "
                f"| {'yes' if human_arm.get('visible_subject_appearance_change') else 'no'} "
                f"| {'yes' if arm.get('channel_effective') else 'no'} |"
            )
        lines += [
            "",
            "Guard/motion/composition changes are causal evidence in this mini-test, not failure gates.",
            "All four α=1.0 arms remained visually coherent; the preregistered α=0.5 collapse fallback was not used.",
            "Signed per-shot DINO direction and selected guard deltas:",
            "",
        ]
        for arm_id, arm in channel.get("arms", {}).items():
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
            f"Visual reviewer: `{channel.get('human_reviewer', 'pending')}`. Arm-level evidence:",
            "",
        ]
        for arm_id, arm in channel.get("arms", {}).items():
            human_arm = arm.get("human_verdict", {})
            visuals = arm.get("visuals", {})
            visual_links = [
                f"[2-up]({rel(visuals['two_up'])})" if visuals.get("two_up") else None,
                (
                    f"[subject closeups]({rel(visuals['subject_closeups'])})"
                    if visuals.get("subject_closeups")
                    else None
                ),
            ]
            lines.append(
                f"- `{arm_id}`: {human_arm.get('evidence', 'pending review')} "
                + "; ".join(link for link in visual_links if link)
            )
        lines += [
            "",
            f"Detailed channel report: [CHANNEL_REPORT.md]({rel(args.pilot_root + '/channel_test/CHANNEL_REPORT.md')}).",
            f"Machine result: [channel_results.json]({rel(args.pilot_root + '/channel_test/channel_results.json')}); human verdict: [human_verdict.json]({rel(args.pilot_root + '/channel_test/human_verdict.json')}).",
        ]

    if alpha_numeric is not None:
        lines += [
            "",
            "## Alpha stability calibration",
            "",
            f"Real-V numerical prefilter: **{'PASS' if alpha_numeric.get('gate_pass') else 'FAIL'}**; eligible alphas: `{alpha_numeric.get('eligible_alphas')}`.",
            f"Numeric record: [{args.pilot_root}/p2/alpha_numeric_prefilter.json]({rel(args.pilot_root + '/p2/alpha_numeric_prefilter.json')}).",
        ]
    if alpha_scan is not None:
        lines += [
            f"Rendered/visual stability: **{'PASS' if alpha_scan.get('gate_pass') else 'FAIL'}**; selected alpha: `{alpha_scan.get('selected_alpha')}`.",
            f"Rendered record: [{args.pilot_root}/p2/alpha_scan.json]({rel(args.pilot_root + '/p2/alpha_scan.json')}).",
            f"Calibration evidence: [contact sheet]({rel(args.pilot_root + '/p2/alpha_scan/alpha_contact_sheet.png')}), [real-time 4-up]({rel(args.pilot_root + '/p2/alpha_scan/alpha_4up.mp4')}), [metrics]({rel(args.pilot_root + '/p2/alpha_scan/rendered_metrics.json')}), [visual verdict]({rel(args.pilot_root + '/p2/alpha_scan/visual_verdict.json')}).",
        ]

    if g23 is not None and not g23.get("g2", {}).get("gate_pass"):
        lines += [
            "",
            "## G2 — intervention delivery/stability",
            "",
            "Verdict: **BLOCKED**. Delivery/stability failed, so G3 numeric and human hypothesis verdicts are inadmissible.",
        ]
    if g23 is not None and g23.get("g2", {}).get("gate_pass"):
        g2, g3 = g23["g2"], g23["g3"]
        lines += [
            "",
            "## G2/G3 — in-place causal intervention",
            "",
            f"G2 delivery/stability: **{'PASS' if g2['gate_pass'] else 'FAIL'}**. G3 numeric E/G/S: **{'PASS' if g3['gate_pass'] else 'FAIL'}**.",
            "",
            "| arm | identity delta | background preservation | RAFT motion Δ | CLIP background similarity Δ | diversity Δ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            record = g3["arms"][arm]
            leakage = record["leakage_vector"]
            lines.append(
                f"| {arm} | {record['mean_identity_consistency_delta']:.4f} "
                f"| {record['mean_background_preservation']:.4f} "
                f"| {leakage['raft_motion_delta_mean']:.4f} "
                f"| {leakage['clip_background_similarity_delta_mean']:.4f} "
                f"| {leakage['inter_shot_diversity_delta_mean']:.4f} |"
            )
        lines += [
            "",
            "The leakage vector is reported componentwise; it is not collapsed into a weighted scalar.",
            f"Detailed report: [G2_G3_REPORT.md]({rel(args.pilot_root + '/p2/evaluation/G2_G3_REPORT.md')}).",
            "",
            "## Human verdict",
            "",
            f"Reviewer: `{human.get('reviewer')}`. Overall: **{human.get('overall_verdict')}**. Numeric/human conflict: **{human.get('numeric_human_conflict')}**.",
            "",
            "| scene | identity | composition/action leakage | motion | boundary artifact | placebo |",
            "|---|---|---|---|---|---|",
        ]
        for scene in SCENES:
            checks = human["scenes"][scene]
            lines.append(
                f"| {scene} | {checks['identity_recovery']['rating']} "
                f"| {checks['composition_action_leakage']['rating']} "
                f"| {checks['motion_quality']['rating']} "
                f"| {checks['boundary_artifacts']['rating']} "
                f"| {checks['random_placebo']['rating']} |"
            )
        lines += ["", "Decisive visual evidence (shot-specific):", ""]
        for scene in SCENES:
            checks = human["scenes"][scene]
            lines.append(f"- `{scene}`")
            for key, label in (
                ("identity_recovery", "identity"),
                ("composition_action_leakage", "composition/action"),
                ("motion_quality", "motion"),
                ("boundary_artifacts", "boundary"),
                ("random_placebo", "placebo"),
            ):
                lines.append(
                    f"  - {label} ({checks[key]['rating']}): {checks[key]['evidence']}"
                )
        lines += [
            "",
            f"Machine-readable evidence sentences: [human_verdict.json]({rel(args.pilot_root + '/qualitative/human_verdict.json')}).",
            "",
            "## Mandatory human viewing order",
            "",
            "1. Mask QC: `masks/<scene>_mask_qc.png` (and seed1/2 sheets for estimator validity).",
            "2. Entity details: `qualitative/<scene>_entity_crops.png`.",
            "3. Shot evolution: `qualitative/<scene>_shotstrip.png`.",
            "4. Static ±0.5 s boundary companion: `qualitative/<scene>_boundary_qc.png`.",
            "5. Real-time motion and boundaries: `grids/<scene>_4up.mp4`.",
            "6. Metric curves: `qualitative/per_shot_dino_clip_curves.png`.",
            "",
            "| scene | mask QC | entity crops | shotstrip | boundary QC | real-time 4-up |",
            "|---|---|---|---|---|---|",
        ]
        for scene in SCENES:
            lines.append(
                f"| {scene} "
                f"| [mask]({rel(args.pilot_root + '/masks/' + scene + '_mask_qc.png')}) "
                f"| [crops]({rel(args.pilot_root + '/qualitative/' + scene + '_entity_crops.png')}) "
                f"| [strip]({rel(args.pilot_root + '/qualitative/' + scene + '_shotstrip.png')}) "
                f"| [boundary]({rel(args.pilot_root + '/qualitative/' + scene + '_boundary_qc.png')}) "
                f"| [4-up]({rel(args.pilot_root + '/grids/' + scene + '_4up.mp4')}) |"
            )
        lines.append(
            f"Per-shot metric curves: [per_shot_dino_clip_curves.png]({rel(args.pilot_root + '/qualitative/per_shot_dino_clip_curves.png')})."
        )

    if v3_complete:
        lines += [
            "",
            "## Final evidence-chain summary",
            "",
            "1. **G0 replay transparency survived:** probe-on/off decoded frames match exactly; the bear discrepancy was a stale archive, not probe-induced drift.",
            "2. **G1 killed the stronger universal claim:** no pooled 12-class layer×rank candidate beat complement, random null, and gating LOSO.",
            f"3. **G1b {'survived' if g1b['gate_pass'] else 'also died'}:** the final within-scene, no-shot-pooling estimator {'found a gated low-rank signal' if g1b['gate_pass'] else 'still produced no candidate passing ≥3/4 scenes'}.",
            f"4. **The causal V channel was {'effective' if channel['channel_effective'] else 'ineffective'}:** the four P=I arms {'produced at least one identity-metric or visible subject change' if channel['channel_effective'] else 'produced no nontrivial identity or visible subject change'}.",
            "",
            "The decision matrix is terminal. No additional estimator variant or v4 is allowed.",
            "",
            "## Terminal decision-matrix application",
            "",
            "| causal channel | G1b pass | G1b fails against random |",
            "|---|---|---|",
            "| effective | enter full three-arm P2 | permanently close low-rank linear capsule; retain channel validity and end pilot |",
            "| ineffective | close capsule (channel overrides estimator) | close capsule; double-negative confirmation |",
            "",
            f"Observed cell: **channel {'effective' if channel['channel_effective'] else 'ineffective'} × G1b {'pass' if g1b['gate_pass'] else 'fail'}** → **{verdict}**.",
            "",
            "## Operations and incident controls",
            "",
            "- Main and channel orchestrators hold the same non-blocking `logs/orchestrator.lock`; resume scripts probe that lock before starting.",
            "- Queue claim files are protected by `flock`, and a nonzero config is archived with a `.failed.<timestamp>` suffix before exactly one retry; the second failure is fatal.",
            "- Long-render admission uses `memory.free >= 61440 MiB` (60 GiB).",
            "- Long renders run in tmux, write per-config logs/exit codes, and retain frozen-input SHA auditing.",
        ]
        if archived_channel_failures:
            lines.append(
                "- Exercised retry evidence: "
                + ", ".join(
                    f"[{path.name}]({rel(str(path.relative_to(ROOT)))})"
                    for path in archived_channel_failures
                )
                + "; each corresponding log/runtime/render failure trace was archived before the sole retry."
            )
        if channel["channel_effective"] and not g1b["gate_pass"]:
            lines += [
                "",
                "## Future work (outside this pilot)",
                "",
                "The low-rank linear capsule is permanently closed; this is not an invitation to try another estimator variant. The retained causal-channel result supports only two distinct follow-up families: (1) a nonlinear or learned capsule, evaluated as a new study with its own controls, or (2) whole-vector prototype steering that explicitly accepts the composition/motion leakage measured here. Neither path reopens this pilot or creates a v4.",
            ]

    lines += [
        "",
        "## Reproduction ledger",
        "",
        "```bash",
        "python scripts/prepare_capsule_clipseg_masks.py --seeds 0 1 2",
        "python scripts/build_capsule_mask_qc.py --mask_source precomputed --seeds 0 1 2 --human_qc accepted",
        "python scripts/build_capsule_pilot_configs.py",
        "# tmux: scripts/run_capsule_probe_replays.sh GPU CONFIG...",
        "python scripts/build_capsule_g0_bear_diagnostic.py",
        "# tmux: scripts/run_capsule_g0_bear_diagnostic.sh GPU",
        "python scripts/audit_capsule_probe_replays.py",
        "python scripts/evaluate_capsule_g0.py",
        "# If G0 is blocked, diagnose/fix/substitute before continuing.",
        "# Only if G0 passes:",
        "python scripts/estimate_capsule_subspace.py --random_trials 100 --bootstrap_repeats 1000",
        "python scripts/build_capsule_g1_visuals.py",
        "# Protocol v3 final dual track (G1 remains preserved):",
        "python scripts/estimate_capsule_subspace_g1b.py --random_trials 100 --bootstrap_repeats 1000",
        "python scripts/build_capsule_channel_test.py",
        "# tmux: scripts/orchestrate_capsule_channel_test.sh",
        "python scripts/evaluate_capsule_channel_test.py --device cuda:0",
        "# Fill channel_test/human_verdict.json after viewing every 2-up and subject-closeup sheet",
        "python scripts/finalize_capsule_channel_test.py",
        "# Only if G1b and the causal channel both pass:",
        "python scripts/calibrate_capsule_alpha.py",
        "# tmux: scripts/run_capsule_alpha_scan.sh GPU",
        "python scripts/evaluate_capsule_alpha_scan.py --device cuda:0",
        "# Fill p2/alpha_scan/visual_verdict.json after viewing the contact sheet and 4-up",
        "python scripts/finalize_capsule_alpha_scan.py",
        "python scripts/build_capsule_p2_configs.py",
        "python scripts/promote_capsule_alpha_render.py",
        "python scripts/list_capsule_p2_remaining_configs.py",
        "# tmux: scripts/run_capsule_p2.sh GPU REMAINING_CONFIG...",
        "python scripts/audit_capsule_p2_runs.py",
        "python scripts/evaluate_capsule_p2.py",
        "python scripts/build_capsule_p3_artifacts.py",
        "python scripts/build_capsule_pilot_report.py",
        "```",
        "",
        f"- Probe config/command manifest: [{args.pilot_root}/probe_run_manifest.json]({rel(args.pilot_root + '/probe_run_manifest.json')}).",
        f"- Probe audit: [{args.pilot_root}/{replay_audit_name}]({rel(args.pilot_root + '/' + replay_audit_name)}).",
        f"- Bear G0 diagnostic manifest: [{args.pilot_root}/g0/bear_diagnostic_manifest.json]({rel(args.pilot_root + '/g0/bear_diagnostic_manifest.json')}).",
        f"- G0 visual review: [{args.pilot_root}/g0/visual_review.json]({rel(args.pilot_root + '/g0/visual_review.json')}).",
        "- Evaluation environment: [environment_versions.txt](environment_versions.txt).",
    ]
    if g1b is not None:
        lines.append(
            f"- G1b final estimator: [{args.pilot_root}/g1b/g1b_results.json]({rel(args.pilot_root + '/g1b/g1b_results.json')})."
        )
    if channel is not None:
        lines.append(
            f"- Channel-test manifest/results/human review: [manifest]({rel(args.pilot_root + '/channel_test/channel_run_manifest.json')}), [results]({rel(args.pilot_root + '/channel_test/channel_results.json')}), [human verdict]({rel(args.pilot_root + '/channel_test/human_verdict.json')})."
        )
    if g23 is not None:
        lines.append(
            f"- Rendered alpha calibration: [{args.pilot_root}/p2/alpha_scan.json]({rel(args.pilot_root + '/p2/alpha_scan.json')}); [4-up]({rel(args.pilot_root + '/p2/alpha_scan/alpha_4up.mp4')})."
        )
        lines.append(
            f"- P2 completeness/frozen-input audit: [{args.pilot_root}/p2/p2_run_audit.json]({rel(args.pilot_root + '/p2/p2_run_audit.json')})."
        )
        lines.append(
            f"- Alpha render manifest/metrics/verdict: [render_manifest.json]({rel(args.pilot_root + '/p2/alpha_scan/render_manifest.json')}), [rendered_metrics.json]({rel(args.pilot_root + '/p2/alpha_scan/rendered_metrics.json')}), [visual_verdict.json]({rel(args.pilot_root + '/p2/alpha_scan/visual_verdict.json')})."
        )
        lines.append(
            f"- P2 config/command manifest: [{args.pilot_root}/p2/p2_run_manifest.json]({rel(args.pilot_root + '/p2/p2_run_manifest.json')})."
        )
    (report_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "report": str((report_dir / 'REPORT.md').relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
