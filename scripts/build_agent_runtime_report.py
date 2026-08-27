#!/usr/bin/env python3
"""Build the evidence-linked Agentic Boundary Runtime v1 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def link(relative: str, label: str | None = None) -> str:
    path = Path(relative)
    return f"[{label or path.name}](../../{path.as_posix()})"


def yn(value: bool) -> str:
    return "PASS" if value else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/agentic_runtime_v1/REPORT.md")
    args = parser.parse_args()
    feature = load("videos/agentic_runtime_v1/feature_off/audit.json")
    protocol = load("videos/agentic_runtime_v1/protocol_gates/results.json")
    protocol_ledger = load("videos/agentic_runtime_v1/protocol_gates/ledger_audit.json")
    track = load("videos/agentic_runtime_v1/track_w/results.json")
    demo = load("videos/agentic_runtime_v1/demos/summary.json")
    ledger = load("videos/agentic_runtime_v1/demos/ledger_audit.json")

    feature_hashes = ", ".join(
        f"`{scene}` `{record['explicit_decoded_sha256'][:12]}…`"
        for scene, record in feature["scenes"].items()
    )
    demo_lines = []
    for scene, record in demo["scenes"].items():
        demo_lines.append(
            f"| `{scene}` | {record['shots']} | {record['true_action_match_count']} | "
            f"{record['quarantined_shots']} | {link(record['two_up'], '2-up')} | "
            f"{link(record['ledger'], 'ledger')} | {yn(record['passed'])} |"
        )
    fold_lines = []
    for scene, record in track["evaluation"]["folds"].items():
        fold_lines.append(
            f"| `{scene}` | {record['n']} | {record['shot_index_mae']:.4f} | "
            f"{record['available_signal_mae']:.4f} | {record['mae_improvement']:+.4f} |"
        )

    gates = [
        (
            "1. feature-off decoded-bit equality",
            feature["passed"],
            link("videos/agentic_runtime_v1/feature_off/audit.json"),
            "4 pilot scenes, seed 0, 480 decoded frames each",
        ),
        (
            "2. snapshot/restore no-action equality",
            protocol["snapshot_restore"]["passed"],
            link("videos/agentic_runtime_v1/protocol_gates/results.json"),
            "2 boundaries per scene; latent and post-generation cache bit equality",
        ),
        (
            "3. quarantine invisible",
            protocol["quarantine_invisible_e2e"]["passed"],
            link("videos/agentic_runtime_v1/protocol_gates/results.json"),
            "unit + deterministic end-to-end partition assertion",
        ),
        (
            "4. no future leak",
            ledger["no_future_leak_pass"] and protocol_ledger["no_future_leak_pass"],
            link("videos/agentic_runtime_v1/demos/ledger_audit.json"),
            "session ledgers plus VLM transcript source/hash availability audit",
        ),
        (
            "5. ledger replay",
            ledger["replay_pass"],
            link("videos/agentic_runtime_v1/demos/ledger_audit.json"),
            "3 manifest-seeded random actions, offline cache replay with an unreachable endpoint",
        ),
        (
            "6. retry budget forced fail",
            protocol["retry_budget_forced_fail"]["passed"],
            link("videos/agentic_runtime_v1/protocol_gates/results.json"),
            "exactly one restore/retry then accept-with-quarantine; no trusted-memory pollution",
        ),
        (
            "7. true-VLM / true-action demos",
            demo["passed"],
            link("videos/agentic_runtime_v1/demos/summary.json"),
            "watchmaker + driftwood; replay-backed audited action streams",
        ),
    ]
    gate_lines = [
        f"| {name} | **{yn(passed)}** | {evidence} | {note} |"
        for name, passed, evidence, note in gates
    ]
    overall = all(passed for _, passed, _, _ in gates)
    ci = track["evaluation"]["cluster_bootstrap_improvement_ci95"]
    lines = [
        "# Agentic Boundary Runtime v1 — engineering closure report",
        "",
        f"Status: **{'ENGINEERING GATES PASS' if overall else 'INCOMPLETE'}**.",
        "",
        "This report makes **no video-quality or consistency-improvement claim**. The demos show that the protocol, real VLM calls, bounded action selection, rollback, quarantine, and ledgers execute. Formal baseline/static/prompt-only/agent/matched-compute comparisons remain out of scope.",
        "",
        "## Delivered runtime",
        "",
        "`CausalShotSession` implements `pre_shot → decide → apply → snapshot → generate provisional → observe → verify → commit`, with one restore/repair retry and final accept-with-quarantine. Normative state is compiled before generation and exposed as an immutable as-of-shot view; visual and model-native payloads transition together from candidate to persistent or quarantine. Trusted reads return persistent payloads only.",
        "",
        "Bounded tools are validated locally after schema-constrained VLM decoding: append-only reviewed prompt patches; entity KV injection with at most four anchors and a region mask; V steering on layers `{0,7,14,21,29}` with alpha `{0.25,0.5,1.0}`; scene anchors; RoPE remap; logit bias `{0,1,2}`; quarantine; one retry; and abstention. Every action ledger entry carries the diagnostic fingerprint, params, reason, attention-mass check status, and result metrics.",
        "",
        "The pipeline parses `inference.agent_runtime` with `enabled=false` by default and exposes explicit session construction. v1 fails fast if active shot buffering is combined with streaming VAE. Legacy inference takes no agent branch when disabled.",
        "",
        "Snapshot inventory includes committed output latent, current-shot noise, positive/negative KV and cross-attention caches, KV-RAG transient/persistent state plus candidate/quarantine trusted partitions, indices, shot/RoPE position, scheduler, CPU/CUDA RNG, and streaming-VAE pending state/output cursor.",
        "",
        "## Acceptance gates",
        "",
        "| gate | status | evidence | scope |",
        "|---|:---:|---|---|",
        *gate_lines,
        "",
        f"Feature-off decoded hashes: {feature_hashes}.",
        "",
        "The snapshot gate uses the exact state adapter and deterministic same-noise regeneration harness over all four real scene specifications; it does not rerun the 5B model for every rollback cell. The feature-off gate is the full 5B decoded-frame render gate.",
        "",
        "## Track W — offline observer test",
        "",
        f"Decision: **{track['decision']}**. Track-W gate: **{yn(track['gate_pass'])}**; v1 therefore verifies every fixed shot boundary.",
        "",
        f"The 4-scene × 3-seed frozen probe corpus yielded {track['rows']} usable next-shot boundary records. Only V-norm drift was genuinely present. Attention entropy, sink share, model per-frame attention mass, and identity-word cross-attention mass were not in the frozen dumps (`cross_attention_calls=0`; masks were precomputed CLIPSeg). A CLIPSeg salience distribution was retained only as a named proxy and was not counted as requested telemetry.",
        "",
        f"LOSO MAE: shot-index baseline `{track['evaluation']['shot_index_baseline_mae']:.4f}`, available V signals `{track['evaluation']['available_signal_mae']:.4f}`. Paired improvement `{track['evaluation']['paired_mae_improvement']:+.4f}`; scene-seed cluster-bootstrap 95% CI `[{ci[0]:+.4f}, {ci[2]:+.4f}]`, which crosses zero.",
        "",
        "| held-out scene | n | shot-index MAE | V-signal MAE | improvement |",
        "|---|---:|---:|---:|---:|",
        *fold_lines,
        "",
        f"Evidence: {link('videos/agentic_runtime_v1/track_w/results.json')}.",
        "",
        "## True-VLM / true-action demonstrations",
        "",
        "The local Qwen3-VL-8B server performed authored-state compilation, boundary decisions, and current-shot verification through strict JSON schemas and exact fingerprint caches. All 50 VLM calls referenced by the final demonstrations (18 compiler, 16 controller, 16 verifier) have server-side schema-enforcement markers and matching cache fingerprints. Controller/verifier output is capped at 512 tokens, and the server enforces compact constrained JSON to eliminate whitespace-only decoder stalls. The decoded action side reuses previously rendered and audited full-vector V intervention streams; their runtime manifests prove the model action reached the graph. This is replay-backed integration, not a new quality experiment.",
        "",
        "| scene | shots | VLM-selected actions matching audited true action | quarantined shots | video | ledger | status |",
        "|---|---:|---:|---|---|---|:---:|",
        *demo_lines,
        "",
        "Development validation failures (immutable-state serialization, over-broad schemas, constrained-decoder whitespace, and the bounded-string performance probe) are retained under `videos/agentic_runtime_v1/demos/failures/`. The final locked driver completed on attempt 0; its wrapper permits exactly one operational retry and archives the failed attempt before retrying.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "/home/chendong/miniconda3/bin/python -m pytest -q tests --ignore=tests/test_vlm_optimizer_arms.py",
        "/home/chendong/miniconda3/bin/python scripts/run_agent_protocol_gates.py",
        "/home/chendong/miniconda3/bin/python scripts/evaluate_agent_track_w.py --device cuda:0",
        "tmux new-session -d -s agent_feature_off 'bash scripts/run_agent_feature_off_gate.sh'",
        "/home/chendong/miniconda3/bin/python scripts/audit_agent_feature_off.py",
        "tmux new-session -d -s agent_vlm_demo 'bash scripts/run_agent_vlm_demo_gate.sh'",
        "/home/chendong/miniconda3/bin/python scripts/audit_agent_ledger.py \\",
        "  videos/agentic_runtime_v1/demos/bearded_watchmaker_workshop/ledger.jsonl \\",
        "  videos/agentic_runtime_v1/demos/sandy_beach_driftwood/ledger.jsonl \\",
        "  --output videos/agentic_runtime_v1/demos/ledger_audit.json",
        "```",
        "",
        "Long jobs require a nonblocking driver/session lock and at least 60 GiB free GPU memory. Both GPU drivers check the threshold, archive a failed attempt, and retry at most once before fatal exit.",
        "",
        "## Known limitations",
        "",
        "- No quality benefit is claimed; the inherited V action has known uncontrolled sign.",
        "- Track W failed, so adaptive `when` triggering is disabled in favor of full verification at fixed boundaries.",
        "- The demonstration is shot-by-shot transaction/VLM replay over real pre-rendered action streams. Fresh 5B generation was used for feature-off equivalence, but the true-action demo did not rerender the intervention online.",
        "- Strict streaming is not implemented in v1. The action schema records forward-only compatibility, and the snapshot interface retains VAE pending/cursor fields for a later degradation mode.",
        "- The exact snapshot harness validates tensor/cache/RNG equivalence with deterministic scene cells; a larger full-model rollback render matrix is future hardening, not a v1 quality experiment.",
        "",
        "## Test result",
        "",
        "`264 passed` under the repository Miniconda environment, excluding the pre-existing heavyweight `test_vlm_optimizer_arms.py` integration module. Agent-specific coverage includes bounded domains, immutable normative as-of views, candidate transitions, quarantine invisibility, forced-fail retry budget, temporal fail-closed behavior, exact audited-action schemas, bounded verifier output, full snapshot inventory, and KV-RAG state restoration.",
    ]
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
