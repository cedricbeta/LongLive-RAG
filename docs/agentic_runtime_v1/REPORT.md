# Agentic Boundary Runtime v1 — engineering closure report

Status: **ENGINEERING GATES PASS**.

This report makes **no video-quality or consistency-improvement claim**. The demos show that the protocol, real VLM calls, bounded action selection, rollback, quarantine, and ledgers execute. Formal baseline/static/prompt-only/agent/matched-compute comparisons remain out of scope.

## Delivered runtime

`CausalShotSession` implements `pre_shot → decide → apply → snapshot → generate provisional → observe → verify → commit`, with one restore/repair retry and final accept-with-quarantine. Normative state is compiled before generation and exposed as an immutable as-of-shot view; visual and model-native payloads transition together from candidate to persistent or quarantine. Trusted reads return persistent payloads only.

Bounded tools are validated locally after schema-constrained VLM decoding: append-only reviewed prompt patches; entity KV injection with at most four anchors and a region mask; V steering on layers `{0,7,14,21,29}` with alpha `{0.25,0.5,1.0}`; scene anchors; RoPE remap; logit bias `{0,1,2}`; quarantine; one retry; and abstention. Every action ledger entry carries the diagnostic fingerprint, params, reason, attention-mass check status, and result metrics.

The pipeline parses `inference.agent_runtime` with `enabled=false` by default and exposes explicit session construction. v1 fails fast if active shot buffering is combined with streaming VAE. Legacy inference takes no agent branch when disabled.

Snapshot inventory includes committed output latent, current-shot noise, positive/negative KV and cross-attention caches, KV-RAG transient/persistent state plus candidate/quarantine trusted partitions, indices, shot/RoPE position, scheduler, CPU/CUDA RNG, and streaming-VAE pending state/output cursor.

## Acceptance gates

| gate | status | evidence | scope |
|---|:---:|---|---|
| 1. feature-off decoded-bit equality | **PASS** | [audit.json](../../videos/agentic_runtime_v1/feature_off/audit.json) | 4 pilot scenes, seed 0, 480 decoded frames each |
| 2. snapshot/restore no-action equality | **PASS** | [results.json](../../videos/agentic_runtime_v1/protocol_gates/results.json) | 2 boundaries per scene; latent and post-generation cache bit equality |
| 3. quarantine invisible | **PASS** | [results.json](../../videos/agentic_runtime_v1/protocol_gates/results.json) | unit + deterministic end-to-end partition assertion |
| 4. no future leak | **PASS** | [ledger_audit.json](../../videos/agentic_runtime_v1/demos/ledger_audit.json) | session ledgers plus VLM transcript source/hash availability audit |
| 5. ledger replay | **PASS** | [ledger_audit.json](../../videos/agentic_runtime_v1/demos/ledger_audit.json) | 3 manifest-seeded random actions, offline cache replay with an unreachable endpoint |
| 6. retry budget forced fail | **PASS** | [results.json](../../videos/agentic_runtime_v1/protocol_gates/results.json) | exactly one restore/retry then accept-with-quarantine; no trusted-memory pollution |
| 7. true-VLM / true-action demos | **PASS** | [summary.json](../../videos/agentic_runtime_v1/demos/summary.json) | watchmaker + driftwood; replay-backed audited action streams |

Feature-off decoded hashes: `bearded_watchmaker_workshop` `c655ee66d5cc…`, `brown_bear_river` `77c755a8c8e4…`, `sandy_beach_driftwood` `cb8980f83b1a…`, `tattooed_noodle_chef` `36e340bdfb3d…`.

The snapshot gate uses the exact state adapter and deterministic same-noise regeneration harness over all four real scene specifications; it does not rerun the 5B model for every rollback cell. The feature-off gate is the full 5B decoded-frame render gate.

## Track W — offline observer test

Decision: **downgrade_fixed_boundary_full_verification**. Track-W gate: **FAIL**; v1 therefore verifies every fixed shot boundary.

The 4-scene × 3-seed frozen probe corpus yielded 76 usable next-shot boundary records. Only V-norm drift was genuinely present. Attention entropy, sink share, model per-frame attention mass, and identity-word cross-attention mass were not in the frozen dumps (`cross_attention_calls=0`; masks were precomputed CLIPSeg). A CLIPSeg salience distribution was retained only as a named proxy and was not counted as requested telemetry.

LOSO MAE: shot-index baseline `0.1577`, available V signals `0.1530`. Paired improvement `+0.0047`; scene-seed cluster-bootstrap 95% CI `[-0.0178, +0.0256]`, which crosses zero.

| held-out scene | n | shot-index MAE | V-signal MAE | improvement |
|---|---:|---:|---:|---:|
| `bearded_watchmaker_workshop` | 21 | 0.1990 | 0.1722 | +0.0268 |
| `brown_bear_river` | 21 | 0.1563 | 0.1379 | +0.0184 |
| `sandy_beach_driftwood` | 13 | 0.1702 | 0.1548 | +0.0154 |
| `tattooed_noodle_chef` | 21 | 0.1103 | 0.1479 | -0.0376 |

Evidence: [results.json](../../videos/agentic_runtime_v1/track_w/results.json).

## True-VLM / true-action demonstrations

The local Qwen3-VL-8B server performed authored-state compilation, boundary decisions, and current-shot verification through strict JSON schemas and exact fingerprint caches. All 50 VLM calls referenced by the final demonstrations (18 compiler, 16 controller, 16 verifier) have server-side schema-enforcement markers and matching cache fingerprints. Controller/verifier output is capped at 512 tokens, and the server enforces compact constrained JSON to eliminate whitespace-only decoder stalls. The decoded action side reuses previously rendered and audited full-vector V intervention streams; their runtime manifests prove the model action reached the graph. This is replay-backed integration, not a new quality experiment.

| scene | shots | VLM-selected actions matching audited true action | quarantined shots | video | ledger | status |
|---|---:|---:|---|---|---|:---:|
| `bearded_watchmaker_workshop` | 8 | 3 | [] | [2-up](../../videos/agentic_runtime_v1/demos/bearded_watchmaker_workshop/two_up.mp4) | [ledger](../../videos/agentic_runtime_v1/demos/bearded_watchmaker_workshop/ledger.jsonl) | PASS |
| `sandy_beach_driftwood` | 8 | 5 | [] | [2-up](../../videos/agentic_runtime_v1/demos/sandy_beach_driftwood/two_up.mp4) | [ledger](../../videos/agentic_runtime_v1/demos/sandy_beach_driftwood/ledger.jsonl) | PASS |

Development validation failures (immutable-state serialization, over-broad schemas, constrained-decoder whitespace, and the bounded-string performance probe) are retained under `videos/agentic_runtime_v1/demos/failures/`. The final locked driver completed on attempt 0; its wrapper permits exactly one operational retry and archives the failed attempt before retrying.

## Reproduction

```bash
/home/chendong/miniconda3/bin/python -m pytest -q tests --ignore=tests/test_vlm_optimizer_arms.py
/home/chendong/miniconda3/bin/python scripts/run_agent_protocol_gates.py
/home/chendong/miniconda3/bin/python scripts/evaluate_agent_track_w.py --device cuda:0
tmux new-session -d -s agent_feature_off 'bash scripts/run_agent_feature_off_gate.sh'
/home/chendong/miniconda3/bin/python scripts/audit_agent_feature_off.py
tmux new-session -d -s agent_vlm_demo 'bash scripts/run_agent_vlm_demo_gate.sh'
/home/chendong/miniconda3/bin/python scripts/audit_agent_ledger.py \
  videos/agentic_runtime_v1/demos/bearded_watchmaker_workshop/ledger.jsonl \
  videos/agentic_runtime_v1/demos/sandy_beach_driftwood/ledger.jsonl \
  --output videos/agentic_runtime_v1/demos/ledger_audit.json
```

Long jobs require a nonblocking driver/session lock and at least 60 GiB free GPU memory. Both GPU drivers check the threshold, archive a failed attempt, and retry at most once before fatal exit.

## Known limitations

- No quality benefit is claimed; the inherited V action has known uncontrolled sign.
- Track W failed, so adaptive `when` triggering is disabled in favor of full verification at fixed boundaries.
- The demonstration is shot-by-shot transaction/VLM replay over real pre-rendered action streams. Fresh 5B generation was used for feature-off equivalence, but the true-action demo did not rerender the intervention online.
- Strict streaming is not implemented in v1. The action schema records forward-only compatibility, and the snapshot interface retains VAE pending/cursor fields for a later degradation mode.
- The exact snapshot harness validates tensor/cache/RNG equivalence with deterministic scene cells; a larger full-model rollback render matrix is future hardening, not a v1 quality experiment.

## Test result

`264 passed` under the repository Miniconda environment, excluding the pre-existing heavyweight `test_vlm_optimizer_arms.py` integration module. Agent-specific coverage includes bounded domains, immutable normative as-of views, candidate transitions, quarantine invisibility, forced-fail retry budget, temporal fail-closed behavior, exact audited-action schemas, bounded verifier output, full snapshot inventory, and KV-RAG state restoration.
