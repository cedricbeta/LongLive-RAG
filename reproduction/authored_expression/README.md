# Script-bound garment action and paired-review diagnostics

This opt-in action appends an authored color/type phrase only when active script quotes and the current shot bind it unambiguously under the implemented text rules. Default behavior remains `literal_clauses_v1`; select `--target-expression authored_garment_v1` in the [native progressive workflow](../native_progressive/MIXED_EVIDENCE_ROUTING.md). The compiler's semantic entity/pronoun attribution is not certified by these rules. No acceptance gate changes.

Read the [measured development record](../../docs/authored-expression-and-paired-review-20260917.zh.md). The actual CPU bridge reproduces a previously generated native action; it does not run a visual planner, generate another video or accept that candidate. The two GPU diagnostic protocols are not adopted: the joint task fails its schema and an identical-input control, while the shorter task invents outfits on uniform gray images.

## Offline evidence verification

From the repository root use Python **3.10**, Torch, PyYAML, NumPy, Pillow 11.1.0 and OpenCV 4.11.0.86. Python 3.10 preserves the recorded monitor's exact floating-point summation. No model weights or GPU are needed.

```bash
python implementation/scripts/verify_authored_expression_export.py \
  --root . --output /tmp/authored-expression-audit.json
cmp /tmp/authored-expression-audit.json \
  results/authored_expression_policy_contracts/export_audit.json
```

This checks the 86 current test-bound files and explicit historical source mapping, all three bridge attempts, final policy derivation, actual mask/action fingerprints, published source-video pixels, 39 raw responses, nine unattempted cases, telemetry and byte-exact summary/gallery. It does not regenerate missing raw parent latents, reload weights or establish semantic accuracy. The [current test command and log](../../results/authored_expression_policy_contracts/exported_tests.json) record 255 tests plus 11 subtests.

## Repeat the real CPU bridge

Regenerate the [official Iris baseline](../official_timed_60s/README.md) and retain its raw latents; install the [native repair adapter](../official_repair/README.md) into a separate official checkout. Set `PARENT_RUN`, `NATIVE_CHECKOUT` and a fresh `BRIDGE_RUN` to those paths. The archived mask is valid only for its exact hashed parent; a different regenerated parent requires a newly derived mask and a new experiment, not silently changed provenance.

```bash
python implementation/scripts/probe_authored_expression_bridge.py \
  --feedback results/native_iris_mixed_evidence_v1/run \
  --candidate results/native_iris_natural_expression_v1/candidate \
  --parent "$PARENT_RUN" --checkout "$NATIVE_CHECKOUT" \
  --mask reproduction/native_expression_control/automatic_mask.tensor \
  --assets-manifest results/native_iris_natural_expression_v1/assets_manifest.json \
  --output "$BRIDGE_RUN" --gpu 0
```

Use the native environment from the baseline reproduction; `--gpu` is recorded in preparation only. This command does not execute the prepared generator. It programmatically selects the sole legal expression and checks the first action against the archived actual action. v1 failed after preparation due to a result-field mismatch; v2 completed before final source hardening; v3 freezes and verifies the final six bridge source files before/after execution. All versions remain available.

## Repeat the paired diagnostics

Use the pinned Molmo model and Python 3.12.2 model environment from [native observations](../native_observations/README.md). Set `MODEL_PATH`, `MODEL_PYTHON`, and fresh `PAIR_RUN` / `SHORT_RUN`. `MODEL_PYTHON` must retain its environment executable path. Canonical paired code differs from the first frozen runner only by an optional validator hook used by the shorter task; each published plan binds its actual frozen runner.

```bash
"$MODEL_PYTHON" implementation/scripts/probe_native_paired_review.py --prepare \
  --root . --folder "$PAIR_RUN" --model-path "$MODEL_PATH" \
  --python "$MODEL_PYTHON" --gpu 1
```

Start `monitor_native_gpu.py` with a fresh telemetry folder in a second terminal as documented in native observations, then execute `"$MODEL_PYTHON" "$PAIR_RUN/runner.py" --folder "$PAIR_RUN"` once. The plan freezes 12 cases, a 900-second deadline and a three-consecutive-invalid stop. Do not edit an archived plan or replace failed calls. After the observer exits, use `audit_native_paired_reviews.py --root . --folder "$PAIR_RUN" --telemetry "$PAIR_TELEMETRY" --output "$PAIR_RUN/audit.json"` under Python 3.10.

Preparation reads the pinned model manifest from the repository's original observation folder and verifies all model files. The shorter probe explicitly follows a failed joint task. Use the **new** joint run above as its reference, carrying the correct local model paths. Its preparer requires an actual terminal result with three malformed calls; if the new joint run behaves differently, stop and record it rather than forcing the archived outcome.

```bash
"$MODEL_PYTHON" implementation/scripts/probe_native_pair_descriptions.py --prepare \
  --reference "$PAIR_RUN" --folder "$SHORT_RUN"
```

Use a fresh observer and execute `"$MODEL_PYTHON" "$SHORT_RUN/runner.py" --folder "$SHORT_RUN"` once. The shorter task uses two images, a 512-token ceiling and 36 cases. The joint task uses 6/10 images and a 2048-token ceiling. Both use seed 0 per call, bf16/SDPA, greedy decoding and no parsing retries. The shorter prompt supplies no target or branch identity. Input count, prompt, output schema and token ceiling change jointly, so this is not a single-factor ablation.

Admission is limited to idle GPUs 0/1/2 on the recorded host; GPU 3 is reserved. Observers are read-only. New paths require new frozen plans; historical absolute paths are provenance only. The complete gallery is [here](../../artifacts/native_iris_paired_reviews/index.html), intended for local browser viewing after checkout.
