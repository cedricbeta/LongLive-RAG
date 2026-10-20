The probe reuses ns040 seed 0 from the [twelve-script pilot](../native_development_pilot12/README.md). All variants use official LongLive-RAG 1.3B with six continuous ten-second prompts, 832×480, 16 fps. The target was selected after seeing the original. It is fixed-target development evidence, with no automatic, independent or human acceptance.

| Arm | Actual intervention |
|---|---|
| B | Reused original, no new generation. |
| P | Append the standing sentence only to the ordinary 10–20-second prompt. |
| R | One private masked replay over latents [41,81), strength 1.0, seed 4245; same strengthened prompt. Ordinary authored prompts and later text stay unchanged. |

The appendix is `Nora and Ethan are both standing upright on their feet throughout this interaction.` It elaborates the existing joint standing requirement. There are no new hair, clothing or identity specifications. Both variants start changing conditioning at ten seconds. Different ordinary versus private denoising paths mean this is not an isolated mask ablation or a budget-matched comparison.

CLIPSeg segments `people` in all 160 original frames from ten to twenty seconds. At each latent, the four associated frame masks are united and enclosed by a program rectangle padded by 5% horizontally and 15% vertically. Three published overlays were reviewed before inspecting either new output. Faces are included because posture can move the whole body. The regions also include parts of the table, props and background; mask provenance does not establish semantic protection.

The maximum is one P generation, one R generation, and one segmentation attempt. No replacement mask, seed, prompt, sampler or case is allowed after outcomes. The mask, all attempted outputs, full later video, source bindings and measured costs are retained in the [records](../../results/native_ns040_posture_probe_v1/summary.json).

The preceding [planner preflight](../../results/narrastream_action_preflight_v1/official_narrastream_planner_preflight_v2/summary.json) made two fixed text calls with Qwen3.5-27B revision `fc05daec18b0a78c049392ed2e771dde82bdf654`. Its public weights were hash verified; local BF16 greedy decoding used the official extractor prompt/parser with thinking disabled. It did not reproduce the original remote provider or its JSON-constrained decoder. The initial download timeout and CPU import failures are retained separately.

The subsequent [two-call visual diagnostic](../../results/narrastream_action_preflight_v1/native_pilot12_action_diagnostic_v1/summary.json) used Qwen3-VL-30B-A3B-Instruct revision `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`. It gave contradictory hand-contact statements in ns040 and credited a visible held remote as a completed retrieval in ns130. Its raw labels are not validated successes or official NarraStream scores. Both diagnostic budgets are closed. The posture probe does not convert those outputs into an automatic admission.

Use the [native pilot setup](../native_development_pilot12/README.md) to regenerate the complete parent, including its original latent tensor, and the [native repair setup](../official_repair/README.md) for the repair checkout. The parent and candidate must pass the existing exact source/config/weight checks. Obtain public `CIDAS/clipseg-rd64-refined` at revision `999e0328d9e10b484360c477313983f9afdd7050`. Set paths for your machine; the archived runtime absolute paths are provenance only.

```bash
# REPO: this repository; NATIVE_PYTHON: the documented native environment.
# PARENT: regenerated pilot/ns040/baseline; REPAIR_CHECKOUT: installed native repair checkout.
# CLIPSEG: the pinned local snapshot; PROBE: a fresh output directory.
"$NATIVE_PYTHON" "$REPO/implementation/scripts/prepare_native_posture_probe.py" \
  --root "$PROBE" --parent "$PARENT" --repair-checkout "$REPAIR_CHECKOUT" \
  --clipseg "$CLIPSEG" \
  --supervisor "$REPO/results/native_ns040_posture_probe_v1/P_prompt/supervisor.py"
"$NATIVE_PYTHON" "$REPO/implementation/scripts/prepare_native_group_mask.py" --root "$PROBE"
"$NATIVE_PYTHON" "$PROBE/P_prompt/supervisor.py" "$PROBE/P_prompt"
```

Check the three mask previews and the predeclared geometry gate before admitting R. This developer review is part of the disclosed fixed-target procedure. With an admitted region, prepare R using the existing adapter:

```bash
"$NATIVE_PYTHON" "$REPO/adapters/official_longlive_rag/prepare_repair.py" \
  --parent "$PARENT" --parent-branch native_ns040_seed0 --checkout "$REPAIR_CHECKOUT" \
  --output "$PROBE/R_group" --assets-manifest "$PROBE/assets_manifest.json" \
  --mask "$PROBE/group_mask/mask.pt" --prompt "$PROBE/repair_prompt.txt" \
  --noise-seed 4245 --gpu 2 --timeout 1800
```

Copy the study plan into R and bind its SHA-256 in R's manifest before launch. This is provenance metadata; it does not change the generated condition:

```bash
"$NATIVE_PYTHON" - "$PROBE" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
shutil.copyfile(root / 'plan.json', root / 'R_group/plan.json')
path = root / 'R_group/manifest.json'
record = json.loads(path.read_text())
record['study_plan_sha256'] = hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest()
path.write_text(json.dumps(record, indent=2) + '\n')
PY
"$NATIVE_PYTHON" "$PROBE/R_group/supervisor.py" "$PROBE/R_group"
```

The original study starts the [NVML observer](../../implementation/scripts/monitor_native_gpu.py) before each supervisor; retain its samples and distinguish wall time, utilization proxy and device energy. Use free GPU 0/2 and preserve GPU 3.

After both attempts complete, run the [fixed-target auditor](../../implementation/scripts/audit_native_fixed_target.py), [paired input checks](../../implementation/scripts/audit_native_posture_probe.py), and [full video exporter](../../implementation/scripts/review_native_posture_probe.py):

```bash
"$CPU_PYTHON" "$REPO/implementation/scripts/audit_native_fixed_target.py" \
  --folder "$PROBE/R_group" --parent "$PARENT"
"$CPU_PYTHON" "$REPO/implementation/scripts/audit_native_posture_probe.py" --root "$PROBE"
"$CPU_PYTHON" "$REPO/implementation/scripts/review_native_posture_probe.py" \
  --root "$PROBE" --arms B_original P_prompt R_group --output "$PROBE/review"
```

`CPU_PYTHON` is the documented CPU audit environment. The preparer creates `PROBE/B_original` as a reference to the complete parent. The exporter preserves all 960 source frames and the 18 frozen review times per arm; encoded pixels can differ despite equal parent latents.

The comparison gallery and raw observations support only the inspected qualitative statements. They do not complete the twelve-script multi-strategy pilot, establish identity preservation, or establish superiority to fixed retry or Best-of-N. Historical 5B results and latest official automatic progressive remain separate.
