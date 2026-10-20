# Kai classroom: local repair and a later literal reminder

The [full 60-second four-way video](../../artifacts/native_kai_retention_factorial_v1/full_60s_comparison.mp4) places **B/P above and R/RP below**, with native 832×480 images in each pane. [All 18 fixed times](../../artifacts/native_kai_retention_factorial_v1/index.html), [observations](../../results/native_kai_retention_factorial_v1/developer_observations.json), [costs](../../results/native_kai_retention_factorial_v1/summary.json) and [actual paired checks](../../results/native_kai_retention_factorial_v1/pair_check.json) report the completed outcome separately from this protocol.

**Observed outcome:** RP is silver-white at the 25-second and approximately 30-second samples and remains so in the later 35–60-second samples, while R is brown again by 35 seconds. P also becomes silver by the 35-second sample and retains it later. The clearest added local benefit is therefore the earlier 25–30-second correction under this timing protocol. Later retention also occurs without local repair. RP has visible fragments and soft face detail near 39.94 seconds, and later paper handling/body poses differ; age and Ava-departure errors remain. There is no automatic or independent acceptance. Two new generations took 523.22/570.82 seconds, with the latter including 56 extra private forwards.

This is a new, explicitly post-outcome development comparison. B is the previously completed ns280 seed-0 official LongLive-RAG 1.3B classroom baseline. R is the previously completed single native local hair-color repair, which visibly improves sampled 25–30-second hair color before brown recurs. Both outputs are reused unchanged. Two new arms are allowed once each:

| Arm | Local 20–30-second repair | Cue at 30/40/50-second boundaries |
| --- | --- | --- |
| B, archived | No | No |
| P, new | No | `Kai has silver hair.` |
| R, archived | Identical frozen hair mask, original third prompt plus silver sentence, sigma 1, private seed 4245 | No |
| RP, new | Exactly the archived R action | Exactly the P cue |

The purpose is to distinguish local color correction from the effect of simply restating an authored attribute. The cue is derived from the original named-actor description, preserves each later prompt verbatim, and adds no reference image or new prompt boundary. Both new arms receive the same later text. No mask is recomputed, no private seed or sampler is changed, and no candidate is selected by an evaluator. All attempted generations, failures and costs are retained. Older Kai, Noah, Iris and ns235 budgets remain closed; this comparison is neither a replacement ns235 case nor a held-out result.

This controls later conditioning, not every alternative repair strategy. P begins at 30 seconds and therefore cannot change the earlier local interval. The study does **not** compare local repair against a full-shot prompt rewrite at 20 seconds, budget-matched fixed retry, best-of-N or an adaptive feedback policy. Any local temporal advantage must be described with that timing limitation. Identity, age, other characters, actions and scene damage remain separate from the hair-color target.

## State and execution checks

P uses the same timed runtime as B. RP uses the isolated cue installer. Preparation proves that its pipeline and runner equal the exact frozen transformation of R's native runtime; the private repair adapter is byte-identical. The original parent/config/timing/video/latent/mask bindings and twelve model assets are validated before GPU admission. Actual P/B and RP/R prefixes through latent 120 must match exactly; a mismatch invalidates paired attribution and triggers no retry.

The repair validator binds authored timing to the parent. `timing_0000.json` consequently retains those authored prompts in RP; it must not be reported as the effective text. `continuation_cue_0000.json` records the intervention, and all actual prompt-switch hashes are checked against the effective schedule. The pair check also verifies identical R/RP local requests apart from the cue, shared non-intervention settings and both cue logs/action hashes. Native prefix, outside-mask, RNG, clean-recache, cursor and retrieval-pool guards remain active. Optional resident/history byte hashes and raw-VAE prefix pixel checks remain unmeasured; latent equality does not imply equality of compressed video pixels or semantic preservation.

Helper tests are the unchanged [21 tests already executed in the ns235 study](../../results/native_ns235_continuation_cue_v1/cue_unit_tests.txt). Their narrow presence grammar is not a general natural-language actor-presence detector. The completed new RP run is the first real local repair to execute this cue integration. No test result supplies quality acceptance or human labels.

## Reproduce

Use the pinned official revision `973884a3cd3ad4b314c3d4ab42274c52e7a0b22a`, native environment and assets from [the fast-case instructions](../native_fast_cases/README.md). First obtain the exact B and R files following [the Kai local repair reproduction](../native_kai_silver_hair/README.md). Retain completed status, manifests, configurations, timing, audits, videos and raw latents. The published original mask tensor may only be reused with its matching parent hashes. Different images require a new mask/study, not a silent rebinding. Large raw latents and model weights are not in Git.

Use a separate fresh official checkout for the opt-in cue renderer. From this delivery repository, using the native generation Python:

```bash
python adapters/official_longlive_rag/install_continuation_cue.py /path/to/fresh-cue-checkout
python implementation/scripts/prepare_native_retention_pair.py \
  --root /path/to/new-retention-study \
  --baseline /path/to/ns280/baseline --local-repair /path/to/kai/candidate \
  --mask /path/to/kai/hair_mask/mask.pt \
  --checkout /path/to/fresh-cue-checkout --adapter adapters/official_longlive_rag
python /path/to/new-retention-study/P_cue/supervisor.py /path/to/new-retention-study/P_cue
python /path/to/new-retention-study/RP_local_cue/supervisor.py /path/to/new-retention-study/RP_local_cue
python implementation/scripts/audit_native_retention_cue.py --folder /path/to/new-retention-study/P_cue
python implementation/scripts/audit_native_fixed_target.py \
  --folder /path/to/new-retention-study/RP_local_cue --parent /path/to/ns280/baseline
python implementation/scripts/audit_native_retention_pair.py --root /path/to/new-retention-study
python implementation/scripts/review_native_retention_fourway.py \
  --root /path/to/new-retention-study --arms B_original P_cue R_local RP_local_cue \
  --output /path/to/new-retention-study/four_way_review
```

The preparer checks the existing runs and creates read-only reference symlinks for B/R; never execute supervisors or audits through those references. It freezes a new plan before preparing the two new arms. Supervisors use idle GPU 0/2 with source/configuration checks and no silent restart; GPU 3 remains reserved. Start the existing NVML observer before each supervisor for telemetry. Absolute paths in archived records are provenance, while command arguments above provide portable entry points. Reproduction requires the original B and R native runtime directories as well as their outputs so their source binding can be checked.

New and previously incurred costs are reported separately. Reusing B/R avoids rerendering them but does not erase their original generation, segmentation or preparation costs. Fixed samples are assistant/developer observations, not continuous-motion review or anonymous independent human labels. Latest official progressive acceptance and submission readiness are separate from this small factorial diagnostic.
