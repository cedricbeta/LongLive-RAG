# A script-derived reminder retains silver hair in one exposed video

Watch the [complete 60-second paired video](../../artifacts/native_ns235_continuation_cue_v1/full_60s_comparison.mp4) or [gallery with all 18 fixed times](../../artifacts/native_ns235_continuation_cue_v1/index.html). Left B uses the six original public ns235 prompts; right P appends exactly `Kai has silver hair.` to shots 4–6, starting at 30 seconds. Both use the same official LongLive-RAG 1.3B weights, timed adapter, seed 0, 832×480 resolution and 16 fps.

In the sampled 45–60-second views, B's hair is brown/dark blond while P's remains silver-white. Earlier 35–40-second samples already differ in color. This is a **cue-only retention effect**, with visible differences in actions, equipment and composition. Both versions become younger-looking than the initial older Kai and retain Ava in sampled later views despite the Kai-alone instruction. Silver hair does not establish identity or age preservation. All 960 frames are included in the comparison; assistant/developer review covered 18 fixed times per arm and three additional full-resolution views, not continuous motion or independent human evaluation.

**Two generations completed; zero local repairs, zero automatically accepted improvements and zero human labels.** The frozen plan allowed R/RP only if B showed clearly nonsilver hair in the 20–30-second edit interval. Frames 320/400/479 remain light silver/white or cream under warm lighting, without a clear conflict. R/RP were therefore not admitted; no mask, retry or replacement case was run. The planned four-cell local-repair × reminder factorial remains incomplete. P starts at 30 seconds, so it does not test a prompt rewrite within the local edit interval. This is not feedback superiority, budget-matched best-of-N, a held-out result or a success-rate estimate. The public script was selected before rendering, but its actors and narrative pattern were development-exposed.

The actual first 121 latent frames match exactly; the remaining 120 change. All five executed prompt-switch hashes match each arm's intended text, other generation settings match, and retrieval records through query 121 match. Native refresh, cursor and pool-length guards run; optional history/resident tensor hashes and raw VAE prefix pixel equality were not measured. Lossy MP4 pixels must not be assumed equal from latent equality. See [the pair check](../../results/native_ns235_continuation_cue_v1/baseline_pair_check.json), [observations](../../results/native_ns235_continuation_cue_v1/developer_observations.json) and [admission decision](../../results/native_ns235_continuation_cue_v1/repair_admission.json).

B and P take 548.90 and 518.19 seconds on GPUs 0/2, overlapping within 548.90 seconds. Their summed worker wall time is 1067.08 seconds, not kernel-active GPU time. Preparation takes 20.86 seconds; audits take 2.56/2.52 seconds; full comparison export takes 28.08 seconds. Raw logs, device telemetry and [costs](../../results/native_ns235_continuation_cue_v1/summary.json) are retained. No segmentation or semantic-judge call is hidden in this comparison.

## Reproduce B/P

Use the pinned official revision `973884a3cd3ad4b314c3d4ab42274c52e7a0b22a`, assets and native generation environment from [the fast-case instructions](../native_fast_cases/README.md). The new wrapper rebinds archived inputs to local paths and invokes the exact study preparer; it verifies the revision, four adapted source files and twelve assets. The original absolute paths are provenance only. Use a fresh pinned official checkout and a new output directory:

```bash
python adapters/official_longlive_rag/install.py --timed /path/to/fresh-official-checkout
python reproduction/native_ns235_continuation_cue/prepare.py \
  --repository . --checkout /path/to/fresh-official-checkout \
  --output /path/to/new-ns235-study
python /path/to/new-ns235-study/B_original/supervisor.py /path/to/new-ns235-study/B_original
python /path/to/new-ns235-study/P_cue/supervisor.py /path/to/new-ns235-study/P_cue
python implementation/scripts/audit_native_cue_case.py --folder /path/to/new-ns235-study/B_original
python implementation/scripts/audit_native_cue_case.py --folder /path/to/new-ns235-study/P_cue
python implementation/scripts/review_native_cue_study.py \
  --root /path/to/new-ns235-study --arms B_original P_cue \
  --output /path/to/new-ns235-study/baseline_review
```

Prepared supervisors use GPUs 0/2, require empty compute admission and refuse restarting a recorded attempt. Run independently only on available devices. Start the existing NVML observer before each supervisor to reproduce device telemetry; neither process uses GPU 3. Weights and raw latents are excluded from Git. The gallery includes both original encoded videos and the full comparison, with no shortened replacement for the complete clips.

## Scope of the unused local-cue adapter

The pure-Python helper derives only a literal hair-color sentence bound to an earlier named-actor definition and a narrow explicit actor-presence grammar. It preserves authored action text and adds no boundaries, reference images or new attributes. Unsupported or ambiguous presence is rejected; this is not a general natural-language presence detector. [21 helper tests](../../results/native_ns235_continuation_cue_v1/cue_unit_tests.txt) pass; they establish code contracts, not video quality.

A separate isolated installer and conditional pair preparer are included for review. They keep the original private repair sampler unchanged and would record authored timing separately from the effective suffix text. This integration was installed and source-checked but **never exercised in a real repair**, because admission failed. Both actual B/P videos instead use the existing timed runtime, with P's changed prompts explicitly in its configuration. Do not treat their success as execution coverage of the unused integration or alter the archived admission to force R/RP.

The earlier Kai classroom local repair remains a [separate fixed-target example](../native_kai_silver_hair/README.md): visible local silver color around 25–30 seconds, brown again by 35 seconds. Historical local 5B cases and latest official automatic progressive results retain their original scope; the latter still has zero retained improvements. Main fair comparisons, credible independent/human evaluation, multi-seed/held-out evidence and submission-ready conclusions remain outstanding.
