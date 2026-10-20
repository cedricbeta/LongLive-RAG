# Kai silver-hair fixed-target probe

Watch the [complete 60-second comparison](../../artifacts/native_kai_silver_hair_v1/full_60s_comparison.mp4) and [all fixed views](../../artifacts/native_kai_silver_hair_v1/index.html). See [developer observations](../../results/native_kai_silver_hair_v1/developer_observations.json) and [actual costs](../../results/native_kai_silver_hair_v1/summary.json) for the outcome. This is a separately frozen, post-baseline development probe, not automatic acceptance or held-out evaluation. The earlier Kai garment batch remains closed with zero repairs.

**There is visible local color improvement:** Kai's hair changes from light brown to silver/white at the fixed 25-second and approximately 30-second views, while Ava's hair remains blonde and the broad scene arrangement stays close. The 20-second onset remains brown, and Kai's hair is brown again by the 35-second sample. Later pose/paper-handling differences and pre-existing age/presence errors remain. Keep this as a qualitative fixed-target capability example, not a clean persistent repair or an automatically accepted result. The sole candidate takes 535.08 seconds; its budget is closed.

All 81 earlier latent frames match exactly, as do edited-shot latents outside the mask. This does **not** establish a pixel-exact prefix: decoded lossy MP4 frames 281–319 differ, with maximum per-frame mean absolute difference 1.583 on the 0–255 scale. No raw VAE prefix comparison was run and the cause of those encoded differences was not isolated. Optional history/resident byte hashes remain unmeasured.

The reused ns280 seed-0 original changes from an older silver-haired man to a younger man with light brown hair. This probe targets only the third-shot hair color, preserving the original prompt and appending `Kai has silver hair.`. Existing age drift and unwanted later Ava presence remain unresolved requirements. A local color change must not be presented as fixing identity or the full script.

The generator is official LongLive-RAG 1.3B, revision `973884a3cd3ad4b314c3d4ab42274c52e7a0b22a`, with the disclosed timing and native masked-replay adapters. The sampler is unchanged native x0/re-noise, full historical attention, four native steps, sigma 1.0, private seed 4245. The edited latent interval is `[81,121)`; the first 81 latents belong to the parent prefix, and the full later 30 seconds are regenerated. There is one candidate, no retries, no candidate-dependent mask/prompt tuning and no model-weight update.

## Reproduction

Use the pinned environment/assets from [native fast-case reproduction](../native_fast_cases/README.md). Regenerate only the ns280 baseline using [the existing baseline preparation](../native_sampler_cases/README.md); retain config, timing, latents, manifest and completed status. The [study plan](../../results/native_kai_silver_hair_v1/plan.json) binds the original parent video and latent hashes. Copy it into a fresh study directory, rebind only `parent` and its manifest hash to actual local files, and record that path-binding change separately. A different parent hash requires a new plan and mask. Weights and large raw latents are not Git assets.

Use `implementation/scripts/preview_native_hair_mask.py` with that plan, entity phrase `man in a brown coat`, and the pinned CLIPSeg model. The first `man` preview produced 56/0/27 cells at frames 320/400/479. The one allowed grounded phrase produced 61/85/54; both preflights and all raw arrays are retained. No threshold sweep or manual pixel coordinates were used. The full mask uses `CIDAS/clipseg-rd64-refined` revision `999e0328d9e10b484360c477313983f9afdd7050`:

```bash
python adapters/official_longlive_rag/prepare_mask.py \
  --parent /path/to/ns280/baseline --segment 2 \
  --entity-phrase 'man in a brown coat' --part-phrase hair \
  --exclude-phrase face --threshold 0.35 --part-context entity_box \
  --model /path/to/pinned-clipseg --output /path/to/new-study/hair_mask
```

Inspect the actual full-mask overlays before admission; the archived inspection is not ground truth and cannot be copied to certify different images. Use a separate fresh pinned official checkout with `install_repair.py`, and prepare with the canonical native-generation Python:

```bash
python adapters/official_longlive_rag/prepare_repair.py \
  --parent /path/to/ns280/baseline --parent-branch native_ns280_seed0 \
  --checkout /path/to/native-repair-checkout --output /path/to/new-study/candidate \
  --assets-manifest results/native_kai_silver_hair_v1/assets_manifest.json \
  --mask /path/to/new-study/hair_mask/mask.pt \
  --prompt results/native_kai_silver_hair_v1/repair_prompt.txt \
  --noise-seed 4245 --gpu 2 --timeout 1800
```

Copy the fresh study plan to `candidate/plan.json`, start the existing NVML observer, then run the generated `candidate/supervisor.py` once with the candidate directory as its argument. It checks actual source/configuration and admits only an idle GPU; GPU 3 is reserved. Do not silently restart an attempt.

After completion, use `implementation/scripts/audit_native_fixed_target.py --folder ... --parent ...`, then `implementation/scripts/review_native_fixed_target.py --root /path/to/new-study`. The auditor's only code change from the earlier fast-case auditor is conditioning a limitation sentence on actual prefix length; measured checks remain the same. It is exported under a distinct name to preserve older source bindings. History/resident byte hashes remain unmeasured. Latent equality is not a semantic or decoded-pixel guarantee.

The exact archived automatic tensor is included as `automatic_hair_mask.tensor`; never reuse it on a differing parent. Raw command paths are provenance and must be rebound locally. The full comparison includes all 960 source frames and 36 predeclared images. Fixed-image developer inspection is not independent human annotation or continuous-motion evaluation. Costs include reused baseline cost, both preflights, full mask, preparation, generation, nested private sampling and CPU audit/review, while untimed developer/publication work is disclosed.
