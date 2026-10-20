# Noah shaved-head sampler comparison

Watch the [complete 60-second three-way video](../../artifacts/native_noah_hair_sampler_v1/full_60s_three_way.mp4) or [gallery with all fixed samples](../../artifacts/native_noah_hair_sampler_v1/index.html). The [observations](../../results/native_noah_hair_sampler_v1/developer_observations.json) and [measured costs](../../results/native_noah_hair_sampler_v1/summary.json) describe the actual outcome. This is a separately frozen, post-baseline development choice of a real visible hair error. The previous Noah/Kai garment study remains closed with zero repairs; its targets and decisions were not changed.

**Both candidates remove much of the curly crown, but neither produces a clean persistent repair.** Dark forelock remnants, an elongated scalp or boundary artifacts remain in the first shot. Euler looks closer to shaved at the fixed 15–20-second views, but hair reappears in both candidates by the 25-second sample. Later poses, facial appearance and garment shading diverge; the baseline also has appearance drift and artifacts. Neither candidate is adopted, and this two-attempt study is closed. Generation takes 566.74/566.53 seconds concurrently; the full comparison includes all 960 frames from every branch. These observations are developer inspection of 18 fixed times, not independent acceptance or continuous-motion review.

Both candidates use official LongLive-RAG 1.3B, revision `973884a3cd3ad4b314c3d4ab42274c52e7a0b22a`, the same actual ns000 seed-0 parent, automatic binary mask, first-shot latent interval `[0,41)`, full authored prompt plus `Noah has a shaved head.`, private seed 4243, sigma 1.0 and four native timesteps. One uses native x0 prediction/re-noising; the other uses fixed initial private background noise and Euler flow updates. This jointly changes integration and noise refresh, and does not isolate either factor alone. Ordinary generation, full attention history, clean recache and subsequent retrieval are unchanged. It is not the old local 5B model or schedule.

There are exactly two candidates, one per mode, with no retries or candidate-dependent tuning. No automatic controller acceptance or human labels are claimed. The default sampler is not changed by this experiment. The [existing contracts](../../results/repair_sampler_contracts_v1/unit_tests.txt) and [native CUDA check](../../results/repair_sampler_contracts_v1/native_kernel_v2.json) cover sampler mechanics, not pretrained video quality. First-shot repair has an empty prefix, so this run cannot demonstrate protection of a nonempty prefix. Outside-mask latent equality does not guarantee unchanged decoded pixels. Optional history/resident byte hashes were not enabled.

## Reproduce the inputs and pair

Use the pinned assets/environment from [native fast-case reproduction](../native_fast_cases/README.md). Regenerate only the ns000 baseline using the [Noah/Kai preparation instructions](../native_sampler_cases/README.md), retaining its config, timing, latents, manifest and completed status. Large weights and raw latents are not Git assets. The original parent video and latent file hashes are recorded in [plan.json](../../results/native_noah_hair_sampler_v1/plan.json).

Create a fresh study directory and copy that plan into it. Set `parent` to your actual baseline directory and `parent_manifest_sha256` to the SHA-256 of its manifest; record this path-binding change separately and preserve the original plan. Require the regenerated parent video and latent hashes to match the recorded hashes before reusing these exact study claims. A differing parent is a new run requiring a new plan and mask. Do not edit archived results or statuses to force admission.

In a separate fresh official checkout, install the isolated sampler adapter:

```bash
python adapters/official_longlive_rag/install_repair_sampler.py /path/to/sampler-checkout
python implementation/scripts/preview_native_hair_mask.py \
  --plan /path/to/new-study/plan.json --entity-phrase person \
  --model /path/to/pinned-clipseg --output /path/to/new-study/person_preflight
python adapters/official_longlive_rag/prepare_mask.py \
  --parent /path/to/ns000/baseline --segment 0 --entity-phrase person \
  --part-phrase hair --exclude-phrase face --threshold 0.35 \
  --part-context entity_box --model /path/to/pinned-clipseg \
  --output /path/to/new-study/hair_mask
```

CLIPSeg is `CIDAS/clipseg-rd64-refined` revision `999e0328d9e10b484360c477313983f9afdd7050`; all model hashes are in the mask manifest. The original `man` preview produced 0/54/0 cells; one separately documented `person` preview produced 65/62/74 cells at frames 0/80/159. No manual pixel coordinates were used. The full mask has 41 nonempty latent rows and mean coverage 0.0123632. Inspect actual overlays before recording a fresh `mask_admission.json` with decision `render_pair`, hashes, reviewer type and coverage limitations. The archived admission is developer inspection, not ground truth. The exact archived mask tensor is also provided as `automatic_hair_mask.tensor`; do not apply it to a different parent.

With the native generation Python, prepare once and launch each supervisor once:

```bash
python implementation/scripts/prepare_noah_hair_sampler_pair.py \
  --root /path/to/new-study --checkout /path/to/sampler-checkout \
  --adapter adapters/official_longlive_rag --gpu-native 0 --gpu-euler 2
python /path/to/new-study/sampler_pair/native/supervisor.py /path/to/new-study/sampler_pair/native
python /path/to/new-study/sampler_pair/euler/supervisor.py /path/to/new-study/sampler_pair/euler
```

Launch the existing NVML observer before each supervisor to reproduce device-energy measurements. The two supervisors may run concurrently on separately admitted idle GPUs; GPU 3 is reserved. Each enforces a 1,800-second bound, source/config checks and parent-bound preparation. Do not restart an already recorded attempt.

After completion, run `implementation/scripts/audit_native_fast_repair.py --folder ... --parent ...` for each actual candidate, then `implementation/scripts/review_native_sampler_pair.py --root /path/to/new-study`. The exported study also includes the original orchestration source and commands for traceability; its absolute paths must be rebound to local paths, not executed unchanged. The comparison includes all 960 frames, and all 54 fixed images remain available. Cost totals distinguish reused baseline cost, mask/preparation, concurrent rendering, nested private sampling and CPU review. They are not a complete end-to-end human evaluation cost.
