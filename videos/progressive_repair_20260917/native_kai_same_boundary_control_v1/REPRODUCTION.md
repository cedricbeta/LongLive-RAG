# A prompt-only control starting at the same 20-second boundary

Watch the [complete 60-second comparison](../../artifacts/native_kai_same_boundary_control_v1/full_60s_comparison.mp4) or [all fixed views](../../artifacts/native_kai_same_boundary_control_v1/index.html). The upper row is original B and cue-only P30, which begins at 30 seconds. The lower row is the new cue-only P20, which begins at 20 seconds, and the archived local-repair-plus-cue RP. See [observations](../../results/native_kai_same_boundary_control_v1/developer_observations.json), [costs](../../results/native_kai_same_boundary_control_v1/summary.json) and [actual paired checks](../../results/native_kai_same_boundary_control_v1/pair_check.json) for the outcome.

**Observed result:** ordinary P20 also makes Kai silver-white at the sampled 25–30-second views and retains light gray/silver hair later. Thus the earlier correction relative to P30 does not demonstrate local-method color superiority. P20 has fewer fragments around 30 seconds, but changes poses/composition and has artifacts near 40 seconds and the end. Ava hair appears cooler gray/white in the last two samples; appearance leakage versus lighting is unresolved. Age/departure errors remain, and no arm has independent or automatic acceptance. The one new run took 530.13 seconds versus the archived RP's 570.82 seconds; that single-run difference is not an isolated sampler speed estimate. This budget is closed.

The earlier [Kai retention comparison](../native_kai_retention_factorial/README.md) showed local silver-hair correction at sampled 25–30 seconds and later retention with reminders. Its cue-only control started at 30 seconds, so the earlier local benefit could reflect the different start times. This new study permits exactly **one ordinary native generation** to address that confound. It does not change or reopen any archived experiment.

P20 appends exactly `Kai has silver hair.` to the original third prompt at 20 seconds, then uses the same augmented fourth, fifth and sixth prompts as P30/RP. Its third ordinary prompt equals RP's private repair prompt. The initial two prompts, generation seed 0, official LongLive-RAG 1.3B weights, native sampler, timing and all other generation settings are shared. There is no mask, private repair request, extra private denoising, reference image or new semantic-judge call. This is a static prompt baseline derived from literal script text, not an automated diagnosis or planning result. The narrow continuation helper is unchanged; it is not extended to parse the plural third-shot subject.

The central comparison is P20 versus RP at the same intervention boundary, including current hair color, other actors, actions, identity/age, visual artifacts and later behavior. If ordinary conditioning also repairs sampled 25–30-second hair, the former timing advantage does not establish a benefit of local repair. Color alone cannot establish net quality. This compares complete interventions—whole-shot text conditioning versus masked private repair—not a mask-only ablation. It is one exposed script/seed chosen after previous outcomes, with no success-rate, held-out, adaptive-feedback or independent human claim. No candidate selection or retry is allowed.

## Checks and costs

The preparer freezes actual B/P30/RP video, latent and manifest hashes; verifies the original runtime and twelve assets; and records the exact six prompts before rendering. The actual first 81 latent frames must match all three references. The five executed prompt-switch hashes must match P20's intended condition, with the later three matching P30/RP. Configurations must agree outside paths and disclosed interventions. Native cursor/pool guards remain active; optional history/resident byte hashes and raw-VAE prefix equality are unmeasured. Latent equality does not guarantee compressed video pixel equality or semantic preservation.

P20's full generation/preparation/audit/export and device telemetry are retained. Previously incurred B/P30/RP and mask costs remain linked separately, not erased by reuse. P20 has zero private repair forwards, while archived RP includes 56; wall time is measured rather than inferred from that count. Device energy includes idle power, and sampled utilization is not kernel-active GPU time. Developer inspection and publication are unmeasured. The complete comparison includes all 960 frames, while assistant/developer visual review uses the same 18 predeclared times per arm rather than claiming continuous-motion or anonymous human evaluation.

## Reproduce

Use the pinned official revision `973884a3cd3ad4b314c3d4ab42274c52e7a0b22a` and native environment/assets from [the fast-case instructions](../native_fast_cases/README.md). First reproduce or obtain the completed [Kai retention study](../native_kai_retention_factorial/README.md), including B/P30/RP configurations, manifests, audits, timing, videos and raw latents. Keep its original ordinary timed-runtime directory accessible. Weights and large raw latents are not included in Git.

The archived reference already has a closed-budget summary. When reproducing the reference from scratch, its generation/audit commands produce the pair check but do not write the developer's outcome summary. After both intended reference runs and the pair check pass, record only the completed budget state below. This does not manufacture an outcome review or acceptance:

```bash
python - /path/to/completed-retention-study <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
read = lambda path: json.loads(path.read_text())
assert read(root / 'plan.json')['new_generation_ceiling'] == 2
assert read(root / 'pair_check.json')['status'] == 'passed'
for arm in ['P_cue', 'RP_local_cue']:
    assert read(root / arm / 'status.json')['status'] == 'completed'
summary = root / 'summary.json'
if summary.exists():
    assert read(summary)['remaining_generation_budget'] == 0
else:
    with summary.open('x') as stream:
        json.dump({'phase': 'reproduction_budget_closed', 'remaining_generation_budget': 0,
                   'quality_acceptance': False, 'human_labels': 0}, stream, indent=2)
        stream.write('\n')
PY
```

Then run from this delivery repository with the native generation Python:

```bash
python implementation/scripts/prepare_native_prompt_start_control.py \
  --root /path/to/new-same-boundary-study \
  --reference-study /path/to/completed-retention-study \
  --supervisor adapters/official_longlive_rag/run_prepared.py
python /path/to/new-same-boundary-study/P20_prompt/supervisor.py /path/to/new-same-boundary-study/P20_prompt
python implementation/scripts/audit_native_prompt_start_control.py \
  --folder /path/to/new-same-boundary-study/P20_prompt
python implementation/scripts/review_native_prompt_start_control.py \
  --root /path/to/new-same-boundary-study \
  --arms B_original P30_prompt P20_prompt RP_local_cue \
  --output /path/to/new-same-boundary-study/four_way_review
```

The reference study must be completed and closed; its archived paths are validated, not silently rewritten. The preparer creates reference symlinks only. Never launch supervisors or rewrite audits through those archived symlinks. The one new supervisor requires an idle GPU 0 and refuses restarting an existing attempt; GPU 3 remains reserved. Start the existing NVML observer before the supervisor to reproduce telemetry. Command arguments provide portable entry points; historical absolute paths are provenance only. Default production generation/repair code and all previous study outcomes remain unchanged.
