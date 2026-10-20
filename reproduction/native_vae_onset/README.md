# Native VAE onset intervention and paired-input replay

This is a five-decode development diagnostic on existing latents, not a new repair sampler or an acceptance result. See the [measured record](../../docs/native-vae-onset-20260917.zh.md), [all fixed views](../../artifacts/native_vae_onset_v1/index.html), and [numerical traces](../../results/native_vae_onset_v1/result.json).

## Verify the published export

Use Python 3.10, Torch, NumPy, Pillow, Matplotlib 3.10.8 and the repository's existing scientific environment. No GPU, weights, or raw latents are required:

```bash
python implementation/scripts/verify_native_onset_exports.py \
  --root . --output /tmp/onset-export.json
cmp /tmp/onset-export.json results/native_vae_onset_v1/verification.json
```

This checks published raw PNGs and all recorded traces, recomputes NVML summaries, reconstructs the figures/gallery byte-exactly, and binds 39 preprocessing records to the original actual requests. It does **not** reload the VAE or processor, recompute excluded float windows, or establish garment accuracy. The separate [runtime audit](../../results/native_vae_onset_v1/runtime_audit.json) records actual reanalysis of 360 raw frames and reconstruction of 80 PNGs.

## Repeat the five real VAE decodes

Follow the [official Iris baseline](../official_timed_60s/README.md) and [natural-expression replay](../native_expression_control/README.md), retaining raw latents. Use the pinned native checkout and VAE asset. The experiment verifies actual parent/candidate latent hashes against their manifest/audit, plus the automatic mask and VAE weight. Regenerated artifacts with different hashes constitute a newly frozen experiment, not a byte-identical historical replay; do not rewrite archived bindings.

The measured model environment was Python 3.12.2, Torch 2.9.1+cu128, Torchvision 0.24.1, NumPy 1.26.4 and Pillow 11.1.0. Set `MODEL_PYTHON`, `NATIVE_CHECKOUT`, `PARENT_RUN`, `CANDIDATE_RUN`, `MASK_FILE`, and fresh `ONSET_RUN` / `ONSET_TELEMETRY` paths. Preserve the model environment executable path without resolving its symlink.

```bash
"$MODEL_PYTHON" implementation/scripts/probe_native_vae_onset.py --prepare \
  --folder "$ONSET_RUN" --checkout "$NATIVE_CHECKOUT" \
  --parent "$PARENT_RUN" --candidate "$CANDIDATE_RUN" --mask "$MASK_FILE" \
  --python "$MODEL_PYTHON" --gpu 1
```

The plan freezes the four early/later block combinations and a parent repeat. Start the read-only observer in a second terminal using Python 3.10 with `nvidia-ml-py` 12.570.86:

```bash
"$OBSERVER_PYTHON" implementation/scripts/monitor_native_gpu.py \
  --folder "$ONSET_RUN" --output "$ONSET_TELEMETRY" --interval 2 --max-wait 1100
```

After it records its first sample, execute the frozen runner once:

```bash
"$MODEL_PYTHON" "$ONSET_RUN/runner.py" --folder "$ONSET_RUN"
```

Wait for the terminal observer record. Each decode uses continuous VAE caching over two 60-latent chunks, dropping one initial output frame. Cache is cleared between conditions. It saves the full 476-frame numerical trace, 16 predetermined PNGs per condition and local float windows `[312,384)`. GPU admission allows idle GPUs 0/1/2 on the recorded host; GPU 3 is reserved. The 900-second deadline terminates only the owned child process group. There is no diffusion sampling, VLM call, retry or adoption.

To recheck the local raw windows, run under the Python 3.10 audit environment:

```bash
python implementation/scripts/summarize_native_vae_onset.py \
  --folder "$ONSET_RUN" --telemetry "$ONSET_TELEMETRY" \
  --raw-windows "$ONSET_RUN/raw_local_only" --mask "$MASK_FILE" \
  --output "$ONSET_RUN/runtime_audit.json"
```

Omit `--raw-windows` for the exported-evidence audit. `--plot`, `--gallery` and `--mask-plot` write the published display types; `--mask-plot` also needs `--mask`. Raw float windows and weights are intentionally excluded from Git. The actual automatic mask is already available in [native expression reproduction](../native_expression_control/automatic_mask.tensor).

## Repeat the actual paired preprocessing on CPU

Use the pinned Molmo processor and environment from [native observations](../native_observations/README.md). The nonweight file list/checksums are recorded in [the CPU plan](../../results/paired_preprocessing_v1/plan.json); no safetensors weights are loaded. The directory must contain the pinned configuration, tokenizer, processor and custom code files. Set `MODEL_PATH` accordingly and choose a fresh `PREPROCESS_RUN`:

```bash
"$MODEL_PYTHON" implementation/scripts/audit_paired_preprocessing.py \
  --root . --model-path "$MODEL_PATH" --output "$PREPROCESS_RUN"
```

It replays all 39 actual requests in their archived order and requires exact equality of every processed tensor. It checks normalized crop equality for the four identical two-image controls, including uniform gray. Any mismatch fails the audit; it never repairs a reply or calls a language model. Its 5.578-second measured CPU stage overlapped the VAE run; these stages are not additive end-to-end timing.
