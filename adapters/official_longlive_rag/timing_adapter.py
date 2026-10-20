"""Opt-in exact-duration temporal adaptation; never changes the export FPS."""
from fractions import Fraction

import torch


def compile_timing(segments, initial_prompts, fps=16, maximum_block=3):
    if fps != 16 or maximum_block != 3 or not segments:
        raise ValueError('This protocol supports 16 fps and native maximum block size 3')
    start, pixels, schedule, blocks, records = 0, 0, {}, [], []
    for index, segment in enumerate(segments):
        count = Fraction(str(segment['seconds'])) * fps
        prompts = list(segment['prompts'])
        if count.denominator != 1 or count <= 0 or int(count) % 4:
            raise ValueError('Segment durations must have a positive frame count divisible by four')
        if len(prompts) != len(initial_prompts) or any(not isinstance(x, str) or not x.strip() for x in prompts):
            raise ValueError('Each segment needs one nonempty prompt per batch member')
        if index == 0 and prompts != list(initial_prompts):
            raise ValueError('First segment must equal the baseline initial prompt')
        latent_count = int(count) // 4 + (1 if index == 0 else 0)
        schedule[start] = prompts
        records.append({'segment': index, 'start_latent_frame': start, 'latent_frames': latent_count,
                        'start_export_frame': pixels, 'export_frames': int(count),
                        'start_seconds': pixels / fps, 'seconds': int(count) / fps})
        remaining = latent_count
        while remaining:
            size = min(remaining, maximum_block)
            blocks.append(size)
            remaining -= size
        start += latent_count
        pixels += int(count)
    return {'protocol': 'continuous_vae_drop_first_frame_v1', 'fps': fps,
            'latent_frames': start, 'decoded_frames_before_crop': 4 * start - 3,
            'crop_start_frames': 1, 'export_frames': pixels, 'blocks': blocks,
            'schedule': schedule, 'segments': records,
            'changes_from_official_default': [
                'Short final generation blocks at authored prompt boundaries',
                'Continuous VAE cache across decode chunks', 'One initial warm-up output frame omitted']}


def decode_continuous(vae, latent, chunk_size=60):
    if chunk_size < 1 or latent.ndim != 5 or latent.shape[1] < 1:
        raise ValueError('Nonempty [B,T,C,H,W] latent and positive chunk size required')
    device, dtype = latent.device, latent.dtype
    scale = [vae.mean.to(device=device, dtype=dtype), 1.0 / vae.std.to(device=device, dtype=dtype)]
    videos = []
    for sample in latent.permute(0, 2, 1, 3, 4):
        chunks = []
        vae.model.clear_cache()
        try:
            for start in range(0, sample.shape[1], chunk_size):
                chunk = sample[:, start:start + chunk_size].unsqueeze(0)
                decoded = vae.model.cached_decode(chunk, scale).float().clamp_(-1, 1)
                expected = 4 * chunk.shape[2] - (3 if start == 0 else 0)
                if decoded.shape[2] != expected:
                    raise RuntimeError('Actual VAE frame count differs from continuous timing contract')
                chunks.append(decoded.squeeze(0).cpu())
            videos.append(torch.cat(chunks, dim=1))
        finally:
            vae.model.clear_cache()
    return torch.stack(videos).permute(0, 2, 1, 3, 4)
