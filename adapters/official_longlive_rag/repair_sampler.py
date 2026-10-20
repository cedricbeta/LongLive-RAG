"""Fixed-noise Euler diagnostic for private repair calls only.

The native distilled generator, its four timesteps, retrieval context, clean
recache and ordinary suffix generation are unchanged. This is not a replica of
the historical 5B model or its schedule, and makes no efficacy claim.
"""
import math

import torch


NATIVE = 'native_x0_renoise_v1'
EULER = 'fixed_noise_euler_v1'


def sample_euler(pipeline, parent, mask, eps, conditional, start, memory_indices, trace):
    steps, scheduler = pipeline.denoising_step_list, pipeline.scheduler
    if len(steps) != 4:
        raise ValueError('Sampler diagnostic requires the four native timesteps')
    sigmas = []
    for step in steps:
        index = (scheduler.timesteps.detach().cpu() - step.detach().cpu()).abs().argmin()
        sigmas.append(float(scheduler.sigmas[index]))
    if (any(not math.isfinite(s) for s in sigmas) or abs(sigmas[0] - 1.) > 1e-6
            or not all(a > b > 0 for a, b in zip(sigmas, sigmas[1:]))):
        raise ValueError('Expected sigma1 followed by strictly decreasing positive native sigmas')
    if eps.shape != parent.shape or mask.dtype != torch.bool:
        raise ValueError('Invalid noise or frozen binary mask')
    x = eps
    for index, step in enumerate(steps):
        t = torch.ones(parent.shape[:2], device=parent.device, dtype=torch.int64) * step
        background = scheduler.add_noise(parent.flatten(0, 1), eps.flatten(0, 1), t.flatten()).unflatten(0, parent.shape[:2])
        x = torch.where(mask, x, background)
        flow, _ = pipeline.generator(noisy_image_or_video=x, conditional_dict=conditional,
            timestep=t, kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
            current_start=start * pipeline.frame_seq_length, memory_indices=memory_indices)
        if flow.shape != parent.shape or flow.dtype != parent.dtype or not torch.isfinite(flow).all():
            raise ValueError('Native flow output has invalid shape, dtype or values')
        next_sigma = sigmas[index + 1] if index + 1 < len(steps) else 0.
        x = x + flow * (next_sigma - sigmas[index])
        trace.append({'step': index, 'native_timestep': float(step), 'sigma': sigmas[index],
                      'next_sigma': next_sigma, 'background_noise': 'same_initial_private_noise'})
    return torch.where(mask, x, parent)
