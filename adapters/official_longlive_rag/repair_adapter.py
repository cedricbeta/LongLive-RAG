"""Offline masked parent replay for the pinned official 1.3B RAG generator.

The ordinary denoising pass still runs to preserve its RNG stream. A strict
prefix is replaced by saved parent latents before clean recache. Edited blocks
use a private RNG and the native four-step x0/re-noise sampler at strength 1.
Descriptors and clean KV recache consume the resulting latents; the suffix is
generated normally. This does not serialize KV/RNG snapshots or select branches.
"""
import hashlib
import json
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def cross_hash(cache):
    return [[tensor_hash(c['k']), tensor_hash(c['v']), bool(c['is_init'])] for c in cache]


def positions(pipeline):
    return [[int(c['global_end_index']), int(c['local_end_index']),
             len(c['cpu_k_frames']), len(c['cpu_v_frames'])] for c in pipeline.kv_cache1]


def rng_hash(device):
    result = {'cpu': tensor_hash(torch.get_rng_state())}
    if device.type == 'cuda':
        result['cuda'] = tensor_hash(torch.cuda.get_rng_state(device))
    return result


class MaskedParentRepair:
    def __init__(self, request, noise, timing, args):
        self.request = dict(request)
        if timing is None or timing['protocol'] != 'continuous_vae_drop_first_frame_v1':
            raise ValueError('Repair requires the audited continuous timing protocol')
        if noise.shape[0] != 1 or noise.ndim != 5:
            raise ValueError('Only one parent branch per inference call is supported')
        if request.get('schema_version') != 1 or request.get('entry_sigma') != 1.0:
            raise ValueError('Repair schema 1 with fixed entry_sigma 1.0 is required')
        if not isinstance(request.get('parent_branch_id'), str) or not request['parent_branch_id']:
            raise ValueError('A named parent branch is required')
        for name in ['parent_latents', 'parent_video', 'parent_config', 'parent_timing', 'mask']:
            if file_hash(request[name + '_path']) != request[name + '_sha256']:
                raise ValueError('Changed repair input: ' + name)
        parent_timing = json.loads(Path(request['parent_timing_path']).read_text())
        if json.dumps(parent_timing, sort_keys=True) != json.dumps(
                json.loads(json.dumps(timing)), sort_keys=True):
            raise ValueError('Parent and child timing/prompts differ')
        parent_config = OmegaConf.to_container(OmegaConf.load(request['parent_config_path']), resolve=True)
        current_config = OmegaConf.to_container(args, resolve=True)
        output_only = {'data_path', 'output_folder', 'adapter_diagnostics_dir', 'skip_existing',
                       'audit_prompt_switch', 'repair_request'}
        for key, value in parent_config.items():
            if key not in output_only and current_config.get(key) != value:
                raise ValueError('Parent generation configuration differs: ' + key)
        if int(current_config.get('num_frame_per_block', 0)) != 3:
            raise ValueError('Repair supports the measured native maximum block size 3')
        self.parent = torch.load(request['parent_latents_path'], map_location='cpu', weights_only=True)
        if not isinstance(self.parent, torch.Tensor) or self.parent.shape != noise.shape or self.parent.dtype != noise.dtype:
            raise ValueError('Parent latent shape or dtype differs from the generator input')
        if not torch.isfinite(self.parent).all():
            raise ValueError('Parent contains nonfinite latents')
        interval = request['latent_range']
        if len(interval) != 2 or any(type(x) is not int for x in interval):
            raise ValueError('Use integer half-open latent range')
        self.start, self.end = interval
        boundaries, end = {0}, 0
        for size in timing['blocks']:
            end += size
            boundaries.add(end)
        if not 0 <= self.start < self.end <= noise.shape[1] or not {self.start, self.end} <= boundaries:
            raise ValueError('Edit range must align with actual generation blocks')
        # One repair transaction targets one authored segment. It cannot silently
        # replace two different authored prompts with a single repair prompt.
        if not any(s['start_latent_frame'] == self.start and
                   s['start_latent_frame'] + s['latent_frames'] == self.end for s in timing['segments']):
            raise ValueError('Edit range must cover exactly one authored segment')
        self.seed = request['noise_seed']
        if type(self.seed) is not int or not 0 <= self.seed < 2**63 - noise.shape[1]:
            raise ValueError('Invalid private repair seed')
        self.prompts = request['prompts']
        if (not isinstance(self.prompts, list) or len(self.prompts) != 1
                or not isinstance(self.prompts[0], str) or not self.prompts[0].strip()):
            raise ValueError('One nonempty repair prompt is required')
        payload = torch.load(request['mask_path'], map_location='cpu', weights_only=True)
        for field, expected in [('parent_video_sha256', request['parent_video_sha256']),
                                ('parent_latents_sha256', request['parent_latents_sha256']),
                                ('parent_timing_sha256', request['parent_timing_sha256']),
                                ('latent_range', interval), ('grid', list(noise.shape[-2:]))]:
            if payload.get(field) != expected:
                raise ValueError('Mask belongs to another parent or coordinate system: ' + field)
        self.mask = payload['mask']
        if not isinstance(self.mask, torch.Tensor) or list(self.mask.shape) != [self.end - self.start, *noise.shape[-2:]]:
            raise ValueError('Mask must be [edited latent frames, H, W]')
        if not torch.isfinite(self.mask).all() or not ((self.mask == 0) | (self.mask == 1)).all():
            raise ValueError('Mask must be finite and binary; threshold it before freezing the request')
        self.mask = self.mask.bool()
        self.log = []
        self.conditional = None
        self.pending = None

    def apply(self, pipeline, native_result, start, memory_indices):
        count = native_result.shape[1]
        if self.pending is not None:
            raise RuntimeError('Previous block did not finish clean recache')
        phase = 'prefix' if start < self.start else 'edit' if start < self.end else 'suffix'
        if start < self.end < start + count or start < self.start < start + count:
            raise ValueError('A generator block crosses the edit boundary')
        parent = self.parent[:, start:start + count].to(native_result.device)
        result = parent.clone() if phase == 'prefix' else native_result
        report = {'start_latent_frame': start, 'latent_frames': count, 'phase': phase,
                  'parent_branch_id': self.request['parent_branch_id']}
        if phase == 'edit':
            began = time.monotonic()
            mask = self.mask[start - self.start:start - self.start + count].to(native_result.device)[None, :, None]
            initial_rng = rng_hash(native_result.device)
            original_positions = positions(pipeline)
            authored_cross = cross_hash(pipeline.crossattn_cache)
            if not all(row[2] for row in authored_cross):
                raise RuntimeError('Authored cross-attention was not initialized before repair')
            generator = torch.Generator(device=native_result.device).manual_seed(self.seed + start)
            steps = pipeline.denoising_step_list
            scheduler = pipeline.scheduler
            entry_index = (scheduler.timesteps.cpu() - steps[0].cpu()).abs().argmin()
            if abs(float(scheduler.sigmas[entry_index]) - 1.0) > 1e-6:
                raise ValueError('Native first denoising step is not sigma 1.0')
            try:
                if self.conditional is None:
                    self.conditional = pipeline.text_encoder(text_prompts=self.prompts)
                pipeline._initialize_crossattn_cache(batch_size=1, dtype=parent.dtype, device=parent.device)
                eps = torch.randn(parent.shape, generator=generator, device=parent.device, dtype=parent.dtype)
                x = eps
                for index, step in enumerate(steps):
                    t = torch.ones(parent.shape[:2], device=parent.device, dtype=torch.int64) * step
                    background = scheduler.add_noise(parent.flatten(0, 1), eps.flatten(0, 1), t.flatten()).unflatten(0, parent.shape[:2])
                    x = torch.where(mask, x, background)
                    _, x0 = pipeline.generator(noisy_image_or_video=x, conditional_dict=self.conditional,
                        timestep=t, kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                        current_start=start * pipeline.frame_seq_length, memory_indices=memory_indices)
                    result = torch.where(mask, x0, parent)
                    if index + 1 < len(steps):
                        eps = torch.randn(parent.shape, generator=generator, device=parent.device, dtype=parent.dtype)
                        next_t = steps[index + 1] * torch.ones([count], device=parent.device, dtype=torch.long)
                        x = scheduler.add_noise(result.flatten(0, 1), eps.flatten(0, 1), next_t).unflatten(0, parent.shape[:2])
            finally:
                # The caller still holds the authored conditional_dict. Its
                # normal clean recache must repopulate this fresh text cache.
                pipeline._initialize_crossattn_cache(batch_size=1, dtype=parent.dtype, device=parent.device)
            if positions(pipeline) != original_positions:
                raise RuntimeError('Repair mutated cache positions or retrieval pool lengths')
            if rng_hash(native_result.device) != initial_rng:
                raise RuntimeError('Repair consumed the native global RNG stream')
            if not torch.isfinite(result).all():
                raise RuntimeError('Repair produced nonfinite latents')
            outside = (~mask).expand_as(result)
            if not torch.equal(result[outside], parent[outside]):
                raise RuntimeError('Repair changed a latent outside its frozen mask')
            report.update(mask_fraction=float(mask.float().mean()), private_seed=self.seed + start,
                          entry_sigma=1.0, denoising_steps=len(steps), global_rng_preserved=True,
                          cache_positions_preserved=True, outside_mask_exact=True,
                          parent_block_sha256=tensor_hash(parent), result_block_sha256=tensor_hash(result),
                          authored_cross_before=authored_cross,
                          repair_wall_seconds=time.monotonic() - began)
        self.pending = report
        return result

    def finish_block(self, pipeline, result, start):
        report = self.pending
        if report is None or report['start_latent_frame'] != start:
            raise RuntimeError('No matching pending block')
        if len(pipeline.latent_descriptors) != start + result.shape[1]:
            raise RuntimeError('Clean descriptors do not cover repaired output')
        if report['phase'] == 'edit':
            after = cross_hash(pipeline.crossattn_cache)
            if after != report['authored_cross_before']:
                raise RuntimeError('Authored cross-attention was not restored exactly')
            report['authored_cross_restored_exact'] = True
        report['descriptor_count_after'] = len(pipeline.latent_descriptors)
        report['descriptor_sha256_after'] = [tensor_hash(x) for x in pipeline.latent_descriptors[-result.shape[1]:]]
        report['cache_positions_after_clean_recache'] = positions(pipeline)
        self.log.append(report)
        self.pending = None
