"""Parent integrity, private RNG, background pinning and authored-text guards."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

r = module('repair_adapter')
timing_module = module('timing_adapter')


def fixture(tmp_path, *, empty=False):
    segments = [{'seconds': 0.5, 'prompts': ['first']}, {'seconds': 0.5, 'prompts': ['second']}, {'seconds': 0.5, 'prompts': ['third']}]
    timing = timing_module.compile_timing(segments, ['first'])
    parent = torch.arange(56, dtype=torch.float32).reshape(1, 7, 2, 2, 2)
    paths = {name: tmp_path / name for name in ['parent_latents', 'parent_video', 'parent_timing', 'parent_config', 'mask']}
    torch.save(parent, paths['parent_latents'])
    paths['parent_video'].write_bytes(b'video checksum fixture')
    paths['parent_timing'].write_text(json.dumps(timing))
    args = OmegaConf.create({'num_frame_per_block': 3, 'seed': 0, 'timed_segments': segments})
    OmegaConf.save(args, paths['parent_config'])
    request = {'schema_version': 1, 'entry_sigma': 1.0, 'parent_branch_id': 'parent0',
               'latent_range': [3, 5], 'noise_seed': 4243, 'prompts': ['repair']}
    for name in paths:
        request[name + '_path'] = str(paths[name])
        if name != 'mask':
            request[name + '_sha256'] = r.file_hash(paths[name])
    mask = torch.zeros(2, 2, 2)
    if not empty:
        mask[:, 0, 0] = 1
    payload = {'mask': mask, 'latent_range': [3, 5], 'grid': [2, 2],
               **{key: request[key] for key in ['parent_video_sha256', 'parent_latents_sha256', 'parent_timing_sha256']}}
    torch.save(payload, paths['mask'])
    request['mask_sha256'] = r.file_hash(paths['mask'])
    return request, torch.zeros_like(parent), timing, args, parent


def pipeline():
    p = SimpleNamespace(frame_seq_length=2, denoising_step_list=torch.tensor([1000., 750., 500., 250.]),
        kv_cache1=[{'global_end_index': torch.tensor(10), 'local_end_index': torch.tensor(10),
                    'cpu_k_frames': [torch.tensor([1.])], 'cpu_v_frames': [torch.tensor([2.])]}],
        latent_descriptors=[torch.zeros(1) for _ in range(3)], calls=0)
    def initialize(**unused):
        p.crossattn_cache = [{'k': torch.zeros(1), 'v': torch.zeros(1), 'is_init': False}]
    p._initialize_crossattn_cache = initialize
    p.text_encoder = lambda text_prompts: {'text': text_prompts}
    p.scheduler = SimpleNamespace(timesteps=p.denoising_step_list, sigmas=p.denoising_step_list / 1000.)
    p.scheduler.add_noise = lambda x, eps, t: (1 - t[:, None, None, None] / 1000.) * x + t[:, None, None, None] / 1000. * eps
    def generator(**kwargs):
        p.calls += 1
        value = 9. if kwargs['conditional_dict']['text'] == ['repair'] else 3.
        c = p.crossattn_cache[0]
        c['k'].fill_(value); c['v'].fill_(value + 1); c['is_init'] = True
        return None, kwargs['noisy_image_or_video'] + 7.
    p.generator = generator
    initialize()
    p.crossattn_cache[0].update(k=torch.tensor([3.]), v=torch.tensor([4.]), is_init=True)
    return p


@pytest.mark.parametrize('empty', [False, True])
def test_private_repair_pins_background_and_restores_authored_text(tmp_path, empty):
    request, noise, timing, args, parent = fixture(tmp_path, empty=empty)
    repair = r.MaskedParentRepair(request, noise, timing, args)
    p = pipeline()
    rng = torch.get_rng_state().clone()
    result = repair.apply(p, noise[:, 3:5], 3, None)
    assert p.calls == 4  # Zero masks exercise actual prompt/cache/RNG restoration.
    assert torch.equal(torch.get_rng_state(), rng)
    outside = (~repair.mask)[None, :, None].expand_as(result)
    assert torch.equal(result[outside], parent[:, 3:5][outside])
    assert torch.equal(result, parent[:, 3:5]) == empty
    assert not p.crossattn_cache[0]['is_init']
    p.latent_descriptors += [result[:, i].mean().reshape(1) for i in range(2)]
    p.generator(noisy_image_or_video=result, conditional_dict={'text': ['second']})
    repair.finish_block(p, result, 3)
    assert repair.log[0]['authored_cross_restored_exact']


def test_nonempty_prefix_uses_saved_parent_and_suffix_stays_native(tmp_path):
    request, noise, timing, args, parent = fixture(tmp_path)
    repair = r.MaskedParentRepair(request, noise, timing, args)
    p = pipeline()
    prefix = repair.apply(p, noise[:, :3], 0, None)
    assert torch.equal(prefix, parent[:, :3])
    assert p.calls == 0
    repair.finish_block(p, prefix, 0)
    assert repair.log[0]['phase'] == 'prefix'
    native_suffix = noise[:, 5:7] + 123.
    suffix = repair.apply(p, native_suffix, 5, None)
    assert suffix is native_suffix
    assert not torch.equal(suffix, parent[:, 5:7])


def test_mask_rejects_other_parent_even_with_a_fresh_mask_file_hash(tmp_path):
    request, noise, timing, args, _ = fixture(tmp_path)
    payload = torch.load(request['mask_path'], weights_only=True)
    payload['parent_video_sha256'] = 'another_parent'
    torch.save(payload, request['mask_path'])
    request['mask_sha256'] = r.file_hash(request['mask_path'])
    with pytest.raises(ValueError, match='another parent'):
        r.MaskedParentRepair(request, noise, timing, args)


def test_changed_seed_or_timing_cannot_reuse_parent(tmp_path):
    request, noise, timing, args, _ = fixture(tmp_path)
    args.seed = 1
    with pytest.raises(ValueError, match='configuration differs: seed'):
        r.MaskedParentRepair(request, noise, timing, args)
    args.seed = 0
    timing['schedule'][3] = ['different authored scene']
    with pytest.raises(ValueError, match='timing/prompts differ'):
        r.MaskedParentRepair(request, noise, timing, args)


def test_missing_authored_recache_is_not_silently_recorded_as_success(tmp_path):
    request, noise, timing, args, _ = fixture(tmp_path)
    repair = r.MaskedParentRepair(request, noise, timing, args)
    p = pipeline()
    result = repair.apply(p, noise[:, 3:5], 3, None)
    p.latent_descriptors += [torch.zeros(1), torch.zeros(1)]
    with pytest.raises(RuntimeError, match='not restored exactly'):
        repair.finish_block(p, result, 3)
    assert not repair.log


def test_accidental_global_rng_use_fails_transaction(tmp_path):
    request, noise, timing, args, _ = fixture(tmp_path)
    repair = r.MaskedParentRepair(request, noise, timing, args)
    p = pipeline()
    original = p.generator
    def bad(**kwargs):
        torch.rand(1)
        return original(**kwargs)
    p.generator = bad
    with pytest.raises(RuntimeError, match='global RNG'):
        repair.apply(p, noise[:, 3:5], 3, None)
    assert not p.crossattn_cache[0]['is_init']
