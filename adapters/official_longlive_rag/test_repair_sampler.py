"""Legacy-path equivalence and independent Euler/transaction invariants."""
from pathlib import Path
from types import ModuleType

import pytest
import torch

import test_repair as legacy
from install_repair_sampler import patch_adapter
from repair_sampler import EULER, NATIVE


@pytest.fixture
def patched():
    module = ModuleType('private_sampler_test_adapter')
    source = Path(__file__).with_name('repair_adapter.py').read_text()
    exec(compile(patch_adapter(source), 'patched_adapter', 'exec'), module.__dict__)
    return module.MaskedParentRepair


@pytest.mark.parametrize('empty', [False, True])
def test_native_default_exactly_matches_original_class(tmp_path, patched, empty):
    request, noise, timing, args, parent = legacy.fixture(tmp_path, empty=empty)
    original = legacy.r.MaskedParentRepair(request, noise, timing, args)
    new = patched(request, noise, timing, args)
    p, q = legacy.pipeline(), legacy.pipeline()
    inputs = [[], []]
    for pipeline, record in zip((p, q), inputs):
        call = pipeline.generator
        def capture(_call=call, _record=record, **kwargs):
            _record.append(kwargs['noisy_image_or_video'].clone())
            return _call(**kwargs)
        pipeline.generator = capture
    before = torch.get_rng_state().clone()
    a = original.apply(p, noise[:, 3:5], 3, [0])
    b = new.apply(q, noise[:, 3:5], 3, [0])
    assert torch.equal(a, b) and p.calls == q.calls == 4
    assert all(torch.equal(x, y) for x, y in zip(*inputs))
    assert torch.equal(torch.get_rng_state(), before)
    assert legacy.r.positions(p) == legacy.r.positions(q)
    assert legacy.r.cross_hash(p.crossattn_cache) == legacy.r.cross_hash(q.crossattn_cache)
    assert new.pending['repair_sampler'] == NATIVE


@pytest.mark.parametrize('empty', [False, True])
def test_euler_solves_constant_flow_and_reuses_background_noise(tmp_path, patched, empty):
    request, noise, timing, args, parent = legacy.fixture(tmp_path, empty=empty)
    request['repair_sampler'] = EULER
    edit = patched(request, noise, timing, args)
    p, inputs = legacy.pipeline(), []
    def constant_flow(**kwargs):
        inputs.append(kwargs['noisy_image_or_video'].clone())
        p.crossattn_cache[0]['is_init'] = True
        return torch.full_like(inputs[-1], 4.), torch.zeros_like(inputs[-1])
    p.generator = constant_flow
    before = torch.get_rng_state().clone()
    result = edit.apply(p, noise[:, 3:5], 3, [0])
    eps = torch.randn(parent[:, 3:5].shape, generator=torch.Generator().manual_seed(4246))
    mask = edit.mask[None, :, None].expand_as(result)
    # Integral of constant dX/dsigma=4 from sigma1 to sigma0 is -4.
    assert torch.allclose(result[mask], (eps - 4)[mask], atol=1e-6, rtol=0)
    assert torch.equal(result[~mask], parent[:, 3:5][~mask])
    for index, x in enumerate(inputs):
        sigma = 1 - index / 4
        expected = (1 - sigma) * parent[:, 3:5] + sigma * eps
        assert torch.equal(x[~mask], expected[~mask])
    assert len(inputs) == 4 and len(edit.pending['sampler_trace']) == 4
    assert torch.equal(before, torch.get_rng_state())
    assert not p.crossattn_cache[0]['is_init']
    assert torch.equal(result, parent[:, 3:5]) == empty
    p.latent_descriptors += [torch.zeros(1), torch.zeros(1)]
    p.crossattn_cache[0].update(k=torch.tensor([3.]), v=torch.tensor([4.]), is_init=True)
    edit.finish_block(p, result, 3)
    assert edit.log[0]['authored_cross_restored_exact']


def test_euler_failure_clears_private_text_without_committing(tmp_path, patched):
    request, noise, timing, args, parent = legacy.fixture(tmp_path)
    request['repair_sampler'] = EULER
    edit = patched(request, noise, timing, args)
    p = legacy.pipeline()
    def fail(**kwargs):
        p.crossattn_cache[0]['is_init'] = True
        raise RuntimeError('model error fixture')
    p.generator = fail
    with pytest.raises(RuntimeError, match='model error fixture'):
        edit.apply(p, noise[:, 3:5], 3, None)
    assert not p.crossattn_cache[0]['is_init'] and not edit.log and edit.pending is None


@pytest.mark.parametrize('steps', [[1000., 500., 750., 250.], [750., 500., 250.], [750., 600., 500., 250.]])
def test_invalid_or_non_sigma1_schedule_refused(tmp_path, patched, steps):
    request, noise, timing, args, parent = legacy.fixture(tmp_path)
    request['repair_sampler'] = EULER
    edit = patched(request, noise, timing, args)
    p = legacy.pipeline()
    p.denoising_step_list = torch.tensor(steps)
    with pytest.raises(ValueError):
        edit.apply(p, noise[:, 3:5], 3, None)
    assert p.calls == 0 and not edit.log


def test_unknown_mode_refused_before_any_execution(tmp_path, patched):
    request, noise, timing, args, parent = legacy.fixture(tmp_path)
    request['repair_sampler'] = 'unfrozen_method'
    with pytest.raises(ValueError, match='Unknown'):
        patched(request, noise, timing, args)


def test_euler_does_not_touch_prefix_or_suffix(tmp_path, patched):
    request, noise, timing, args, parent = legacy.fixture(tmp_path)
    request['repair_sampler'] = EULER
    edit = patched(request, noise, timing, args)
    p = legacy.pipeline()
    assert torch.equal(edit.apply(p, noise[:, :3], 0, None), parent[:, :3])
    edit.finish_block(p, parent[:, :3], 0)
    suffix = noise[:, 5:7] + 123
    assert edit.apply(p, suffix, 5, None) is suffix and p.calls == 0
