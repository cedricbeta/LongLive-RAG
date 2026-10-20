"""Context selection, true OFF identity, failure cleanup and invalid layouts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location('repair_context_tested', Path(__file__).with_name('repair_context.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fixture():
    state = []
    def attention(q, k, v):
        state.append((q, k, v))
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    g = torch.Generator().manual_seed(32)
    q = torch.randn(1, 3, 2, 4, generator=g)
    k = torch.randn(1, 12, 2, 4, generator=g)
    v = torch.randn(1, 12, 2, 4, generator=g)
    return SimpleNamespace(attention=attention), (q, k, v), state


def test_off_uses_identical_callable_tensors_and_result():
    m, args, state = fixture()
    original, trace = m.attention, []
    expected = original(*args)
    before = [v.clone() for v in args]
    with r.attention_scope(m, 'full_context_v1', 3, 30, trace):
        assert m.attention is original
        actual = m.attention(*args)
    assert m.attention is original and torch.equal(actual, expected)
    assert all(a is b for a, b in zip(state[-1], args))
    assert all(torch.equal(a, b) for a, b in zip(args, before))
    assert trace[0]['intercepted_calls'] == []


def test_on_uses_current_kv_without_mutating_inputs_and_restores():
    m, args, state = fixture()
    original, trace = m.attention, []
    q, k, v = args
    expected = original(q, k[:, -3:], v[:, -3:])
    before = [a.clone() for a in args]
    with r.attention_scope(m, 'current_block_only_v1', 3, 1, trace):
        actual = m.attention(*args)
    assert torch.equal(actual, expected) and m.attention is original
    assert state[-1][0] is q
    assert all(torch.equal(a, b) for a, b in zip(args, before))
    assert trace[0]['intercepted_calls'] == [{'query_tokens': 3, 'native_key_tokens': 12, 'used_key_tokens': 3}]
    assert not torch.equal(actual, original(*args))


@pytest.mark.parametrize('failure', ['body', 'missing_layer', 'extra_layer', 'bad_query', 'nested', 'replace_callable'])
def test_every_failure_restores_callable_and_unlocks(failure):
    m, args, _ = fixture()
    original, trace = m.attention, []
    with pytest.raises(RuntimeError):
        with r.attention_scope(m, 'current_block_only_v1', 3, 1, trace):
            if failure == 'body':
                raise RuntimeError('model failure')
            if failure == 'missing_layer':
                pass
            elif failure == 'extra_layer':
                m.attention(*args); m.attention(*args)
            elif failure == 'bad_query':
                m.attention(args[0][:, :2], *args[1:])
            elif failure == 'nested':
                with r.attention_scope(m, 'full_context_v1', 3, 1, []):
                    pass
            elif failure == 'replace_callable':
                m.attention(*args)
                m.attention = lambda *a: None
    assert m.attention is original and trace == []
    with r.attention_scope(m, 'current_block_only_v1', 3, 1, trace):
        m.attention(*args)
    assert trace[0]['callable_restored']


def test_mode_is_closed_and_crossattention_binding_is_untouched():
    m, args, _ = fixture()
    cross = SimpleNamespace(attention=m.attention)
    original = m.attention
    with pytest.raises(ValueError):
        with r.attention_scope(m, 'disable_rag', 3, 1, []):
            pass
    with r.attention_scope(m, 'current_block_only_v1', 3, 1, []):
        m.attention(*args)
        assert cross.attention is original
