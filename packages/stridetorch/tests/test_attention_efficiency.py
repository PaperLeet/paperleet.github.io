"""Selective tape lifetime, independent grouped dropout math and cache guards."""
import gc
import math
import weakref
import numpy as np
import pytest
from stridetorch import torch


SHAPES = [(6, 3, 4), (2, 5, 4), (3, 5, 6), (3, 5)]


def scalar_attention(arrays, probability, seed):
    q, k, v, mask = arrays
    output = np.zeros((6, 3, 6)); gradients = [np.zeros(a.shape) for a in arrays]
    draws = (np.arange(90).reshape(6, 3, 5) + .5) / 90
    for head in range(6):
        kh, vh = head // 3, head // 2
        for row in range(3):
            logits = [sum(float(q[head, row, e]) * float(k[kh, col, e]) for e in range(4)) * .37 + float(mask[row, col]) for col in range(5)]
            maximum = max(logits)
            p = [0.] * 5 if maximum == -math.inf else [math.exp(x - maximum) for x in logits]
            total = sum(p)
            p = [x / total for x in p] if total else p
            multipliers = [0. if probability == 1 else float(draws[head, row, col] >= probability) / (1 - probability) for col in range(5)]
            incoming = [sum(float(seed[head, row, c]) * float(v[vh, col, c]) for c in range(6)) * multipliers[col] for col in range(5)]
            center = sum(x * y for x, y in zip(p, incoming))
            for col in range(5):
                ds = p[col] * (incoming[col] - center)
                gradients[3][row, col] += ds
                for e in range(4):
                    gradients[0][head, row, e] += ds * float(k[kh, col, e]) * .37
                    gradients[1][kh, col, e] += ds * float(q[head, row, e]) * .37
                for c in range(6):
                    output[head, row, c] += p[col] * multipliers[col] * float(v[vh, col, c])
                    gradients[2][vh, col, c] += p[col] * multipliers[col] * float(seed[head, row, c])
    return output, gradients


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('probability', [0., .25, 1.])
@pytest.mark.parametrize('active', range(16))
def test_selective_dropout_pullbacks_match_independent_scalar_math(monkeypatch, dtype, probability, active):
    from stridetorch import _random
    rng = np.random.default_rng(5110)
    arrays = [rng.normal(size=s).astype(dtype) * .2 for s in SHAPES]; arrays[-1][1] = -np.inf
    values = [torch.tensor(a, requires_grad=bool(active & (1 << i))) for i, a in enumerate(arrays)]
    draws = []
    def sample(generator, shape, dtype):
        draws.append(shape)
        return ((np.arange(np.prod(shape)).reshape(shape) + .5) / np.prod(shape)).astype(dtype)
    monkeypatch.setattr(_random, 'uniform_samples', sample)
    y = torch.nn.functional.scaled_dot_product_attention(*values[:3], attn_mask=values[3], enable_gqa=True, dropout_p=probability, scale=.37)
    assert y.requires_grad == bool(active)
    def reject(*args, **kwargs):
        raise AssertionError('backward sampled a new dropout mask')
    monkeypatch.setattr(_random, 'uniform_samples', reject)
    for _ in range(2):
        seed = rng.normal(size=y.shape).astype(dtype)
        if active:
            for x in values: x.grad = None
            y.backward(torch.tensor(seed), retain_graph=True)
        expected, gradients = scalar_attention(arrays, probability, seed)
        np.testing.assert_allclose(y.numpy(), expected, rtol=4e-6, atol=2e-6)
        for x, gradient in zip(values, gradients):
            if x.requires_grad: np.testing.assert_allclose(x.grad.numpy(), gradient, rtol=6e-6, atol=3e-6)
            else: assert x.grad is None
    assert len(draws) == (1 if probability == .25 else 0)


@pytest.mark.parametrize('active', [0, 1, 2, 4, 8, 5, 6, 15])
@pytest.mark.parametrize('probability', [0., .25, 1.])
def test_tape_releases_unneeded_score_and_scaled_operand_arrays(monkeypatch, active, probability):
    from stridetorch import _attention
    refs = {}; probabilities = _attention._probabilities; multiplier = _attention._dropout_multiplier; product = _attention._head_product
    def record_probability(*args):
        result = probabilities(*args); refs['p'] = weakref.ref(result); return result
    def record_multiplier(*args):
        result = multiplier(*args)
        if result is not None: refs['multiplier'] = weakref.ref(result)
        return result
    def record_product(left, right, grouped):
        if left.shape == SHAPES[0] and right.shape == (2, 4, 5):
            refs['scaled_q'] = weakref.ref(left); refs['scaled_k'] = weakref.ref(right.base if isinstance(right.base, np.ndarray) else right)
        if left.shape == (6, 3, 5) and right.shape == SHAPES[2]: refs['used'] = weakref.ref(left)
        return product(left, right, grouped)
    monkeypatch.setattr(_attention, '_probabilities', record_probability)
    monkeypatch.setattr(_attention, '_dropout_multiplier', record_multiplier)
    monkeypatch.setattr(_attention, '_head_product', record_product)
    values = [torch.ones(*s, dtype=torch.float64, requires_grad=bool(active & (1 << i))) for i, s in enumerate(SHAPES)]
    y = torch.nn.functional.scaled_dot_product_attention(*values[:3], attn_mask=values[3], enable_gqa=True, dropout_p=probability)
    gc.collect()
    logits = bool(active & 11); v_only = bool(active & 4) and not logits
    expected = dict(p=logits or (v_only and probability == 0), used=v_only or (logits and probability == 0), scaled_q=bool(active & 2), scaled_k=bool(active & 1))
    if probability: expected['multiplier'] = logits
    assert set(refs) == set(expected)
    for name, needed in expected.items(): assert (refs[name]() is not None) == needed, name
    if active:
        # Pullback needs stay captured when leaf flags subsequently change.
        for x in values: x.requires_grad_(True)
        result = torch.autograd.grad(y, values, torch.ones_like(y), allow_unused=True)
        for i, x in enumerate(result):
            assert (x is not None) == bool(active & (1 << i))
            if x is not None: assert np.isfinite(x.numpy()).all()


def test_geometry_cache_does_not_retain_numpy_dtype_metadata():
    from stridetorch import _attention
    class Owner: pass
    owner = Owner(); ref = weakref.ref(owner)
    dtype = np.dtype(np.float32, metadata={'owner': owner})
    arrays = [np.ones(s, dtype=dtype) for s in SHAPES[:3]]
    values = [torch.from_numpy(x) for x in arrays]
    assert values[0]._data.dtype.metadata['owner'] is owner
    _attention._geometry.cache_clear()
    with torch.no_grad():
        output = torch.nn.functional.scaled_dot_product_attention(*values, enable_gqa=True)
    del owner, dtype, arrays, values, output
    gc.collect()
    assert ref() is None
    assert _attention._geometry.cache_info().currsize == 1


@pytest.mark.parametrize('changed', ['query_dtype', 'key_dtype', 'value_dtype', 'mask_dtype', 'feature', 'head', 'dropout', 'causal_type', 'gqa_type'])
def test_warm_metadata_cache_rechecks_types_shapes_and_scalar_options(changed):
    values = [torch.ones(*s) for s in SHAPES[:3]]; options = dict(enable_gqa=True, attn_mask=torch.ones(3, 5, dtype=torch.bool))
    torch.nn.functional.scaled_dot_product_attention(*values, **options)
    if changed.endswith('_dtype'):
        if changed == 'mask_dtype': options['attn_mask'] = torch.ones(3, 5, dtype=torch.int32)
        else: values[{'query_dtype': 0, 'key_dtype': 1, 'value_dtype': 2}[changed]] = values[{'query_dtype': 0, 'key_dtype': 1, 'value_dtype': 2}[changed]].double()
    if changed == 'feature': values[1] = torch.ones(2, 5, 3)
    if changed == 'head': values[1] = torch.ones(4, 5, 4)
    if changed == 'dropout': options['dropout_p'] = float('nan')
    if changed == 'causal_type': options['is_causal'] = 1
    if changed == 'gqa_type': options['enable_gqa'] = 1
    with pytest.raises((RuntimeError, TypeError)):
        torch.nn.functional.scaled_dot_product_attention(*values, **options)


def test_cache_accepts_same_shape_new_values_and_dtypes_without_reusing_results():
    from stridetorch import _attention
    _attention._geometry.cache_clear()
    outputs = []
    for dtype, value in [(torch.float32, 1.), (torch.float32, 2.), (torch.float64, 3.)]:
        q = torch.ones(1, 2, 3, 4, dtype=dtype); k = torch.ones(1, 1, 5, 4, dtype=dtype); v = torch.full((1, 1, 5, 6), value, dtype=dtype)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        outputs.append(y.numpy().copy()); assert y.dtype == dtype
        np.testing.assert_allclose(y.numpy(), value, rtol=2e-6, atol=2e-6)
    assert _attention._geometry.cache_info().currsize == 2
    assert not np.array_equal(outputs[0], outputs[1])
