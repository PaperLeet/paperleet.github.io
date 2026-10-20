"""Independent math attention, grouped heads, mixed AD and stochastic tape rules."""
import numpy as np
import pytest
from torchwasm import torch


def sum_to(value, shape):
    while value.ndim > len(shape):
        value = value.sum(0)
    for axis, size in enumerate(shape):
        if size == 1 and value.shape[axis] != 1:
            value = value.sum(axis, keepdims=True)
    return value.reshape(shape)


def reference(arrays, *, grouped=False, mask=None, causal=False, factor=.37, seed=None, directions=None, multiplier=None):
    q, k, v = arrays
    kh, vh = (k.shape[-3], v.shape[-3]) if grouped else (None, None)
    if grouped:
        k = np.repeat(k, q.shape[-3] // kh, axis=-3)
        v = np.repeat(v, q.shape[-3] // vh, axis=-3)
    scores = q @ k.swapaxes(-1, -2) * factor
    if causal:
        scores = np.where(np.arange(k.shape[-2]) <= np.arange(q.shape[-2])[:, None], scores, -np.inf)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf) if mask.dtype.kind == 'b' else scores + mask
    if scores.shape[-1]:
        maximum = np.max(scores, axis=-1, keepdims=True)
        exp = np.exp(scores - np.where(np.isneginf(maximum), 0, maximum))
        total = exp.sum(-1, keepdims=True)
        p = np.divide(exp, total, out=np.zeros_like(exp), where=total != 0)
    else:
        p = np.zeros_like(scores)
    used = p if multiplier is None else p * multiplier
    y = used @ v
    gradients = None
    if seed is not None:
        incoming = seed @ v.swapaxes(-1, -2)
        if multiplier is not None:
            incoming *= multiplier
        ds = p * (incoming - (p * incoming).sum(-1, keepdims=True))
        dq = ds @ k * factor
        dk = ds.swapaxes(-1, -2) @ q * factor
        dv = used.swapaxes(-1, -2) @ seed
        if grouped:
            dk = dk.reshape((*dk.shape[:-3], kh, q.shape[-3] // kh, *dk.shape[-2:])).sum(-3)
            dv = dv.reshape((*dv.shape[:-3], vh, q.shape[-3] // vh, *dv.shape[-2:])).sum(-3)
        gradients = [sum_to(g, a.shape) for g, a in zip((dq, dk, dv), arrays)]
        if mask is not None and mask.dtype.kind == 'f':
            gradients.append(sum_to(ds, mask.shape))
    tangent = None
    if directions is not None:
        dq, dk, dv = directions[:3]
        if grouped:
            dk = np.repeat(dk, q.shape[-3] // kh, axis=-3)
            dv = np.repeat(dv, q.shape[-3] // vh, axis=-3)
        ds = (dq @ k.swapaxes(-1, -2) + q @ dk.swapaxes(-1, -2)) * factor
        if mask is not None and mask.dtype.kind == 'f':
            ds += directions[3]
        dp = p * (ds - (p * ds).sum(-1, keepdims=True))
        if multiplier is not None:
            dp *= multiplier
        tangent = dp @ v + used @ dv
    return y, gradients, tangent


SHAPES = [
    ((3, 4), (5, 4), (5, 2)),
    ((2, 3, 4), (1, 5, 4), (2, 5, 2)),
    ((2, 3, 4, 5), (1, 3, 6, 5), (2, 1, 6, 7)),
    ((2, 1, 3, 4, 5), (1, 2, 3, 6, 5), (2, 1, 1, 6, 7)),
]
GQA_SHAPES = [
    ((4, 3, 4), (2, 5, 4), (2, 5, 6)),
    ((2, 6, 3, 4), (1, 2, 5, 4), (2, 3, 5, 6)),
    ((2, 1, 4, 3, 4), (1, 2, 1, 5, 4), (2, 1, 2, 5, 6)),
    ((2, 4, 3, 4), (1, 4, 5, 4), (2, 1, 5, 6)),
]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shapes,grouped', [(s, False) for s in SHAPES] + [(s, True) for s in GQA_SHAPES])
@pytest.mark.parametrize('kind', ['none', 'bool', 'float', 'causal'])
def test_values_vjp_jvp_and_repeated_seeds(dtype, shapes, grouped, kind):
    rng = np.random.default_rng(5010)
    arrays = [rng.normal(size=s).astype(dtype) for s in shapes]
    options = dict(enable_gqa=grouped, scale=.37)
    mask = None
    if kind in ('bool', 'float'):
        keep = rng.random((shapes[0][-2], shapes[1][-2])) > .3
        keep[1] = False
        mask = keep if kind == 'bool' else np.where(keep, rng.normal(size=keep.shape), -np.inf).astype(dtype)
        options['attn_mask'] = torch.tensor(mask, requires_grad=kind == 'float')
    options['is_causal'] = kind == 'causal'
    values = [torch.from_numpy(np.asfortranarray(a)).requires_grad_() for a in arrays]
    active = values + ([options['attn_mask']] if kind == 'float' else [])
    output = torch.nn.functional.scaled_dot_product_attention(*values, **options)
    for _ in range(2):
        seed = rng.normal(size=output.shape).astype(dtype)
        got = torch.autograd.grad(output, active, torch.tensor(seed), retain_graph=True)
        want, gradients, _ = reference(arrays, grouped=grouped, mask=mask, causal=kind == 'causal', seed=seed)
        for a, b in zip([output, *got], [want, *gradients]):
            np.testing.assert_allclose(a.numpy(), b, rtol=8e-6, atol=4e-6)
            assert a.dtype == dtype
    directions = [rng.normal(size=x.shape).astype(dtype) for x in active]
    def call(*x):
        opts = dict(options)
        if kind == 'float':
            opts['attn_mask'] = x[3]
        return torch.nn.functional.scaled_dot_product_attention(*x[:3], **opts)
    p, t = torch.func.jvp(call, tuple(active), tuple(torch.tensor(d) for d in directions))
    _, _, expected = reference(arrays, grouped=grouped, mask=mask, causal=kind == 'causal', directions=directions)
    np.testing.assert_allclose(p.numpy(), want, rtol=8e-6, atol=4e-6)
    np.testing.assert_allclose(t.numpy(), expected, rtol=1e-5, atol=8e-6)


@pytest.mark.parametrize('kind', ['none', 'bool', 'float', 'float_masked', 'causal'])
@pytest.mark.parametrize('grouped', [False, True])
def test_reverse_of_jvp_all_primal_and_direction_inputs(kind, grouped):
    rng = np.random.default_rng(5011)
    shapes = [(1, 2, 2, 2), (1, 1 if grouped else 2, 3, 2), (1, 1 if grouped else 2, 3, 2)]
    arrays = [rng.normal(size=s) for s in shapes]
    mask = None
    if kind == 'bool':
        mask = np.array([[False, False, False], [True, False, True]])
    if kind in ('float', 'float_masked'):
        mask = rng.normal(size=(2, 3))
        if kind == 'float_masked':
            mask[0] = -np.inf
        arrays.append(mask)
    arrays += [rng.normal(size=a.shape) for a in arrays]
    n = len(arrays) // 2
    values = [torch.tensor(a, requires_grad=True) for a in arrays]
    def call(*x):
        return torch.nn.functional.scaled_dot_product_attention(*x[:3], attn_mask=x[3] if kind.startswith('float') else (torch.tensor(mask) if mask is not None else None), is_causal=kind == 'causal', enable_gqa=grouped, scale=.37)
    p, t = torch.func.jvp(call, tuple(values[:n]), tuple(values[n:]))
    seeds = [rng.normal(size=p.shape), rng.normal(size=p.shape)]
    (p * torch.tensor(seeds[0]) + t * torch.tensor(seeds[1])).sum().backward()
    def evaluate(raw):
        y, _, tangent = reference(raw[:3], grouped=grouped, mask=raw[3] if kind.startswith('float') else mask, causal=kind == 'causal', directions=raw[n:])
        return float((y * seeds[0] + tangent * seeds[1]).sum())
    for position, value in enumerate(values):
        gradient = np.empty_like(arrays[position])
        for index in np.ndindex(gradient.shape):
            plus = [a.copy() for a in arrays]; minus = [a.copy() for a in arrays]
            plus[position][index] += 1e-6; minus[position][index] -= 1e-6
            gradient[index] = (evaluate(plus) - evaluate(minus)) / 2e-6
        np.testing.assert_allclose(value.grad.numpy(), gradient, rtol=8e-6, atol=2e-7)


@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('probability', [0., .25, 1.])
def test_dropout_uses_one_mask_for_forward_reverse_and_jvp(monkeypatch, grouped, probability):
    from torchwasm import _random
    rng = np.random.default_rng(5012)
    shapes = GQA_SHAPES[0] if grouped else SHAPES[1]
    arrays = [rng.normal(size=s) for s in shapes]
    draws = []
    def sample(generator, shape, dtype):
        draws.append(shape)
        return np.linspace(.01, .99, np.prod(shape)).reshape(shape).astype(dtype)
    monkeypatch.setattr(_random, 'uniform_samples', sample)
    shape = (shapes[0][0], shapes[0][-2], shapes[1][-2])
    multiplier = None if probability == 0 else np.zeros(shape) if probability == 1 else (sample(None, shape, np.float64) >= probability) / (1 - probability)
    draws.clear()
    values = [torch.tensor(a, requires_grad=True) for a in arrays]
    fn = lambda *x: torch.nn.functional.scaled_dot_product_attention(*x, dropout_p=probability, enable_gqa=grouped, scale=.37)
    y = fn(*values); seed = rng.normal(size=y.shape)
    y.backward(torch.tensor(seed), retain_graph=True)
    want, gradients, _ = reference(arrays, grouped=grouped, multiplier=multiplier, seed=seed)
    for a, b in zip([y, *[x.grad for x in values]], [want, *gradients]):
        np.testing.assert_allclose(a.numpy(), b, rtol=1e-12, atol=1e-12)
    directions = [rng.normal(size=a.shape) for a in arrays]
    p, t = torch.func.jvp(fn, tuple(values), tuple(torch.tensor(a) for a in directions))
    _, _, expected = reference(arrays, grouped=grouped, multiplier=multiplier, directions=directions)
    np.testing.assert_allclose(p.numpy(), want, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t.numpy(), expected, rtol=1e-12, atol=1e-12)
    assert len(draws) == (2 if probability == .25 else 0)


def test_gqa_never_repeats_kv_storage(monkeypatch):
    def reject(*a, **k):
        raise AssertionError('repeated K/V storage')
    monkeypatch.setattr(np, 'repeat', reject)
    monkeypatch.setattr(np, 'tile', reject)
    monkeypatch.setattr(torch.Tensor, 'repeat_interleave', reject)
    monkeypatch.setattr(torch.Tensor, 'repeat', reject)
    values = [torch.ones(*s, requires_grad=True) for s in GQA_SHAPES[1]]
    p, t = torch.func.jvp(lambda *x: torch.nn.functional.scaled_dot_product_attention(*x, enable_gqa=True), tuple(values), tuple(torch.ones_like(x) for x in values))
    (p + t).sum().backward()
    torch.nn.functional.scaled_dot_product_attention(*values, enable_gqa=True).sum().backward()
    assert all(np.isfinite(x.grad.numpy()).all() for x in values)


@pytest.mark.parametrize('heads', [(4, 0, 0), (0, 1, 1), (0, 2, 3)])
def test_empty_gqa_heads_keep_zero_gradients(heads):
    values = [torch.ones(2, h, n, d, dtype=torch.float64, requires_grad=True) for h, n, d in zip(heads, [3, 5, 5], [4, 4, 6])]
    y = torch.nn.functional.scaled_dot_product_attention(*values, enable_gqa=True)
    y.sum().backward()
    assert y.shape == (2, heads[0], 3, 6)
    np.testing.assert_array_equal(y.numpy(), 0.)
    for x in values:
        np.testing.assert_array_equal(x.grad.numpy(), 0.)


@pytest.mark.parametrize('changed', [0, 1, 2, 3])
@pytest.mark.parametrize('forward', [False, True])
def test_saved_versions_survive_mask_dropout_and_grouping(changed, forward):
    values = [torch.ones(*s, requires_grad=True) for s in GQA_SHAPES[0]] + [torch.zeros(3, 5, requires_grad=True)]
    fn = lambda *x: torch.nn.functional.scaled_dot_product_attention(*x[:3], attn_mask=x[3], enable_gqa=True, dropout_p=.25)
    y = torch.func.jvp(fn, tuple(values), tuple(torch.ones_like(x) for x in values))[1] if forward else fn(*values)
    values[changed].detach().add_(1)
    with pytest.raises(RuntimeError, match='modified in-place'):
        y.sum().backward()


@pytest.mark.parametrize('option', [dict(dropout_p=-.1), dict(dropout_p=1.1), dict(dropout_p=float('nan')), dict(is_causal=True, attn_mask=torch.ones(2, 2, dtype=torch.bool)), dict(enable_gqa=True)])
def test_invalid_arguments(option):
    shapes = [(2, 4, 2, 3), (2, 3 if option.get('enable_gqa') else 4, 2, 3), (2, 4, 2, 3)]
    with pytest.raises(RuntimeError):
        torch.nn.functional.scaled_dot_product_attention(*(torch.ones(*s) for s in shapes), **option)


@pytest.mark.parametrize('active', range(16))
def test_frozen_inputs_shared_gradients_and_no_grad(active):
    rng = np.random.default_rng(5013)
    arrays = [rng.normal(size=s) for s in GQA_SHAPES[0]] + [rng.normal(size=(3, 5))]
    values = [torch.tensor(a, requires_grad=bool(active & (1 << i))) for i, a in enumerate(arrays)]
    y = torch.nn.functional.scaled_dot_product_attention(*values[:3], attn_mask=values[3], enable_gqa=True, scale=.37)
    assert y.requires_grad == bool(active)
    if active:
        seed = rng.normal(size=y.shape); y.backward(torch.tensor(seed))
        _, expected, _ = reference(arrays[:3], grouped=True, mask=arrays[3], seed=seed)
        for a, b in zip(values, expected):
            if a.requires_grad: np.testing.assert_allclose(a.grad.numpy(), b, rtol=1e-12, atol=1e-12)
            else: assert a.grad is None
    with torch.no_grad():
        assert not torch.nn.functional.scaled_dot_product_attention(*values[:3], attn_mask=values[3], enable_gqa=True).requires_grad


def test_float32_mask_with_double_inputs_and_mask_only_jvp():
    q = torch.ones(2, 4, 3, 4, dtype=torch.float64, requires_grad=True)
    k = torch.ones(1, 2, 5, 4, dtype=torch.float64, requires_grad=True)
    v = torch.arange(60, dtype=torch.float64).reshape(2, 2, 5, 3).requires_grad_()
    m = torch.zeros(3, 5, dtype=torch.float32, requires_grad=True)
    p, t = torch.func.jvp(lambda mm: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mm, enable_gqa=True), (m,), (torch.ones_like(m),))
    np.testing.assert_allclose(t.numpy(), 0., atol=1e-12)
    (p + t).sum().backward()
    assert m.grad.dtype == np.dtype('float32') and p.dtype == t.dtype == np.dtype('float64')
    assert torch._browser_capabilities['torch.nn.functional.scaled_dot_product_attention'].forward


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('empty', ['query', 'key', 'feature', 'value'])
def test_empty_sequence_and_features(dtype, empty):
    shapes = [(2, 4, 3, 4), (1, 2, 5, 4), (2, 2, 5, 6)]
    axis = {'query': (0, -2), 'key': (1, -2), 'feature': (0, -1), 'value': (2, -1)}[empty]
    shapes = [list(s) for s in shapes]; shapes[axis[0]][axis[1]] = 0
    if empty == 'key': shapes[2][-2] = 0
    if empty == 'feature': shapes[1][-1] = 0
    arrays = [np.ones(s, dtype=dtype) for s in shapes]
    values = [torch.tensor(a, requires_grad=True) for a in arrays]
    p, t = torch.func.jvp(lambda *x: torch.nn.functional.scaled_dot_product_attention(*x, enable_gqa=True), tuple(values), tuple(torch.ones_like(x) for x in values))
    expected, _, derivative = reference(arrays, grouped=True, factor=1. if empty == 'feature' else .5, directions=arrays)
    np.testing.assert_allclose(p.numpy(), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(t.numpy(), derivative, rtol=2e-6, atol=2e-6)
    (p + t).sum().backward()
    assert all(np.isfinite(x.grad.numpy()).all() for x in values)


def test_grouped_encoder_sgd_mixed_ad_checkpoint_and_dropout_continuation():
    import copy
    class Encoder(torch.nn.Module):
        def __init__(self, grouped):
            super().__init__(); self.grouped = grouped
            self.q = torch.nn.Linear(8, 16, dtype=torch.float64)
            self.k = torch.nn.Linear(8, 4, dtype=torch.float64)
            self.v = torch.nn.Linear(8, 6, dtype=torch.float64)
            self.out = torch.nn.Linear(12, 8, dtype=torch.float64)
            self.norm = torch.nn.LayerNorm(8, dtype=torch.float64)
        def forward(self, x):
            b, length = x.shape[:2]
            q = self.q(x).reshape(b, length, 4, 4).transpose(1, 2)
            k = self.k(x).reshape(b, length, 1, 4).transpose(1, 2)
            v = self.v(x).reshape(b, length, 2, 3).transpose(1, 2)
            if not self.grouped:
                k = k.unsqueeze(2).expand(b, 1, 4, length, 4).reshape(b, 4, length, 4)
                v = v.unsqueeze(2).expand(b, 2, 2, length, 3).reshape(b, 4, length, 3)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=self.grouped, is_causal=True, dropout_p=.25 if self.training else 0.)
            return self.norm(x + self.out(y.transpose(1, 2).reshape(b, length, 12)))
    torch.manual_seed(5014); first = Encoder(True); second = Encoder(False)
    second.load_state_dict(first.state_dict())
    models = [first, second]; optimizers = [torch.optim.SGD(m.parameters(), lr=.01, momentum=.7) for m in models]
    x = torch.randn(2, 3, 8, dtype=torch.float64); tangent = torch.randn_like(x)
    for step in range(4):
        losses = []
        for model, optimizer in zip(models, optimizers):
            torch.manual_seed(800 + step); optimizer.zero_grad()
            p, t = torch.func.jvp(model, (x,), (tangent,))
            loss = p.square().mean() + .03 * t.square().mean(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        np.testing.assert_allclose(losses[0], losses[1], rtol=1e-12, atol=1e-12)
        for a, b in zip(first.parameters(), second.parameters()):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-12, atol=1e-12)
        if step == 1:
            saved = copy.deepcopy({'model': second.state_dict(), 'optimizer': optimizers[1].state_dict()})
            models[1] = second = Encoder(False); second.load_state_dict(saved['model'])
            optimizers[1] = torch.optim.SGD(second.parameters(), lr=.01, momentum=.7); optimizers[1].load_state_dict(saved['optimizer'])
    first.eval(); second.eval()
    np.testing.assert_allclose(first(x).numpy(), second(x).numpy(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('masked', [False, True])
def test_shared_qkv_and_mask_view_mixed_derivatives(grouped, masked):
    rng = np.random.default_rng(5016)
    array = rng.normal(size=(2, 3, 3)); direction = rng.normal(size=array.shape)
    x = torch.tensor(array, requires_grad=True); dx = torch.tensor(direction, requires_grad=True)
    fn = lambda a: torch.nn.functional.scaled_dot_product_attention(a, a, a, attn_mask=a[0] if masked else None, enable_gqa=grouped, scale=.37)
    p, t = torch.func.jvp(fn, (x,), (dx,)); pseed = rng.normal(size=p.shape); tseed = rng.normal(size=p.shape)
    (p * torch.tensor(pseed) + t * torch.tensor(tseed)).sum().backward()
    def evaluate(a, d):
        mask = a[0] if masked else None
        y, _, tangent = reference([a, a, a], mask=mask, directions=[d, d, d] + ([d[0]] if masked else []), grouped=grouped)
        return float((y * pseed + tangent * tseed).sum())
    for position, (raw, value) in enumerate([(array, x), (direction, dx)]):
        expected = np.empty_like(raw)
        for index in np.ndindex(raw.shape):
            plus = [array.copy(), direction.copy()]; minus = [array.copy(), direction.copy()]
            plus[position][index] += 1e-6; minus[position][index] -= 1e-6
            expected[index] = (evaluate(*plus) - evaluate(*minus)) / 2e-6
        np.testing.assert_allclose(value.grad.numpy(), expected, rtol=8e-6, atol=3e-7)


def test_dropout_rng_replay_and_single_call_transform():
    values = [torch.ones(*s) for s in GQA_SHAPES[0]]
    fn = lambda *x: torch.nn.functional.scaled_dot_product_attention(*x, enable_gqa=True, dropout_p=.3)
    torch.manual_seed(812); y = fn(*values)
    torch.manual_seed(812); p, t = torch.func.jvp(fn, tuple(values), tuple(torch.ones_like(x) for x in values))
    np.testing.assert_array_equal(y.numpy(), p.numpy())
    np.testing.assert_allclose(t.numpy(), p.numpy(), rtol=2e-6, atol=2e-6)
    torch.manual_seed(812)
    with torch.random.fork_rng():
        np.testing.assert_array_equal(y.numpy(), fn(*values).numpy())
    np.testing.assert_array_equal(y.numpy(), fn(*values).numpy())


def test_geometry_cache_is_bounded_and_retains_no_tensor_state():
    import gc
    import weakref
    from torchwasm import _attention
    _attention._geometry.cache_clear()
    values = [torch.ones(*s) for s in GQA_SHAPES[0]]
    refs = [weakref.ref(x) for x in values]
    with torch.no_grad():
        torch.nn.functional.scaled_dot_product_attention(*values, enable_gqa=True)
    del values
    gc.collect()
    assert all(ref() is None for ref in refs)
    for length in range(1, 80):
        _attention._geometry((1, 2, length, 3), (1, 1, 4, 3), (1, 1, 4, 3), None, True)
    assert _attention._geometry.cache_info().currsize == 64


@pytest.mark.parametrize('dtype,value', [(np.float32, 1e19), (np.float64, 7e153)])
@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('scale', [None, .5, -.5, 0.])
@pytest.mark.parametrize('masked', [False, True])
def test_prescaling_prevents_premature_dot_overflow(dtype, value, grouped, scale, masked):
    heads = (4, 2, 1) if grouped else (2, 2, 2)
    arrays = [np.full((1, heads[0], 2, 4), value, dtype=dtype), np.full((1, heads[1], 3, 4), value, dtype=dtype), np.ones((1, heads[2], 3, 2), dtype=dtype)]
    values = [torch.tensor(a, requires_grad=True) for a in arrays]
    options = dict(enable_gqa=grouped, scale=scale)
    if masked: options['attn_mask'] = torch.zeros(2, 3, dtype=torch.bool)
    y = torch.nn.functional.scaled_dot_product_attention(*values, **options)
    y.sum().backward()
    np.testing.assert_allclose(y.numpy(), 0. if masked else 1., rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(values[0].grad.numpy(), 0.)
    np.testing.assert_array_equal(values[1].grad.numpy(), 0.)
    np.testing.assert_allclose(values[2].grad.numpy(), 0. if masked else heads[0] / heads[2] * 2 / 3, rtol=2e-6, atol=2e-6)
    for x in values: x.grad = None
    directions = [torch.zeros_like(values[0]).requires_grad_(), torch.zeros_like(values[1]).requires_grad_(), torch.ones_like(values[2]).requires_grad_()]
    p, t = torch.func.jvp(lambda *x: torch.nn.functional.scaled_dot_product_attention(*x, **options), tuple(values), tuple(directions))
    np.testing.assert_allclose(t.numpy(), p.numpy(), rtol=2e-6, atol=2e-6)
    (p + t).sum().backward()
    assert all(np.isfinite(x.grad.numpy()).all() for x in [*values, *directions])
