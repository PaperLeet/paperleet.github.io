"""Independent one-dimensional region oracles and unchanged-import workflows."""
import pickle

import numpy as np
import pytest

from stridetorch import UnsupportedOperation, as_torch, torch


@pytest.fixture(params=[('scan', 'loop'), ('packed', 'scatter'), ('matrix', 'packed'),
                       ('block', 'block'), ('auto', 'auto')], autouse=True)
def algorithm(request, monkeypatch):
    from stridetorch import _average_pooling
    monkeypatch.setattr(_average_pooling, '_ALGORITHM', request.param[0])
    monkeypatch.setattr(_average_pooling, '_BACKWARD_ALGORITHM', request.param[1])


def regions(length, kind):
    if kind == 'fixed':
        return [(i, i + 2, 2) for i in range(0, length - 1, 2)]
    if kind == 'ceil':
        # Kernel 3, stride 2, padding 1, ceil mode, excluding padding.
        return [(max(0, i - 1), min(length, i + 2),
                 min(length, i + 2) - max(0, i - 1))
                for i in range(0, length + 1, 2) if i - 1 < length]
    count = {'irregular': 3, 'global': 1, 'upsample': 11}[kind]
    return [(i * length // count, ((i + 1) * length + count - 1) // count,
             ((i + 1) * length + count - 1) // count - i * length // count)
            for i in range(count)]


def operation(api, x, kind):
    if kind == 'fixed': return api.nn.functional.avg_pool1d(x, 2)
    if kind == 'ceil': return api.nn.functional.avg_pool1d(x, 3, 2, 1, True, False)
    return api.nn.functional.adaptive_avg_pool1d(x, {'irregular': 3, 'global': 1, 'upsample': 11}[kind])


def oracle(data, windows, seed=None):
    if seed is None:
        result = np.empty((*data.shape[:-1], len(windows)), dtype=data.dtype)
        for i, (start, stop, count) in enumerate(windows):
            result[..., i] = data[..., start:stop].sum(axis=-1) / count
    else:
        result = np.zeros_like(data)
        for i, (start, stop, count) in enumerate(windows):
            result[..., start:stop] += seed[..., i, None] / count
    return result


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(2, 7), (2, 3, 7), (0, 2, 7)])
@pytest.mark.parametrize('kind', ['fixed', 'ceil', 'irregular', 'global', 'upsample'])
@pytest.mark.parametrize('layout', ['contiguous', 'fortran', 'strided'])
def test_region_values_pullback_jvp_and_reverse_tangent(dtype, shape, kind, layout):
    rng = np.random.default_rng(4101)
    a = rng.normal(size=shape).astype(dtype)
    if layout == 'fortran': a = np.asfortranarray(a)
    if layout == 'strided': a = np.repeat(a, 2, axis=-1)[..., ::2]
    before = a.copy(); v = rng.normal(size=shape).astype(dtype)
    x = torch.from_numpy(a).requires_grad_(); direction = torch.tensor(v, requires_grad=True)
    y = operation(torch, x, kind); windows = regions(shape[-1], kind)
    expected = oracle(a, windows); seed = rng.normal(size=y.shape).astype(dtype)
    np.testing.assert_allclose(y.numpy(), expected, rtol=3e-6, atol=3e-7)
    y.backward(torch.tensor(seed))
    np.testing.assert_allclose(x.grad.numpy(), oracle(a, windows, seed), rtol=3e-6, atol=3e-7)
    x.grad = None
    p, tangent = torch.func.jvp(lambda z: operation(torch, z, kind), (x,), (direction,))
    np.testing.assert_allclose(p.numpy(), expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(tangent.numpy(), oracle(v, windows), rtol=3e-6, atol=3e-7)
    tangent.backward(torch.tensor(seed))
    assert x.grad is None
    np.testing.assert_allclose(direction.grad.numpy(), oracle(v, windows, seed), rtol=3e-6, atol=3e-7)
    np.testing.assert_array_equal(a, before)
    assert not np.shares_memory(a, y.numpy())


@pytest.mark.parametrize('kind', ['fixed', 'ceil', 'irregular', 'global', 'upsample'])
def test_directional_derivative_is_independent_of_graph(kind):
    rng = np.random.default_rng(4102)
    a = rng.normal(size=(1, 2, 7)); direction = rng.normal(size=a.shape)
    _, tangent = torch.func.jvp(lambda z: operation(torch, z, kind),
                               (torch.tensor(a),), (torch.tensor(direction),))
    windows = regions(7, kind); epsilon = 1e-5
    numerical = (oracle(a + epsilon * direction, windows) - oracle(a - epsilon * direction, windows)) / (2 * epsilon)
    np.testing.assert_allclose(tangent.numpy(), numerical, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('name', ['avg_pool1d', 'adaptive_avg_pool1d'])
@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
def test_nonfinite_windows_and_seeds_remain_local(name, value):
    a = np.ones((1, 1, 8)); a[..., 0] = value
    x = torch.tensor(a, requires_grad=True)
    y = getattr(torch.nn.functional, name)(x, 2 if name == 'avg_pool1d' else 4)
    np.testing.assert_allclose(y.numpy(), [[[value, 1., 1., 1.]]], equal_nan=True)
    y.backward(torch.tensor([[[1., 1., value, 1.]]]))
    np.testing.assert_allclose(x.grad.numpy(), [[[.5, .5, .5, .5, value, value, .5, .5]]], equal_nan=True)


@pytest.mark.parametrize('shape,backward_error', [((0, 7), False), ((1, 0, 7), True), ((0, 2, 7), False)])
def test_empty_channels_distinguish_unbatched_and_batched_input(shape, backward_error):
    x = torch.empty(shape, requires_grad=True)
    y = torch.nn.functional.adaptive_avg_pool1d(x, 2)
    assert y.shape == (*shape[:-1], 2)
    if backward_error:
        with pytest.raises(RuntimeError): y.sum().backward()
    else:
        y.sum().backward(); assert x.grad.shape == shape


def test_zero_output_and_explicit_support_boundaries():
    x = torch.ones(1, 2, 7, requires_grad=True)
    y = torch.nn.functional.adaptive_avg_pool1d(x, 0)
    assert y.shape == (1, 2, 0)
    with pytest.raises(RuntimeError): y.sum().backward()
    with pytest.raises(UnsupportedOperation): torch.nn.functional.adaptive_avg_pool1d(x, 4_000_001)
    with pytest.raises(UnsupportedOperation): torch.nn.functional.adaptive_avg_pool1d(torch.ones(1, 2, 7, dtype=torch.int64), 2)
    for shape in [(7,), (1, 2, 0), (1, 2, 3, 7)]:
        with pytest.raises(RuntimeError): torch.nn.functional.avg_pool1d(torch.empty(shape), 2)
    for size in [True, None, 2., [None]]:
        with pytest.raises(TypeError): torch.nn.functional.adaptive_avg_pool1d(x, size)
    for size in [[], [2, 3], -1]:
        with pytest.raises(RuntimeError): torch.nn.functional.adaptive_avg_pool1d(x, size)
    with pytest.raises(ValueError): torch.nn.functional.avg_pool1d(x, 2**63)
    empty = torch.empty(1, 2, 0, requires_grad=True)
    global_mean = torch.nn.functional.adaptive_avg_pool1d(empty, 1)
    assert global_mean.shape == (1, 2, 1) and np.isnan(global_mean.numpy()).all()
    global_mean.sum().backward(); assert empty.grad.shape == empty.shape


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64, torch.int64])
def test_inference_dtypes_and_padding(dtype):
    x = torch.arange(8, dtype=dtype).reshape(1, 1, 8)
    y = torch.nn.functional.avg_pool1d(x, 2)
    expected = np.asarray([[[0, 2, 4, 6]]]) if dtype == torch.int64 else np.asarray([[[.5, 2.5, 4.5, 6.5]]])
    np.testing.assert_array_equal(y.numpy(), expected)
    assert y.dtype == dtype
    ones = torch.ones(1, 1, 4, dtype=torch.float64)
    np.testing.assert_allclose(torch.nn.functional.avg_pool1d(ones, 3, 2, 1, True).numpy(), [[[2/3, 1., .5]]])
    np.testing.assert_array_equal(torch.nn.functional.avg_pool1d(ones, 3, 2, 1, True, False).numpy(), 1.)


@pytest.mark.parametrize('name,shape,size', [
    ('avg_pool1d', (0, 2, 7), 3), ('adaptive_avg_pool1d', (0, 2, 7), 3),
    ('adaptive_avg_pool1d', (0, 7), 3), ('adaptive_avg_pool1d', (1, 0, 7), 3),
    ('adaptive_avg_pool1d', (1, 2, 7), 0), ('adaptive_avg_pool1d', (0, 2, 7), 1),
    ('avg_pool2d', (0, 2, 5, 7), 2), ('adaptive_avg_pool2d', (0, 2, 5, 7), (2, 3)),
    ('adaptive_avg_pool2d', (1, 0, 5, 7), (2, 3)), ('adaptive_avg_pool2d', (1, 2, 5, 7), (0, 3)),
    ('adaptive_avg_pool2d', (0, 2, 5, 7), (1, 1)), ('adaptive_avg_pool2d', (1, 0, 5, 7), (1, 1)),
])
def test_empty_pool_outputs_have_conventional_contiguous_strides(name, shape, size):
    y = getattr(torch.nn.functional, name)(torch.empty(shape), size)
    assert y.numel() == 0 and y.is_contiguous()
    strides = []; step = 1
    for length in reversed(y.shape): strides.append(step); step *= max(1, length)
    assert y.stride() == tuple(reversed(strides))


def test_aliases_modules_metadata_and_pickle():
    with as_torch() as api:
        import torch as imported
        from torch.nn.modules.pooling import AvgPool1d, AdaptiveAvgPool1d
        from torch.nn.functional import avg_pool1d, adaptive_avg_pool1d
        assert imported is api and AvgPool1d is api.nn.AvgPool1d
        x = api.arange(14, dtype=api.float64).reshape(1, 2, 7)
        for cls, function, root in [(AvgPool1d, avg_pool1d, api.avg_pool1d),
                                    (AdaptiveAvgPool1d, adaptive_avg_pool1d, api.adaptive_avg_pool1d)]:
            model = cls(3); restored = pickle.loads(pickle.dumps(model))
            assert not list(model.parameters()) and not model.state_dict()
            np.testing.assert_array_equal(model(x).numpy(), restored(x).numpy())
            np.testing.assert_array_equal(model(x).numpy(), function(x, 3).numpy())
            np.testing.assert_array_equal(model(x).numpy(), root(x, 3).numpy())
        assert AvgPool1d(3, 2, 1).kernel_size == (3,)
        assert str(AvgPool1d(3, 2, 1)) == 'AvgPool1d(kernel_size=(3,), stride=(2,), padding=(1,))'


def test_convolution_pooling_training_checkpoint_resume_and_tangent():
    def make():
        return torch.nn.Sequential(torch.nn.Conv1d(2, 3, 3, padding=1), torch.nn.SiLU(),
                                   torch.nn.AvgPool1d(3, 2, 1, True, False),
                                   torch.nn.AdaptiveAvgPool1d(3), torch.nn.Flatten(), torch.nn.Linear(9, 2)).double()
    torch.manual_seed(4103); model = make(); x = torch.ones(2, 2, 17, dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=.02, momentum=.9)
    for _ in range(3):
        optimizer.zero_grad(); model(x).square().mean().backward(); optimizer.step()
    saved = pickle.loads(pickle.dumps((model.state_dict(), optimizer.state_dict())))
    restored = make(); restored.load_state_dict(saved[0])
    resumed = torch.optim.SGD(restored.parameters(), lr=.02, momentum=.9); resumed.load_state_dict(saved[1])
    for m, opt in [(model, optimizer), (restored, resumed)]:
        opt.zero_grad(); m(x).square().mean().backward(); opt.step()
    for a, b in zip(model.parameters(), restored.parameters()): np.testing.assert_array_equal(a.numpy(), b.numpy())
    primal, tangent = torch.func.jvp(restored, (x,), (torch.ones_like(x),))
    epsilon = 1e-5
    numerical = (restored(x + epsilon).numpy() - restored(x - epsilon).numpy()) / (2 * epsilon)
    np.testing.assert_allclose(primal.numpy(), restored(x).numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tangent.numpy(), numerical, rtol=1e-8, atol=1e-10)
