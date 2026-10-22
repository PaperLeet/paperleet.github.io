"""Native 1D average-pooling values, linear tapes and mutation contracts."""
import pickle
import numpy as np
import pytest
import torch
from torchwasm import torch as lite, UnsupportedOperation


def dtype_for(api, dtype): return getattr(api, np.dtype(dtype).name)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(2, 7), (1, 2, 7), (0, 2, 7)])
@pytest.mark.parametrize('kind', ['fixed', 'ceil', 'adaptive', 'global', 'upsample'])
@pytest.mark.parametrize('layout', ['contiguous', 'strided'])
def test_values_vjp_jvp_and_reverse_tangent(dtype, shape, kind, layout):
    rng = np.random.default_rng(4001)
    a = rng.normal(size=shape).astype(dtype); v = rng.normal(size=shape).astype(dtype)
    if layout == 'strided': a = np.repeat(a, 2, axis=-1)[..., ::2]
    rows = []
    for api in (lite, torch):
        x = api.from_numpy(a).requires_grad_(); direction = api.tensor(v, requires_grad=True)
        if kind == 'fixed': fn = lambda z: api.nn.functional.avg_pool1d(z, 2)
        elif kind == 'ceil': fn = lambda z: api.nn.functional.avg_pool1d(z, 3, 2, 1, True, False)
        else: fn = lambda z: api.nn.functional.adaptive_avg_pool1d(z, {'adaptive': 3, 'global': 1, 'upsample': 12}[kind])
        y = fn(x); y.square().sum().backward()
        p, tangent = api.func.jvp(fn, (x,), (direction,)); tangent.square().sum().backward()
        rows.append([z.detach().numpy() for z in (y, x.grad, p, tangent, direction.grad)])
    for a, b in zip(*rows): np.testing.assert_allclose(a, b, rtol=4e-6, atol=2e-7)


@pytest.mark.parametrize('name', ['avg_pool1d', 'adaptive_avg_pool1d'])
@pytest.mark.parametrize('global_size', [False, True])
@pytest.mark.parametrize('target', ['input', 'output', 'tangent', 'tangent_input', 'jvp_input'])
def test_storage_mutations(name, global_size, target):
    rows = []
    for api in (lite, torch):
        x = api.ones(1, 2, 5, dtype=api.float64, requires_grad=True)
        v = api.ones_like(x, requires_grad=True)
        fn = (lambda z: api.nn.functional.adaptive_avg_pool1d(z, 1 if global_size else 3)) if name == 'adaptive_avg_pool1d' else (lambda z: api.nn.functional.avg_pool1d(z, 5 if global_size else 2))
        if target in ('tangent', 'tangent_input', 'jvp_input'): p, t = api.func.jvp(fn, (x,), (v,)); y = t
        else: p = fn(x); y = p
        with api.no_grad(): {'input': x, 'output': p, 'tangent': t if target == 'tangent' else p, 'tangent_input': v, 'jvp_input': x}[target].detach().add_(1)
        try: y.sum().backward(); rows.append(('success', x.grad is not None, v.grad is not None))
        except RuntimeError: rows.append(('RuntimeError',))
    assert rows[0] == rows[1]


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int64])
@pytest.mark.parametrize('name', ['avg_pool1d', 'adaptive_avg_pool1d'])
def test_inference_types_and_modules(dtype, name):
    a = np.arange(14, dtype=dtype).reshape(1, 2, 7); rows = []
    if dtype is np.int64 and name == 'adaptive_avg_pool1d':
        with pytest.raises(RuntimeError): torch.nn.AdaptiveAvgPool1d(3)(torch.tensor(a))
        with pytest.raises(UnsupportedOperation): lite.nn.AdaptiveAvgPool1d(3)(lite.tensor(a))
        return
    for api in (lite, torch):
        cls = api.nn.AvgPool1d if name == 'avg_pool1d' else api.nn.AdaptiveAvgPool1d
        model = cls(3); rows.append(model(api.tensor(a)).numpy())
    np.testing.assert_allclose(*rows, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize('nonfinite', [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize('name', ['avg_pool1d', 'adaptive_avg_pool1d'])
def test_nonfinite_region_locality(name, nonfinite):
    a = np.ones((1, 1, 8)); a[..., 0] = nonfinite
    rows = [getattr(api.nn.functional, name)(api.tensor(a), 2 if name == 'avg_pool1d' else 4).numpy() for api in (lite, torch)]
    np.testing.assert_array_equal(*rows)


def test_module_pickle_and_metadata():
    for model in (lite.nn.AvgPool1d(3, 2, 1, True, False), lite.nn.AdaptiveAvgPool1d(3)):
        x = lite.ones(1, 2, 7)
        np.testing.assert_array_equal(model(x).numpy(), pickle.loads(pickle.dumps(model))(x).numpy())


@pytest.mark.parametrize('cls_name', ['AvgPool1d', 'AdaptiveAvgPool1d'])
@pytest.mark.parametrize('arg', [2, 2., None, True, [], [2], (2, 3), np.int64(2), '2'])
def test_constructor_normalization_repr_and_validation_phase(cls_name, arg):
    rows = []
    for api in (lite, torch):
        model = getattr(api.nn, cls_name)(arg)
        row = [str(model)]
        try: row += [tuple(model(api.ones(1, 2, 7)).shape)]
        except Exception as error: row += [type(error).__name__]
        rows.append(row)
    assert rows[0] == rows[1]


@pytest.mark.parametrize('name', ['avg_pool1d', 'adaptive_avg_pool1d'])
@pytest.mark.parametrize('arg', [[], (), [2, 3], True, np.bool_(True), 2., None, [None], 2**63, -(2**63)-1, -1, 0, [2], np.int64(2)])
def test_function_parameter_outcomes(name, arg):
    rows = []
    for api in (lite, torch):
        try: rows.append(('value', tuple(getattr(api.nn.functional, name)(api.ones(1, 2, 7), arg).shape)))
        except Exception as error: rows.append(('error', type(error).__name__))
    assert rows[0] == rows[1]


@pytest.mark.parametrize('kw', [dict(stride=[]), dict(stride=()), dict(stride=0), dict(stride=[2, 3]),
                               dict(padding=2), dict(padding=-1), dict(padding=True), dict(padding=2**63),
                               dict(ceil_mode=1), dict(count_include_pad=1), dict(divisor_override=3)])
def test_fixed_keyword_validation(kw):
    rows = []
    for api in (lite, torch):
        try: rows.append(('value', tuple(api.nn.functional.avg_pool1d(api.ones(1, 2, 7), 3, **kw).shape)))
        except Exception as error: rows.append(('error', type(error).__name__))
    assert rows[0] == rows[1]


@pytest.mark.parametrize('shape', [(0, 7), (1, 0, 7), (0, 2, 7), (1, 2, 0), (7,), (1, 2, 3, 7)])
@pytest.mark.parametrize('size', [0, 1, 2, 7])
def test_adaptive_empty_shape_forward_backward_phases(shape, size):
    rows = []
    for api in (lite, torch):
        row = []
        try:
            x = api.empty(shape, requires_grad=True); y = api.nn.functional.adaptive_avg_pool1d(x, size)
            row += [('value', tuple(y.shape))]
            try: y.sum().backward(); row += [('grad', tuple(x.grad.shape))]
            except Exception as error: row += [('backward', type(error).__name__)]
        except Exception as error: row += [('forward', type(error).__name__)]
        rows.append(row)
    assert rows[0] == rows[1]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_seeded_random_regions_values_pullbacks_and_tangents(dtype):
    rng = np.random.default_rng(4110)
    for _ in range(120):
        length = int(rng.integers(1, 45)); channels = int(rng.integers(1, 4))
        shape = (channels, length) if rng.random() < .5 else (int(rng.integers(0, 3)), channels, length)
        a = rng.normal(size=shape).astype(dtype); v = rng.normal(size=shape).astype(dtype)
        if rng.random() < .5: a = np.asfortranarray(a)
        if rng.random() < .5: a = np.repeat(a, 2, axis=-1)[..., ::2]
        adaptive = rng.random() < .5
        if adaptive: args = (int(rng.integers(1, 60)),)
        else:
            kernel = int(rng.integers(1, length + 1)); step = int(rng.integers(1, length + 1))
            args = (kernel, step, int(rng.integers(0, kernel // 2 + 1)), bool(rng.integers(0, 2)), bool(rng.integers(0, 2)))
        rows = []; seed = None
        for api in (lite, torch):
            x = api.from_numpy(a).requires_grad_(); direction = api.tensor(v, requires_grad=True)
            function = api.nn.functional.adaptive_avg_pool1d if adaptive else api.nn.functional.avg_pool1d
            y = function(x, *args)
            if seed is None: seed = rng.normal(size=y.shape).astype(dtype)
            y.backward(api.tensor(seed))
            p, tangent = api.func.jvp(lambda z: function(z, *args), (x,), (direction,))
            tangent.backward(api.tensor(seed))
            rows.append([z.detach().numpy() for z in (y, x.grad, p, tangent, direction.grad)])
        for aa, bb in zip(*rows): np.testing.assert_allclose(aa, bb, rtol=3e-5, atol=3e-6)


def test_native_convolution_pooling_training_and_resumed_sgd():
    rng = np.random.default_rng(4111); state = None; rows = []
    a = rng.normal(size=(2, 2, 17)); target = rng.normal(size=(2, 2))
    for api in (lite, torch):
        def make():
            return api.nn.Sequential(api.nn.Conv1d(2, 3, 3, padding=1), api.nn.SiLU(),
                                     api.nn.AvgPool1d(3, 2, 1, True, False), api.nn.AdaptiveAvgPool1d(3),
                                     api.nn.Flatten(), api.nn.Linear(9, 2)).double()
        model = make()
        if state is None: state = {name: value.numpy().copy() for name, value in model.state_dict().items()}
        model.load_state_dict({name: api.tensor(value) for name, value in state.items()})
        optimizer = api.optim.SGD(model.parameters(), lr=.02, momentum=.9, weight_decay=.01)
        x = api.tensor(a); wanted = api.tensor(target); history = []
        for _ in range(3):
            optimizer.zero_grad(); loss = (model(x) - wanted).square().mean()
            history.append(loss.item()); loss.backward(); optimizer.step()
        saved = pickle.loads(pickle.dumps((model.state_dict(), optimizer.state_dict())))
        resumed = make(); resumed.load_state_dict(saved[0])
        resumed_opt = api.optim.SGD(resumed.parameters(), lr=.02, momentum=.9, weight_decay=.01)
        resumed_opt.load_state_dict(saved[1]); resumed_opt.zero_grad()
        (resumed(x) - wanted).square().mean().backward(); resumed_opt.step()
        p, tangent = api.func.jvp(resumed, (x,), (api.ones_like(x),))
        rows.append([np.asarray(history), p.detach().numpy(), tangent.detach().numpy(),
                     *[value.detach().numpy() for value in resumed.parameters()]])
    for aa, bb in zip(*rows): np.testing.assert_allclose(aa, bb, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize('name,shape,size', [
    ('avg_pool1d', (0, 2, 7), 3), ('adaptive_avg_pool1d', (0, 2, 7), 3),
    ('adaptive_avg_pool1d', (0, 7), 3), ('adaptive_avg_pool1d', (1, 0, 7), 3),
    ('adaptive_avg_pool1d', (1, 2, 7), 0), ('adaptive_avg_pool1d', (0, 2, 7), 1),
    ('avg_pool2d', (0, 2, 5, 7), 2), ('adaptive_avg_pool2d', (0, 2, 5, 7), (2, 3)),
    ('adaptive_avg_pool2d', (1, 0, 5, 7), (2, 3)), ('adaptive_avg_pool2d', (1, 2, 5, 7), (0, 3)),
    ('adaptive_avg_pool2d', (0, 2, 5, 7), (1, 1)), ('adaptive_avg_pool2d', (1, 0, 5, 7), (1, 1)),
])
def test_native_empty_pool_output_layout(name, shape, size):
    values = [getattr(api.nn.functional, name)(api.empty(shape), size) for api in (lite, torch)]
    assert values[0].shape == tuple(values[1].shape)
    assert values[0].stride() == values[1].stride()
    assert values[0].is_contiguous() == values[1].is_contiguous()
