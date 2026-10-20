"""Independent native checks for modules used by browser distillation lessons."""
import operator
import copy

import numpy as np
import pytest
import torch

from paperleet.browser import numpy_autograd as wasm


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('op', [operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne])
def test_comparison_scalar_boundaries_match_native_without_precision_loss(dtype, op):
    for boundary in (.8, 1.2, .875, 1.75, .5, 2., 1 + 2**-40):
        middle = np.asarray(boundary, dtype=dtype)
        data = np.asarray([np.nextafter(middle, -np.inf, dtype=dtype), middle, np.nextafter(middle, np.inf, dtype=dtype)], dtype=dtype)
        actual, expected = op(wasm.Tensor(data), boundary), op(torch.tensor(data), boundary)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        assert not actual.requires_grad


@pytest.mark.parametrize('op', [operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne])
def test_comparison_explicit_scalar_tensor_precision_and_broadcasting(op):
    values = np.asarray([[.8, 1.2], [np.nan, np.inf]], dtype=np.float32)
    for scalar in (np.asarray(1.2, dtype=np.float64), np.asarray([.8, 1.2], dtype=np.float64)):
        np.testing.assert_array_equal(op(wasm.Tensor(values), wasm.Tensor(scalar)).numpy(), op(torch.tensor(values), torch.tensor(scalar)).numpy())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape,bias', [((3,), False), ((5, 3), True), ((2, 4, 3), False), ((0, 3), True)])
@pytest.mark.parametrize('wasm_kernel', [False, True])
def test_linear_identical_parameters_match_native_values_and_all_gradients(dtype, shape, bias, wasm_kernel, monkeypatch):
    from torchwasm import _nn
    monkeypatch.setattr(_nn, '_WASM_LINEAR_GRADS', wasm_kernel)
    rng = np.random.RandomState(73)
    data, weight, offset = rng.randn(*shape).astype(dtype), rng.randn(2, 3).astype(dtype), rng.randn(2).astype(dtype)
    actual = wasm.Linear(3, 2, bias=bias, dtype=dtype)
    expected = torch.nn.Linear(3, 2, bias=bias, dtype=torch.float32 if dtype == np.float32 else torch.float64)
    with wasm.no_grad(), torch.no_grad():
        actual.weight[:] = wasm.Tensor(weight); expected.weight.copy_(torch.tensor(weight))
        if bias: actual.bias[:] = wasm.Tensor(offset); expected.bias.copy_(torch.tensor(offset))
    x, xx = wasm.Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    y, yy = actual(x), expected(xx)
    y.square().sum().backward(); yy.square().sum().backward()
    tolerance = 3e-6 if dtype == np.float32 else 1e-12
    pairs = [(y, yy.detach()), (x.grad, xx.grad)] + [(a.grad, b.grad) for a, b in zip(actual.parameters(), expected.parameters())]
    for a, b in pairs: np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_silu_forward_and_analytical_vjp_match_native(dtype):
    data = np.asarray([-100, -7, -1, 0, .01, 2, 100], dtype=dtype)
    x, xx = wasm.Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    y, yy = wasm.SiLU()(x), torch.nn.SiLU()(xx)
    y.sum().backward(); yy.sum().backward()
    for a, b in [(y, yy.detach()), (x.grad, xx.grad)]:
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=3e-6, atol=1e-7)
    with pytest.raises(wasm.BrowserUnsupported): wasm.SiLU(inplace=True)


@pytest.mark.parametrize('dtype', [np.float32,np.float64])
@pytest.mark.parametrize('shape,outputs', [((8,128),32),((2,4,128),32),((128,),256)])
@pytest.mark.parametrize('wasm_kernel', [False,True])
def test_large_linear_native_values_all_vjps_and_reverse_over_forward(dtype,shape,outputs,wasm_kernel,monkeypatch):
    from torchwasm import _nn
    monkeypatch.setattr(_nn,'_WASM_LINEAR_GRADS',wasm_kernel)
    rng=np.random.default_rng(1521)
    data=[rng.normal(size=shape).astype(dtype),(.1*rng.normal(size=(outputs,shape[-1]))).astype(dtype),rng.normal(size=outputs).astype(dtype)]
    actual=[wasm.Tensor(value,requires_grad=True) for value in data]
    expected=[torch.tensor(value,requires_grad=True) for value in data]
    direction=rng.normal(size=shape).astype(dtype)
    a=wasm.Tensor(direction,requires_grad=True);b=torch.tensor(direction,requires_grad=True)
    first=wasm.jvp(lambda x:wasm.linear(x,*actual[1:]),(actual[0],),(a,))
    second=torch.func.jvp(lambda x:torch.nn.functional.linear(x,*expected[1:]),(expected[0],),(b,))
    seed=rng.normal(size=shape[:-1]+(outputs,)).astype(dtype)
    (first[0]*wasm.Tensor(seed)+first[1].square()).sum().backward()
    (second[0]*torch.tensor(seed)+second[1].square()).sum().backward()
    tolerance=3e-5 if dtype==np.float32 else 3e-12
    for left,right in [*zip(first,second),*[(x.grad,y.grad) for x,y in zip([*actual,a],[*expected,b])]]:
        np.testing.assert_allclose(left.numpy(),right.detach().numpy(),rtol=tolerance,atol=tolerance)


def test_nested_shared_parameters_deepcopy_freezing_and_ema_alias_versions():
    shared = wasm.Linear(2, 2)
    online = wasm.Sequential(shared, wasm.SiLU(), shared)
    target = copy.deepcopy(online)
    assert target[0] is target[2] and target[0] is not shared
    assert list(online.parameters(recurse=False)) == []
    assert len(list(online.parameters())) == 2
    assert len(online) == 3 and len(online[:2]) == 2
    target.requires_grad_(False)
    for a, b in zip(online.parameters(), target.parameters()):
        assert not b.requires_grad and a.requires_grad
        assert not np.shares_memory(a.numpy(), b.numpy())
    x = wasm.Tensor([[.2, -.4]], dtype=np.float32, requires_grad=True)
    output = target(x).sum()
    with wasm.no_grad():
        # A detached alias of a deep-copied parameter must share its new counter.
        target[0].weight.detach().fill_(2.)
    with pytest.raises(wasm.BrowserUnsupported, match='modified in-place'): output.backward()
    with wasm.no_grad(): shared.weight.fill_(4.)
    assert np.all(target[0].weight.numpy() == 2.)
    fresh = target(x).sum(); fresh.backward()
    assert x.grad is not None and all(p.grad is None for p in target.parameters())
    with pytest.raises(wasm.BrowserUnsupported): copy.deepcopy(x.square())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_named_ema_updates_match_native_and_preserve_parameter_identity(dtype):
    x = wasm.Parameter(wasm.Tensor(np.asarray([2., -1.], dtype=dtype)))
    xx = torch.nn.Parameter(torch.tensor(x.numpy()))
    source = np.asarray([-.5, 1.5], dtype=dtype)
    original = x
    with wasm.no_grad(), torch.no_grad():
        assert x.mul_(.8).add_(wasm.Tensor(source), alpha=.2) is original
        xx.mul_(.8).add_(torch.tensor(source), alpha=.2)
        x.sub_(wasm.Tensor(source), alpha=.1).div_(2.)
        xx.sub_(torch.tensor(source), alpha=.1).div_(2.)
    np.testing.assert_allclose(x.numpy(), xx.detach().numpy(), rtol=2e-6, atol=1e-7)
    with pytest.raises(wasm.BrowserUnsupported): x.mul_(2.)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('options', [{}, {'momentum':.8}, {'momentum':.8,'dampening':.2,'weight_decay':.1}, {'momentum':.8,'nesterov':True,'weight_decay':.1,'maximize':True}])
def test_sgd_native_multistep_momentum_missing_gradients_and_zeroing(dtype, options):
    values = np.asarray([.4, -1.2], dtype=dtype)
    actual = [wasm.Parameter(wasm.Tensor(values)), wasm.Parameter(wasm.Tensor(values + 1))]
    expected = [torch.nn.Parameter(torch.tensor(values)), torch.nn.Parameter(torch.tensor(values + 1))]
    opt, reference = wasm.SGD(actual, lr=.04, **options), torch.optim.SGD(expected, lr=.04, **options)
    for step in range(6):
        for i, (a, b) in enumerate(zip(actual, expected)):
            grad = np.asarray([step * .1 - i, .3 + i], dtype=dtype)
            a.grad = None if i == 1 and step % 2 == 0 else wasm.Tensor(grad)
            b.grad = None if a.grad is None else torch.tensor(grad)
        opt.step(); reference.step()
        for a, b in zip(actual, expected): np.testing.assert_allclose(a.numpy(), b.detach().numpy(), rtol=3e-6, atol=1e-7)
        opt.zero_grad(set_to_none=False); reference.zero_grad(set_to_none=False)
        for a, b in zip(actual, expected):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None: np.testing.assert_array_equal(a.grad.numpy(), b.grad.numpy())
    assert opt.step(closure=lambda: None) is None
    assert reference.step(closure=lambda: None) is None


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('replacement', [False, True])
def test_single_draw_categorical_law_reproducibility_and_native_distribution(dtype, replacement):
    probabilities = np.asarray([[0., .15, .25, .6]] * 20000, dtype=dtype)
    generator = wasm.Generator().manual_seed(319)
    got = wasm.multinomial(wasm.Tensor(probabilities, requires_grad=True), 1, replacement, generator=generator)
    repeat = wasm.multinomial(wasm.Tensor(probabilities), 1, replacement, generator=wasm.Generator().manual_seed(319))
    want = torch.multinomial(torch.tensor(probabilities), 1, replacement, generator=torch.Generator().manual_seed(91)).numpy()
    assert got.shape == (20000, 1) and got.dtype == np.int64 and not got.requires_grad
    np.testing.assert_array_equal(got.numpy(), repeat.numpy())
    assert not (got.numpy() == 0).any()
    for values in (got.numpy(), want):
        np.testing.assert_allclose(np.bincount(values.ravel(), minlength=4)/len(values), probabilities[0], rtol=0, atol=.012)
    with pytest.raises(RuntimeError): wasm.multinomial(wasm.Tensor(np.zeros((2, 3), dtype=dtype)), 1)
    with pytest.raises(wasm.BrowserUnsupported): wasm.multinomial(wasm.Tensor(probabilities[:2]), 2, replacement=False)


@pytest.mark.parametrize('values,minlength', [([], 3), ([0, 4, 1, 1, 0], 0), ([2, 2, 2], 6)])
def test_unweighted_bincount_matches_native(values, minlength):
    data = np.asarray(values, dtype=np.int64)
    actual = wasm.bincount(wasm.Tensor(data), minlength=minlength)
    np.testing.assert_array_equal(actual.numpy(), torch.bincount(torch.tensor(data), minlength=minlength).numpy())
    with pytest.raises(wasm.BrowserUnsupported): wasm.bincount(wasm.Tensor(data), weights=wasm.Tensor(data.astype(float)))


def test_testing_assert_close_preserves_native_dtype_shape_exactness_and_nan_rules():
    for actual, expected, options in [
        (np.asarray([1.], dtype=np.float32), np.asarray([1. + 5e-6], dtype=np.float32), {}),
        (np.asarray([1.]), np.asarray([1. + 5e-6]), {}),
        (np.asarray([np.nan]), np.asarray([np.nan]), {}),
        (np.asarray([np.nan]), np.asarray([np.nan]), {'equal_nan':True}),
        (np.asarray([1], dtype=np.int64), np.asarray([1.]), {}),
        (np.asarray([2**60], dtype=np.int64), np.asarray([2**60+1], dtype=np.int64), {}),
        (np.zeros((2, 1)), np.zeros((2,)), {}),
    ]:
        try: torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), **options)
        except AssertionError:
            with pytest.raises(AssertionError): wasm.assert_close(wasm.Tensor(actual), wasm.Tensor(expected), **options)
        else: wasm.assert_close(wasm.Tensor(actual), wasm.Tensor(expected), **options)
