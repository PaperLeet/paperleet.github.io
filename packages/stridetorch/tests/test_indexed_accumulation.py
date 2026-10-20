"""Indexed linear maps: conservation, adjoints, nonlinear AD and storage rules."""
import gc
import weakref

import numpy as np
import pytest

from stridetorch import UnsupportedOperation, torch


def indexed(name, x, src, axis, index, alpha=.7):
    if name == 'index_select':
        return x.index_select(axis, index)
    if name == 'index_add':
        return x.index_add(axis, index, src, alpha=alpha)
    return x.scatter_add(axis, index, src)


@pytest.mark.parametrize('name', ['index_select', 'index_add', 'scatter_add'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('axis', [0, 1])
def test_indexed_adjoint_and_nonlinear_hvp(name, dtype, axis):
    rng = np.random.default_rng(418)
    shape = (3, 4)
    source_shape = (5, 4) if axis == 0 else (3, 5)
    x = torch.tensor(rng.normal(size=shape), dtype=dtype, requires_grad=True)
    src = torch.tensor(rng.normal(size=source_shape), dtype=dtype, requires_grad=True)
    dx, ds = torch.full_like(x, .21), torch.full_like(src, -.13)
    ids = np.array([0, 0, 1, 2, 1])
    if name == 'scatter_add':
        ids = np.broadcast_to(ids.reshape((5, 1) if axis == 0 else (1, 5)), source_shape).copy()
    index = torch.tensor(ids)
    f = lambda a, b: indexed(name, a, b, axis, index)
    value, tangent = torch.func.jvp(f, (x, src), (dx, ds))
    seed = torch.tensor(rng.normal(size=value.shape), dtype=dtype)
    gradients = torch.autograd.grad(value, (x, src), seed, allow_unused=True)
    lhs = sum(float((g.numpy() * d.numpy()).sum()) for g, d in zip(gradients, (dx, ds)) if g is not None)
    np.testing.assert_allclose(lhs, (seed.numpy() * tangent.numpy()).sum(), rtol=3e-6, atol=3e-6)

    def loss(a, b):
        return torch.linalg.vector_norm(f(a, b).sin())

    _, hvp = torch.autograd.functional.hvp(loss, (x, src), (dx, ds))
    h = 2e-3 if dtype == torch.float32 else 1e-5
    finite = []
    for sign in [-1, 1]:
        a = (x.detach() + sign * h * dx).requires_grad_()
        b = (src.detach() + sign * h * ds).requires_grad_()
        derivs = torch.autograd.grad(loss(a, b), (a, b), allow_unused=True)
        finite.append([g.numpy() if g is not None else np.zeros(t.shape, dtype=dtype) for g, t in zip(derivs, (a, b))])
    tolerance = 5e-4 if dtype == torch.float32 else 2e-8
    for actual, before, after in zip(hvp, *finite):
        np.testing.assert_allclose(actual.numpy(), (after - before) / (2 * h), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('name', ['index_add', 'scatter_add'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.int64])
def test_repeated_aggregation_conserves_total_and_inplace_view(name, dtype):
    base = torch.arange(24, dtype=dtype)
    view = base[::2].reshape(3, 4)
    src = torch.arange(20, dtype=dtype).reshape(5, 4)
    index = torch.tensor([0, 0, 2, 1, 0])
    if name == 'scatter_add':
        index = index.reshape(5, 1).expand(5, 4)
    expected_total = view.sum().item() + src.sum().item()
    before, pointer = view._version, view.data_ptr()
    result = getattr(view, name + '_')(0, index, src)
    assert result is view and view.data_ptr() == pointer
    assert view._version == before + 1
    assert view.sum().item() == expected_total
    np.testing.assert_array_equal(base.numpy()[1::2], np.arange(1, 24, 2))


@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('rank,axis', [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_half_repeated_index_cpu_rounding_profiles(index_dtype, rank, axis):
    shape = (1,) * rank
    src_shape = list(shape)
    src_shape[axis] = 1000
    index = torch.zeros(1000, dtype=index_dtype)
    x = torch.ones(shape, dtype=torch.float16, requires_grad=True)
    selected = x.index_select(axis, index)
    dx = torch.autograd.grad(selected, x, torch.full_like(selected, .1))[0]
    expected = 100. if rank > 1 and axis in (0, rank - 1) and index_dtype == torch.int64 else 105.1875
    assert dx.item() == expected
    src = torch.full(src_shape, .1, dtype=torch.float16)
    assert torch.zeros_like(x).index_add(axis, index, src).item() == expected
    assert torch.zeros_like(x).scatter_add(axis, torch.zeros(src_shape, dtype=index_dtype), src).item() == 100.


@pytest.mark.parametrize('name', ['index_add_', 'scatter_add_'])
def test_invalid_mutation_is_atomic_and_disjoint_views_are_allowed(name):
    x = torch.arange(4, dtype=torch.float64)
    before, version = x.numpy().copy(), x._version
    with pytest.raises((IndexError, RuntimeError)):
        getattr(x, name)(0, torch.tensor([0, 9]), torch.ones(2, dtype=x.dtype))
    np.testing.assert_array_equal(x.numpy(), before)
    assert x._version == version
    for src in [x, x[1:]]:
        with pytest.raises(RuntimeError):
            getattr(x, name)(0, torch.zeros(src.numel(), dtype=torch.int64), src)
    base = torch.arange(8, dtype=torch.float64)
    getattr(base[::2], name)(0, torch.arange(4), base[1::2])
    np.testing.assert_array_equal(base.numpy(), [1., 1., 5., 3., 9., 5., 13., 7.])


@pytest.mark.parametrize('name', ['index_select', 'index_add', 'scatter_add'])
def test_saved_indices_reject_raw_negative_mutation(name):
    x, src = torch.ones(3, requires_grad=True), torch.ones(2, requires_grad=True)
    index = torch.tensor([0, 0])
    y = indexed(name, x, src, 0, index)
    index.numpy()[0] = -1
    with pytest.raises((IndexError, RuntimeError)):
        y.sum().backward()


@pytest.mark.parametrize('name', ['index_select', 'index_add', 'scatter_add'])
def test_no_intermediate_array_retention(name):
    leaf = torch.ones(256, 8, requires_grad=True)
    middle = leaf + 1
    reference = weakref.ref(middle._data)
    ids = torch.zeros(4, dtype=torch.int64)
    if name == 'index_select':
        y = middle.index_select(0, ids)
    else:
        if name == 'scatter_add':
            ids = ids.reshape(4, 1).expand(4, 8)
        y = indexed(name, torch.zeros(256, 8), middle[:4], 0, ids)
    del middle
    gc.collect()
    assert reference() is None
    y.sum().backward()
    assert leaf.grad.shape == leaf.shape


@pytest.mark.parametrize('index_shape,source_shape', [((1,), ()), ((0,), ()), ((0,), (1,)), ((0, 3), (1, 3)), ((2, 0), ())])
def test_empty_and_scalar_scatter_source_gradient_reduction(index_shape, source_shape):
    shape = (2, 3) if len(index_shape) == 2 else (3,)
    x = torch.ones(shape, requires_grad=True)
    src = torch.ones(source_shape, requires_grad=True)
    index = torch.zeros(index_shape, dtype=torch.int64)
    y = x.scatter_add(0 if len(shape) == 1 else 1, index, src)
    gx, gs = torch.autograd.grad(y, (x, src), torch.ones_like(y))
    np.testing.assert_array_equal(gx.numpy(), np.ones(shape))
    np.testing.assert_array_equal(gs.numpy(), np.full(source_shape, index.numel()))


def test_complex_alpha_real_gradient_boundary_and_inactive_source():
    x = torch.ones(3, requires_grad=True)
    src = torch.ones(2, requires_grad=True)
    index = torch.tensor([0, 0])
    with pytest.raises(RuntimeError):
        x.index_add(0, index, src, alpha=1+0j).sum().backward()
    x = torch.ones(3, requires_grad=True)
    x.index_add(0, index, src.detach(), alpha=1+0j).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.ones(3))
    with pytest.raises(UnsupportedOperation):
        x.index_add_(0, index, src)
    for name in ['index_select', 'index_add', 'scatter_add']:
        args = (x, 0, index) if name == 'index_select' else (x, 0, index, src)
        with pytest.raises(UnsupportedOperation):
            getattr(torch, name)(*args, out=torch.empty(3))
