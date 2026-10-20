"""Independent scalar-clamp contracts, including saved-value lifetimes."""
import gc
import weakref
import numpy as np
import pytest
from torchwasm import torch


def assert_exact(actual, expected):
    actual = actual.detach().numpy()
    expected = np.asarray(expected)
    np.testing.assert_array_equal(actual, expected)
    if actual.dtype.kind == 'f':
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'uint8', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('shape', [(), (3,), (2, 0, 3)])
@pytest.mark.parametrize('floating_bound', [False, True])
def test_checked_dtype_shape_and_values(dtype, shape, floating_bound):
    data = np.full(shape, 3, dtype=dtype)
    value = torch.from_numpy(data).clamp(max=1.5 if floating_bound else 2)
    expected_dtype = np.dtype('float32') if floating_bound and data.dtype.kind in 'iu' else data.dtype
    assert value.dtype == expected_dtype and value.shape == shape
    assert_exact(value, np.full(shape, 1.5 if floating_bound else 2, dtype=expected_dtype))
    assert not value.requires_grad
    if not data.size:
        assert value.stride() == (3, 3, 1)


@pytest.mark.parametrize('default', [torch.float32, torch.float64])
def test_floating_bound_uses_current_default_for_integral_input(default):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default)
        result = torch.tensor([-2, 0, 3]).clamp(-.5, .5)
        assert result.dtype == default
        assert_exact(result, [-.5, 0., .5])
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize('dtype,bound', [('int8', 128), ('int16', -32769), ('int32', 2**31),
                                        ('int64', 2**63), ('uint8', 256),
                                        ('float16', 65505.), ('float32', 1e50)])
@pytest.mark.parametrize('lower', [False, True])
def test_scalar_bound_overflow_is_checked(dtype, bound, lower):
    x = torch.tensor([1], dtype=getattr(torch, dtype))
    with pytest.raises(RuntimeError, match='overflow'):
        x.clamp(**{'min' if lower else 'max': bound})


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('bounds', [(-1., 1.), (-1., None), (None, 1.), (1., -1.)])
@pytest.mark.parametrize('strided', [False, True])
def test_ieee_cotangent_selection_and_inclusive_boundaries(dtype, bounds, strided):
    data = np.array([-2., -1., 0., 1., 2., np.nan], dtype=dtype)
    if strided:
        storage = np.empty((6, 2), dtype=dtype); storage[:, 0] = data
        data = storage[:, 0]
    x = torch.from_numpy(data).requires_grad_(True)
    seed = np.array([np.nan, -np.inf, -0., np.inf, np.nan, -np.inf], dtype=dtype)
    gradient, = torch.autograd.grad(x.clamp(*bounds), x, torch.from_numpy(seed))
    # Explicit finite input membership; an unordered NaN is never active.
    membership = [i != 5 and (bounds[0] is None or v >= bounds[0]) and
                  (bounds[1] is None or v <= bounds[1]) for i, v in enumerate(data)]
    expected = np.array([v if active else 0. for v, active in zip(seed, membership)], dtype=dtype)
    assert_exact(gradient, expected)
    assert x.grad is None


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('method', ['backward', 'grad', 'inputs'])
def test_pullback_observes_mutation_through_shared_numpy_input(dtype, method):
    data = np.array([-2., -.5, .5, 2.], dtype=dtype)
    x = torch.from_numpy(data).requires_grad_(True)
    result = x.clamp(-1., 1.)
    data[:] = [.25, 2., -2., -.25]
    seed = torch.tensor([2., 3., 5., 7.], dtype=dtype)
    if method == 'grad':
        actual, = torch.autograd.grad(result, x, seed)
        assert x.grad is None
    else:
        result.backward(seed, **({'inputs': [x]} if method == 'inputs' else {}))
        actual = x.grad
    assert_exact(actual, [2., 0., 0., 7.])


@pytest.mark.parametrize('target', ['input', 'output', 'downstream'])
def test_storage_guards_cover_only_values_needed_by_each_pullback(target):
    x = torch.tensor([-.5, .5], dtype=torch.float64, requires_grad=True)
    output = x.clamp(-1., 1.)
    loss = output.square().sum() if target == 'downstream' else output.sum()
    with torch.no_grad():
        (x if target == 'input' else output).detach().add_(1.)
    if target == 'output':
        loss.backward(); assert_exact(x.grad, [1., 1.])
    else:
        with pytest.raises(RuntimeError, match='modified'):
            loss.backward()
        assert x.grad is None


@pytest.mark.parametrize('retained', [False, True])
@pytest.mark.parametrize('method', ['backward', 'grad', 'inputs'])
def test_saved_input_array_freed_after_consumption_and_not_before(retained, method):
    x = torch.tensor([-.75, -.25, .25, .75], dtype=torch.float64, requires_grad=True)
    middle = x + .125
    array_ref, wrapper_ref = weakref.ref(middle._data), weakref.ref(middle)
    output = middle.clamp(-.5, .5)
    del middle; gc.collect()
    assert wrapper_ref() is None and array_ref() is not None
    for keep in ([True, False] if retained else [False]):
        x.grad = None
        if method == 'grad':
            value, = torch.autograd.grad(output, x, torch.ones_like(output), retain_graph=keep)
            assert x.grad is None
        else:
            output.backward(torch.ones_like(output), retain_graph=keep,
                            **({'inputs': [x]} if method == 'inputs' else {}))
            value = x.grad
        assert_exact(value, [0., 1., 1., 0.])
        gc.collect(); assert (array_ref() is not None) == keep


def test_storage_guard_failure_keeps_payload_until_graph_is_dropped():
    x = torch.tensor([-.25, .25], dtype=torch.float64, requires_grad=True)
    middle = x + .125; array_ref = weakref.ref(middle._data)
    output = middle.clamp(-.5, .5)
    with torch.no_grad(): middle.detach().add_(1.)
    del middle
    with pytest.raises(RuntimeError, match='modified'): output.sum().backward()
    gc.collect(); assert array_ref() is not None
    del output; gc.collect(); assert array_ref() is None


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('interface', ['direct', 'functional'])
def test_jvp_selects_infinite_and_nan_directions_and_keeps_direction_reverse(dtype, interface):
    x = torch.tensor([-2., -.5, .5, 2.], dtype=dtype)
    direction = torch.tensor([np.inf, -0., 3., np.nan], dtype=dtype, requires_grad=True)
    if interface == 'direct':
        primal, tangent = torch.func.jvp(lambda a: a.clamp(-1., 1.), (x,), (direction,))
    else:
        primal, tangent = torch.autograd.functional.jvp(lambda a: a.clamp(-1., 1.), x, direction, create_graph=True)
    assert_exact(primal, [-1., -.5, .5, 1.])
    assert_exact(tangent, [0., -0., 3., 0.])
    gradient, = torch.autograd.grad(tangent, direction, torch.ones_like(tangent))
    assert_exact(gradient, [0., 1., 1., 0.])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('strided', [False, True])
def test_quadratic_clamp_hvp_away_from_kinks(dtype, strided):
    data = np.array([-2., -.5, .5, 2.], dtype=dtype)
    if strided: data = np.stack((data, data), axis=1)[:, 0]
    x = torch.from_numpy(data); direction = torch.tensor([2., 3., 5., 7.], dtype=dtype)
    value, actual = torch.autograd.functional.hvp(lambda a: a.clamp(-1., 1.).square().sum(), x, direction)
    assert_exact(value, 2.5)
    assert_exact(actual, [0., 6., 10., 0.])


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('method', ['backward', 'grad', 'inputs'])
def test_mutation_after_owned_intermediate_alias_is_dropped(dtype_name, strided, method):
    # A weak-reference-plus-forward-snapshot shortcut loses these updates when
    # the last user-held NumPy alias dies before backward. The result must still
    # use the final saved values, including for a view of a temporary allocation.
    dtype = getattr(torch, dtype_name)
    data = np.array([-2., -.5, .5, 2.], dtype=dtype_name)
    if strided:
        expanded = np.zeros(8, dtype=dtype_name); expanded[::2] = data; data = expanded
    x = torch.tensor(data, dtype=dtype, requires_grad=True)
    middle = x + .125
    if strided: middle = middle[::2]
    output = middle.clamp(-1., 1.)
    alias = middle.detach().numpy()
    alias[:] = [.25, 2., -2., -.25]
    del middle, alias
    gc.collect()
    seed = torch.tensor([2., 3., 5., 7.], dtype=dtype)
    if method == 'grad':
        actual, = torch.autograd.grad(output, x, seed)
        assert x.grad is None
    else:
        output.backward(seed, **({'inputs': [x]} if method == 'inputs' else {}))
        actual = x.grad
    expected = np.array([2., 0., 0., 7.], dtype=dtype_name)
    if strided:
        expanded = np.zeros(8, dtype=dtype_name); expanded[::2] = expected; expected = expanded
    assert_exact(actual, expected)
