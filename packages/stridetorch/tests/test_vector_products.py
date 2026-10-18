"""Vector/matrix entry points, independent derivative algebra and native empty controls."""
import copy
import gc
import pickle
import tracemalloc
import weakref
import numpy as np
import pytest
from stridetorch import torch, as_torch, UnsupportedOperation, _tensor


# Measured native PyTorch 2.6 controls, independent of the implementation's
# dispatch planner. L/R select the original left/right dtype; error is a
# native rejection. Both float32/64 orders were independently checked.
# Native record SHA-256: 742413bffac229d958a8f129aae134d1ffe3de397277d51adb1f40ee1bdde1a3
EMPTY_CONTRACTS = [
    ((0,), (0,), (False, False), 'C', 'error', ()),
    ((0,), (0,), (False, False), 'F', 'error', ()),
    ((0,), (0,), (True, False), 'C', 'error', ()),
    ((0,), (0,), (True, False), 'F', 'error', ()),
    ((0,), (0,), (False, True), 'C', 'error', ()),
    ((0,), (0,), (False, True), 'F', 'error', ()),
    ((2, 0), (0,), (False, False), 'C', 'R', (2,)),
    ((2, 0), (0,), (False, False), 'F', 'R', (2,)),
    ((2, 0), (0,), (True, False), 'C', 'R', (2,)),
    ((2, 0), (0,), (True, False), 'F', 'R', (2,)),
    ((2, 0), (0,), (False, True), 'C', 'R', (2,)),
    ((2, 0), (0,), (False, True), 'F', 'R', (2,)),
    ((0, 3), (3,), (False, False), 'C', 'R', (0,)),
    ((0, 3), (3,), (False, False), 'F', 'R', (0,)),
    ((0, 3), (3,), (True, False), 'C', 'R', (0,)),
    ((0, 3), (3,), (True, False), 'F', 'R', (0,)),
    ((0, 3), (3,), (False, True), 'C', 'R', (0,)),
    ((0, 3), (3,), (False, True), 'F', 'R', (0,)),
    ((0,), (0, 2), (False, False), 'C', 'error', ()),
    ((0,), (0, 2), (False, False), 'F', 'error', ()),
    ((0,), (0, 2), (True, False), 'C', 'error', ()),
    ((0,), (0, 2), (True, False), 'F', 'error', ()),
    ((0,), (0, 2), (False, True), 'C', 'error', ()),
    ((0,), (0, 2), (False, True), 'F', 'error', ()),
    ((2, 0), (0, 2), (False, False), 'C', 'error', ()),
    ((2, 0), (0, 2), (False, False), 'F', 'error', ()),
    ((2, 0), (0, 2), (True, False), 'C', 'error', ()),
    ((2, 0), (0, 2), (True, False), 'F', 'error', ()),
    ((2, 0), (0, 2), (False, True), 'C', 'error', ()),
    ((2, 0), (0, 2), (False, True), 'F', 'error', ()),
    ((2, 2, 0), (0, 2), (False, False), 'C', 'error', ()),
    ((2, 2, 0), (0, 2), (False, False), 'F', 'error', ()),
    ((2, 2, 0), (0, 2), (True, False), 'C', 'error', ()),
    ((2, 2, 0), (0, 2), (True, False), 'F', 'error', ()),
    ((2, 2, 0), (0, 2), (False, True), 'C', 'error', ()),
    ((2, 2, 0), (0, 2), (False, True), 'F', 'error', ()),
    ((2, 2, 3), (3, 0), (False, False), 'C', 'error', ()),
    ((2, 2, 3), (3, 0), (False, False), 'F', 'R', (2, 2, 0)),
    ((2, 2, 3), (3, 0), (True, False), 'C', 'error', ()),
    ((2, 2, 3), (3, 0), (True, False), 'F', 'R', (2, 2, 0)),
    ((2, 2, 3), (3, 0), (False, True), 'C', 'error', ()),
    ((2, 2, 3), (3, 0), (False, True), 'F', 'error', ()),
    ((2, 0), (2, 0, 2), (False, False), 'C', 'R', (2, 2, 2)),
    ((2, 0), (2, 0, 2), (False, False), 'F', 'R', (2, 2, 2)),
    ((2, 0), (2, 0, 2), (True, False), 'C', 'error', ()),
    ((2, 0), (2, 0, 2), (True, False), 'F', 'error', ()),
    ((2, 0), (2, 0, 2), (False, True), 'C', 'R', (2, 2, 2)),
    ((2, 0), (2, 0, 2), (False, True), 'F', 'R', (2, 2, 2)),
    ((0, 3), (2, 3, 2), (False, False), 'C', 'R', (2, 0, 2)),
    ((0, 3), (2, 3, 2), (False, False), 'F', 'R', (2, 0, 2)),
    ((0, 3), (2, 3, 2), (True, False), 'C', 'error', ()),
    ((0, 3), (2, 3, 2), (True, False), 'F', 'error', ()),
    ((0, 3), (2, 3, 2), (False, True), 'C', 'R', (2, 0, 2)),
    ((0, 3), (2, 3, 2), (False, True), 'F', 'R', (2, 0, 2)),
    ((0,), (2, 0, 2), (False, False), 'C', 'L', (2, 2)),
    ((0,), (2, 0, 2), (False, False), 'F', 'L', (2, 2)),
    ((0,), (2, 0, 2), (True, False), 'C', 'L', (2, 2)),
    ((0,), (2, 0, 2), (True, False), 'F', 'L', (2, 2)),
    ((0,), (2, 0, 2), (False, True), 'C', 'L', (2, 2)),
    ((0,), (2, 0, 2), (False, True), 'F', 'L', (2, 2)),
    ((2, 2, 0), (2, 0, 2), (False, False), 'C', 'R', (2, 2, 2)),
    ((2, 2, 0), (2, 0, 2), (False, False), 'F', 'R', (2, 2, 2)),
    ((2, 2, 0), (2, 0, 2), (True, False), 'C', 'R', (2, 2, 2)),
    ((2, 2, 0), (2, 0, 2), (True, False), 'F', 'R', (2, 2, 2)),
    ((2, 2, 0), (2, 0, 2), (False, True), 'C', 'R', (2, 2, 2)),
    ((2, 2, 0), (2, 0, 2), (False, True), 'F', 'R', (2, 2, 2)),
    ((1, 2, 0), (3, 0, 2), (False, False), 'C', 'R', (3, 2, 2)),
    ((1, 2, 0), (3, 0, 2), (False, False), 'F', 'R', (3, 2, 2)),
    ((1, 2, 0), (3, 0, 2), (True, False), 'C', 'error', ()),
    ((1, 2, 0), (3, 0, 2), (True, False), 'F', 'error', ()),
    ((1, 2, 0), (3, 0, 2), (False, True), 'C', 'R', (3, 2, 2)),
    ((1, 2, 0), (3, 0, 2), (False, True), 'F', 'R', (3, 2, 2)),
    ((3, 2, 0), (1, 0, 2), (False, False), 'C', 'R', (3, 2, 2)),
    ((3, 2, 0), (1, 0, 2), (False, False), 'F', 'R', (3, 2, 2)),
    ((3, 2, 0), (1, 0, 2), (True, False), 'C', 'R', (3, 2, 2)),
    ((3, 2, 0), (1, 0, 2), (True, False), 'F', 'R', (3, 2, 2)),
    ((3, 2, 0), (1, 0, 2), (False, True), 'C', 'error', ()),
    ((3, 2, 0), (1, 0, 2), (False, True), 'F', 'error', ()),
    ((0, 2, 3), (1, 3, 2), (False, False), 'C', 'R', (0, 2, 2)),
    ((0, 2, 3), (1, 3, 2), (False, False), 'F', 'R', (0, 2, 2)),
    ((0, 2, 3), (1, 3, 2), (True, False), 'C', 'R', (0, 2, 2)),
    ((0, 2, 3), (1, 3, 2), (True, False), 'F', 'R', (0, 2, 2)),
    ((0, 2, 3), (1, 3, 2), (False, True), 'C', 'error', ()),
    ((0, 2, 3), (1, 3, 2), (False, True), 'F', 'error', ()),
    ((1, 2, 3), (0, 3, 2), (False, False), 'C', 'R', (0, 2, 2)),
    ((1, 2, 3), (0, 3, 2), (False, False), 'F', 'R', (0, 2, 2)),
    ((1, 2, 3), (0, 3, 2), (True, False), 'C', 'error', ()),
    ((1, 2, 3), (0, 3, 2), (True, False), 'F', 'error', ()),
    ((1, 2, 3), (0, 3, 2), (False, True), 'C', 'R', (0, 2, 2)),
    ((1, 2, 3), (0, 3, 2), (False, True), 'F', 'R', (0, 2, 2)),
    ((2, 1, 0, 3), (1, 3, 3, 2), (False, False), 'C', 'R', (2, 3, 0, 2)),
    ((2, 1, 0, 3), (1, 3, 3, 2), (False, False), 'F', 'R', (2, 3, 0, 2)),
    ((2, 1, 0, 3), (1, 3, 3, 2), (True, False), 'C', 'R', (2, 3, 0, 2)),
    ((2, 1, 0, 3), (1, 3, 3, 2), (True, False), 'F', 'R', (2, 3, 0, 2)),
    ((2, 1, 0, 3), (1, 3, 3, 2), (False, True), 'C', 'R', (2, 3, 0, 2)),
    ((2, 1, 0, 3), (1, 3, 3, 2), (False, True), 'F', 'R', (2, 3, 0, 2)),
]

def arranged(shape, dtype, layout, rng=None):
    value = (rng.uniform(-1, 1, shape) if rng is not None else np.ones(shape)).astype(dtype)
    if layout == 'F':return np.asfortranarray(value)
    if layout == 'stride':
        storage = np.empty((*shape[:-1], shape[-1] * 2), dtype=dtype)
        storage[..., ::2] = value
        return storage[..., ::2]
    return value


def operation(name, method):
    keyword = {'dot': 'tensor', 'mv': 'vec', 'matmul': 'other'}[name]
    if method:return lambda a, b:getattr(a, name)(**{keyword: b})
    return lambda a, b:getattr(torch, name)(input=a, **{keyword: b}, out=None)


def algebra(a, b, ta, tb, seed):
    a, b, ta, tb, seed = [np.asarray(x, dtype='float64') for x in (a, b, ta, tb, seed)]
    value, direction = a @ b, ta @ b + a @ tb
    ga, gb, ha, hb = [np.zeros_like(x) for x in (a, b, a, b)]
    # Independent columns of the bilinear map's Jacobian and its directional
    # derivative. No adapter gradients or finite-difference tolerances enter.
    for i in range(a.size):
        basis = np.zeros_like(a);basis.flat[i] = 1
        column = basis @ b
        ga.flat[i] = np.sum(column * seed)
        ha.flat[i] = np.sum(column * direction + (basis @ tb) * value)
    for i in range(b.size):
        basis = np.zeros_like(b);basis.flat[i] = 1
        column = a @ basis
        gb.flat[i] = np.sum(column * seed)
        hb.flat[i] = np.sum(column * direction + (ta @ basis) * value)
    return value, ga, gb, direction, ha, hb, .5 * np.sum(value * value)


@pytest.mark.parametrize('name,shapes', [
    ('dot', ((5,), (5,))), ('dot', ((0,), (0,))),
    ('mv', ((3, 5), (5,))), ('mv', ((0, 5), (5,))), ('mv', ((3, 0), (0,))),
    ('matmul', ((3, 5), (5,))), ('matmul', ((5,), (5, 2))),
    ('matmul', ((2, 3, 5), (5,))), ('matmul', ((5,), (2, 5, 3)))])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('layout', ['C', 'F', 'stride'])
@pytest.mark.parametrize('method', [False, True])
@pytest.mark.parametrize('wasm', [False, True])
def test_vector_values_reverse_forward_and_curvature(name, shapes, dtype, layout, method, wasm, monkeypatch):
    monkeypatch.setattr(_tensor, '_WASM_MATMUL', wasm)
    rng = np.random.default_rng(79011)
    a, b, ta, tb = [arranged(s, dtype, layout, rng) for s in (*shapes, *shapes)]
    seed = rng.uniform(-1, 1, np.shape(a @ b)).astype(dtype)
    expected = algebra(a, b, ta, tb, seed)
    x, w = [torch.from_numpy(z).requires_grad_() for z in (a, b)]
    dx, dw = map(torch.from_numpy, (ta, tb));fn = operation(name, method)
    y = fn(x, w);ga, gb = torch.autograd.grad(y, (x, w), torch.from_numpy(seed))
    primal, direction = torch.func.jvp(fn, (x, w), (dx, dw))
    loss, (ha, hb) = torch.autograd.functional.hvp(lambda x, w:fn(x, w).square().sum() * .5, (x, w), (dx, dw))
    tolerance = 4e-5 if dtype == 'float32' else 3e-12
    for actual, target in zip((y, ga, gb, direction, ha, hb, loss), expected):
        np.testing.assert_allclose(actual.detach().numpy(), target, atol=tolerance, rtol=tolerance)
    np.testing.assert_array_equal(primal.numpy(), y.numpy())


@pytest.mark.parametrize('case', EMPTY_CONTRACTS)
@pytest.mark.parametrize('reverse_dtype', [False, True])
@pytest.mark.parametrize('method', [False, True])
def test_empty_matmul_matches_native_dtype_grad_and_layout(case, reverse_dtype, method):
    sa, sb, flags, layout, code, expected_shape = case
    dtypes = ('float64', 'float32') if reverse_dtype else ('float32', 'float64')
    a, b = [torch.from_numpy(arranged(s, d, layout)).requires_grad_(g) for s, d, g in zip((sa, sb), dtypes, flags)]
    fn = operation('matmul', method)
    if code == 'error':
        with pytest.raises(RuntimeError):fn(a, b)
        return
    y = fn(a, b);assert tuple(y.shape) == expected_shape
    assert y.dtype == np.dtype(dtypes[0 if code == 'L' else 1])
    assert y.requires_grad == any(flags)
    np.testing.assert_array_equal(y.numpy(), np.zeros(expected_shape))
    if any(flags):
        gradients = torch.autograd.grad(y, tuple(v for v, active in zip((a, b), flags) if active), torch.ones_like(y))
        for g, v in zip(gradients, (v for v, active in zip((a, b), flags) if active)):
            assert g.dtype == v.dtype
            np.testing.assert_array_equal(g.numpy(), np.zeros(v.shape))


@pytest.mark.parametrize('name,shapes', [('dot', ((0,), (0,))), ('dot', ((3,), (3,))), ('mv', ((2, 3), (3,))), ('matmul', ((2, 3), (3, 2)))])
@pytest.mark.parametrize('dtype', ['bool', 'uint16', 'uint32', 'uint64'])
@pytest.mark.parametrize('method', [False, True])
def test_unavailable_native_cpu_dtypes_do_not_use_numpy_boolean_or_wide_unsigned_product(name, shapes, dtype, method):
    a, b = [torch.from_numpy(np.ones(s, dtype=dtype)) for s in shapes]
    with pytest.raises(RuntimeError):operation(name, method)(a, b)


@pytest.mark.parametrize('name,shapes', [('dot', ((3,), (3,))), ('mv', ((2, 3), (3,))), ('matmul', ((2, 3), (3, 2)))])
@pytest.mark.parametrize('bad', ['input_array', 'other_array', 'input_list', 'other_list', 'scalar', 'unknown_keyword', 'extra_positional', 'method_out', 'method_array', 'unbound_array'])
def test_public_vector_matrix_arguments_are_strict(name, shapes, bad):
    a, b = [torch.from_numpy(np.ones(s, dtype='float32')) for s in shapes]
    fn = getattr(torch, name)
    with pytest.raises(TypeError):
        if bad == 'input_array':fn(a.numpy(), b)
        elif bad == 'other_array':fn(a, b.numpy())
        elif bad == 'input_list':fn(a.tolist(), b)
        elif bad == 'other_list':fn(a, b.tolist())
        elif bad == 'scalar':fn(2, b)
        elif bad == 'unknown_keyword':fn(a, b, unknowable=True)
        elif bad == 'extra_positional':fn(a, b, None)
        elif bad == 'method_out':getattr(a, name)(b, out=None)
        elif bad == 'method_array':getattr(a, name)(b.numpy())
        else:getattr(torch.Tensor, name)(a.numpy(), b)


@pytest.mark.parametrize('name,shapes', [('dot', ((3,), (3,))), ('mv', ((2, 3), (3,))), ('matmul', ((2, 3), (3, 2)))])
def test_real_output_buffers_remain_explicit_and_unmodified(name, shapes):
    a, b = [torch.from_numpy(np.ones(s, dtype='float32')) for s in shapes]
    out = torch.zeros_like(getattr(torch, name)(a, b))
    with pytest.raises(UnsupportedOperation):getattr(torch, name)(a, b, out=out)
    np.testing.assert_array_equal(out.numpy(), np.zeros(out.shape))


def test_dot_uses_native_tensor_keyword_and_scalar_operands_raise_runtime_error():
    a = torch.ones(3)
    assert torch.dot(input=a, tensor=a).item() == 3
    with pytest.raises(TypeError):torch.dot(input=a, other=a)
    with pytest.raises(TypeError):a.dot(other=a)
    for name in ('dot', 'mv', 'matmul'):
        with pytest.raises(IndexError if name == 'mv' else RuntimeError):getattr(torch, name)(torch.tensor(2.), a)
    with pytest.raises(RuntimeError):torch.tensor(2.) @ a
    with pytest.raises(RuntimeError):torch.ones(2, 3) @ torch.ones(4, 2)
    with pytest.raises(RuntimeError):torch.ones(2, 2, 3) @ torch.ones(3, 3, 2)


def test_matrix_python_reflection_and_numpy_inference_do_not_fake_tensor_arguments():
    a = torch.ones(2, 3)
    class Reflected:
        def __rmatmul__(self, other):return 79
    assert a @ Reflected() == 79
    for other in (2, [[1]], np.ones((3, 2))):
        assert a.__matmul__(other) is NotImplemented
        assert a.__rmatmul__(other) is NotImplemented
    with pytest.raises(TypeError):torch.Tensor.__matmul__(a.numpy(), a)
    with pytest.raises(TypeError):np.ones((3, 2)) @ a
    for dtype in ('float32', 'float64'):
        b = np.ones((3, 2), dtype=dtype)
        for result in (a @ b, np.matmul(a, b)):
            assert isinstance(result, torch.Tensor)
            assert result.dtype == np.result_type(a.dtype, b.dtype)
            np.testing.assert_array_equal(result.numpy(), np.ones((2, 2)) * 3)
    a.requires_grad_()
    with pytest.raises(RuntimeError):a @ np.ones((3, 2), dtype='float32')
    with torch.no_grad():assert not (a @ np.ones((3, 2), dtype='float32')).requires_grad


@pytest.mark.parametrize('dtype', ['int64', 'bool'])
@pytest.mark.parametrize('name', ['matmul', 'bmm'])
def test_empty_integer_result_cannot_silently_discard_a_float_gradient(dtype, name):
    a = torch.ones(2, 2, 0, requires_grad=True)
    b = torch.ones(2, 0, 2, dtype=getattr(torch, dtype))
    with pytest.raises(RuntimeError):getattr(torch, name)(a, b)
    with torch.no_grad():result = getattr(torch, name)(a, b)
    assert result.dtype == b.dtype and not result.requires_grad
    np.testing.assert_array_equal(result.numpy(), np.zeros((2, 2, 2)))


@pytest.mark.parametrize('name,shapes,strides', [
    ('matmul', ((0, 3), (3, 2)), (2, 1)),
    ('mv', ((0, 3), (3,)), (1,)),
    ('matmul', ((0, 2, 3), (1, 3, 2)), (4, 2, 1)),
    ('bmm', ((0, 2, 3), (0, 3, 2)), (4, 2, 1))])
def test_empty_result_has_native_contiguous_element_strides(name, shapes, strides):
    a, b = [torch.ones(*s) for s in shapes]
    result = getattr(torch, name)(a, b)
    assert result.stride() == strides and result.is_contiguous()


def test_empty_broadcast_uses_metadata_without_casting_or_materializing_repeated_input(monkeypatch):
    source = np.ones((1, 8, 128, 256), dtype='float64');source_ref = weakref.ref(source)
    a = torch.ones(32, 1, 0, 128, requires_grad=True);b = torch.from_numpy(source)
    def forbidden(*args, **kwargs):raise AssertionError('Empty products need no cast or numerical contraction')
    monkeypatch.setattr(torch.Tensor, 'to', forbidden)
    monkeypatch.setattr(_tensor._matmul, 'product', forbidden)
    gc.collect();tracemalloc.start()
    try:
        result = torch.matmul(a, b)
        _, peak = tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    assert result.shape == (32, 8, 0, 256) and result.dtype == b.dtype
    # Casting the input needs 1 MiB; expanding it physically needs 64 MiB.
    assert peak < 128 * 1024
    with torch.no_grad():b.fill_(7)
    del source, b;gc.collect();assert source_ref() is None
    gradient, = torch.autograd.grad(result, (a,), torch.ones_like(result))
    assert gradient.shape == a.shape and gradient.dtype == a.dtype


class VectorOperationModule(torch.nn.Module):
    def __init__(self, operation):
        super().__init__();self.operation = operation
    def forward(self, a, b):return self.operation(a, b)


@pytest.mark.parametrize('name,shapes', [('dot', ((3,), (3,))), ('mv', ((2, 3), (3,))), ('matmul', ((2, 3), (3, 2)))])
def test_new_functions_survive_module_pickle_copy_and_torch_aliases(name, shapes):
    with as_torch():
        import torch as alias
        a, b = [alias.ones(*s) for s in shapes]
        operation = getattr(alias, name);expected = operation(a, b).numpy()
        module = VectorOperationModule(operation)
        for restored in (copy.deepcopy(module), pickle.loads(pickle.dumps(module))):
            np.testing.assert_array_equal(restored(a, b).numpy(), expected)
        np.testing.assert_array_equal(getattr(a, name)(b).numpy(), expected)


def test_empty_geometry_does_not_retain_user_dtype_metadata():
    class Payload:pass
    payload = Payload();reference = weakref.ref(payload)
    dtype = np.dtype('float64', metadata={'payload': payload})
    left = np.ones((0, 4), dtype=dtype);right = np.ones((4, 3), dtype=dtype)
    a, b = map(torch.from_numpy, (left, right));result = a @ b
    assert result.shape == (0, 3) and result.stride() == (3, 1)
    del result, a, b, left, right, dtype, payload
    gc.collect();assert reference() is None
