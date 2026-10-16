"""Native differential regressions for browser ownership, masking and AD boundaries."""
import ast
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from paperleet.browser import numpy_autograd as wasm
from paperleet.browser import schedules_numpy as schedules


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(2, 4), (0, 3), ()])
def test_nan_to_num_preserves_values_and_gradient_paths(dtype, shape):
    data = np.resize(np.array([np.nan, np.inf, -np.inf, -2., 0., 3.], dtype=dtype), shape)
    x = wasm.Tensor(data.copy(), requires_grad=True)
    native = torch.tensor(data.copy(), requires_grad=True)
    actual = wasm.nan_to_num(x, nan=.5, posinf=9., neginf=-7.)
    expected = torch.nan_to_num(native, nan=.5, posinf=9., neginf=-7.)
    weights = np.resize(np.array([.2, -.5, 1., 2.], dtype=dtype), shape)
    actual.backward(wasm.Tensor(weights)); expected.backward(torch.tensor(weights))
    np.testing.assert_array_equal(actual.numpy(), expected.detach().numpy())
    np.testing.assert_array_equal(x.grad.numpy(), native.grad.numpy())
    finite = wasm.tensor([2., 3.], dtype=dtype, requires_grad=True)
    (finite.nan_to_num() * finite).sum().backward()
    np.testing.assert_array_equal(finite.grad.numpy(), [4., 6.])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_nan_to_num_noncontiguous_no_grad_and_stale_operand(dtype):
    source = np.array([[np.nan, 1., np.inf], [-np.inf, -2., 3.]], dtype=dtype).T
    x, native = wasm.Tensor(source.copy(order='K'), requires_grad=True), torch.tensor(source, requires_grad=True)
    x.nan_to_num().sum().backward(); native.nan_to_num().sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), native.grad.numpy())
    with wasm.no_grad(): assert not x.nan_to_num().requires_grad
    output = x.nan_to_num()
    with wasm.no_grad(): x[...] = 0.
    with pytest.raises(wasm.BrowserUnsupported, match='modified in-place'): output.sum().backward()
    with pytest.raises(wasm.BrowserUnsupported): wasm.nan_to_num(x, out=x)
    with pytest.raises(wasm.BrowserUnsupported): wasm.jvp(wasm.nan_to_num, (x,), (wasm.Tensor(np.ones_like(source)),))


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.bool_, np.complex64, np.complex128])
def test_nan_to_num_preserves_existing_nondifferentiable_dtype_support(dtype):
    data = np.array([complex(np.nan, np.inf), complex(-np.inf, 2.)], dtype=dtype) if np.dtype(dtype).kind == 'c' else np.array([0, 1], dtype=dtype)
    actual = wasm.nan_to_num(wasm.Tensor(data))
    expected = torch.nan_to_num(torch.tensor(data))
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    assert not actual.requires_grad and actual.dtype == data.dtype


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_nan_to_num_nonfinite_incoming_gradients_match_native_multiplication(dtype):
    data = np.array([np.nan, np.inf, -np.inf, -2., 3.], dtype=dtype)
    seed = np.array([np.inf, np.nan, -np.inf, np.nan, np.inf], dtype=dtype)
    actual = wasm.Tensor(data.copy(), requires_grad=True)
    expected = torch.tensor(data.copy(), requires_grad=True)
    with np.errstate(invalid='ignore'):
        wasm.nan_to_num(actual).backward(wasm.Tensor(seed))
        torch.nan_to_num(expected).backward(torch.tensor(seed))
    np.testing.assert_array_equal(actual.grad.numpy(), expected.grad.numpy())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('kind', ['contiguous', 'strided', 'readonly'])
def test_tensor_copies_numpy_inputs_before_forward_and_backward(dtype, kind):
    source = np.arange(12, dtype=dtype).reshape(3, 4)
    if kind == 'strided': source = source.T[:, ::2]
    if kind == 'readonly': source.setflags(write=False)
    expected_data = source.copy()
    actual, expected = wasm.tensor(source, requires_grad=True), torch.tensor(source.copy(), requires_grad=True)
    output, native_output = actual.square().sum(), expected.square().sum()
    if kind == 'readonly': source.setflags(write=True)
    source[...] = 100.
    output.backward(); native_output.backward()
    np.testing.assert_array_equal(actual.numpy(), expected_data)
    np.testing.assert_array_equal(actual.grad.numpy(), expected.grad.numpy())
    assert not np.shares_memory(actual.numpy(), source)


def test_tensor_detaches_and_copies_but_as_tensor_and_from_numpy_keep_sharing():
    source = np.array([2., 3.], dtype=np.float32)
    for constructor in (wasm.as_tensor, wasm.from_numpy):
        value = constructor(source)
        assert np.shares_memory(value.numpy(), source)
        value[0] = 4.
        assert source[0] == 4.
    x = wasm.tensor(source, requires_grad=True)
    assert wasm.as_tensor(x) is x
    converted = wasm.as_tensor(x, dtype=np.float64)
    converted.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [1., 1.])
    detached = wasm.tensor(x)
    assert not detached.requires_grad and not np.shares_memory(detached.numpy(), x.numpy())
    assert not np.shares_memory(wasm.as_tensor(source, dtype=np.float64).numpy(), source)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64])
def test_schedules_tensor_also_owns_its_numpy_input_and_preserves_inferred_dtype(dtype):
    data = np.arange(12, dtype=dtype).reshape(3, 4).T[:, ::2]
    actual, expected = schedules.tensor(data), torch.tensor(data)
    data[...] = 99
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    assert actual.dtype == expected.numpy().dtype and not np.shares_memory(actual.numpy(), data)
    assert schedules.tensor([1.]).dtype == np.dtype('float32')


@pytest.mark.parametrize('operation', ['triu', 'tril'])
@pytest.mark.parametrize('diagonal', [-1, 0, 1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_triangular_transforms_zero_masked_nonfinite_values_and_gradients(operation, diagonal, dtype):
    data = np.array([[[np.inf, np.nan, -np.inf], [np.nan, -np.inf, np.inf], [np.inf, np.inf, np.nan]]], dtype=dtype)
    x, native = wasm.Tensor(data.copy(), requires_grad=True), torch.tensor(data, requires_grad=True)
    actual = getattr(x, operation)(diagonal)
    expected = getattr(native, operation)(diagonal)
    weights = np.array([[[np.inf, 2., -np.inf], [3., 4., 5.], [np.nan, 6., 7.]]], dtype=dtype)
    actual.backward(wasm.Tensor(weights)); expected.backward(torch.tensor(weights))
    np.testing.assert_array_equal(actual.numpy(), expected.detach().numpy())
    np.testing.assert_array_equal(x.grad.numpy(), native.grad.numpy())
    assert not np.shares_memory(actual.numpy(), x.numpy())


def test_triangular_vector_input_is_rejected_and_tensor_methods_are_not_shadowed():
    for operation in ('triu', 'tril'):
        with pytest.raises(RuntimeError): getattr(wasm.Tensor([1., 2.]), operation)()
    import paperleet_torch
    for source in Path(paperleet_torch.__file__).parent.glob('*.py'):
        tree = ast.parse(source.read_text())
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            methods = []
            for node in (item for item in cls.body if isinstance(item, ast.FunctionDef)):
                accessor = next((decorator.attr for decorator in node.decorator_list
                                 if isinstance(decorator, ast.Attribute)
                                 and isinstance(decorator.value, ast.Name)
                                 and decorator.value.id == node.name
                                 and decorator.attr in ('setter', 'deleter')), None)
                methods.append((node.name, accessor))
            assert len(methods) == len(set(methods)), f'Duplicate methods in {cls.name}'


def test_registry_fails_closed_without_hiding_student_errors():
    module = types.ModuleType('example')
    registry = wasm.OperatorRegistry({'example': module})
    registry.add('example.unvalidated', lambda x: wasm.Tensor(x.numpy().copy()))
    with pytest.raises(wasm.BrowserUnsupported, match='reverse-mode'):
        module.unvalidated(wasm.Tensor([1.], requires_grad=True))
    with pytest.raises(ValueError, match='Duplicate'): registry.add('example.unvalidated', lambda x: x)
    registry.add('example.sum', lambda x: x.sum(), reverse='analytic', unsupported_keywords=('out',))
    with pytest.raises(wasm.BrowserUnsupported, match='out'): module.sum(wasm.Tensor([1.]), out=wasm.Tensor([0.]))
    assert module.sum(wasm.Tensor([1.]), out=None).item() == 1.
    with pytest.raises(TypeError): module.sum(wasm.Tensor([1.]), typo=True)
    registry.add('example.broken', lambda x: (_ for _ in ()).throw(TypeError('student bug')), reverse='analytic')
    with pytest.raises(TypeError, match='student bug'): module.broken(wasm.Tensor([1.]))
