"""Native operator promotion, shape, scalar and graph-lifetime contracts."""
import gc
import itertools
import warnings
import weakref

import numpy as np
import pytest
import torch as native

from torchwasm import torch, UnsupportedOperation


DTYPES = ['bool', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16',
          'int32', 'int64', 'float16', 'float32', 'float64', 'complex64', 'complex128']
MODES = ['vectors', 'left_scalar', 'right_scalar', 'both_scalar']


def operand(lib, dtype, scalar=False):
    data = np.array([1, 2], dtype=dtype)
    if data.dtype.kind == 'c': data += np.array([.25j, .5j], dtype=dtype)
    return lib.tensor(data[:1].reshape(()) if scalar else data)


def compare(make, *, dtype_result=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            expected = make(native)
        except Exception as error:
            with pytest.raises(type(error)): make(torch)
            return
        kind = str(expected if dtype_result else expected.dtype).removeprefix('torch.')
        if kind == 'complex32':
            with pytest.raises(UnsupportedOperation): make(torch)
            return
        actual = make(torch)
        if dtype_result:
            assert actual == np.dtype(kind)
        else:
            assert actual.dtype == expected.numpy().dtype
            assert actual.shape == tuple(expected.shape)
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())


@pytest.mark.parametrize('left,right', tuple(itertools.product(DTYPES, repeat=2)))
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('operation', ['where', 'result_type'])
def test_tensor_operand_categories_match_native(left, right, mode, operation):
    def run(lib):
        a = operand(lib, left, mode in ('left_scalar', 'both_scalar'))
        b = operand(lib, right, mode in ('right_scalar', 'both_scalar'))
        return lib.result_type(a, b) if operation == 'result_type' else lib.where(lib.tensor([True, False]), a, b)
    compare(run, dtype_result=operation == 'result_type')


@pytest.mark.parametrize('left,right', tuple(itertools.product(DTYPES, repeat=2)))
@pytest.mark.parametrize('operation,scalar', [('cat', False), ('stack', False), ('stack', True)])
def test_concatenation_promotes_all_tensor_dtypes(left, right, operation, scalar):
    compare(lambda lib: getattr(lib, operation)([operand(lib, left, scalar), operand(lib, right, scalar)]))


SCALARS = [True, 2, -.5, 1+.5j, np.bool_(True), np.int16(2), np.uint64(1),
           np.uint64(2**63), np.float16(.5), np.float32(.5), np.float64(.5),
           np.complex64(1+.5j), np.complex128(1+.5j), 2**63, 2**64-1, 2**64,
           -2**63-1, 1e10, 1e300, float('nan'), float('inf'), 1+1e300j]


@pytest.mark.parametrize('dtype', ['bool', 'int8', 'uint8', 'int64', 'uint64', 'float16', 'float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('value', SCALARS)
@pytest.mark.parametrize('scalar_tensor', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('operation', ['where', 'result_type'])
def test_wrapped_number_arguments_match_native(dtype, value, scalar_tensor, reverse, operation):
    def run(lib):
        tensor = operand(lib, dtype, scalar_tensor)
        a, b = (value, tensor) if reverse else (tensor, value)
        return lib.result_type(a, b) if operation == 'result_type' else lib.where(lib.tensor([True, False]), a, b)
    compare(run, dtype_result=operation == 'result_type')


@pytest.mark.parametrize('a,b', tuple(itertools.product([True, 2, -.5, 1+.5j, np.bool_(True), np.int16(2), np.uint64(1), 2**63], repeat=2)))
def test_scalar_only_result_type_and_selection(a, b):
    compare(lambda lib: lib.result_type(a, b), dtype_result=True)
    compare(lambda lib: lib.where(lib.tensor([True, False]), a, b))


def test_native_keyword_overloads_and_invalid_signatures():
    for lib in (native, torch):
        a, b = lib.ones(2), lib.ones(2, dtype=lib.float64)
        assert lib.result_type(tensor=a, other=b) == lib.float64
        assert lib.result_type(tensor=a, other=2.) == lib.float32
        assert lib.result_type(scalar=2., tensor=a) == lib.float32
        assert lib.result_type(scalar1=2, scalar2=.5) == lib.float32
        for options in [dict(tensor1=a, tensor2=b), dict(tensor=2, other=a),
                        dict(scalar=2, tensor=3), dict(tensor=a), dict(input=a, other=b)]:
            with pytest.raises(TypeError): lib.result_type(**options)
        with pytest.raises(TypeError): lib.result_type(a, b, tensor=a)
        mask = lib.tensor([True, False])
        np.testing.assert_array_equal(lib.where(condition=mask, input=a, other=2).numpy(), [1., 2.])
        np.testing.assert_array_equal(a.where(condition=mask, other=2).numpy(), [1., 2.])
        for args in [(mask, None, None), (mask, a)]:
            with pytest.raises(TypeError): lib.where(*args)
        for name in ['cat', 'concat', 'concatenate', 'stack']:
            value = getattr(lib, name)(tensors=[a, a], dim=0)
            assert value.shape == ((2, 2) if name == 'stack' else (4,))
            np.testing.assert_array_equal(getattr(lib, name)([a, a], axis=0).numpy(), value.numpy())
            with pytest.raises(TypeError): getattr(lib, name)([a, a], dim=0, axis=0)
        with pytest.raises(TypeError): lib.where(mask, out=None)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(), (2,), (0,), (2, 0), (2, 3)])
def test_single_argument_where_returns_long_indices(dtype, shape):
    data = np.arange(np.prod(shape, dtype=int) if shape else 1).reshape(shape).astype(dtype)
    if dtype in ('uint16', 'uint32', 'uint64'):
        for lib in (native, torch):
            with pytest.raises(NotImplementedError): lib.where(lib.tensor(data))
        return
    for lib in (native, torch):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = lib.where(lib.tensor(data))
        expected = np.nonzero(np.atleast_1d(data))
        assert isinstance(result, tuple) and len(result) == len(expected)
        for actual, index in zip(result, expected):
            assert actual.dtype == lib.int64 and not actual.requires_grad
            np.testing.assert_array_equal(actual.numpy(), index)


@pytest.mark.parametrize('dtype', ['int16', 'int64', 'float16', 'float32', 'complex64'])
def test_three_argument_where_requires_boolean_condition(dtype):
    for lib in (native, torch):
        with pytest.raises(RuntimeError): lib.where(lib.tensor([0, 1], dtype=getattr(lib, dtype)), lib.ones(2), lib.zeros(2))


@pytest.mark.parametrize('operation', ['cat', 'stack'])
@pytest.mark.parametrize('dim', [True, .5, 2, -3, np.int64(0)])
def test_concatenation_axis_protocol_matches_native(operation, dim):
    compare(lambda lib: getattr(lib, operation)([lib.ones(2), lib.ones(2)], dim=dim))


@pytest.mark.parametrize('operation', ['cat', 'stack'])
@pytest.mark.parametrize('case', ['empty_first', 'empty_last', 'all_empty', 'rank_error', 'shape_error', 'scalar', 'generator', 'array_element'])
def test_concatenation_empty_and_invalid_shapes(operation, case):
    def run(lib):
        inputs = {
            'empty_first': lambda: [lib.empty(0, dtype=lib.float64), lib.ones(2, 3)],
            'empty_last': lambda: [lib.ones(2, 3), lib.empty(0, dtype=lib.float64)],
            'all_empty': lambda: [lib.empty(0), lib.empty(0, dtype=lib.float64)],
            'rank_error': lambda: [lib.ones(2, 3), lib.ones(2)],
            'shape_error': lambda: [lib.ones(2, 3), lib.ones(2, 2)],
            'scalar': lambda: [lib.tensor(1.), lib.tensor(2.)],
            'generator': lambda: (lib.ones(2) for _ in range(2)),
            'array_element': lambda: [lib.ones(2), np.ones(2)],
        }[case]()
        return getattr(lib, operation)(inputs, dim=0)
    compare(run)


@pytest.mark.parametrize('left_dtype,right_dtype', [('float32', 'float64'), ('float64', 'float32'), ('float16', 'float64')])
def test_broadcast_selection_and_join_gradients_match_native(left_dtype, right_dtype):
    records = []
    for lib in (native, torch):
        a = lib.tensor([[1.], [2.]], dtype=getattr(lib, left_dtype), requires_grad=True)
        b = lib.tensor([[3., 4., 5.]], dtype=getattr(lib, right_dtype), requires_grad=True)
        mask = lib.tensor([[True, False, True], [False, True, False]])
        result = lib.where(mask, a, b)
        result.square().sum().backward()
        records.append((result.detach().numpy(), a.grad.numpy(), b.grad.numpy()))
        assert a.grad.dtype == a.dtype and b.grad.dtype == b.dtype
    for actual, expected in zip(records[1], records[0]): np.testing.assert_array_equal(actual, expected)
    for op in ['cat', 'stack']:
        records = []
        for lib in (native, torch):
            a = lib.tensor([1., 2.], dtype=getattr(lib, left_dtype), requires_grad=True)
            b = lib.tensor([3., 4.], dtype=getattr(lib, right_dtype), requires_grad=True)
            result = getattr(lib, op)([a, b, a]); result.square().sum().backward()
            records.append((a.grad.numpy(), b.grad.numpy()))
        for actual, expected in zip(records[1], records[0]): np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('target', ['left', 'right', 'output'])
@pytest.mark.parametrize('retained', [False, True])
def test_where_does_not_save_branch_or_output_values(target, retained):
    for lib in (native, torch):
        a = lib.tensor([2., 3.], requires_grad=True); b = lib.tensor([5., 7.], requires_grad=True)
        mask = lib.tensor([True, False]); out = lib.where(mask, a, b)
        if retained: out.sum().backward(retain_graph=True); a.grad = None; b.grad = None
        with lib.no_grad(): {'left': a, 'right': b, 'output': out}[target].detach()[0] = 9.
        out.sum().backward()
        np.testing.assert_array_equal(a.grad.numpy(), [1., 0.]); np.testing.assert_array_equal(b.grad.numpy(), [0., 1.])


def test_where_saves_condition_and_releases_numerical_state():
    for lib in (native, torch):
        x = lib.tensor([2., 3.], requires_grad=True); condition = lib.tensor([True, False])
        y = lib.where(condition, x, 0); y.sum().backward()
        with pytest.raises(RuntimeError): y.sum().backward()
        y = lib.where(condition, x, 0)
        condition[0] = False
        with pytest.raises(RuntimeError if lib is native else UnsupportedOperation): y.sum().backward()


def test_where_does_not_keep_intermediate_tensor_objects_alive():
    x = torch.tensor([2., 3.], requires_grad=True); condition = torch.tensor([True, False])
    a, b = x + 1, x * .5
    y = torch.where(condition, a, b); loss = y.square().sum()
    # Square legitimately saves its input values; test the branch objects that
    # where itself must not retain, independently of that downstream operator.
    references = [weakref.ref(value) for value in (a, b)]
    del a, b, y
    gc.collect(); assert all(reference() is None for reference in references)
    loss.backward(); np.testing.assert_array_equal(x.grad.numpy(), [6., 1.5])


@pytest.mark.parametrize('operation', ['where', 'cat', 'stack'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_forward_and_reverse_over_forward_selection_join(operation, dtype):
    records = []
    for lib in (native, torch):
        x = lib.tensor([1., 2.], dtype=getattr(lib, dtype), requires_grad=True)
        tangent = lib.tensor([.25, -.5], dtype=getattr(lib, dtype))
        mask = lib.tensor([True, False])
        def function(value):
            a, b = value.square(), value * 3
            return lib.where(mask, a, b) if operation == 'where' else getattr(lib, operation)([a, b])
        y, dy = lib.func.jvp(function, (x,), (tangent,))
        dy.sum().backward()
        records.append((y.detach().numpy(), dy.detach().numpy(), x.grad.numpy()))
    for actual, expected in zip(records[1], records[0]): np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('value', [0, np.bool_(True), np.complex64(1+.5j), 1e300, 2**64])
def test_forward_where_uses_the_same_checked_scalar_arguments(value):
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            reference = native.func.jvp(lambda x: native.where(native.tensor([True, False]), x, value),
                                        (native.ones(2),), (native.ones(2),))
        except Exception as error:
            with pytest.raises(type(error)):
                torch.func.jvp(lambda x: torch.where(torch.tensor([True, False]), x, value),
                               (torch.ones(2),), (torch.ones(2),))
            return
        result = torch.func.jvp(lambda x: torch.where(torch.tensor([True, False]), x, value),
                               (torch.ones(2),), (torch.ones(2),))
    for actual, expected in zip(result, reference):
        assert actual.dtype == expected.numpy().dtype
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_masked_two_layer_training_keeps_float32_and_matches_native():
    records = []
    for lib in (native, torch):
        first = lib.nn.Linear(2, 2, bias=False); second = lib.nn.Linear(2, 1, bias=False)
        with lib.no_grad():
            first.weight.copy_(lib.tensor([[.5, -.25], [.75, .5]]))
            second.weight.copy_(lib.tensor([[.25, -.5]]))
        optimizer = lib.optim.SGD([*first.parameters(), *second.parameters()], lr=.05)
        inputs = lib.tensor([[1., 2.], [3., -1.]]); target = lib.tensor([[.5], [-.25]])
        mask = lib.tensor([[True, False], [False, True]])
        losses = []
        for _ in range(6):
            optimizer.zero_grad()
            middle = lib.where(mask, first(inputs), 0)
            assert middle.dtype == lib.float32
            loss = (second(middle) - target).square().mean()
            loss.backward(); optimizer.step(); losses.append(loss.item())
        records.append((losses, first.weight.detach().numpy(), second.weight.detach().numpy()))
    for actual, expected in zip(records[1], records[0]): np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
def test_scalar_ieee_values_ignore_numpy_error_policy(dtype):
    with np.errstate(all='raise'):
        for value in [0., 1e-45, 1e-300, float('inf'), -float('inf'), float('nan')]:
            compare(lambda lib: lib.where(lib.tensor([True, False]), operand(lib, dtype), value))
        assert np.geterr() == dict.fromkeys(np.geterr(), 'raise')


def test_uint8_condition_warns_once_and_keeps_its_version_guard():
    for lib in (native, torch):
        x = lib.tensor([2., 3.], requires_grad=True); condition = lib.tensor([1, 0], dtype=lib.uint8)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            y = lib.where(condition, x, 0)
            lib.where(condition, x, 0)
        assert len([item for item in seen if 'uint8' in str(item.message)]) == 1
        condition[0] = 0
        with pytest.raises(RuntimeError if lib is native else UnsupportedOperation): y.sum().backward()
