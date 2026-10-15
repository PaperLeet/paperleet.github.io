"""Native-reference contracts at the external NumPy/Tensor storage boundary."""
import subprocess
import sys
import warnings
import operator

import numpy as np
import pytest
import torch as native

from paperleet_torch import torch


APIS = ['tensor', 'as_tensor', 'from_numpy']
DTYPES = ['bool', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'complex64', 'complex128']


@pytest.mark.parametrize('api', APIS)
@pytest.mark.parametrize('dtype', DTYPES)
def test_supported_numpy_storage_types_match_native(api, dtype):
    data = np.array([0, 1, 2], dtype=dtype)
    actual, expected = getattr(torch, api)(data), getattr(native, api)(data)
    assert actual.dtype == expected.numpy().dtype
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    assert actual.is_leaf and not actual.requires_grad


def invalid_array(kind):
    return {
        'unicode': lambda: np.array(['1', '2']),
        'bytes': lambda: np.array([b'1', b'2']),
        'object_numbers': lambda: np.array([1., 2.], dtype=object),
        'object_mixed': lambda: np.array([1., object()], dtype=object),
        'datetime': lambda: np.array(['2026-01-01'], dtype='datetime64[D]'),
        'timedelta': lambda: np.array([1, 2], dtype='timedelta64[D]'),
        'structured': lambda: np.array([(1,), (2,)], dtype=[('x', 'i4')]),
        'void': lambda: np.array([b'ab'], dtype='V2'),
        'longdouble': lambda: np.ones(2, dtype=np.longdouble),
        'clongdouble': lambda: np.ones(2, dtype=np.clongdouble),
    }[kind]()


@pytest.mark.parametrize('api, cast', [('tensor', False), ('tensor', True), ('as_tensor', False), ('as_tensor', True), ('from_numpy', False)])
@pytest.mark.parametrize('kind', ['unicode', 'bytes', 'object_numbers', 'object_mixed', 'datetime', 'timedelta', 'structured', 'void', 'longdouble', 'clongdouble'])
def test_invalid_numpy_types_rejected_before_explicit_cast(api, cast, kind):
    data = invalid_array(kind)
    # Some NumPy builds alias longdouble to double instead of exposing NPY_LONGDOUBLE.
    if kind in ('longdouble', 'clongdouble') and data.dtype.char in 'dD': pytest.skip('No extended dtype on this build')
    before = data.copy()
    for library in [native, torch]:
        options = {'dtype': library.float32} if cast else {}
        with pytest.raises(TypeError): getattr(library, api)(data, **options)
    np.testing.assert_array_equal(data, before)
    with pytest.raises(TypeError): torch.Tensor(data)


def layout_array(layout):
    if layout == 'negative': return np.arange(6, dtype='f4')[::-1]
    if layout == 'negative_singleton': return np.arange(3, dtype='f4')[2:1:-1]
    if layout == 'odd_stride': return np.ndarray((2,), dtype='f4', buffer=bytearray(12), strides=(3,))
    if layout == 'odd_singleton': return np.ndarray((1,), dtype='f4', buffer=bytearray(4), strides=(3,))
    if layout == 'multi_singleton': return np.ndarray((1, 2, 2), dtype='f4', buffer=bytearray(16), strides=(-16, 8, 4))
    if layout == 'empty_negative': return np.ndarray((0,), dtype='f4', buffer=bytearray(0), strides=(-4,))
    if layout == 'empty_odd': return np.ndarray((0, 2, 2), dtype='f4', buffer=bytearray(0), strides=(3, 8, 4))
    if layout == 'complex_odd_singleton': return np.ndarray((1,), dtype='c8', buffer=bytearray(8), strides=(4,))
    if layout == 'opposite_endian': return np.arange(3, dtype=np.dtype('f4').newbyteorder('S'))
    if layout == 'fortran': return np.asfortranarray(np.arange(12, dtype='f4').reshape(3, 4))
    if layout == 'transpose': return np.arange(12, dtype='f4').reshape(3, 4).T
    if layout == 'stride': return np.arange(30, dtype='f4').reshape(5, 6)[::2, 1::2]
    if layout == 'broadcast': return np.broadcast_to(np.array([1., 2., 3.], dtype='f4'), (2, 3))
    if layout == 'unaligned': return np.ndarray((3,), dtype='f4', buffer=bytearray(13), offset=1)
    if layout == 'scalar': return np.array(2., dtype='f4')
    if layout == 'empty': return np.empty((2, 0), dtype='f4')
    raise AssertionError(layout)


@pytest.mark.parametrize('api, cast', [('tensor', False), ('tensor', True), ('as_tensor', False), ('as_tensor', True), ('from_numpy', False)])
@pytest.mark.parametrize('layout', ['negative', 'negative_singleton', 'odd_stride', 'odd_singleton', 'opposite_endian', 'multi_singleton', 'empty_negative', 'empty_odd', 'complex_odd_singleton'])
def test_invalid_numpy_layout_is_not_hidden_by_copy_or_cast(api, cast, layout):
    data = layout_array(layout)
    for library in [native, torch]:
        options = {'dtype': library.float64} if cast else {}
        with pytest.raises(ValueError): getattr(library, api)(data, **options)


@pytest.mark.parametrize('api', APIS)
@pytest.mark.parametrize('layout', ['fortran', 'transpose', 'stride', 'broadcast', 'unaligned', 'scalar', 'empty'])
def test_valid_numpy_views_keep_values_and_ownership(api, layout):
    for library in [native, torch]:
        data = layout_array(layout); expected = data.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = getattr(library, api)(data)
        np.testing.assert_array_equal(result.numpy(), expected)
        if not data.size: continue
        if api == 'tensor':
            assert result.data_ptr() != data.__array_interface__['data'][0]
            result.reshape(-1)[0] = 9.
            np.testing.assert_array_equal(data, expected)
        else:
            assert result.data_ptr() == data.__array_interface__['data'][0]
            if data.flags.writeable:
                data.flat[0] = 9.
                assert result.numpy().flat[0] == 9.


def test_readonly_import_warns_once_but_copy_does_not_in_fresh_process():
    script = '''
import warnings, numpy as np, torch as native
from paperleet_torch import torch
for api in (native, torch):
    data=np.arange(3,dtype='f4');data.flags.writeable=False
    with warnings.catch_warnings(record=True) as warnings_seen:
        warnings.simplefilter('always')
        copied=api.tensor(data)
    assert not warnings_seen
    with warnings.catch_warnings(record=True) as warnings_seen:
        warnings.simplefilter('always')
        first=api.as_tensor(data)
        second=api.from_numpy(data)
        converted=api.as_tensor(data,dtype=api.float64)
    assert len(warnings_seen)==1, warnings_seen
    assert 'not writable' in str(warnings_seen[0].message)
    assert first.data_ptr()==second.data_ptr()==data.__array_interface__['data'][0]
    assert converted.data_ptr()!=first.data_ptr()
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('value', [0., -0., 1.25, 1e300, 1e-300, float('nan'), float('inf'), -float('inf'), 0, -(2**63), 2**63-1, True, False])
def test_builtin_scalar_defaults_and_ieee_values_ignore_numpy_error_mode(value):
    with np.errstate(all='raise'):
        actual, expected = torch.tensor(value), native.tensor(value)
    assert actual.dtype == expected.numpy().dtype
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    if actual.dtype.kind == 'f': assert np.signbit(actual.item()) == np.signbit(expected.item())


@pytest.mark.parametrize('value', [-(2**63)-1, 2**63, 2**64])
def test_default_python_integer_overflow_matches_native(value):
    for library in [native, torch]:
        with pytest.raises(ValueError): library.tensor(value)


@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64'])
@pytest.mark.parametrize('scalar', [2**63, 2**64-1])
@pytest.mark.parametrize('operation', [operator.add, operator.sub, operator.mul])
def test_weak_integer_operands_are_distinct_from_default_tensor_construction(dtype, scalar, operation):
    for reverse in [False, True]:
        outputs = []
        for library in [native, torch]:
            value = library.tensor([1, 2, -3], dtype=getattr(library, dtype) if library is native else np.dtype(dtype))
            result = operation(scalar, value) if reverse else operation(value, scalar)
            outputs.append(result.numpy())
        np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize('api', ['tensor', 'as_tensor'])
@pytest.mark.parametrize('target', ['float32', 'int64', 'complex64'])
def test_numpy_casting_obeys_torch_ieee_conversion_not_numpy_error_mode(api, target):
    # Out-of-range float→integer conversions have no portable Torch value
    # guarantee; use finite in-range values for the integer reference contract.
    values = [-3.9, -.1, 0., 2.9, 1e6] if target == 'int64' else [1e300, 1e-300, float('nan'), float('inf')]
    data = np.array(values, dtype='f8')
    with np.errstate(all='raise'):
        actual = getattr(torch, api)(data, dtype=getattr(torch, target))
        expected = getattr(native, api)(data, dtype=getattr(native, target))
        assert np.geterr()['over'] == 'raise'
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_copying_tensor_stays_detached_and_keeps_dtype():
    for api in [native, torch]:
        original = api.tensor([2., 3.], dtype=api.float64, requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); copied = api.tensor(original)
        assert copied.is_leaf and not copied.requires_grad and copied.dtype == original.dtype
        assert copied.data_ptr() != original.data_ptr()
        differentiable = api.tensor(2., requires_grad=True)
        differentiable.square().backward()
        assert differentiable.grad.item() == 4.


@pytest.mark.parametrize('kind', ['unicode', 'bytes', 'object_numbers', 'object_mixed', 'datetime', 'timedelta', 'structured', 'void'])
def test_new_tensor_cannot_hide_invalid_numpy_input_with_inherited_dtype(kind):
    for api in [native, torch]:
        with pytest.raises(TypeError): api.ones(1).new_tensor(invalid_array(kind))


@pytest.mark.parametrize('layout', ['negative', 'negative_singleton', 'odd_stride', 'odd_singleton', 'opposite_endian', 'multi_singleton', 'empty_negative', 'empty_odd', 'complex_odd_singleton'])
def test_new_tensor_checks_external_layout_before_copy(layout):
    for api in [native, torch]:
        with pytest.raises(ValueError): api.ones(1).new_tensor(layout_array(layout))


@pytest.mark.parametrize('tensor_input', [False, True])
def test_new_tensor_inherits_or_overrides_dtype_with_independent_leaf_storage(tensor_input):
    outputs = []
    for api in [native, torch]:
        source = api.tensor([2., 3.], requires_grad=True) if tensor_input else np.array([2., 3.], dtype='f4')
        owner = api.ones(1, dtype=api.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = owner.new_tensor(source, requires_grad=True)
            overridden = owner.new_tensor(source, dtype=api.float32)
        assert result.is_leaf and result.requires_grad and result.dtype == api.float64
        assert not overridden.requires_grad and overridden.dtype == api.float32
        assert result.data_ptr() != (source.data_ptr() if tensor_input else source.__array_interface__['data'][0])
        result.square().sum().backward(); outputs.append(result.grad.numpy())
        if tensor_input: assert source.grad is None
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize('mode', ['from_numpy', 'as_tensor', 'Tensor', 'numpy', 'asarray'])
@pytest.mark.parametrize('metadata', ['shape', 'dtype', 'strides', 'readonly'])
def test_numpy_descriptor_mutation_does_not_change_tensor_metadata(mode, metadata):
    outputs = []
    for api in [native, torch]:
        data = np.ndarray((4,), dtype='f4', buffer=bytearray(32), offset=12)
        data[:] = [0., 1., 2., 3.]
        if mode in ('from_numpy', 'as_tensor', 'Tensor'):
            value = getattr(api, mode)(data); external = data
        else:
            value = api.tensor([0., 1., 2., 3.])
            external = value.numpy() if mode == 'numpy' else np.asarray(value)
        pointer = value.data_ptr()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            if metadata == 'shape': external.shape = (2, 2)
            elif metadata == 'dtype': external.dtype = np.dtype('U1')
            elif metadata == 'strides':
                if mode in ('numpy', 'asarray'):
                    # The exported dense buffer has no prefix for negative strides.
                    external.strides = (0,)
                else: external.strides = (-4,)
            else: external.flags.writeable = False
        assert value.shape == (4,) and value.dtype == api.float32
        assert value.data_ptr() == pointer
        np.testing.assert_array_equal(value.numpy(), [0., 1., 2., 3.])
        before = value._version; value[0] = 9.
        assert value._version == before + 1
        outputs.append(value.numpy().copy())
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize('size', [1, 3, 8, 9])
@pytest.mark.parametrize('container', [list, tuple])
def test_small_float_sequence_allocation_keeps_values_owned_and_gradients(size, container):
    outputs = []
    for api in [native, torch]:
        values = [(-1.)**i * (i + .25) for i in range(size)]
        data = container(values)
        value = api.tensor(data, requires_grad=True)
        assert value.dtype == api.float32 and value.shape == (size,)
        value.square().sum().backward()
        np.testing.assert_array_equal(value.grad.numpy(), 2 * np.asarray(values, dtype='f4'))
        if container is list:
            data[0] = 999.; assert value[0].item() == .25
        outputs.append(value.detach().numpy())
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize('values', [[1e300, 1e-300, float('nan')], [float('inf'), -float('inf'), -0.], [1e-45, -1e-45], [1.25, 2, True]])
def test_float_list_casting_and_fallback_ignore_numpy_error_mode(values):
    with np.errstate(all='raise'):
        actual, expected = torch.tensor(values), native.tensor(values)
        assert np.geterr()['over'] == 'raise'
    assert actual.dtype == expected.numpy().dtype
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


@pytest.mark.parametrize('metadata', ['dtype', 'strides', 'shape'])
def test_import_validation_reads_current_metadata_instead_of_caching_success(metadata):
    for api in [native, torch]:
        for factory in [api.from_numpy, api.as_tensor]:
            data = np.ndarray((4,), dtype='f4', buffer=bytearray(32), offset=12)
            data[:] = [0., 1., 2., 3.]
            first = factory(data)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                if metadata == 'dtype': data.dtype = np.dtype('U1')
                elif metadata == 'strides': data.strides = (-4,)
                else: data.shape = (2, 2)
            if metadata == 'shape': assert factory(data).shape == (2, 2)
            else:
                with pytest.raises(TypeError if metadata == 'dtype' else ValueError): factory(data)
            assert first.shape == (4,) and first.dtype == api.float32
            np.testing.assert_array_equal(first.numpy(), [0., 1., 2., 3.])
