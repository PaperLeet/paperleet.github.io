"""Native-reference contracts for scalar promotion and sequence storage."""
import itertools
import warnings

import numpy as np
import pytest
import torch as native

from torchwasm import torch


DTYPES = ['bool', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16',
          'int32', 'int64', 'float16', 'float32', 'float64', 'complex64', 'complex128']
VALUES = {'bool': True, 'int': 2, 'float': 2.25, 'complex': 2.25 + .5j}
VALUES.update({'numpy_' + name: np.dtype(name).type(1) for name in DTYPES})


def compare(make, *, dtype=None, api='tensor'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        options = {'dtype': getattr(native, dtype)} if dtype else {}
        try:
            expected = getattr(native, api)(make(native), **options)
        except Exception as error:
            with pytest.raises(type(error)):
                getattr(torch, api)(make(torch), **({'dtype': getattr(torch, dtype)} if dtype else {}))
            return
        actual = getattr(torch, api)(make(torch), **({'dtype': getattr(torch, dtype)} if dtype else {}))
        assert actual.dtype == expected.numpy().dtype
        assert actual.shape == tuple(expected.shape)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        assert actual.is_leaf == expected.is_leaf and actual.requires_grad == expected.requires_grad


@pytest.mark.parametrize('left,right', tuple(itertools.product(DTYPES, repeat=2)))
def test_public_type_promotion_matches_native(left, right):
    try:
        expected = native.promote_types(getattr(native, left), getattr(native, right))
    except RuntimeError:
        with pytest.raises(RuntimeError): torch.promote_types(getattr(torch, left), getattr(torch, right))
    else:
        assert torch.promote_types(getattr(torch, left), getattr(torch, right)) == np.dtype(str(expected).removeprefix('torch.'))


@pytest.mark.parametrize('left,right', [(None, None), (None, 'float32'), ('float32', None)])
def test_public_type_promotion_rejects_missing_dtypes(left, right):
    for lib in (native, torch):
        with pytest.raises(TypeError):
            lib.promote_types(getattr(lib, left) if left else None,
                              getattr(lib, right) if right else None)


def test_public_type_promotion_accepts_native_keyword_names():
    for lib in (native, torch):
        assert lib.promote_types(type1=lib.int8, type2=lib.uint8) == lib.int16
        with pytest.raises(TypeError): lib.promote_types(left=lib.int8, right=lib.uint8)
        with pytest.raises(TypeError): lib.promote_types(lib.int8, type1=lib.int8, type2=lib.uint8)


@pytest.mark.parametrize('left,right', tuple(itertools.product(VALUES, repeat=2)))
@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('api', ['tensor', 'as_tensor'])
def test_mixed_sequence_promotion_matches_native(left, right, nested, api):
    a, b = VALUES[left], VALUES[right]
    compare(lambda lib: [[a], [b]] if nested else [a, b], api=api)


@pytest.mark.parametrize('name', VALUES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_explicit_sequence_scalar_storage_matches_native(name, dtype):
    compare(lambda lib: [VALUES[name]], dtype=dtype)


@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'int8', 'int16', 'int64', 'uint64', 'bool'])
@pytest.mark.parametrize('value', [-2**63-1, -2**63, -65537, -65536, -256, -255,
                                 -129, -128, 127, 128, 255, 256, 2**63-1, 2**63,
                                 2**64-1, 2**64, -1., -.5, 127.5, 255.5,
                                 float(2**63), float('nan'), float('inf')])
def test_scalar_integer_cast_range_and_protocol_matches_native(dtype, value):
    compare(lambda lib: [value], dtype=dtype)


@pytest.mark.parametrize('dtype', [None, 'float32', 'float64', 'int64', 'bool', 'uint64', 'complex128'])
@pytest.mark.parametrize('case', ['numpy0d_first', 'numpy0d_second', 'negative_array', 'foreign_array',
                                 'array_rows', 'scalar_tensor', 'one_element_tensor',
                                 'multi_element_tensor', 'empty_rows', 'ragged', 'none', 'object',
                                 'string', 'string_sequence', 'range', 'memoryview', 'bytearray'])
def test_sequence_shapes_and_scalar_protocols_match_native(case, dtype):
    def make(lib):
        return {
            'numpy0d_first': lambda: [np.array(2), 1],
            'numpy0d_second': lambda: [1, np.array(2)],
            'negative_array': lambda: [np.arange(3, dtype='f8')[::-1]],
            'foreign_array': lambda: [np.array([1, 2], dtype='>i4')],
            'array_rows': lambda: [np.array([1, 2], dtype='i2'), np.array([3, 4], dtype='i2')],
            'scalar_tensor': lambda: [lib.tensor(1.)],
            'one_element_tensor': lambda: [lib.tensor([[1.]])],
            'multi_element_tensor': lambda: [lib.tensor([1., 2.])],
            'empty_rows': lambda: [[], []],
            'ragged': lambda: [[1], [2, 3]],
            'none': lambda: [None], 'object': lambda: [object()],
            'string': lambda: '1.5', 'string_sequence': lambda: ['1.5'],
            'range': lambda: range(3), 'memoryview': lambda: memoryview(bytes([1, 2])),
            'bytearray': lambda: bytearray([1, 2]),
        }[case]()
    compare(make, dtype=dtype)


def test_precision_copy_leaf_and_training_gradients_match_native():
    for lib in (native, torch):
        source = [np.float64(1 + 2**-40), np.float64(2 + 2**-39)]
        value = lib.tensor(source, requires_grad=True)
        assert value.detach().numpy().dtype == np.dtype('float64')
        expected = np.array(source)
        source[0] = np.float64(99.)
        np.testing.assert_array_equal(value.detach().numpy(), expected)
        (value.square().sum()).backward()
        np.testing.assert_array_equal(value.grad.numpy(), 2 * expected)
        label = lib.tensor([np.int16(2), np.int16(3)])
        assert label.numpy().dtype == np.dtype('int16')
        copied = lib.ones(1, dtype=lib.float64).new_tensor(label, requires_grad=True)
        assert copied.is_leaf and copied.dtype == lib.float64
        copied.sum().backward()
        np.testing.assert_array_equal(copied.grad.numpy(), [1., 1.])
        np.testing.assert_array_equal(label.numpy(), [2, 3])


def test_inference_maximal_complex_still_validates_remaining_storage():
    for values in [[np.complex128(1), np.uint16(2)],
                   [np.uint16(2), np.complex128(1)],
                   [np.complex128(1), '2'], [np.complex128(1), None]]:
        compare(lambda lib: values)


@pytest.mark.parametrize('size', [1, 3, 8, 9])
@pytest.mark.parametrize('kind', [int, bool])
def test_small_homogeneous_sequences_own_data_and_preserve_bounds(size, kind):
    source = [kind(1)] * size
    for lib in (native, torch):
        value = lib.tensor(source)
        assert value.numpy().dtype == np.dtype('int64' if kind is int else 'bool')
        assert value.shape == (size,) and not value.requires_grad
        value[0] = kind(0)
        assert source[0] == kind(1)
    if kind is int:
        for source in [[2**63] * size, [-2**63-1] * size,
                       [2**100, .5], [1, np.int16(2)], [1, np.uint16(2)]]:
            compare(lambda lib: source)
