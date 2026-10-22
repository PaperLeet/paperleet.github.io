"""NumPy-only construction workflows, also executed in real Pyodide."""
import warnings

import numpy as np
import pytest

from torchwasm import as_torch, torch


def test_scalar_sequences_preserve_precision_copy_and_gradients():
    source = [[np.float64(1 + 2**-40)], [np.float64(2 + 2**-39)]]
    value = torch.tensor(source, requires_grad=True)
    expected = np.array(source)
    assert value.dtype == torch.float64 and value.is_leaf
    source[0][0] = np.float64(99.)
    np.testing.assert_array_equal(value.numpy(), expected)
    value.square().sum().backward()
    np.testing.assert_array_equal(value.grad.numpy(), 2 * expected)
    source = torch.tensor([[3.]], dtype=torch.float64, requires_grad=True)
    copied = torch.tensor([source]); assert copied.shape == (1,) and copied.is_leaf
    assert not copied.requires_grad and copied.dtype == torch.float64
    copied[0] = 5.; assert source.item() == 3.


def test_common_dtype_categories_and_public_aliases():
    assert torch.half == torch.float16 and torch.short == torch.int16 and torch.char == torch.int8
    assert torch.cfloat == torch.complex64 and torch.cdouble == torch.complex128
    for left, right, expected in [(torch.int64, torch.float16, torch.float16),
                                  (torch.float32, torch.complex128, torch.complex128),
                                  (torch.int8, torch.uint8, torch.int16),
                                  (torch.float64, torch.complex64, torch.complex128),
                                  (torch.float16, torch.uint64, torch.float16)]:
        assert torch.promote_types(left, right) == expected
        assert torch.promote_types(right, left) == expected
    with pytest.raises(RuntimeError): torch.promote_types(torch.uint16, torch.int16)
    with pytest.raises(RuntimeError): torch.promote_types(torch.uint64, torch.complex64)
    assert torch.promote_types(torch.uint64, torch.uint64) == torch.uint64
    for left, right in [(None, None), (torch.float32, None), (None, torch.float64)]:
        with pytest.raises(TypeError): torch.promote_types(left, right)
    assert torch.promote_types(type1=torch.int8, type2=torch.uint8) == torch.int16
    with pytest.raises(TypeError): torch.promote_types(left=torch.int8, right=torch.uint8)


def test_numpy_labels_and_explicit_new_tensor_copy():
    labels = [np.int16(2), np.int16(3)]
    value = torch.as_tensor(labels)
    assert value.dtype == torch.int16
    labels[0] = np.int16(99)
    np.testing.assert_array_equal(value.numpy(), [2, 3])
    owner = torch.ones(1, dtype=torch.float64)
    result = owner.new_tensor(value, requires_grad=True)
    result.square().sum().backward()
    assert result.is_leaf and result.dtype == torch.float64
    np.testing.assert_array_equal(result.grad.numpy(), [4., 6.])
    inherited = owner.new_tensor([np.int16(2), np.int16(3)])
    assert inherited.dtype == torch.float64
    np.testing.assert_array_equal(inherited.numpy(), [2., 3.])


def test_integer_scalar_storage_is_distinct_from_array_casts():
    np.testing.assert_array_equal(torch.tensor([-255, -129, 255], dtype=torch.uint8).numpy(), [1, 127, 255])
    for value in [-256, 256, -.5, float('inf'), float('nan')]:
        with pytest.raises(RuntimeError): torch.tensor([value], dtype=torch.uint8)
    for value in [np.float32(1.5), np.float16(1.5), 1+.5j]:
        with pytest.raises(TypeError): torch.tensor([value], dtype=torch.int64)
    for value in [np.uint64(1), np.int16(1), 1.]:
        with pytest.raises(TypeError): torch.tensor([value], dtype=torch.uint64)
    with pytest.raises(OverflowError): torch.tensor([-1], dtype=torch.uint64)
    assert torch.tensor([2**64-1], dtype=torch.uint64).item() == 2**64-1
    # Array import/casting uses the separate ndarray conversion contract.
    value = torch.tensor(np.array([1.5], dtype='f4'), dtype=torch.int64)
    assert value.tolist() == [1]


def test_nested_arrays_values_and_shape_validation():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        value = torch.tensor([np.arange(3, dtype='f8')[::-1]])
        assert value.dtype == torch.float64 and value.shape == (1, 3)
        np.testing.assert_array_equal(value.numpy(), [[2., 1., 0.]])
        value = torch.tensor([np.array([1, 2], dtype='>i4')])
        assert value.dtype == torch.int32
        np.testing.assert_array_equal(value.numpy(), [[1, 2]])
    with pytest.raises(TypeError): torch.tensor([np.array(2), 1])
    np.testing.assert_array_equal(torch.tensor([1, np.array(2)]).numpy(), [1, 2])
    with pytest.raises(ValueError): torch.tensor([[1], [2, 3]])
    with pytest.raises(ValueError): torch.tensor(['1.5'])
    with pytest.raises(RuntimeError): torch.tensor([None])
    with pytest.raises(TypeError): torch.tensor([None], dtype=torch.float64)
    assert torch.tensor([[], []]).shape == (2, 0)


def test_error_modes_and_unchanged_torch_import_workflow():
    previous = np.geterr()
    with np.errstate(all='raise'), as_torch():
        import torch as aliased
        for values in [[np.float64(1e300)], [np.float64(1e-300)]]:
            value = aliased.tensor(values, dtype=aliased.float32)
            assert value.dtype == aliased.float32
            assert np.geterr() == dict.fromkeys(previous, 'raise')
        with pytest.raises(RuntimeError): aliased.tensor([float('nan')], dtype=aliased.int16)
        assert np.geterr() == dict.fromkeys(previous, 'raise')
    assert np.geterr() == previous


def test_small_builtin_sequences_and_mixed_overflow_follow_distinct_paths():
    for size in [1, 3, 8, 9]:
        integers = [2] * size; flags = [True] * size
        value = torch.tensor(integers); mask = torch.tensor(flags)
        assert value.dtype == torch.int64 and mask.dtype == torch.bool
        value[0] = 0; mask[0] = False
        assert integers[0] == 2 and flags[0] is True
        with pytest.raises(ValueError): torch.tensor([2**63] * size)
    assert torch.tensor([2**100, .5]).dtype == torch.float32
    with pytest.raises(RuntimeError): torch.tensor([1, np.uint16(2)])
