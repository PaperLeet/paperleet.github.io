"""Independent array-import, allocation and training contracts without Torch."""
import warnings

import numpy as np
import pytest

from stridetorch import UnsupportedOperation, torch


def test_external_non_numeric_arrays_are_rejected_before_casting():
    samples = [np.array(['1', '2']), np.array([b'1']), np.array([1., object()], dtype=object),
               np.array(['2026-01-01'], dtype='datetime64[D]'),
               np.array([(1,)], dtype=[('x', 'i4')])]
    for data in samples:
        for factory in [torch.tensor, torch.as_tensor, torch.from_numpy]:
            with pytest.raises(TypeError): factory(data)
        for factory in [torch.tensor, torch.as_tensor]:
            with pytest.raises(TypeError): factory(data, dtype=torch.float32)
        with pytest.raises(TypeError): torch.Tensor(data)
        with pytest.raises(TypeError): torch.ones(1).new_tensor(data)


def test_array_layout_validation_and_legal_shared_view_gradient_guard():
    invalid = [np.arange(3, dtype='f4')[::-1], np.arange(3, dtype='f4')[2:1:-1],
               np.arange(3, dtype=np.dtype('f4').newbyteorder('S')),
               np.ndarray((2,), dtype='f4', buffer=bytearray(12), strides=(3,))]
    for data in invalid:
        for factory in [torch.tensor, torch.as_tensor, torch.from_numpy]:
            with pytest.raises(ValueError): factory(data)
        with pytest.raises(ValueError): torch.ones(1).new_tensor(data)
    data = np.arange(30, dtype='f4').reshape(5, 6)[::2, 1::2]
    value = torch.from_numpy(data).requires_grad_()
    loss = value.square().sum()
    loss.backward()
    np.testing.assert_array_equal(value.grad.numpy(), 2 * data)
    value.grad = None; loss = value.square().sum()
    before = value._version
    with torch.no_grad(): value[0, 0] = 99.
    assert data[0, 0] == 99. and value._version == before + 1
    with pytest.raises(UnsupportedOperation, match='modified in-place'): loss.backward()


def test_scalar_and_numpy_casts_keep_ieee_values_and_error_policy():
    before = np.geterr()
    data = np.array([1e300, 1e-300, float('nan'), float('inf')], dtype='f8')
    with np.errstate(all='raise'):
        assert np.isinf(torch.tensor(1e300).item())
        assert torch.tensor(1e-300).item() == 0.
        for factory in [torch.tensor, torch.as_tensor]:
            result = factory(data, dtype=torch.float32)
            np.testing.assert_array_equal(result.numpy(), [float('inf'), 0., float('nan'), float('inf')])
            integer = factory(np.array([-3.9, -.1, 0., 2.9]), dtype=torch.int64)
            np.testing.assert_array_equal(integer.numpy(), [-3, 0, 0, 2])
            # Torch explicitly leaves out-of-range float→integer values platform
            # dependent. Check execution/error-mode isolation, not a sentinel.
            undefined = factory(data, dtype=torch.int64)
            assert undefined.shape == data.shape and undefined.dtype == np.dtype('int64')
        assert np.geterr()['over'] == 'raise'
    assert np.geterr() == before
    for value in [-(2**63)-1, 2**63, 2**64]:
        with pytest.raises(ValueError): torch.tensor(value)
    # Weak scalar operands use the receiving dtype, unlike default construction.
    assert (torch.tensor([1], dtype=torch.int32) + 2**63).tolist() == [1]
    assert (torch.tensor([1], dtype=torch.int64) + 2**63).tolist() == [-(2**63)+1]
    assert ((2**64-1) + torch.tensor([1], dtype=torch.int64)).tolist() == [0]


def test_readonly_array_warning_is_shared_by_import_factories(monkeypatch):
    import stridetorch._state as state
    monkeypatch.setattr(state, '_numpy_readonly_warning_emitted', False)
    data = np.arange(4, dtype='f4'); data.flags.writeable = False
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always'); copied = torch.tensor(data)
    assert not seen and copied.data_ptr() != data.__array_interface__['data'][0]
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        first = torch.as_tensor(data); second = torch.from_numpy(data)
        converted = torch.as_tensor(data, dtype=torch.float64)
    assert len(seen) == 1 and 'not writable' in str(seen[0].message)
    assert first.data_ptr() == second.data_ptr() == data.__array_interface__['data'][0]
    assert converted.data_ptr() != first.data_ptr()
    np.testing.assert_array_equal(first.numpy(), data)


def test_storage_guard_covers_constructor_allocations_and_conversions():
    for dtype in ['O', 'U2', 'S2', 'V4', 'datetime64[D]']:
        for factory in [lambda: torch.zeros(2, dtype=dtype),
                        lambda: torch.full((2,), 1, dtype=dtype),
                        lambda: torch.ones(2).to(dtype=dtype)]:
            with pytest.raises(TypeError): factory()
    model = torch.nn.Linear(2, 1)
    parameter = model.weight; before = parameter.numpy().copy(); version = parameter._version
    with pytest.raises(TypeError): model.to(dtype='O')
    assert model.weight is parameter and parameter._version == version
    np.testing.assert_array_equal(parameter.numpy(), before)


def test_existing_internal_strided_tensor_copy_stays_detached_and_independent():
    # Internal views can have NumPy strides that external Torch imports reject.
    source = torch.Tensor(np.arange(4, dtype='f4')[::-1], requires_grad=True)
    copied = torch.tensor(source)
    assert copied.is_leaf and not copied.requires_grad
    assert copied.data_ptr() != source.data_ptr() and copied.dtype == source.dtype
    np.testing.assert_array_equal(copied.numpy(), source.numpy())
    copied[0] = 9.; assert source[0].item() == 3.
    inherited = torch.ones(1, dtype=torch.float64).new_tensor(source, requires_grad=True)
    assert inherited.is_leaf and inherited.dtype == torch.float64
    assert inherited.data_ptr() != source.data_ptr()
    value = torch.tensor(2., requires_grad=True)
    value.square().backward(); assert value.grad.item() == 4.


def test_shared_bytes_keep_independent_array_and_tensor_metadata():
    data = np.arange(4, dtype='f4')
    value = torch.from_numpy(data)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        data.shape = (2, 2); data.dtype = np.dtype('U1'); data.flags.writeable = False
    assert value.shape == (4,) and value.dtype == torch.float32
    np.testing.assert_array_equal(value.numpy(), [0., 1., 2., 3.])
    value[0] = 9.; assert value.tolist() == [9., 1., 2., 3.]
    for factory in [lambda: value.numpy(), lambda: np.asarray(value)]:
        exported = factory(); assert exported is not value.numpy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            exported.shape = (2, 2); exported.dtype = np.dtype('U1'); exported.flags.writeable = False
        assert value.shape == (4,) and value.dtype == torch.float32
    for size in [1, 3, 8, 9]:
        source = [.25] * size
        result = torch.tensor(source, requires_grad=True)
        source[0] = 99.; assert result[0].item() == .25
        result.square().sum().backward()
        np.testing.assert_array_equal(result.grad.numpy(), [.5] * size)
    with np.errstate(all='raise'):
        for values in [[1e300, 1e-300], [float('nan'), float('inf')], [1e-45, -1e-45]]:
            result = torch.tensor(values)
            assert result.dtype == torch.float32
