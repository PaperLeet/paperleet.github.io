"""CPU/no-grad copy_ values, aliasing and saved-state guards."""
import warnings
import numpy as np
import pytest
import torch as native
from stridetorch import UnsupportedOperation, torch


@pytest.mark.parametrize('target_dtype', ['float32', 'float64', 'int64', 'bool', 'complex64'])
@pytest.mark.parametrize('source_dtype', ['float32', 'float64', 'int64', 'bool', 'complex64'])
def test_copy_broadcast_cast_and_identity(target_dtype, source_dtype):
    arrays = []
    for api in (torch, native):
        values = [[1+2j, 0+3j, -2-1j]] if source_dtype == 'complex64' else [[1.2, 0., -2.4]]
        source = api.tensor(values, dtype=getattr(api, source_dtype))
        target = api.zeros(2, 3, dtype=getattr(api, target_dtype)); version = target._version
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); result = target.copy_(source, non_blocking=True)
        assert result is target and target._version == version + 1
        assert target.dtype == getattr(api, target_dtype)
        arrays.append(target.numpy())
    np.testing.assert_array_equal(*arrays)


@pytest.mark.parametrize('layout', ['transpose', 'slice', 'single_dim_expand', 'disjoint_strided', 'same', 'same_expanded', 'empty'])
def test_copy_layout_and_shared_storage_versions(layout):
    arrays = []
    for api in (torch, native):
        base = api.arange(12., dtype=api.float64).reshape(3, 4)
        if layout == 'transpose': target, source = base.T, api.ones(4, 3, dtype=api.float64) * 2
        elif layout == 'slice': target, source = base[1:], api.ones(1, 4, dtype=api.float64)
        elif layout == 'single_dim_expand': target, source = base[:1].expand(1, 4), api.ones(1, 4, dtype=api.float64)
        elif layout == 'disjoint_strided': target, source = base[:, 0], base[:, 1]
        elif layout == 'same': target, source = base[1:], base[1:]
        elif layout == 'same_expanded': target, source = base[:1, :1].expand(2, 3), base[:1, :1].expand(2, 3)
        else: target, source = base[:0], api.ones(1, 4, dtype=api.float64)
        before = base._version
        assert target.copy_(source) is target and base._version == before + 1
        arrays.append(base.numpy())
    np.testing.assert_array_equal(*arrays)


@pytest.mark.parametrize('layout', ['partial', 'transpose', 'expand', 'scalar_alias', 'broadcast', 'rank'])
def test_copy_overlap_and_invalid_broadcast_fail_without_mutation(layout):
    for api in (torch, native):
        base = api.arange(9.).reshape(3, 3); snapshot = base.clone()
        if layout == 'partial': target, source = base[1:], base[:-1]
        elif layout == 'transpose': target, source = base, base.T
        elif layout == 'expand': target, source = base[:1, :1].expand(2, 3), api.ones(2, 3)
        elif layout == 'scalar_alias': target, source = base[:1, :1], base[0, 0]
        elif layout == 'broadcast': target, source = base, api.ones(2)
        else: target, source = base[0, 0], api.ones(1)
        version = base._version
        with pytest.raises(RuntimeError): target.copy_(source)
        assert base._version == version
        np.testing.assert_array_equal(base.numpy(), snapshot.numpy())


@pytest.mark.parametrize('operation', ['square', 'exp', 'add'])
def test_no_grad_copy_preserves_history_and_operator_mutation_guards(operation):
    for api in (torch, native):
        x = api.tensor([2., 3.], requires_grad=True)
        value = {'square': lambda: x.square(), 'exp': lambda: x.exp(), 'add': lambda: x + 1}[operation]()
        target = x if operation == 'square' else value
        with api.no_grad(): target.copy_(api.ones(2))
        assert value.requires_grad and not value.is_leaf
        if operation == 'add': value.sum().backward(); assert x.grad.tolist() == [1., 1.]
        else:
            with pytest.raises((RuntimeError, UnsupportedOperation)): value.sum().backward()


def test_copy_alias_detach_and_live_parameter_callbacks():
    for api in (torch, native):
        x = api.tensor([2., 3.], requires_grad=True); detached = x.detach(); seen = []
        x.register_hook(lambda g: seen.append(g.tolist()))
        with api.no_grad(): assert x.copy_(api.tensor(4.)) is x
        assert detached.tolist() == x.tolist() == [4., 4.]
        x.square().sum().backward(); assert seen == [[8., 8.]]


def test_copy_validation_and_differentiable_boundary_are_explicit():
    for api in (torch, native):
        x = api.ones(2)
        with pytest.raises(TypeError): x.copy_([1., 2.])
        with pytest.raises(TypeError): x.copy_(api.ones(2), non_blocking=1)
    with pytest.raises(UnsupportedOperation): torch.ones(2).copy_(torch.ones(2, requires_grad=True))
    x = torch.ones(2, requires_grad=True)
    with pytest.raises(UnsupportedOperation): torch.func.jvp(lambda v: v.copy_(torch.ones(2)), (x,), (torch.ones(2),))


@pytest.mark.parametrize('values', [np.array([1, object()], dtype=object), np.array(['1', 'bad']), np.array(['2026-01-01', '2026-01-02'], dtype='datetime64[D]'), np.array([(1,), (2,)], dtype=[('value', 'i4')])])
@pytest.mark.parametrize('invalid_source', [False, True])
def test_unsupported_numpy_storage_cannot_partially_mutate_copy(values, invalid_source):
    # Native Torch rejects these NumPy storages before they can enter a copy.
    with pytest.raises(TypeError): native.tensor(values)
    with pytest.raises(TypeError): torch.tensor(values)
    # Keep the old partial-write regression as defense against corrupted private
    # storage, even though public construction now rejects it at the boundary.
    invalid = torch.tensor([0., 0.]); invalid._data = values
    valid = torch.tensor([5., 6.])
    target, source = (valid, invalid) if invalid_source else (invalid, valid)
    snapshot = target.numpy().copy(); version = target._version
    with pytest.raises(UnsupportedOperation, match='numeric'): target.copy_(source)
    assert target._version == version
    np.testing.assert_array_equal(target.numpy(), snapshot)


@pytest.mark.parametrize('source_dtype,target_dtype', [('float64', 'float32'), ('float32', 'float32'), ('float32', 'float64'), ('complex128', 'complex64'), ('complex64', 'complex64'), ('complex64', 'complex128')])
def test_copy_casts_do_not_inherit_numpy_floating_point_error_policy(source_dtype, target_dtype):
    results = []
    values = [1e300, 1e-300, float('nan'), float('inf')] if source_dtype in ('float64', 'complex128') else [float('inf'), 0., float('nan'), -float('inf')]
    if source_dtype.startswith('complex'): values = [complex(value, value) for value in values]
    for api in (torch, native):
        source = api.tensor(values, dtype=getattr(api, source_dtype)); target = api.zeros(4, dtype=getattr(api, target_dtype))
        version = target._version
        with np.errstate(all='raise'):
            assert target.copy_(source) is target
            assert all(policy == 'raise' for policy in np.geterr().values())
        assert target._version == version + 1
        results.append(target.numpy())
    np.testing.assert_array_equal(*results)
