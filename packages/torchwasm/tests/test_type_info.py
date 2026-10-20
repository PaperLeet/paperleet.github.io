"""Numeric limits consumed by tensor programs, without native Torch."""
import copy
import gc
import pickle
import weakref

import numpy as np
import pytest

from torchwasm import as_torch, create_torch, torch


@pytest.fixture(autouse=True)
def restore_default():
    dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'complex64', 'complex128'])
def test_float_limits_describe_real_storage_and_rounding(dtype):
    real = 'float32' if dtype == 'complex64' else 'float64' if dtype == 'complex128' else dtype
    info = torch.finfo(getattr(torch, dtype))
    assert isinstance(info, torch.finfo) and info.dtype == real and type(info.dtype) is str
    assert type(info.bits) is int and info.bits == np.dtype(real).itemsize * 8
    for name in ('eps', 'tiny', 'smallest_normal', 'resolution', 'min', 'max'):
        assert type(getattr(info, name)) is float
    assert info.min == -info.max and info.tiny == info.smallest_normal > 0
    assert info.resolution == {'float16': .001, 'float32': 1e-6, 'float64': 1e-15}[real]
    one = torch.tensor(1., dtype=getattr(torch, real))
    assert (one + info.eps).item() > 1
    assert (one + info.eps / 2).item() == 1
    assert info == torch.finfo(getattr(torch, real))
    assert info is not torch.finfo(getattr(torch, dtype))
    assert not hasattr(info, 'smallest_subnormal')


@pytest.mark.parametrize('dtype', ['uint8', 'int8', 'int16', 'int32', 'int64', 'uint16', 'uint32', 'uint64'])
def test_integer_limits_are_exact_and_can_construct_boundary_values(dtype):
    info = torch.iinfo(getattr(torch, dtype))
    assert isinstance(info, torch.iinfo) and info.dtype == dtype
    assert type(info.bits) is type(info.min) is type(info.max) is int
    unsigned = dtype.startswith('u')
    assert info.min == (0 if unsigned else -(1 << (info.bits - 1)))
    assert info.max == (1 << (info.bits if unsigned else info.bits - 1)) - 1
    values = torch.tensor([info.min, info.max], dtype=getattr(torch, dtype))
    assert values.tolist() == [info.min, info.max]
    assert info == torch.iinfo(type=getattr(torch, dtype))
    assert info is not torch.iinfo(getattr(torch, dtype))
    assert not hasattr(info, 'eps')


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
def test_default_policy_aliases_and_scalar_inference(dtype):
    other = create_torch('limits_example')
    torch.set_default_dtype(dtype)
    before = torch.finfo()
    assert before.dtype == other.finfo().dtype == dtype.name
    for explicit in (torch.float16, torch.float32, torch.float64):
        assert torch.tensor([torch.finfo(explicit).eps]).dtype == dtype
    with as_torch():
        from torch import finfo, iinfo
        assert isinstance(finfo(), finfo) and isinstance(iinfo(torch.int64), iinfo)
    torch.set_default_dtype(torch.float64 if dtype != torch.float64 else torch.float32)
    assert before.dtype == dtype.name
    assert torch.finfo().dtype == torch.get_default_dtype().name


@pytest.mark.parametrize('kind,dtype', [('finfo', torch.float32), ('iinfo', torch.int32)])
@pytest.mark.parametrize('field', ['min', 'max', 'dtype', 'bits', 'custom'])
def test_info_attributes_cannot_be_replaced_or_deleted(kind, dtype, field):
    info = getattr(torch, kind)(dtype)
    with pytest.raises(AttributeError): setattr(info, field, 1)
    with pytest.raises(AttributeError): delattr(info, field)
    assert not hasattr(info, '__dict__')


@pytest.mark.parametrize('kind,dtype', [('finfo', torch.float32), ('iinfo', torch.int32)])
def test_constructor_objects_match_immutable_value_lifecycle(kind, dtype):
    cls = getattr(torch, kind)
    info = cls(dtype)
    assert info == cls(dtype) and info != 3 and info != object()
    assert info != (torch.iinfo(torch.int32) if kind == 'finfo' else torch.finfo(torch.float32))
    for operation in (hash, copy.copy, copy.deepcopy, pickle.dumps, weakref.ref):
        with pytest.raises(TypeError): operation(info)
    with pytest.raises(TypeError): type('UnsupportedChild', (cls,), {})
    info.__init__(torch.float64)
    assert info.dtype == dtype.name
    with pytest.raises(TypeError): cls.__new__(object, dtype)


@pytest.mark.parametrize('kind', ['finfo', 'iinfo'])
@pytest.mark.parametrize('argument', [None, 1, True, 'float32', np.float32, np.float32(1),
                                     [], {}, str, bytes, object, bool])
def test_non_dtype_constructor_arguments_are_rejected(kind, argument):
    with pytest.raises(TypeError): getattr(torch, kind)(argument)


@pytest.mark.parametrize('default', [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize('kind,value,expected', [('finfo', float, 'float64'),
                                               ('finfo', complex, 'float64'),
                                               ('iinfo', int, 'int64')])
def test_builtin_type_aliases_are_independent_of_default_dtype(default, kind, value, expected):
    torch.set_default_dtype(default)
    cls = getattr(torch, kind)
    assert cls(value).dtype == cls(type=value).dtype == expected


@pytest.mark.parametrize('kind,value', [('finfo', int), ('iinfo', float), ('iinfo', complex)])
def test_builtin_type_aliases_still_require_the_right_numeric_kind(kind, value):
    with pytest.raises(TypeError): getattr(torch, kind)(value)


@pytest.mark.parametrize('kind', ['finfo', 'iinfo'])
@pytest.mark.parametrize('bad_form', ['wrong_keyword', 'duplicate', 'extra', 'too_many'])
def test_constructor_keyword_and_arity_validation(kind, bad_form):
    cls = getattr(torch, kind)
    dtype = torch.float32 if kind == 'finfo' else torch.int32
    with pytest.raises(TypeError):
        if bad_form == 'wrong_keyword': cls(dtype=dtype)
        elif bad_form == 'duplicate': cls(dtype, type=dtype)
        elif bad_form == 'extra': cls(type=dtype, typo=1)
        else: cls(dtype, dtype)
    assert cls(type=dtype) == cls(dtype)


@pytest.mark.parametrize('dtype', [torch.bool, torch.int64, torch.uint8])
def test_finfo_rejects_non_floating_types(dtype):
    with pytest.raises(TypeError): torch.finfo(dtype)


@pytest.mark.parametrize('dtype', [torch.bool, torch.float32, torch.complex64])
def test_iinfo_requires_an_explicit_integer_type(dtype):
    with pytest.raises(TypeError): torch.iinfo(dtype)
    with pytest.raises(TypeError): torch.iinfo()


@pytest.mark.parametrize('kind,name', [('finfo', 'float32'), ('iinfo', 'int64')])
def test_numpy_dtype_metadata_is_not_retained(kind, name):
    owner = np.ones(64)
    reference = weakref.ref(owner)
    dtype = np.dtype(name, metadata={'owner': owner})
    info = getattr(torch, kind)(dtype)
    del owner, dtype
    gc.collect()
    assert reference() is None and info.dtype == name


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_limits_remain_constants_inside_forward_reverse_and_hvp(dtype):
    x = torch.tensor([.3, -.7], dtype=dtype, requires_grad=True)
    direction = torch.tensor([.2, .4], dtype=dtype)
    def loss(value):
        return (value.square() * torch.finfo(value.dtype).resolution).sum()
    value, tangent = torch.func.jvp(loss, (x,), (direction,))
    gradient = torch.autograd.grad(value, x)[0]
    _, product = torch.autograd.functional.hvp(loss, x, direction)
    scale = torch.finfo(dtype).resolution
    np.testing.assert_allclose(gradient.numpy(), 2 * x.numpy() * scale, rtol=2e-6)
    np.testing.assert_allclose(product.numpy(), 2 * direction.numpy() * scale, rtol=2e-6)
    np.testing.assert_allclose(tangent.numpy(), (2 * x.numpy() * direction.numpy() * scale).sum(), rtol=2e-6)


def test_integer_limits_feed_checked_indexing_without_rounding():
    index = torch.tensor([0, torch.iinfo(torch.int64).max])
    assert index.tolist() == [0, 9223372036854775807]
    with pytest.raises(RuntimeError): torch.ones(2).gather(0, index)
