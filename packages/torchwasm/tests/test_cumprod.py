import itertools
import warnings

import numpy as np
import pytest

from torchwasm import torch, UnsupportedOperation


def tensor(raw, grad=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return torch.from_numpy(raw).requires_grad_(grad)


def assert_values(actual, expected, *, rtol=2e-5, atol=2e-7):
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=rtol, atol=atol)


def polynomial(raw, axis):
    """Enumerate monomials and omitted factors, independent of scan kernels."""
    coords = list(np.ndindex(raw.shape))
    values = np.zeros(raw.size)
    jacobian = np.zeros((raw.size, raw.size))
    hessians = np.zeros((raw.size, raw.size, raw.size))
    for row, out in enumerate(coords):
        factors = [i for i, point in enumerate(coords)
                   if all(point[d] == out[d] for d in range(raw.ndim) if d != axis)
                   and point[axis] <= out[axis]]
        values[row] = np.prod([raw[coords[k]] for k in factors], dtype=np.float64)
        for first in factors:
            jacobian[row, first] = np.prod([raw[coords[k]] for k in factors if k != first], dtype=np.float64)
            for second in factors:
                if first != second:
                    hessians[row, first, second] = np.prod(
                        [raw[coords[k]] for k in factors if k not in (first, second)], dtype=np.float64)
    return values, jacobian, hessians


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape,axis', [((4,),0), ((2,4),0), ((2,4),1), ((2,0,3),1), ((2,0,3),2)])
@pytest.mark.parametrize('kind', ['C', 'F', 'strided', 'readonly', 'broadcast'])
@pytest.mark.parametrize('zeros', [0, 1, 2])
@pytest.mark.parametrize('interface', ['method', 'function'])
def test_values_reverse_layout_and_zero_factors(dtype, shape, axis, kind, zeros, interface):
    raw = np.resize(np.array([1.125, -.75, 1.5, .5], dtype=dtype), shape)
    if zeros and raw.size:
        key = [slice(None)] * raw.ndim
        key[axis] = slice(0,zeros)
        raw[tuple(key)] = 0
    if kind == 'F':
        raw = np.asfortranarray(raw)
    if kind == 'strided':
        storage = np.empty(tuple(2*n for n in shape), dtype=dtype)
        view = storage[tuple(slice(None,None,2) for _ in shape)]
        view[...] = raw
        raw = view
    if kind == 'readonly':
        raw.flags.writeable = False
    if kind == 'broadcast' and raw.size:
        raw = np.broadcast_to(np.take(raw,[0],axis=axis),shape)
    before = raw.copy()
    expected, jacobian, _ = polynomial(raw,axis)
    x = tensor(raw,True)
    y = x.cumprod(axis) if interface == 'method' else torch.cumprod(x,axis)
    seed = (np.arange(raw.size,dtype=dtype).reshape(shape)+1)/4
    seed.flags.writeable = False
    gradient, = torch.autograd.grad(y,x,tensor(seed))
    assert_values(y,expected.reshape(shape))
    assert_values(gradient,(seed.reshape(-1)@jacobian).reshape(shape))
    assert y.shape == gradient.shape == shape and y.is_contiguous()
    assert x.grad is None
    np.testing.assert_array_equal(raw,before)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('target', [torch.float32, torch.float64])
@pytest.mark.parametrize('axis', [0,1])
@pytest.mark.parametrize('point', [[1.125,-.75,1.5,.5], [0.,2.,3.,4.], [2.,0.,3.,0.]])
@pytest.mark.parametrize('mode', ['direct','functional','hvp_square','hvp_sin','hvp_norm'])
def test_jvp_and_composed_hvp_against_independent_polynomials(dtype,target,axis,point,mode):
    raw = np.array([point,point[::-1]],dtype=dtype)
    direction = np.resize(np.array([.5,-.25,.75,1.],dtype=dtype),raw.shape)
    values, jacobian, curvature = polynomial(raw,axis)
    x, v = tensor(raw,True), tensor(direction)
    f = lambda a: torch.cumprod(a,axis,dtype=target)
    if mode in ('direct','functional'):
        if mode == 'direct':
            primal, tangent = torch.func.jvp(f,(x,),(v,))
        else:
            primal, tangent = torch.autograd.functional.jvp(f,x,v)
        assert_values(primal,values.reshape(raw.shape))
        assert_values(tangent,(jacobian@direction.reshape(-1)).reshape(raw.shape))
    else:
        if mode == 'hvp_square':
            loss = lambda a: f(a).square().sum()
            expected_value = values@values
            slope, outer = 2*values, 2*np.eye(raw.size)
        elif mode == 'hvp_sin':
            loss = lambda a: f(a).sin().sum()
            expected_value = np.sin(values).sum()
            slope, outer = np.cos(values), np.diag(-np.sin(values))
        else:
            if not np.any(values):
                pytest.skip('Zero norm has a separate native boundary contract')
            loss = lambda a: torch.linalg.vector_norm(f(a))
            radius = np.linalg.norm(values)
            expected_value = radius
            slope = values/radius
            outer = np.eye(raw.size)/radius-np.outer(values,values)/radius**3
        expected_hessian = jacobian.T@outer@jacobian+np.einsum('i,ijk->jk',slope,curvature)
        primal, tangent = torch.autograd.functional.hvp(loss,x,v)
        assert_values(primal,expected_value,rtol=5e-5)
        assert_values(tangent,(expected_hessian@direction.reshape(-1)).reshape(raw.shape),rtol=1e-4,atol=2e-6)
    assert x.grad is None


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('point',[[1.125,-.75,1.5,.5],[0.,2.,3.,4.],[2.,0.,3.,0.]])
def test_differentiable_functional_jvp_has_primal_and_vector_edges(dtype,point):
    raw = np.array(point,dtype=dtype)
    values, jacobian, curvature = polynomial(raw,0)
    x = tensor(raw,True)
    direction = np.array([.5,-.25,.75,1.],dtype=dtype)
    v = tensor(direction,True)
    p, t = torch.autograd.functional.jvp(lambda a:a.cumprod(0),x,v,create_graph=True)
    seed = np.array([.25,.5,-.75,1.],dtype=dtype)
    gx, gv = torch.autograd.grad(t,(x,v),tensor(seed))
    assert_values(gx,np.einsum('i,ijk,k->j',seed,curvature,direction))
    assert_values(gv,seed@jacobian)
    assert p.requires_grad and t.requires_grad and x.grad is v.grad is None


@pytest.mark.parametrize('source,target',list(itertools.product(
    [np.float16,np.float32,np.float64],[np.float16,np.float32,np.float64])))
def test_dtype_changing_reverse_is_returned_in_input_space(source,target):
    x = tensor(np.array([2.,0.,3.,4.],dtype=source),True)
    y = x.cumprod(0,dtype=target)
    gradient, = torch.autograd.grad(y,x,tensor(np.array([1.,2.,3.,4.],dtype=target)))
    assert gradient.dtype == np.dtype(source) and y.dtype == np.dtype(target)
    assert_values(gradient,[1.,118.,0.,0.])


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_opmath_accumulation_scalar_ownership_and_saved_versions(dtype):
    values = np.resize(np.array([1+1/1024,1-1/1024,1+3/1024],dtype=dtype),129)
    accumulator = np.float32 if dtype == np.float16 else np.float64
    expected = np.multiply.accumulate(values,dtype=accumulator).astype(dtype)
    np.testing.assert_array_equal(tensor(values).cumprod(0).numpy(),expected)
    x = tensor(np.array(-0.,dtype=dtype),True)
    y = x.cumprod(-1)
    assert y is not x and not np.shares_memory(x._data,y._data)
    assert y.shape == () and np.signbit(y.item())
    y.backward(torch.tensor(2.,dtype=dtype))
    assert x.grad.item() == 2
    x = torch.ones(3,dtype=dtype,requires_grad=True)
    y = x.cumprod(0)
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(UnsupportedOperation):
        y.sum().backward()


@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int8,np.int16,np.int32,np.int64,np.uint64])
@pytest.mark.parametrize('shape',[(),(3,),(2,0),(2,3)])
def test_integer_default_promotion(dtype,shape):
    result = tensor(np.ones(shape,dtype=dtype)).cumprod(-1)
    assert result.dtype == torch.int64 and result.shape == shape
    np.testing.assert_array_equal(result.numpy(),np.ones(shape,dtype=np.int64))


def test_capability_no_grad_errors_and_transform_boundaries():
    assert torch._browser_capabilities['torch.cumprod'].forward
    x = torch.ones(3,requires_grad=True)
    with pytest.raises(RuntimeError):
        x.cumprod(0,dtype=torch.int64)
    with torch.no_grad():
        assert x.cumprod(0,dtype=torch.int64).dtype == torch.int64
        p, t = torch.func.jvp(lambda a:a.cumprod(0),(x,),(torch.ones_like(x),))
        assert_values(t,[1.,2.,3.])
    for axis in (True,1.,'0'):
        with pytest.raises(TypeError):
            x.cumprod(axis)
    for axis in (-2,1):
        with pytest.raises(IndexError):
            x.cumprod(axis)
    with pytest.raises(RuntimeError):
        x.cumprod(2**64)
    with pytest.raises(UnsupportedOperation):
        torch.func.jvp(lambda a:a.cumprod(0,dtype=torch.float16),(x,),(torch.ones_like(x),))
    with pytest.raises(UnsupportedOperation):
        x.cumprod(0,dtype=torch.complex64)
    assert not torch.tensor([1+2j,0j]).cumprod(0).requires_grad


@pytest.mark.parametrize('shape',[(),(1,),(2,1),(0,)])
@pytest.mark.parametrize('source,target',list(itertools.product([np.float32,np.float64],repeat=2)))
def test_single_element_scan_has_constant_functional_jacobian(shape,source,target):
    x = tensor(np.full(shape,1.125,dtype=source),True)
    v = torch.ones_like(x)
    f = lambda a:a.cumprod(-1,dtype=target).sum()
    _, h = torch.autograd.functional.hvp(f,x,v)
    np.testing.assert_array_equal(h.numpy(),np.zeros(shape,dtype=source))
    with pytest.raises(RuntimeError):
        torch.autograd.functional.hvp(f,x,v,strict=True)
    p, t = torch.autograd.functional.jvp(lambda a:a.cumprod(-1,dtype=target),x,v)
    assert_values(t,np.ones(shape,dtype=target))


@pytest.mark.parametrize('values',[[np.inf,0.,1.],[0.,np.inf,1.],[0.,np.nan,1.]])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_nonfinite_zero_hvp_keeps_native_nan_result(values,dtype):
    x = tensor(np.array(values,dtype=dtype),True)
    _, h = torch.autograd.functional.hvp(lambda a:a.cumprod(0).square().sum(),x,torch.ones_like(x))
    assert np.isnan(h.numpy()).all()


@pytest.mark.parametrize('shape',[(),(1,),(65,)])
def test_dtype_overflow_is_independent_of_numpy_error_policy(shape):
    x = tensor(np.full(shape,1e300,dtype=np.float64))
    with np.errstate(all='raise'):
        y = x.cumprod(0,dtype=torch.float32)
    assert np.isposinf(y.numpy()).all()


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('length',[63,64,65,32767,32768,32769,65535,65536,65537])
@pytest.mark.parametrize('layout',['C','strided','readonly_broadcast'])
@pytest.mark.parametrize('rows',[1,3])
def test_single_working_array_and_block_boundaries_against_geometric_prefix(dtype,length,layout,rows):
    step = 1/1024
    row = np.resize(np.array([1+step,1-step],dtype=dtype),length)
    raw = np.broadcast_to(row,(rows,length)).copy()
    if layout == 'strided':
        storage = np.empty((rows,2*length),dtype=dtype)
        storage[:,::2] = raw
        raw = storage[:,::2]
    if layout == 'readonly_broadcast':
        raw = np.broadcast_to(row,(rows,length))
    before = raw.copy()
    y = tensor(raw).cumprod(1)
    # Pairing exact dyadic factors gives a closed-form geometric sequence,
    # independent of accumulation implementation and temporary-array routing.
    powers = (np.arange(length)+1)//2
    expected = (1-step*step)**powers
    expected[::2] *= 1+step
    tolerance = 1e-3 if dtype == np.float16 else 3e-6 if dtype == np.float32 else 1e-10
    assert_values(y,np.broadcast_to(expected,raw.shape),rtol=tolerance,atol=0)
    assert y.is_contiguous() and not np.shares_memory(y._data,raw)
    np.testing.assert_array_equal(raw,before)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_zero_safe_hvp_across_short_scan_boundary(dtype):
    length, second = 66, 33
    raw = np.ones(length,dtype=dtype)
    raw[[0,second]] = 0
    x = tensor(raw,True)
    value, h = torch.autograd.functional.hvp(lambda a:a.cumprod(0).sum(),x,torch.ones_like(x),strict=True)
    # Before the second zero, H[0,k] counts prefixes k..second-1.
    # Afterwards only H[0,second] survives, with length-second prefixes.
    expected = np.zeros(length,dtype=dtype)
    expected[1:second] = np.arange(second-1,0,-1)
    expected[second] = length-second
    expected[0] = second*(second-1)//2+length-second
    assert value.item() == 0
    np.testing.assert_array_equal(h.numpy(),expected)
