"""Native inverse-factor values, ordered JVP, full reverse and saved values."""
import numpy as np
import pytest
import torch as native
from torchwasm import torch as lite, UnsupportedOperation


def close(a, b, dtype):
    single = np.dtype(dtype) in (np.dtype('float32'), np.dtype('complex64'))
    np.testing.assert_allclose(a, b, rtol=8e-5 if single else 8e-11,
                               atol=6e-6 if single else 8e-12, equal_nan=True)


def factor(shape, dtype):
    rng = np.random.default_rng(36)
    n = shape[-1]
    a = (rng.normal(size=shape[:-2] + (2*n, 2*n))*.1).astype(dtype)[..., ::2, ::2]
    if np.dtype(dtype).kind == 'c':a = a + np.asarray(.04j, dtype=dtype)
    a[..., np.arange(n), np.arange(n)] = 2
    return a


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('upper', [False, True])
@pytest.mark.parametrize('shape', [(1,1), (2,2), (2,1,3,3), (0,3,3), (2,0,0), (33,33), (65,65)])
def test_strided_factor_full_vjp_and_retained_seeds(dtype, upper, shape):
    raw = factor(shape, dtype); reports = []
    for lib in (lite, native):
        f = lib.from_numpy(raw.copy()).transpose(-1,-2).contiguous().transpose(-1,-2).requires_grad_()
        x = lib.cholesky_inverse(input=f, upper=upper)
        gradients = []
        for scale in (.3, -1.2):
            seed = np.linspace(.2, .9, x.numel(), dtype=dtype).reshape(shape)*scale
            gradients.append(lib.autograd.grad(x, f, lib.tensor(seed), retain_graph=True)[0].numpy())
        reports.append((x.detach().numpy(), x.stride(), gradients))
    a, b = reports; close(a[0], b[0], dtype); assert a[1] == b[1]
    for x, y in zip(a[2], b[2]):close(x, y, dtype)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('upper', [False, True])
@pytest.mark.parametrize('flags', [(True,True), (True,False), (False,True), (False,False)])
@pytest.mark.parametrize('shape', [(2,2), (2,1,3,3), (0,3,3), (2,0,0)])
def test_ordered_native_jvp_and_reverse_both_channels(dtype, upper, flags, shape):
    raw = factor(shape, dtype); direction = np.full(shape, .13, dtype=dtype); reports = []
    for lib in (lite, native):
        args = tuple(lib.tensor(v, requires_grad=flag) for v,flag in zip((raw,direction), flags))
        x,t = lib.func.jvp(lambda f:f.cholesky_inverse(upper), args[:1], args[1:])
        active = tuple(v for v in args if v.requires_grad)
        gradients = lib.autograd.grad(.17*x.square().sum()+.23*t.square().sum(), active) if active else ()
        reports.append((x.detach().numpy(),t.detach().numpy(),x.stride(),t.stride(),[v.numpy() for v in gradients]))
    a,b=reports;close(a[0],b[0],dtype);close(a[1],b[1],dtype);assert a[2:4]==b[2:4]
    for x,y in zip(a[4],b[4]):close(x,y,dtype)


@pytest.mark.parametrize('upper', [False,True])
@pytest.mark.parametrize('flags', [(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('mutated', ['factor','direction','value','tangent'])
def test_jvp_mutation_guards_and_allowed_source_edits(upper,flags,mutated):
    reports=[]
    for lib in (lite,native):
        f=lib.tensor([[2.,.2],[.4,3.]],dtype=lib.float64,requires_grad=flags[0])
        d=lib.full_like(f,.13,requires_grad=flags[1]);x,t=lib.func.jvp(lambda f:lib.cholesky_inverse(f,upper),(f,),(d,))
        with lib.no_grad():{'factor':f,'direction':d,'value':x,'tangent':t}[mutated].detach().add_(.1)
        try:
            g=lib.autograd.grad(t.sum(),tuple(v for v in (f,d) if v.requires_grad));reports.append([v.numpy() for v in g])
        except RuntimeError:reports.append(None)
    if reports[1] is None:assert reports[0] is None
    else:
        assert reports[0] is not None
        for x,y in zip(*reports):close(x,y,'float64')


@pytest.mark.parametrize('dtype', ['complex64','complex128'])
@pytest.mark.parametrize('upper', [False,True])
@pytest.mark.parametrize('shape', [(1,1),(2,2),(2,3,3),(0,3,3),(2,0,0),(65,65)])
@pytest.mark.parametrize('nonreal_diagonal', [False,True])
def test_complex_inference_and_invalid_nonreal_diagonal_boundary(dtype,upper,shape,nonreal_diagonal):
    a=factor(shape,dtype)
    if nonreal_diagonal:a[...,np.arange(shape[-1]),np.arange(shape[-1])]+=np.asarray(.3j,dtype=dtype)
    if nonreal_diagonal and a.size:
        # Non-real diagonals are not Cholesky factors. Native POTRI's output on
        # these invalid factors depends on the vendor's blocking algorithm.
        with pytest.raises(UnsupportedOperation,match='real diagonals'):lite.tensor(a).cholesky_inverse(upper)
        return
    actual=lite.tensor(a).cholesky_inverse(upper);expected=native.tensor(a).cholesky_inverse(upper)
    close(actual.numpy(),expected.numpy(),dtype);assert actual.stride()==expected.stride()


@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('entry',[0.,np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('n',[1,2,3])
def test_nonfinite_and_singular_diagonal_native_behavior(upper,entry,n):
    a=np.eye(n)*2;a[0,0]=entry
    if not entry:
        for lib in (lite,native):
            with pytest.raises(RuntimeError,match='diagonal element 1'):lib.cholesky_inverse(lib.tensor(a),upper)
    else:close(lite.cholesky_inverse(lite.tensor(a),upper).numpy(),native.cholesky_inverse(native.tensor(a),upper).numpy(),'float64')


@pytest.mark.parametrize('upper',[False,True])
def test_unused_half_nan_negative_diagonal_and_primal_mutation_guards(upper):
    a=np.array([[-2.,0.],[.4,-3.]])
    if upper:a=a.T.copy();a[1,0]=np.nan
    else:a[0,1]=np.nan
    close(lite.cholesky_inverse(lite.tensor(a),upper).numpy(),native.cholesky_inverse(native.tensor(a),upper).numpy(),'float64')
    for mutated in ('input','output'):
        for lib in (lite,native):
            f=lib.tensor([[2.,.2],[.4,3.]],requires_grad=True);x=lib.cholesky_inverse(f,upper)
            with lib.no_grad():{'input':f,'output':x}[mutated].detach().add_(.1)
            with pytest.raises(RuntimeError):x.sum().backward()


def test_invalid_inputs_and_explicit_boundaries():
    for lib in (lite,native):
        for shape in ((),(2,),(2,3)):
            with pytest.raises(RuntimeError):lib.cholesky_inverse(lib.ones(shape))
        with pytest.raises(TypeError):lib.cholesky_inverse([[1.]])
        with pytest.raises(TypeError):lib.cholesky_inverse(lib.eye(2),upper=1)
    for dtype in (lite.int64,lite.float16):
        with pytest.raises(UnsupportedOperation):lite.cholesky_inverse(lite.eye(2,dtype=dtype))
    with pytest.raises(UnsupportedOperation):lite.cholesky_inverse(lite.eye(2),out=lite.empty(2,2))
    with pytest.raises(UnsupportedOperation):lite.cholesky_inverse(lite.eye(2,dtype=lite.complex128,requires_grad=True))
    assert lite._browser_capabilities['torch.cholesky_inverse'].forward
