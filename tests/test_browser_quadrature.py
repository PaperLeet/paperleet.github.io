"""Native-independent value and analytical-gradient checks for new browser APIs."""
import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from paperleet.browser import numpy_autograd as wasm


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(3,3),(2,3,3),(2,1,3,3),(0,3,3),(2,0,0)])
def test_inverse_values_and_batched_vjp_match_native(dtype,shape):
    rng=np.random.RandomState(23)
    data=rng.randn(*shape).astype(dtype)+np.eye(shape[-1],dtype=dtype)*4
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    actual,expected=wasm.inv(a),torch.linalg.inv(b)
    weights=rng.randn(*shape).astype(dtype)
    actual.backward(wasm.Tensor(weights));expected.backward(torch.tensor(weights))
    for got,want in [(actual,expected.detach()),(a.grad,b.grad)]:
        np.testing.assert_allclose(got.numpy(),want.numpy(),atol=2e-6 if dtype==np.float32 else 1e-12,rtol=2e-6)


def test_inverse_singular_shape_and_unsupported_complex():
    for data in ([[1.,2.],[2.,4.]],np.zeros((2,3)),np.zeros(3)):
        with pytest.raises(RuntimeError): wasm.inv(wasm.Tensor(data))
    with pytest.raises(wasm.BrowserUnsupported): wasm.inv(wasm.Tensor(np.eye(2,dtype=np.complex128)))
    with pytest.raises(wasm.BrowserUnsupported): wasm.inv(wasm.Tensor(np.eye(2)),out=wasm.Tensor(np.eye(2)))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_erfinv_native_values_gradients_endpoints_and_extreme_tail(dtype):
    tail=np.nextafter(dtype(1),dtype(0))
    data=np.array([-1.1,-1.,-tail,-.97,-.4,-0.,0.,.4,.97,tail,1.,1.1,np.nan],dtype=dtype)
    small=np.array([-np.finfo(dtype).tiny,-1e-20,1e-20,np.finfo(dtype).tiny],dtype=dtype)
    data=np.concatenate((data,small))
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    actual,expected=a.erfinv(),b.erfinv()
    actual.sum().backward();expected.sum().backward()
    tol=4e-6 if dtype==np.float32 else 5e-14
    # Native CPU's erf-based Newton correction loses tail accuracy. Check that
    # tail independently with erfc, rather than reproduce its rounding error.
    ordinary=np.abs(data)!=tail
    for got,want in [(actual,expected.detach()),(a.grad,b.grad)]:
        np.testing.assert_allclose(got.numpy()[ordinary],want.numpy()[ordinary],atol=tol,rtol=tol,equal_nan=True)
    tail_value=abs(float(actual.numpy()[9]))
    assert math.isclose(math.erfc(tail_value),1-float(tail),rel_tol=2e-6 if dtype==np.float32 else 2e-14)
    np.testing.assert_allclose(actual.numpy()[-4:],small*(math.sqrt(math.pi)/2),atol=0,rtol=3e-7 if dtype==np.float32 else 3e-15)
    assert np.signbit(actual.numpy()[5]) and not np.signbit(actual.numpy()[6])
    finite=(np.abs(data)<1)
    np.testing.assert_allclose([math.erf(float(x)) for x in actual.numpy()[finite]],data[finite],atol=2e-7 if dtype==np.float32 else 2e-15)
    assert wasm.Tensor(np.empty((2,0),dtype=dtype)).erfinv().shape==(2,0)
    assert wasm.Tensor(dtype(.3)).erfinv().shape==()


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape,dim',[((),0),((2,0),-1),((3,4),0),((4,3),-1)])
@pytest.mark.parametrize('descending',[False,True])
def test_sort_stable_values_indices_and_vjp_match_native(dtype,shape,dim,descending):
    data=np.array([2.,-1.,2.,np.nan,0.,0.,np.inf,-np.inf,2.,1.,-1.,3.],dtype=dtype)[:math.prod(shape)].reshape(shape)
    if shape: data=data.T.copy().T  # Retain noncontiguous multidimensional strides.
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    actual,expected=a.sort(dim=dim,descending=descending,stable=True),b.sort(dim=dim,descending=descending,stable=True)
    assert actual.indices.dtype==np.dtype('int64') and not actual.indices.requires_grad
    np.testing.assert_equal(actual.values.numpy(),expected.values.detach().numpy())
    np.testing.assert_array_equal(actual.indices.numpy(),expected.indices.numpy())
    weights=np.arange(math.prod(shape),dtype=dtype).reshape(shape)+1
    actual.values.backward(wasm.Tensor(weights));expected.values.backward(torch.tensor(weights))
    np.testing.assert_array_equal(a.grad.numpy(),b.grad.numpy())


def test_default_unstable_sort_vjp_matches_native_away_from_ties():
    data=np.random.RandomState(53).randn(4,9).T
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    actual,expected=a.sort(0),b.sort(0)
    weights=np.arange(data.size).reshape(data.shape)*.3
    actual.values.backward(wasm.Tensor(weights));expected.values.backward(torch.tensor(weights))
    np.testing.assert_array_equal(actual.indices.numpy(),expected.indices.numpy())
    np.testing.assert_array_equal(a.grad.numpy(),b.grad.numpy())


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape,dim,xshape', [((2,3,5),-1,(5,)),((2,3,5),1,(3,)),((2,3,5),-1,(1,3,5)),((2,3,5),0,(2,1,1)),((3,0),1,(0,)),((3,1),1,(1,))])
def test_irregular_trapezoid_values_and_both_input_gradients(dtype,shape,dim,xshape):
    rng=np.random.RandomState(37)
    y=rng.randn(*shape).astype(dtype);x=rng.randn(*xshape).astype(dtype)
    a,b=wasm.Tensor(y,requires_grad=True),torch.tensor(y,requires_grad=True)
    c,d=wasm.Tensor(x,requires_grad=True),torch.tensor(x,requires_grad=True)
    actual,expected=wasm.trapezoid(a,c,dim=dim),torch.trapezoid(b,d,dim=dim)
    if not shape[dim]:
        assert not actual.requires_grad and not expected.requires_grad
        np.testing.assert_array_equal(actual.numpy(),expected.numpy())
        return
    weights=rng.randn(*actual.shape).astype(dtype)
    actual.backward(wasm.Tensor(weights));expected.backward(torch.tensor(weights))
    for got,want in [(actual,expected.detach()),(a.grad,b.grad),(c.grad,d.grad)]:
        np.testing.assert_allclose(got.numpy(),want.numpy(),atol=1e-6,rtol=2e-6)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('dx',[None,-.7,0.,2.])
def test_regular_trapezoid_on_strided_data(dtype,dx):
    data=np.arange(24,dtype=dtype).reshape(3,8)[:,::2]
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    kwargs={} if dx is None else {'dx':dx}
    actual,expected=wasm.trapezoid(a,**kwargs),torch.trapezoid(b,**kwargs)
    actual.sum().backward();expected.sum().backward()
    np.testing.assert_allclose(actual.numpy(),expected.detach().numpy(),atol=1e-6)
    np.testing.assert_allclose(a.grad.numpy(),b.grad.numpy(),atol=1e-6)


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.complex64,np.complex128])
@pytest.mark.parametrize('otherdtype',[np.float32,np.float64,np.complex64,np.complex128])
@pytest.mark.parametrize('shapes',[((2,),()),((),(2,)),((2,),(2,))])
def test_frozen_complex_arithmetic_retains_imaginary_parts_and_native_promotion(dtype,otherdtype,shapes):
    def values(kind,shape): return np.full(shape,1.2+.7j if np.dtype(kind).kind=='c' else .3,dtype=kind)
    data,other=values(dtype,shapes[0]),values(otherdtype,shapes[1])
    a,b=wasm.Tensor(data),wasm.Tensor(other)
    aa,bb=torch.tensor(data),torch.tensor(other)
    for actual,expected in [(a+b,aa+bb),(a-b,aa-bb),(a*b,aa*bb),(a/b,aa/bb)]:
        assert actual.dtype==expected.numpy().dtype
        np.testing.assert_allclose(actual.numpy(),expected.numpy(),atol=2e-7,rtol=2e-6)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_complex_characteristic_function_and_quadrature_have_independent_native_oracle(dtype):
    phase=np.arange(24,dtype=dtype).reshape(2,3,4)*.13
    nodes=np.array([-.8,-.1,.2,1.7],dtype=dtype)
    actual=(1j*wasm.Tensor(phase)).exp().mean(0)
    expected=(1j*torch.tensor(phase)).exp().mean(0)
    np.testing.assert_allclose(actual.numpy(),expected.numpy(),atol=2e-7)
    a,b=wasm.trapezoid(actual,wasm.Tensor(nodes)),torch.trapezoid(expected,torch.tensor(nodes))
    np.testing.assert_allclose(a.numpy(),b.numpy(),atol=2e-7)
    np.testing.assert_allclose(actual.abs().square().numpy(),expected.abs().square().numpy(),atol=2e-7)
    np.testing.assert_allclose((2/actual).numpy(),(2/expected).numpy(),atol=2e-7)


def test_complex_autograd_and_unsupported_options_explicitly_fall_back():
    real=wasm.Tensor([.3,.6],requires_grad=True)
    for operation in (lambda:1j*real,lambda:real.to(dtype=np.complex128),lambda:real*wasm.Tensor([1j,2j])):
        with pytest.raises(wasm.BrowserUnsupported,match='Complex autograd'):operation()
    with pytest.raises(wasm.BrowserUnsupported):wasm.Tensor([1j],requires_grad=True)
    with pytest.raises(wasm.BrowserUnsupported):wasm.Tensor([1j]).requires_grad_()
    with pytest.raises(wasm.BrowserUnsupported):wasm.Tensor([1j]).float()
    with wasm.no_grad(): assert (1j*real).dtype==np.dtype('complex128')
    with pytest.raises(wasm.BrowserUnsupported):wasm.Tensor([1]).erfinv()
    with pytest.raises(wasm.BrowserUnsupported):real.sort(out=())
    with pytest.raises(IndexError):real.sort(dim=2)
    with pytest.raises(TypeError):wasm.trapezoid(real,real,dx=1.)
    with pytest.raises(RuntimeError):wasm.trapezoid(real,wasm.Tensor([0.]))
    with pytest.raises(wasm.BrowserUnsupported):wasm.trapezoid(wasm.Tensor([1,2]))
    z=wasm.Tensor([1j,1+2j])
    for operation in (lambda:z.var(),lambda:z.std(),lambda:z.sigmoid(),lambda:z.softmax(0),lambda:z.log_softmax(0),lambda:z.max(),lambda:z.argmax(),lambda:z.amax(),lambda:z.relu(),lambda:z.clamp(0),lambda:z<1,lambda:real<1j,lambda:wasm.silu(z),lambda:wasm.elementwise_extreme(z,z)):
        with pytest.raises(wasm.BrowserUnsupported):operation()


def test_installed_exports_preserve_linspace_leaf_gradients_and_complex_values():
    script='''
import numpy as np
from paperleet.browser import numpy_autograd as backend
torch=backend.install()
for dtype in (torch.float32,torch.float64):
    x=torch.linspace(-.4,.8,7,dtype=dtype,requires_grad=True)
    assert x.dtype==dtype and x.requires_grad
    x.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(),np.ones(7))
    np.testing.assert_allclose(torch.erfinv(x.detach()).numpy(),x.detach().erfinv().numpy())
    values,indices=torch.sort(x.detach(),descending=True)
    np.testing.assert_array_equal(indices.numpy(),np.arange(6,-1,-1))
    np.testing.assert_allclose(torch.linalg.inv(torch.eye(2,dtype=dtype)).numpy(),np.eye(2))
z=torch.tensor([1j,1+2j]); assert z.dtype==torch.complex64
assert torch.trapezoid(z).item()==.5+1.5j
assert torch.tensor([1j],dtype=torch.complex128).dtype==torch.complex128
'''
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
