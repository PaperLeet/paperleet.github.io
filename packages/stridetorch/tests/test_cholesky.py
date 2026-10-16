"""Independent scalar formulas, positive-definite training and saved lifetimes."""
import gc
import io
import weakref
import numpy as np
import pytest
from stridetorch import torch, UnsupportedOperation


def close(a,b,dtype='float64'):
    np.testing.assert_allclose(a,b,rtol=7e-5 if dtype=='float32' else 5e-11,
                              atol=5e-6 if dtype=='float32' else 5e-12)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[False,True])
def test_scalar_closed_form_values_vjp_and_nonsymmetric_jvp(dtype,upper):
    data=np.array([[[4.,1.],[1.,3.]],[[3.,-.2],[-.2,2.]]],dtype=dtype)
    l=np.sqrt(data[...,0,0]);b=data[...,1,0]/l;d=np.sqrt(data[...,1,1]-b*b)
    lower=np.zeros_like(data);lower[...,0,0]=l;lower[...,1,0]=b;lower[...,1,1]=d
    a=torch.tensor(data,requires_grad=True);factor=torch.linalg.cholesky(a,upper=upper)
    close(factor.numpy(),lower.swapaxes(-1,-2) if upper else lower,dtype)
    for scale in (.3,-1.4):
        seed=np.arange(data.size,dtype=dtype).reshape(data.shape)*scale+.2
        g=seed.swapaxes(-1,-2) if upper else seed
        expected=np.zeros_like(data)
        expected[...,0,0]=g[...,0,0]/(2*l)-g[...,1,0]*b/(2*l*l)+g[...,1,1]*b*b/(2*l*l*d)
        expected[...,1,1]=g[...,1,1]/(2*d)
        expected[...,1,0]=expected[...,0,1]=(g[...,1,0]/l-g[...,1,1]*b/(d*l))*.5
        a.grad=None;factor.backward(torch.tensor(seed),retain_graph=True);close(a.grad.numpy(),expected,dtype)
    direction=np.array([[[.2,.3],[.7,.4]],[[-.1,.4],[.2,.5]]],dtype=dtype)
    _,t=torch.func.jvp(lambda v:torch.linalg.cholesky(v,upper=upper),(a,),(torch.tensor(direction),))
    dt=np.zeros_like(data);dt[...,0,0]=direction[...,0,0]/(2*l)
    dt[...,1,0]=(direction[...,1,0]-b*dt[...,0,0])/l
    dt[...,1,1]=(direction[...,1,1]-b/l*(direction[...,0,1]+direction[...,1,0])+b*b/(l*l)*direction[...,0,0])/(2*d)
    close(t.numpy(),dt.swapaxes(-1,-2) if upper else dt,dtype)


@pytest.mark.parametrize('upper',[False,True])
def test_spd_parameterization_jvp_reverse_finite_difference_and_checkpoint_resume(upper):
    initial=np.array([[.8,.1],[-.2,.6]])
    direction=np.array([[.2,.3],[.7,.4]])
    def scalar(m,d):
        a=m@m.T+np.eye(2);l=np.zeros((2,2));l[0,0]=np.sqrt(a[0,0]);l[1,0]=a[1,0]/l[0,0];l[1,1]=np.sqrt(a[1,1]-l[1,0]**2)
        da=d@m.T+m@d.T;t=np.zeros((2,2));t[0,0]=da[0,0]/(2*l[0,0]);t[1,0]=(da[1,0]-l[1,0]*t[0,0])/l[0,0];t[1,1]=(da[1,1]-2*l[1,0]*t[1,0])/(2*l[1,1])
        return .17*(l*l).sum()+.23*(t*t).sum()
    m,d=(torch.tensor(v,requires_grad=True) for v in (initial,direction))
    x,t=torch.func.jvp(lambda v:torch.linalg.cholesky(v@v.T+torch.eye(2,dtype=torch.float64),upper=upper),(m,),(d,))
    gm,gd=torch.autograd.grad(.17*x.square().sum()+.23*t.square().sum(),(m,d))
    eps=1e-5
    for which,gradient in enumerate((gm,gd)):
        expected=np.zeros((2,2))
        for index in np.ndindex(2,2):
            plus=[initial.copy(),direction.copy()];minus=[initial.copy(),direction.copy()];plus[which][index]+=eps;minus[which][index]-=eps
            expected[index]=(scalar(*plus)-scalar(*minus))/(2*eps)
        close(gradient.numpy(),expected)
    parameter=torch.nn.Parameter(torch.tensor(initial));opt=torch.optim.SGD([parameter],lr=.03);expected=initial.copy()
    # ||chol(M M^T + I)||_F^2 = ||M||_F^2 + 2, giving an independent
    # exact gradient and update rather than a second autodiff implementation.
    for step in range(6):
        opt.zero_grad();torch.linalg.cholesky(parameter@parameter.T+torch.eye(2,dtype=torch.float64),upper=upper).square().sum().backward();close(parameter.grad.numpy(),2*expected);opt.step();expected*=.94;close(parameter.numpy(),expected)
        if step==2:
            stream=io.BytesIO();torch.save({'m':parameter.detach(),'opt':opt.state_dict()},stream);stream.seek(0);state=torch.load(stream);parameter=torch.nn.Parameter(state['m']);opt=torch.optim.SGD([parameter],lr=.9);opt.load_state_dict(state['opt'])


@pytest.mark.parametrize('upper',[False,True])
def test_saved_factor_only_and_release_after_backward(upper):
    root=torch.tensor([[4.,1.],[1.,3.]],requires_grad=True);a=root*2
    array=weakref.ref(a._data);wrapper=weakref.ref(a)
    factor=torch.linalg.cholesky(a,upper=upper);saved=weakref.ref(factor._data)
    loss=factor.sum();del a,factor;gc.collect()
    assert array() is wrapper() is None and saved() is not None
    loss.backward();gc.collect();assert saved() is None


@pytest.mark.parametrize('dtype',['int64','float16','bool'])
def test_unsupported_input_dtypes(dtype):
    with pytest.raises(UnsupportedOperation):torch.linalg.cholesky(torch.tensor([[1,0],[0,1]],dtype=getattr(torch,dtype)))


@pytest.mark.parametrize('shape',[(2,),(2,3),(2,3,4)])
def test_invalid_matrix_dimensions(shape):
    with pytest.raises(RuntimeError,match='square'):torch.linalg.cholesky(torch.ones(shape))


def test_unsupported_options_and_discovery():
    a=torch.eye(2)
    with pytest.raises(TypeError):torch.linalg.cholesky([[1.]])
    with pytest.raises(TypeError):torch.linalg.cholesky(a,upper=1)
    with pytest.raises(UnsupportedOperation):torch.linalg.cholesky(a,out=torch.empty(2,2))
    assert 'torch.linalg.cholesky' in torch._browser_capabilities
