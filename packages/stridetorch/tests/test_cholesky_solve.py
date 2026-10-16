"""Independent factor algebra, SPD training, tangent finite differences and lifetimes."""
import gc
import io
import weakref
import numpy as np
import pytest
from stridetorch import torch, UnsupportedOperation


def close(a,b,dtype='float64'):
    np.testing.assert_allclose(a,b,rtol=6e-5 if dtype=='float32' else 6e-10,atol=5e-6 if dtype=='float32' else 2e-11)


def inverse2(a):
    det=a[...,0,0]*a[...,1,1]-a[...,0,1]*a[...,1,0];x=np.empty_like(a)
    x[...,0,0]=a[...,1,1];x[...,0,1]=-a[...,0,1];x[...,1,0]=-a[...,1,0];x[...,1,1]=a[...,0,0]
    return x/det[...,None,None]


def reduce(x,shape):
    while x.ndim>len(shape):x=x.sum(0)
    for axis,n in enumerate(shape):
        if n==1 and x.shape[axis]!=1:x=x.sum(axis,keepdims=True)
    return x.reshape(shape)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[False,True])
def test_closed_form_values_full_factor_vjp_and_jvp(dtype,upper):
    f=np.array([[[[2.,0.],[.4,3.]]],[[[3.,0.],[-.2,2.5]]]],dtype=dtype)
    if upper:f=f.swapaxes(-1,-2)
    b=np.arange(12,dtype=dtype).reshape(1,3,2,2)/7+.2;inv=inverse2(f.swapaxes(-1,-2)@f if upper else f@f.swapaxes(-1,-2));x=inv@b
    F=torch.tensor(f,requires_grad=True);B=torch.tensor(b,requires_grad=True);actual=B.cholesky_solve(F,upper);close(actual.numpy(),x,dtype)
    for scale in (.4,-1.3):
        seed=np.full(x.shape,scale,dtype=dtype);db=inv@seed;common=db@x.swapaxes(-1,-2)+x@db.swapaxes(-1,-2);df=-(f@common if upper else common@f)
        F.grad=B.grad=None;actual.backward(torch.tensor(seed),retain_graph=True);close(F.grad.numpy(),reduce(df,f.shape),dtype);close(B.grad.numpy(),reduce(db,b.shape),dtype)
    df=np.full_like(f,.13);db=np.full_like(b,.27);product=df.swapaxes(-1,-2)@f if upper else df@f.swapaxes(-1,-2);da=product+product.swapaxes(-1,-2)
    _,t=torch.func.jvp(lambda b,f:torch.cholesky_solve(b,f,upper),(B,F),(torch.tensor(db),torch.tensor(df)));close(t.numpy(),inv@(db-da@x),dtype)


@pytest.mark.parametrize('upper',[False,True])
def test_full_spd_chain_jvp_reverse_against_finite_differences(upper):
    raw=[np.array([[.8,.1],[-.2,.6]]),np.array([[.3,.7],[.2,.8]]),np.array([[.2,.3],[.7,.4]]),np.array([[.1,.3],[.4,.2]])]
    def scalar(m,b,dm,db):
        inv=inverse2(m@m.T+np.eye(2));x=inv@b;da=dm@m.T+m@dm.T;t=inv@(db-da@x)
        return .17*(x*x).sum()+.23*(t*t).sum()
    args=tuple(torch.tensor(v,requires_grad=True) for v in raw)
    def function(m,b):return torch.cholesky_solve(b,torch.linalg.cholesky(m@m.T+torch.eye(2,dtype=torch.float64),upper=upper),upper)
    x,t=torch.func.jvp(function,args[:2],args[2:]);gradients=torch.autograd.grad(.17*x.square().sum()+.23*t.square().sum(),args)
    for which,g in enumerate(gradients):
        expected=np.zeros_like(raw[which]);eps=1e-5
        for index in np.ndindex(expected.shape):
            plus=[a.copy() for a in raw];minus=[a.copy() for a in raw];plus[which][index]+=eps;minus[which][index]-=eps;expected[index]=(scalar(*plus)-scalar(*minus))/(2*eps)
        close(g.numpy(),expected)


@pytest.mark.parametrize('upper',[False,True])
def test_real_spd_training_and_zip_optimizer_resume(upper):
    m=np.array([[.8,.1],[-.2,.6]]);b=np.array([[.3,.7],[.2,.8]]);M=torch.nn.Parameter(torch.tensor(m));B=torch.nn.Parameter(torch.tensor(b));opt=torch.optim.SGD([M,B],lr=.03)
    for step in range(6):
        inv=inverse2(m@m.T+np.eye(2));x=inv@b;db=inv@(2*x/x.size);dm=-(db@x.T+x@db.T)@m;m-=.03*dm;b-=.03*db
        opt.zero_grad();factor=torch.linalg.cholesky(M@M.T+torch.eye(2,dtype=torch.float64),upper=upper);torch.cholesky_solve(B,factor,upper).square().mean().backward();opt.step();close(M.numpy(),m);close(B.numpy(),b)
        if step==2:
            stream=io.BytesIO();torch.save({'m':M.detach(),'b':B.detach(),'opt':opt.state_dict()},stream);stream.seek(0);state=torch.load(stream);M=torch.nn.Parameter(state['m']);B=torch.nn.Parameter(state['b']);opt=torch.optim.SGD([M,B],lr=.9);opt.load_state_dict(state['opt'])


@pytest.mark.parametrize('upper',[False,True])
def test_unused_rhs_array_released_but_factor_output_saved(upper):
    root_f=torch.tensor([[2.,0.],[.4,3.]],requires_grad=True);root_b=torch.ones(2,2,requires_grad=True);f=root_f*2;b=root_b*3
    fw,bw=weakref.ref(f),weakref.ref(b);fa,ba=weakref.ref(f._data),weakref.ref(b._data);x=torch.cholesky_solve(b,f,upper);xa=weakref.ref(x._data);loss=x.sum();del f,b,x;gc.collect()
    assert fw() is bw() is ba() is None and fa() is not None and xa() is not None
    loss.backward();gc.collect();assert fa() is xa() is None


@pytest.mark.parametrize('shape',[(0,2),(2,0,2),(0,3,2),(2,1,2)])
@pytest.mark.parametrize('upper',[False,True])
def test_empty_and_singleton_values_and_tangent_layout(shape,upper):
    n=shape[-2];f=torch.tensor(np.broadcast_to(np.eye(n),shape[:-2]+(n,n)).copy());b=torch.ones(shape,dtype=torch.float64)
    x,t=torch.func.jvp(lambda b,f:torch.cholesky_solve(b,f,upper),(b,f),(torch.ones_like(b),torch.zeros_like(f)));close(x.numpy(),b.numpy());close(t.numpy(),b.numpy());assert x.stride()==t.stride()
    assert x.stride()[-2:]==(1,max(n,1))


@pytest.mark.parametrize('dtype',['complex64','complex128'])
@pytest.mark.parametrize('upper',[False,True])
def test_constructed_hermitian_system_with_unused_nan(dtype,upper):
    f=np.array([[2.,0.],[.2+.3j,3.]],dtype=dtype)
    if upper:f=f.conj().T
    target=np.array([[.3+.2j,.1],[.4,.6-.3j]],dtype=dtype);a=f.conj().T@f if upper else f@f.conj().T;b=a@target
    f[1,0] = np.nan if upper else f[1,0];f[0,1] = np.nan if not upper else f[0,1]
    actual=torch.cholesky_solve(torch.tensor(b),torch.tensor(f),upper);close(actual.numpy(),target,'float32' if dtype=='complex64' else 'float64')


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('n',[65,96])
@pytest.mark.parametrize('width',[3,40])
def test_constructed_large_system_and_adjoint_seed(dtype,upper,n,width):
    rng=np.random.default_rng(330);f=np.tril((rng.normal(size=(n,n))*.02).astype(dtype));f[np.diag_indices(n)]=2
    if upper:f=f.T.copy()
    a=f.T@f if upper else f@f.T;target=(rng.normal(size=(2,n,width))*.1).astype(dtype);z=(rng.normal(size=target.shape)*.1).astype(dtype);b=a@target;seed=a@z
    F=torch.tensor(f,requires_grad=True);B=torch.tensor(b,requires_grad=True);x=torch.cholesky_solve(B,F,upper);close(x.numpy(),target,dtype);x.backward(torch.tensor(seed));close(B.grad.numpy(),z,dtype)
    common=z@target.swapaxes(-1,-2)+target@z.swapaxes(-1,-2);df=-(f@common if upper else common@f);close(F.grad.numpy(),df.sum(0),dtype)


def test_invalid_inputs_and_capability_metadata():
    b=torch.ones(2,2);f=torch.eye(2)
    for bad in (torch.ones(2),torch.ones(3,2)):
        with pytest.raises(RuntimeError):torch.cholesky_solve(bad,f)
    with pytest.raises(RuntimeError):torch.cholesky_solve(b,torch.ones(2,3))
    with pytest.raises(RuntimeError):torch.cholesky_solve(torch.ones(2,2,2),torch.ones(3,2,2))
    with pytest.raises(RuntimeError):torch.cholesky_solve(b.double(),f)
    with pytest.raises(TypeError):torch.cholesky_solve([[1.]],f)
    with pytest.raises(TypeError):torch.cholesky_solve(b,f,upper=1)
    with pytest.raises(UnsupportedOperation):torch.cholesky_solve(b,f,out=torch.empty(2,2))
    for dtype in (torch.int64,torch.float16):
        with pytest.raises(UnsupportedOperation):torch.cholesky_solve(b.to(dtype=dtype),f.to(dtype=dtype))
    assert torch._browser_capabilities['torch.cholesky_solve'].forward
