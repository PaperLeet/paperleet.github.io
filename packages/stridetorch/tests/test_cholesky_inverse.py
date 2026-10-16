"""Independent inverse-factor algebra, ordered tangent and resumed SPD training."""
import contextlib
import gc
import io
import weakref
import numpy as np
import pytest
from stridetorch import torch, UnsupportedOperation


def close(a,b,dtype='float64'):
    np.testing.assert_allclose(a,b,rtol=8e-5 if dtype=='float32' else 8e-9,
                               atol=6e-6 if dtype=='float32' else 8e-11)


def inverse2(a):
    det=a[...,0,0]*a[...,1,1]-a[...,0,1]*a[...,1,0];x=np.empty_like(a)
    x[...,0,0]=a[...,1,1];x[...,0,1]=-a[...,0,1];x[...,1,0]=-a[...,1,0];x[...,1,1]=a[...,0,0]
    return x/det[...,None,None]


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[False,True])
def test_closed_form_batched_values_and_full_factor_adjoint(dtype,upper):
    f=np.array([[[2.,0.],[.4,3.]],[[3.,0.],[-.2,2.]]],dtype=dtype)
    if upper:f=f.swapaxes(-1,-2)
    tr=lambda a:a.swapaxes(-1,-2);x=inverse2(tr(f)@f if upper else f@tr(f))
    full=f.copy();full[...,1,0] = .3 if upper else full[...,1,0];full[...,0,1] = .3 if not upper else full[...,0,1]
    F=torch.tensor(full,requires_grad=True);actual=torch.cholesky_inverse(F,upper);close(actual.numpy(),x,dtype)
    seed=np.arange(8,dtype=dtype).reshape(2,2,2)/5+.2;common=x@(seed+tr(seed))@x
    actual.backward(torch.tensor(seed));close(F.grad.numpy(),-(full@common if upper else common@full),dtype)
    assert actual.stride()[-2:]==(1,2)


@pytest.mark.parametrize('upper',[False,True])
def test_noncommuting_ordered_jvp_fixture(upper):
    f=np.array([[2.,0.],[.4,3.]]);f=f.T.copy() if upper else f;d=np.array([[.1,.3],[-.2,.4]])
    expected=np.array([[-.02442518518518518,.005076],[.005076,-.03390814814814815]]) if upper else np.array([[-.024425185185185185,-.008812888888888886],[-.008812888888888886,-.022797037037037036]])
    _,t=torch.func.jvp(lambda f:torch.cholesky_inverse(f,upper),(torch.tensor(f),),(torch.tensor(d),));close(t.numpy(),expected)


@pytest.mark.parametrize('upper',[False,True])
def test_spd_chain_ordered_tangent_reverse_finite_differences(upper):
    m=np.array([[.8,.1],[-.2,.6]]);dm=np.array([[.2,.3],[-.1,.4]])
    def scalar(m,dm):
        a=m@m.T+np.eye(2);da=dm@m.T+m@dm.T;l=np.linalg.cholesky(a);li=inverse2(l)
        c=li@da@li.T;p=np.tril(c);p[np.diag_indices(2)]*=.5;dl=l@p
        f=l.T if upper else l;df=dl.T if upper else dl;x=inverse2(a)
        partial=-x@(df.T if upper else df)@x@(f if upper else f.T);t=partial+partial.T
        return .17*(x*x).sum()+.23*(t*t).sum()
    args=tuple(torch.tensor(a,requires_grad=True) for a in (m,dm))
    x,t=torch.func.jvp(lambda m:torch.cholesky_inverse(torch.linalg.cholesky(m@m.T+torch.eye(2,dtype=torch.float64),upper=upper),upper),args[:1],args[1:])
    gradients=torch.autograd.grad(.17*x.square().sum()+.23*t.square().sum(),args)
    for which,g in enumerate(gradients):
        expected=np.zeros((2,2));eps=1e-5
        for index in np.ndindex(2,2):
            plus=[m.copy(),dm.copy()];minus=[m.copy(),dm.copy()];plus[which][index]+=eps;minus[which][index]-=eps
            expected[index]=(scalar(*plus)-scalar(*minus))/(2*eps)
        close(g.numpy(),expected)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('disabled',[False,True])
def test_diagonal_jvp_closed_form_and_no_grad(dtype,upper,disabled):
    f=torch.eye(3,dtype=getattr(torch,dtype))*2;d=torch.arange(9,dtype=getattr(torch,dtype)).reshape(3,3)/10
    f.requires_grad_(disabled);d.requires_grad_(disabled)
    with torch.no_grad() if disabled else contextlib.nullcontext():
        x,t=torch.func.jvp(lambda f:torch.cholesky_inverse(f,upper),(f,),(d,))
    close(x.numpy(),np.eye(3)/4,dtype);close(t.numpy(),-(d.numpy()+d.numpy().T)/8,dtype)
    assert not x.requires_grad and not t.requires_grad


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[False,True])
def test_spd_sgd_and_zip_resume_against_array_updates(dtype,upper):
    m=np.array([[.8,.1],[-.2,.6]],dtype=dtype);M=torch.nn.Parameter(torch.tensor(m));opt=torch.optim.SGD([M],lr=.03)
    for step in range(6):
        x=inverse2(m@m.T+np.eye(2,dtype=dtype));ga=-x@(2*x/x.size)@x;dm=(ga+ga.T)@m;m-=.03*dm
        opt.zero_grad();f=torch.linalg.cholesky(M@M.T+torch.eye(2,dtype=getattr(torch,dtype)),upper=upper);f.cholesky_inverse(upper).square().mean().backward();opt.step();close(M.numpy(),m,dtype)
        if step==2:
            stream=io.BytesIO();torch.save({'m':M.detach(),'optimizer':opt.state_dict()},stream);stream.seek(0);state=torch.load(stream)
            M=torch.nn.Parameter(state['m']);opt=torch.optim.SGD([M],lr=.9);opt.load_state_dict(state['optimizer'])


@pytest.mark.parametrize('upper',[False,True])
def test_saved_array_ownership_and_alias_guard(upper):
    root=torch.tensor([[2.,.2],[.4,3.]],requires_grad=True);f=root*2;fr=weakref.ref(f);fa=weakref.ref(f._data)
    x=torch.cholesky_inverse(f,upper);xa=weakref.ref(x._data);loss=x.sum();del f,x;gc.collect()
    assert fr() is None and fa() is not None and xa() is not None
    loss.backward();gc.collect();assert fa() is xa() is None
    f=root*2;x=torch.cholesky_inverse(f,upper)
    with torch.no_grad():f.transpose(-1,-2).detach().add_(.1)
    with pytest.raises(RuntimeError):x.sum().backward()


@pytest.mark.parametrize('dtype',['float32','float64','complex64','complex128'])
@pytest.mark.parametrize('upper',[False,True])
def test_large_constructed_inverse_identity(dtype,upper):
    n=65;rng=np.random.default_rng(361);f=np.tril((rng.normal(size=(2,n,n))*.02).astype(dtype));f[...,np.arange(n),np.arange(n)]=2
    if np.dtype(dtype).kind=='c':f+=np.tril(np.full_like(f,.01j),-1)
    if upper:f=f.conj().swapaxes(-1,-2)
    h=f.conj().swapaxes(-1,-2);a=h@f if upper else f@h
    x=torch.cholesky_inverse(torch.tensor(f),upper).numpy();close(a@x,np.broadcast_to(np.eye(n),a.shape),'float32' if np.dtype(dtype).itemsize<8 or dtype=='complex64' else 'float64')


@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('shape',[(0,0),(0,3,3),(2,0,0)])
def test_empty_primal_and_tangent_strides(upper,shape):
    f=torch.tensor(np.empty(shape));d=torch.zeros_like(f);x,t=torch.func.jvp(lambda f:torch.cholesky_inverse(f,upper),(f,),(d,))
    assert x.shape==t.shape==shape and x.numel()==t.numel()==0
    assert x.stride()[-2:]==(1,max(shape[-1],1));assert t.stride()[-2:]==(max(shape[-1],1),1)


def test_public_metadata_singular_and_invalid_complex_factor():
    assert torch._browser_capabilities['torch.cholesky_inverse'].forward
    with pytest.raises(RuntimeError,match='diagonal element 2'):torch.cholesky_inverse(torch.tensor([[2.,0.],[.3,0.]]))
    with pytest.raises(UnsupportedOperation,match='real diagonals'):torch.cholesky_inverse(torch.tensor([[2.+.3j]],dtype=torch.complex128))
    with pytest.raises(UnsupportedOperation):torch.cholesky_inverse(torch.eye(2),out=torch.empty(2,2))
