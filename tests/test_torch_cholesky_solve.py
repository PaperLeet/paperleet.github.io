"""Native Cholesky solve contracts, including its full-factor AD extension."""
import numpy as np
import pytest
import torch as native
from stridetorch import torch as lite, UnsupportedOperation

CONFIGS=[((),(),2,3),((2,1),(1,3),3,2),((2,),(),4,3),((),(2,),5,2),((2,),(2,),2,2),((),(),0,2),((),(),2,0),((0,),(1,),3,2),((),(),1,4),((2,),(2,),1,1)]


def arrays(dtype,fbatch,bbatch,n,k):
    rng=np.random.default_rng(33)
    f=(rng.normal(size=fbatch+(2*n,2*n))*.07).astype(dtype)[...,::2,::2]
    f[...,np.arange(n),np.arange(n)]+=2
    b=(rng.normal(size=bbatch+(2*n,2*k))*.3).astype(dtype)[...,::2,::2]
    return f,b


def close(a,b,dtype):
    single=np.dtype(dtype) in (np.dtype('float32'),np.dtype('complex64'))
    np.testing.assert_allclose(a,b,rtol=8e-5 if single else 3e-11,atol=5e-6 if single else 4e-12,equal_nan=True)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('active',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('config',CONFIGS)
def test_values_layout_broadcast_gradients_and_dynamic_seed(dtype,upper,active,config):
    f,b=arrays(dtype,*config);reports=[]
    for lib in (lite,native):
        F=lib.from_numpy(f).requires_grad_(active[1]);B=lib.from_numpy(b).requires_grad_(active[0]);x=lib.cholesky_solve(input=B,input2=F,upper=upper);gradients=[]
        for scale in (.3,-1.2):
            F.grad=B.grad=None;seed=np.linspace(.2,1.1,x.numel(),dtype=dtype).reshape(x.shape)*scale
            x.backward(lib.tensor(seed),retain_graph=True)
            gradients.append([None if a.grad is None else a.grad.numpy().copy() for a in (B,F)])
        reports.append((x.detach().numpy(),x.stride(),gradients))
    a,b=reports;close(a[0],b[0],dtype);assert a[1]==b[1]
    for actual,expected in zip(a[2],b[2]):
        for x,y in zip(actual,expected):
            if y is None:assert x is None
            else:close(x,y,dtype)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('active',[(0,1),(0,),(1,)])
@pytest.mark.parametrize('batched',[False,True])
def test_jvp_and_reverse_through_both_primals_and_directions(dtype,upper,active,batched):
    f,b=arrays(dtype,(2,1) if batched else (),(1,3) if batched else (),3 if batched else 2,2)
    df=np.full_like(f,.13);db=np.full_like(b,.27);reports=[]
    for lib in (lite,native):
        args=tuple(lib.tensor(v,requires_grad=True) for v in (b,f,db,df))
        def function(*values):
            inputs=list(args[:2])
            for index,value in zip(active,values):inputs[index]=value
            return lib.cholesky_solve(*inputs,upper)
        x,t=lib.func.jvp(function,tuple(args[i] for i in active),tuple(args[2+i] for i in active))
        gradients=lib.autograd.grad(.17*x.square().sum()+.23*t.square().sum(),args,allow_unused=True)
        reports.append((x.detach().numpy(),t.detach().numpy(),x.stride(),t.stride(),[None if g is None else g.numpy() for g in gradients]))
    a,b=reports;close(a[0],b[0],dtype);close(a[1],b[1],dtype);assert a[2:4]==b[2:4]
    for x,y in zip(a[4],b[4]):
        if y is None:assert x is None
        else:close(x,y,dtype)


@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('active',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('mutated',['b','factor','out'])
def test_all_three_native_mutation_guards(upper,active,mutated):
    for lib in (lite,native):
        f=lib.tensor([[2.,.2],[.4,3.]],requires_grad=active[1]);b=lib.ones(2,2,requires_grad=active[0]);x=lib.cholesky_solve(b,f,upper)
        with lib.no_grad():{'b':b,'factor':f,'out':x}[mutated].detach().add_(.1)
        with pytest.raises(RuntimeError):x.sum().backward()


@pytest.mark.parametrize('dtype',['complex64','complex128'])
@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('config',[((),(),2,3),((2,1),(1,3),3,2),((),(),0,2),((0,),(1,),3,2),((),(),1,4)])
def test_complex_inference_and_tensor_method(dtype,upper,config):
    f,b=arrays(dtype,*config);f=f+np.asarray(.03j,dtype=dtype);b=b-np.asarray(.1j,dtype=dtype)
    x=lite.tensor(b).cholesky_solve(lite.tensor(f),upper);y=native.tensor(b).cholesky_solve(native.tensor(f),upper)
    close(x.numpy(),y.numpy(),dtype);assert x.stride()==y.stride()


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('n',[65,96])
@pytest.mark.parametrize('width',[3,40,96])
def test_blocked_factor_solves_and_gradients(dtype,upper,n,width):
    f,b=arrays(dtype,(2,),(),n,width);reports=[]
    for lib in (lite,native):
        F=lib.tensor(f,requires_grad=True);B=lib.tensor(b,requires_grad=True);x=lib.cholesky_solve(B,F,upper);x.square().mean().backward();reports.append((x.detach().numpy(),B.grad.numpy(),F.grad.numpy()))
    for x,y in zip(*reports):close(x,y,dtype)


@pytest.mark.parametrize('upper',[False,True])
def test_unused_nan_is_ignored_only_in_primal_solves(upper):
    f=np.array([[2.,.2],[.4,3.]])
    f[1,0] = np.nan if upper else f[1,0]
    f[0,1] = np.nan if not upper else f[0,1]
    reports=[]
    for lib in (lite,native):
        F=lib.tensor(f,requires_grad=True);B=lib.ones(2,2,dtype=lib.float64,requires_grad=True);x=lib.cholesky_solve(B,F,upper);x.sum().backward();reports.append((x.detach().numpy(),B.grad.numpy(),F.grad.numpy()))
    for a,b in zip(*reports):close(a,b,'float64')


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('active',[(0,1),(0,),(1,)])
@pytest.mark.parametrize('triangular',[False,True])
def test_large_narrow_jvp_and_all_reverse_edges(dtype,upper,active,triangular):
    f,b=arrays(dtype,(),(),65,4)
    if triangular:f=np.triu(f) if upper else np.tril(f)
    df=np.full_like(f,.013);db=np.full_like(b,.027);reports=[]
    for lib in (lite,native):
        args=tuple(lib.tensor(v,requires_grad=True) for v in (b,f,db,df))
        def function(*values):
            inputs=list(args[:2])
            for index,value in zip(active,values):inputs[index]=value
            return lib.cholesky_solve(*inputs,upper)
        x,t=lib.func.jvp(function,tuple(args[i] for i in active),tuple(args[2+i] for i in active))
        gradients=lib.autograd.grad(.17*x.square().mean()+.23*t.square().mean(),args,allow_unused=True)
        reports.append((x.detach().numpy(),t.detach().numpy(),[None if g is None else g.numpy() for g in gradients]))
    a,b=reports;close(a[0],b[0],dtype);close(a[1],b[1],dtype)
    for x,y in zip(a[2],b[2]):
        if y is None:assert x is None
        else:close(x,y,dtype)


FLAGS=[(b,f,db,df) for b in (False,True) for f in (False,True)
       for db in (False,True) for df in (False,True) if b or f or db or df]


@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('flags',FLAGS)
@pytest.mark.parametrize('mutated',range(6))
def test_narrow_jvp_mutation_guards_and_permitted_source_snapshots(upper,flags,mutated):
    f,b=arrays('float64',(),(),65,2);reports=[]
    for lib in (lite,native):
        args=tuple(lib.tensor(v,requires_grad=flag) for v,flag in zip(
            (b,f,np.full_like(b,.03),np.full_like(f,.01)),flags))
        x,t=lib.func.jvp(lambda b,f:lib.cholesky_solve(b,f,upper),args[:2],args[2:])
        with lib.no_grad():(*args,x,t)[mutated].detach().add_(.1)
        try:
            t.sum().backward()
            reports.append(('allowed',[None if a.grad is None else a.grad.numpy().copy() for a in args]))
        except RuntimeError:reports.append(('guarded',[]))
    assert reports[0][0]==reports[1][0]
    for actual,expected in zip(reports[0][1],reports[1][1]):
        if expected is None:assert actual is None
        else:close(actual,expected,'float64')


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('flags',FLAGS)
@pytest.mark.parametrize('triangular',[False,True])
def test_narrow_jvp_broadcast_strided_all_activity_patterns(dtype,upper,flags,triangular):
    f,b=arrays(dtype,(2,1),(1,3),64,3)
    if triangular:f=np.triu(f) if upper else np.tril(f)
    rng=np.random.default_rng(340);df=(rng.normal(size=f.shape)*.01).astype(dtype);db=(rng.normal(size=b.shape)*.03).astype(dtype);reports=[]
    for lib in (lite,native):
        args=tuple(lib.from_numpy(v.copy()).requires_grad_(flag) for v,flag in zip((b,f,db,df),flags))
        # Keep final axes strided in the primal and directional inputs.
        args=tuple(a.transpose(-1,-2).contiguous().transpose(-1,-2) for a in args)
        x,t=lib.func.jvp(lambda b,f:lib.cholesky_solve(b,f,upper),args[:2],args[2:]);loss=.17*x.square().sum()+.23*t.square().sum()
        gradients=lib.autograd.grad(loss,tuple(a for a in args if a.requires_grad),retain_graph=True)
        second=lib.autograd.grad(.31*x.sum()-.11*t.sum(),tuple(a for a in args if a.requires_grad))
        reports.append((x.detach().numpy(),t.detach().numpy(),x.stride(),t.stride(),[g.numpy() for g in (*gradients,*second)]))
    a,b=reports;close(a[0],b[0],dtype);close(a[1],b[1],dtype);assert a[2:4]==b[2:4]
    for actual,expected in zip(a[4],b[4]):close(actual,expected,dtype)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('disabled',[False,True])
@pytest.mark.parametrize('config',[((),(),64,2),((0,1),(1,2),64,2),((2,1),(1,3),64,0)])
def test_narrow_jvp_no_reverse_and_empty_batches(dtype,upper,disabled,config):
    f,b=arrays(dtype,*config);reports=[]
    for lib in (lite,native):
        args=tuple(lib.tensor(v,requires_grad=disabled) for v in (b,f,np.full_like(b,.03),np.full_like(f,.01)))
        with lib.no_grad() if disabled else __import__('contextlib').nullcontext():
            x,t=lib.func.jvp(lambda b,f:lib.cholesky_solve(b,f,upper),args[:2],args[2:])
        assert not x.requires_grad and not t.requires_grad
        reports.append((x.detach().numpy(),t.detach().numpy(),x.stride(),t.stride()))
    close(reports[0][0],reports[1][0],dtype);close(reports[0][1],reports[1][1],dtype);assert reports[0][2:]==reports[1][2:]
