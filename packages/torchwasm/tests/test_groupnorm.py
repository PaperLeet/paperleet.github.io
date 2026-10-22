"""NumPy-only GroupNorm oracles, finite differences and model workflows."""
import copy
import math
import numpy as np
import pytest
from torchwasm import torch, as_torch, UnsupportedOperation


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('shape,groups',[((3,6),3),((2,6,4),1),((2,6,3,2),3),((2,6,2,3,2),6),((2,6),6)])
@pytest.mark.parametrize('layout',['contiguous','fortran','stride'])
def test_group_values_all_pullbacks_against_float64_centered_math(dtype,shape,groups,layout):
    rng=np.random.default_rng(1715);raw=rng.normal(size=shape).astype(dtype)
    if layout=='fortran':raw=np.asfortranarray(raw)
    if layout=='stride':raw=np.repeat(raw,2,axis=-1)[...,::2]
    w=rng.normal(size=shape[1]).astype(dtype);b=rng.normal(size=shape[1]).astype(dtype);g=rng.normal(size=shape).astype(dtype)
    x=torch.from_numpy(raw).requires_grad_();weight=torch.tensor(w,requires_grad=True);bias=torch.tensor(b,requires_grad=True)
    y=torch.nn.functional.group_norm(x,groups,weight,bias);y.backward(torch.tensor(g))
    z=raw.astype(np.float64).reshape(shape[0],groups,-1);c=z-z.mean(-1,keepdims=True);inv=1/np.sqrt((c*c).mean(-1,keepdims=True)+1e-5);normal=(c*inv).reshape(shape)
    broadcast=(1,shape[1],*([1]*(len(shape)-2)));scaled=(g.astype(np.float64)*w.reshape(broadcast)).reshape(z.shape)
    dx=(inv*(scaled-scaled.mean(-1,keepdims=True)-c*inv*(scaled*c*inv).mean(-1,keepdims=True))).reshape(shape)
    axes=(0,*range(2,len(shape)));expected=[normal*w.reshape(broadcast)+b.reshape(broadcast),dx,(g*normal).sum(axis=axes),g.astype(np.float64).sum(axis=axes)]
    for actual,value in zip([y,x.grad,weight.grad,bias.grad],expected):
        # Near-constant two-value groups amplify float32 subtraction rounding.
        # Bound it by arithmetic precision and inverse scale, independently of Torch.
        tolerance=8e-12 if dtype==torch.float64 else max(8e-6,8*np.finfo(np.float32).eps*float(inv.max())*float(np.abs(scaled).max()))
        np.testing.assert_allclose(actual.numpy(),value,rtol=1e-5 if dtype==torch.float32 else 8e-12,atol=tolerance)
    assert y.dtype==dtype


@pytest.mark.parametrize('groups',[1,2,4])
def test_primal_and_full_statistics_jvp_reverse_finite_difference(groups):
    rng=np.random.default_rng(1716);arrays=[rng.normal(size=(2,4,3)),rng.normal(size=4),rng.normal(size=4)]
    arrays += [rng.normal(size=a.shape) for a in arrays];seeds=[rng.normal(size=arrays[0].shape) for _ in range(2)]
    values=[torch.tensor(a,requires_grad=True) for a in arrays]
    fn=lambda x,w,b:torch.nn.functional.group_norm(x,groups,w,b)
    assert torch.autograd.gradcheck(fn,tuple(values[:3]))
    p,t=torch.func.jvp(fn,tuple(values[:3]),tuple(values[3:]));(p*torch.tensor(seeds[0])+t*torch.tensor(seeds[1])).sum().backward()
    def objective(a):
        x,w,b,dx,dw,db=a;z=x.reshape(2,groups,-1);d=dx.reshape(z.shape);c=z-z.mean(-1,keepdims=True);inv=1/np.sqrt((c*c).mean(-1,keepdims=True)+1e-5);projected=d-d.mean(-1,keepdims=True)
        normal=(c*inv).reshape(x.shape);j=(projected*inv-c*inv**3*(projected*c).mean(-1,keepdims=True)).reshape(x.shape)
        return float(np.sum((normal*w[None,:,None]+b[None,:,None])*seeds[0]+(j*w[None,:,None]+normal*dw[None,:,None]+db[None,:,None])*seeds[1]))
    for i,v in enumerate(values):
        expected=np.empty_like(arrays[i])
        for index in np.ndindex(v.shape):
            plus=[a.copy() for a in arrays];minus=[a.copy() for a in arrays];plus[i][index]+=1e-6;minus[i][index]-=1e-6
            expected[index]=(objective(plus)-objective(minus))/2e-6
        np.testing.assert_allclose(v.grad.numpy(),expected,rtol=2e-6,atol=2e-6)


def test_ordinary_import_aliases_module_state_conversion_and_reset():
    with as_torch():
        from torch.nn import GroupNorm
        from torch.nn.modules.normalization import GroupNorm as Alias
        import torch.nn.functional as F
        assert Alias is GroupNorm
        model=GroupNorm(2,4,dtype=torch.float64);raw=torch.tensor(np.arange(24).reshape(2,4,3),dtype=torch.float64)
        assert list(model.state_dict())==['weight','bias'] and not list(model.buffers())
        np.testing.assert_array_equal(model(raw).numpy(),F.group_norm(raw,2,model.weight,model.bias).numpy())
        before=model(raw);model.eval();np.testing.assert_array_equal(before.numpy(),model(raw).numpy())
        saved=copy.deepcopy(model.state_dict());model.float();assert model.weight.dtype==torch.float32
        model.double();model.load_state_dict(saved);model.reset_parameters();assert model.weight.tolist()==[1.]*4 and model.bias.tolist()==[0.]*4
        assert list(GroupNorm(2,4,affine=False).state_dict())==[]


def test_empty_batch_tapes_have_shaped_zeros_and_no_affine_gradients():
    x=torch.ones(0,6,4,requires_grad=True);w=torch.ones(6,requires_grad=True);b=torch.zeros(6,requires_grad=True)
    p,t=torch.func.jvp(lambda z:torch.nn.functional.group_norm(z,3,w,b),(x,),(torch.ones_like(x),))
    (p.sum()+t.sum()).backward();assert x.grad.shape==x.shape and w.grad.tolist()==b.grad.tolist()==[0.]*6


@pytest.mark.parametrize('changed',['input','weight','bias','output'])
def test_precise_guards_and_no_grad_output_mutation(changed):
    x=torch.ones(2,6,4,requires_grad=True);w=torch.ones(6,requires_grad=True);b=torch.zeros(6,requires_grad=True)
    y=torch.nn.functional.group_norm(x,3,w,b);target={'input':x,'weight':w,'bias':b,'output':y}[changed].detach()
    with torch.no_grad():target.add_(1)
    if changed in ['input','weight']:
        with pytest.raises(RuntimeError):y.sum().backward()
    else:y.sum().backward();assert x.grad is not None


@pytest.mark.parametrize('layout',['contiguous','fortran','stride'])
def test_large_offset_float32_groups_follow_centered_float64_policy(layout):
    rng=np.random.default_rng(1717);raw=(1e5+rng.normal(size=(2,6,4))*.1).astype(np.float32)
    # Mixed sensitive and ordinary groups check selective wider accumulation.
    raw[:,2:4]=rng.normal(size=(2,2,4)).astype(np.float32)
    if layout=='fortran':raw=np.asfortranarray(raw)
    if layout=='stride':raw=np.repeat(raw,2,axis=-1)[...,::2]
    x=torch.from_numpy(raw).requires_grad_();seed=rng.normal(size=raw.shape).astype(np.float32)
    y=torch.nn.functional.group_norm(x,3);y.backward(torch.tensor(seed))
    z=raw.astype(np.float64).reshape(2,3,8);c=z-z.mean(-1,keepdims=True);inv=1/np.sqrt((c*c).mean(-1,keepdims=True)+1e-5);normal=c*inv;g=seed.astype(np.float64).reshape(z.shape)
    dx=inv*(g-g.mean(-1,keepdims=True)-normal*(g*normal).mean(-1,keepdims=True))
    np.testing.assert_allclose(y.numpy(),normal.reshape(raw.shape),rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(x.grad.numpy(),dx.reshape(raw.shape),rtol=3e-6,atol=5e-6)


def test_numpy_only_model_checkpoint_adam_resume_matches_uninterrupted_model():
    rng=np.random.default_rng(1718)
    def model():return torch.nn.Sequential(torch.nn.Linear(6,6,dtype=torch.float64),torch.nn.GroupNorm(3,6,dtype=torch.float64),torch.nn.SiLU(),torch.nn.Linear(6,2,dtype=torch.float64))
    a=model();b=model();b.load_state_dict(copy.deepcopy(a.state_dict()));oa=torch.optim.Adam(a.parameters(),lr=.003);ob=torch.optim.Adam(b.parameters(),lr=.003)
    for step in range(6):
        x=torch.tensor(rng.normal(size=(4,6)));target=torch.tensor(rng.normal(size=(4,2)))
        for m,opt in [(a,oa),(b,ob)]:opt.zero_grad();loss=(m(x)-target).square().mean();loss.backward();opt.step()
        for p,q in zip(a.parameters(),b.parameters()):np.testing.assert_allclose(p.numpy(),q.numpy(),atol=1e-12)
        if step==2:
            state=copy.deepcopy(b.state_dict());ostate=copy.deepcopy(ob.state_dict());b=model();b.load_state_dict(state);ob=torch.optim.Adam(b.parameters(),lr=99);ob.load_state_dict(ostate)


@pytest.mark.parametrize('kwargs',[{'weight':torch.ones(2)},{'bias':torch.ones(6,dtype=torch.float64)}])
def test_bad_affine_shape_dtype_is_explicit(kwargs):
    with pytest.raises(RuntimeError):torch.nn.functional.group_norm(torch.ones(2,6,4),3,**kwargs)


@pytest.mark.parametrize('dtype',[torch.float16,torch.complex64])
def test_unsupported_storage_dtype_is_explicit(dtype):
    with pytest.raises(UnsupportedOperation):torch.nn.functional.group_norm(torch.ones(2,6,4,dtype=dtype),3)
