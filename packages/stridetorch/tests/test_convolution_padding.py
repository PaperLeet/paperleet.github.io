"""Bounded zero-padding with independent scalar math, IEEE and tape checks."""
import gc
import math
import weakref
import numpy as np
import pytest
from stridetorch import torch


@pytest.fixture(params=['auto','gather','tap','capped'],autouse=True)
def padding_route(request,monkeypatch):
    from stridetorch import _convolution as c
    monkeypatch.setattr(c,'_PADDING_ALGORITHM','auto' if request.param=='capped' else request.param)
    if request.param=='capped':
        monkeypatch.setattr(c,'_PAD_WORKSPACE_BYTES',0)
        monkeypatch.setattr(c,'_GATHER_WORKSPACE_BYTES',0)


def oracle(x,w,b,stride,pairs,dilation,groups,seed=None):
    dim=w.ndim-2;n,ci=x.shape[:2];co=w.shape[0];kernel=w.shape[2:]
    out=tuple((size+a+z-d*(k-1)-1)//s+1 for size,(a,z),d,k,s in zip(x.shape[2:],pairs,dilation,kernel,stride))
    y=np.zeros((n,co,*out));dx=np.zeros(x.shape);dw=np.zeros(w.shape);db=np.zeros(co)
    for batch in range(n):
        for o in range(co):
            group=o//(co//groups)
            for pos in np.ndindex(out):
                key=(batch,o,*pos);value=float(b[o]) if b is not None else 0.
                if seed is not None:db[o]+=float(seed[key])
                for i in range(ci//groups):
                    channel=group*(ci//groups)+i
                    for tap in np.ndindex(kernel):
                        loc=tuple(p*s+k*d-pair[0] for p,s,k,d,pair in zip(pos,stride,tap,dilation,pairs))
                        if any(v<0 or v>=size for v,size in zip(loc,x.shape[2:])):continue
                        xx=(batch,channel,*loc);ww=(o,i,*tap);value+=float(x[xx])*float(w[ww])
                        if seed is not None:dx[xx]+=float(seed[key])*float(w[ww]);dw[ww]+=float(seed[key])*float(x[xx])
                y[key]=value
    return y,dx,dw,db


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('groups',[1,2])
@pytest.mark.parametrize('kind',['halo','anisotropic','same','overlap'])
def test_values_and_vjp_against_scalar_zero_padding(dim,dtype,groups,kind):
    rng=np.random.default_rng(4910);kernel=(2 if kind=='same' else 3,)*dim
    arrays=[rng.normal(size=(1,2,*([3]*dim))).astype(dtype),rng.normal(size=(4,2//groups,*kernel)).astype(dtype),rng.normal(size=4).astype(dtype)]
    stride=(1,)*dim;dilation=(31 if kind=='halo' else 3 if kind=='same' else 2,)*dim
    padding='same' if kind=='same' else dilation
    if kind=='anisotropic':stride=(2,1,3)[:dim];dilation=(7,11,13)[:dim];padding=dilation
    pairs=tuple((d*(k-1)//2,d*(k-1)-d*(k-1)//2) for d,k in zip(dilation,kernel)) if padding=='same' else tuple((p,p) for p in padding)
    values=[torch.from_numpy(np.asfortranarray(a)).requires_grad_() for a in arrays]
    y=getattr(torch.nn.functional,f'conv{dim}d')(*values,stride=stride,padding=padding,dilation=dilation,groups=groups)
    seed=rng.normal(size=y.shape).astype(dtype);y.backward(torch.tensor(seed))
    for actual,expected in zip([y,*[v.grad for v in values]],oracle(*arrays,stride,pairs,dilation,groups,seed)):
        np.testing.assert_allclose(actual.numpy(),expected,rtol=2e-5 if dtype==torch.float32 else 5e-12,atol=8e-6 if dtype==torch.float32 else 5e-12)


@pytest.mark.parametrize('dim',[1,2,3])
def test_bilinear_jvp_reverse_finite_difference_and_gradcheck(dim):
    rng=np.random.default_rng(4911);arrays=[rng.normal(size=(1,2,*([2]*dim))),rng.normal(size=(4,1,*([3]*dim))),rng.normal(size=4)]
    arrays += [rng.normal(size=a.shape) for a in arrays];values=[torch.tensor(a,requires_grad=True) for a in arrays]
    fn=lambda x,w,b:getattr(torch.nn.functional,f'conv{dim}d')(x,w,b,padding=31,dilation=31,groups=2)
    assert torch.autograd.gradcheck(fn,tuple(values[:3]))
    p,t=torch.func.jvp(fn,tuple(values[:3]),tuple(values[3:]));seeds=[rng.normal(size=p.shape),rng.normal(size=p.shape)]
    (p*torch.tensor(seeds[0])+t*torch.tensor(seeds[1])).sum().backward()
    center=(slice(None),slice(None),*([1]*dim))
    def evaluate(raw):
        x,w,b,dx,dw,db=raw;repeat=x.repeat(2,axis=1);direction=dx.repeat(2,axis=1)
        a=w[center].reshape(1,4,*([1]*dim));da=dw[center].reshape(1,4,*([1]*dim));bs=b.reshape(1,4,*([1]*dim));dbs=db.reshape(1,4,*([1]*dim))
        return float(np.sum((repeat*a+bs)*seeds[0]+(direction*a+repeat*da+dbs)*seeds[1]))
    for position,value in enumerate(values):
        expected=np.empty_like(arrays[position])
        for index in np.ndindex(expected.shape):
            plus=[a.copy() for a in arrays];minus=[a.copy() for a in arrays];plus[position][index]+=1e-6;minus[position][index]-=1e-6
            expected[index]=(evaluate(plus)-evaluate(minus))/2e-6
        np.testing.assert_allclose(value.grad.numpy(),expected,rtol=4e-6,atol=4e-6)


@pytest.mark.parametrize('dim',[1,2,3])
def test_huge_padding_never_calls_full_pad_or_padded_adjoint(monkeypatch,dim):
    from stridetorch import _convolution as c
    def reject(*args,**kwargs):raise AssertionError('full padded activation/adjoint allocation')
    monkeypatch.setattr(c,'_pad',reject);monkeypatch.setattr(c,'_matrix_input_gradient',reject)
    x=torch.ones(1,2,*([3]*dim),dtype=torch.float64,requires_grad=True);w=torch.ones(4,1,*([3]*dim),dtype=torch.float64,requires_grad=True)
    y=getattr(torch.nn.functional,f'conv{dim}d')(x,w,padding=1000000,dilation=1000000,groups=2)
    y.sum().backward();assert y.shape==(1,4,*([3]*dim))
    np.testing.assert_array_equal(y.numpy(),1.);np.testing.assert_array_equal(x.grad.numpy(),2.)
    expected=np.zeros(w.shape);expected[(slice(None),slice(None),*([1]*dim))]=3**dim
    np.testing.assert_array_equal(w.grad.numpy(),expected)


def test_zero_workspace_uses_per_tap_fallback_without_gather(monkeypatch):
    from stridetorch import _convolution as c,_convolution_padding as kernels
    monkeypatch.setattr(c,'_PADDING_ALGORITHM','auto');monkeypatch.setattr(c,'_GATHER_WORKSPACE_BYTES',0);monkeypatch.setattr(c,'_PAD_WORKSPACE_BYTES',0);monkeypatch.setattr(c,'_ALGORITHM','matrix')
    def reject(*args,**kwargs):raise AssertionError('forbidden optional full workspace')
    monkeypatch.setattr(kernels,'gather',reject);monkeypatch.setattr(c,'_pad',reject)
    x=torch.ones(1,2,3,dtype=torch.float64,requires_grad=True);w=torch.ones(4,1,3,dtype=torch.float64,requires_grad=True)
    y=torch.nn.functional.conv1d(x,w,padding=31,dilation=31,groups=2);y.sum().backward()
    np.testing.assert_array_equal(y.numpy(),1.);np.testing.assert_array_equal(x.grad.numpy(),2.)


@pytest.mark.parametrize('kind',['weight_inf','input_inf','seed_inf','seed_nan'])
def test_implicit_zero_ieee_rules(kind):
    x=torch.ones(1,1,3,dtype=torch.float64,requires_grad=True);w=torch.ones(1,1,3,dtype=torch.float64,requires_grad=True)
    if kind=='weight_inf':w.detach()[0,0,0]=float('inf')
    if kind=='input_inf':x.detach()[0,0,0]=float('inf')
    y=torch.nn.functional.conv1d(x,w,padding=1000000,dilation=1000000)
    seed=torch.ones_like(y)
    if kind.startswith('seed_'):seed[0,0,0]=float('inf') if kind=='seed_inf' else float('nan')
    y.backward(seed)
    if kind=='weight_inf':assert np.isnan(y.numpy()).all();np.testing.assert_array_equal(x.grad.numpy(),1.)
    elif kind=='input_inf':
        np.testing.assert_array_equal(y.numpy(),[[[np.inf,1.,1.]]]);np.testing.assert_array_equal(w.grad.numpy()[0,0,[0,2]],[0.,0.]);assert np.isinf(w.grad.numpy()[0,0,1])
    else:
        assert np.isnan(w.grad.numpy()[0,0,[0,2]]).all()
        assert (np.isinf if kind=='seed_inf' else np.isnan)(w.grad.numpy()[0,0,1])


def test_expanded_gather_is_not_retained_on_tape(monkeypatch):
    from stridetorch import _convolution as c,_convolution_padding as kernels
    monkeypatch.setattr(c,'_PADDING_ALGORITHM','gather');monkeypatch.setattr(c,'_GATHER_WORKSPACE_BYTES',32*1024*1024)
    refs=[];original=kernels.gather
    def record(*args):
        value=original(*args);refs.append(weakref.ref(value));return value
    monkeypatch.setattr(kernels,'gather',record)
    x=torch.ones(1,2,3,3,dtype=torch.float64,requires_grad=True);w=torch.ones(4,2,3,3,dtype=torch.float64,requires_grad=True)
    y=torch.nn.functional.conv2d(x,w,padding=31,dilation=31);gc.collect();assert len(refs)==1 and refs[0]() is None
    y.sum().backward(retain_graph=True);gc.collect();assert len(refs)==2 and all(ref() is None for ref in refs)
    x.grad=w.grad=None;y.sum().backward();gc.collect();assert len(refs)==3 and all(ref() is None for ref in refs)
    np.testing.assert_array_equal(x.grad.numpy(),4.)


@pytest.mark.parametrize('changed',['input','weight','bias','output'])
def test_version_guards_and_mutation_independence(changed):
    x=torch.ones(1,2,3,dtype=torch.float64,requires_grad=True);w=torch.ones(4,2,3,dtype=torch.float64,requires_grad=True);b=torch.ones(4,dtype=torch.float64,requires_grad=True)
    y=torch.nn.functional.conv1d(x,w,b,padding=31,dilation=31)
    {'input':x,'weight':w,'bias':b,'output':y}[changed].detach().add_(1)
    if changed in ['input','weight']:
        with pytest.raises(RuntimeError):y.sum().backward()
    else:y.sum().backward();np.testing.assert_array_equal(b.grad.numpy(),3.)


@pytest.mark.parametrize('mask',range(8))
def test_active_gradient_subsets_and_bias_only_release(mask):
    values=[torch.ones(1,2,3,dtype=torch.float64,requires_grad=bool(mask&1)),torch.ones(4,2,3,dtype=torch.float64,requires_grad=bool(mask&2)),torch.ones(4,dtype=torch.float64,requires_grad=bool(mask&4))]
    y=torch.nn.functional.conv1d(*values,padding=31,dilation=31);assert y.requires_grad==bool(mask)
    if mask:y.sum().backward()
    for value in values:assert (value.grad is not None)==value.requires_grad
    x=torch.ones(1,2,3);w=torch.ones(4,2,3);b=torch.ones(4,requires_grad=True);refs=[weakref.ref(x),weakref.ref(w)]
    y=torch.nn.functional.conv1d(x,w,b,padding=31,dilation=31);del x,w;gc.collect();assert all(ref() is None for ref in refs)
    y.sum().backward();assert b.grad.tolist()==[3.]*4


def test_strided_unbatched_empty_and_shared_parameter_training():
    raw=np.arange(30,dtype=np.float64).reshape(2,15)[:,::2];x=torch.from_numpy(raw).requires_grad_();w=torch.ones(4,2,3,dtype=torch.float64,requires_grad=True)
    y=torch.nn.functional.conv1d(x,w,padding=31,dilation=31);y.sum().backward();assert y.shape==(4,8)
    np.testing.assert_array_equal(y.numpy(),np.repeat(raw.sum(axis=0,keepdims=True),4,axis=0))
    empty=torch.ones(0,2,3,dtype=torch.float64,requires_grad=True);out=torch.nn.functional.conv1d(empty,w,padding=31,dilation=31);out.sum().backward();assert out.shape==(0,4,3) and out.stride()==(12,3,1)
    model=torch.nn.Conv2d(2,4,3,padding=31,dilation=31,dtype=torch.float64);opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.7);sample=torch.ones(2,2,3,3,dtype=torch.float64)
    before=model.weight.numpy().copy()
    for _ in range(3):opt.zero_grad();model(sample).square().mean().backward();opt.step()
    assert np.any(model.weight.numpy()!=before)
