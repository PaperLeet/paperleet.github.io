"""NumPy-only directional, lifecycle and numerical moment contracts."""
import warnings
import numpy as np
import pytest
from stridetorch import torch


@pytest.mark.parametrize('name,correction',[(name,c) for name in ['var','std','var_mean','std_mean'] for c in [0,.5,1]])
def test_moments_directional_and_duality(name,correction):
    data=np.array([[.2,-1.1,.5],[2.,.1,-.8]],dtype=np.float64)
    tangent=np.array([[.4,.2,-.1],[.7,-.5,.3]])
    def call(x):return getattr(torch,name)(x,dim=1,correction=correction)
    def values(output):return output if isinstance(output,tuple) else (output,)
    x=torch.tensor(data,requires_grad=True)
    outputs=values(call(x));seeds=[np.array([.3,-.7])*(i+1) for i in range(len(outputs))]
    loss=sum((value*torch.tensor(seed)).sum() for value,seed in zip(outputs,seeds));loss.backward()
    p,j=torch.func.jvp(call,(torch.tensor(data),),(torch.tensor(tangent),))
    h=1e-5
    plus=values(call(torch.tensor(data+h*tangent)));minus=values(call(torch.tensor(data-h*tangent)))
    for actual,hi,lo in zip(values(j),plus,minus):
        np.testing.assert_allclose(actual.numpy(),(hi.numpy()-lo.numpy())/(2*h),rtol=3e-9,atol=3e-10)
    lhs=sum((v.numpy()*seed).sum() for v,seed in zip(values(j),seeds))
    np.testing.assert_allclose(lhs,(x.grad.numpy()*tangent).sum(),rtol=2e-12,atol=2e-12)
    mean=data.sum(1)/3;var=((data-mean[:,None])**2).sum(1)/(3-correction)
    expected=[np.sqrt(var) if name.startswith('std') else var]
    if name.endswith('_mean'):expected.append(mean)
    for actual,reference in zip(outputs,expected):np.testing.assert_allclose(actual.numpy(),reference,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('name',['var','std','var_mean','std_mean'])
def test_moments_training_resume(name):
    initial=np.array([[.2,.7,-.4],[1.2,-.6,.9]],dtype=np.float64)
    x=torch.tensor(initial,requires_grad=True);opt=torch.optim.SGD([x],lr=.03)
    expected=initial.copy()
    for _ in range(4):
        opt.zero_grad();outputs=getattr(torch,name)(x,dim=1,correction=.5)
        outputs=outputs if isinstance(outputs,tuple) else (outputs,)
        sum(y.sum() for y in outputs).backward();opt.step()
        center=expected-expected.mean(1,keepdims=True);derivative=2*center/2.5
        if name.startswith('std'):derivative/=2*np.sqrt((center**2).sum(1,keepdims=True)/2.5)
        if name.endswith('_mean'):derivative+=1/3
        expected-=.03*derivative
    np.testing.assert_allclose(x.detach().numpy(),expected,rtol=2e-12,atol=2e-12)
    snapshot=opt.state_dict();restored=torch.tensor(x.detach().numpy(),requires_grad=True)
    resumed=torch.optim.SGD([restored],lr=1.);resumed.load_state_dict(snapshot)
    for parameter,optimizer in [(x,opt),(restored,resumed)]:
        optimizer.zero_grad();torch.var_mean(parameter,dim=1,correction=.5)[0].sum().backward();optimizer.step()
    np.testing.assert_array_equal(x.detach().numpy(),restored.detach().numpy())


@pytest.mark.parametrize('name',['var','std','var_mean','std_mean'])
def test_moments_retain_graph_and_hooks(name):
    x=torch.tensor([.2,-.8,1.5],dtype=torch.float64,requires_grad=True);calls=[]
    x.register_hook(lambda g:calls.append(g.numpy().copy()))
    outputs=getattr(torch,name)(x);outputs=outputs if isinstance(outputs,tuple) else (outputs,)
    loss=sum(v.sum() for v in outputs);loss.backward(retain_graph=True);first=x.grad.numpy().copy()
    loss.backward();np.testing.assert_allclose(x.grad.numpy(),2*first);assert len(calls)==2
    with pytest.raises(RuntimeError):loss.backward()


def test_moments_empty_warning_and_float16_accumulation():
    with pytest.warns(UserWarning,match='Degrees of freedom'):
        output=torch.var(torch.tensor([2.]),correction=1)
    assert np.isnan(output.item())
    data=np.array([1000.,1000.,1001.],dtype=np.float16)
    var,mean=torch.var_mean(torch.tensor(data),correction=1)
    np.testing.assert_allclose(var.item(),np.float16(1/3));assert mean.item()==np.float16(1000+1/3)
    assert var.dtype==mean.dtype==np.float16


def test_moments_typed_boundaries():
    x=torch.tensor([1.,2.])
    for name in ['var','std','var_mean','std_mean']:
        with pytest.raises(torch._browser_unsupported):getattr(torch,name)(x,out=torch.zeros(()))
        with pytest.raises(TypeError):getattr(torch,name)(x,dim=True)
        with pytest.raises(TypeError):getattr(torch,name)(x,unbiased=False,correction=0)
        with pytest.raises(RuntimeError):getattr(torch,name)(torch.tensor([1,2]))
        with pytest.raises(torch._browser_unsupported):getattr(torch,name)(torch.tensor([1+2j]))


@pytest.mark.parametrize('name,values,average,gradient,dtype',[
    (name,values,average,gradient,dtype)
    for name in ['var_mean','std_mean'] for dtype in [np.float32,np.float64]
    for values,average,gradient in [
        ([1.,np.inf],np.inf,[-np.inf,np.nan]),
        ([np.inf,1.],np.nan,[np.nan,-np.inf]),
        ([np.inf,np.inf],np.nan,[np.nan,np.nan]),
        ([-np.inf,1.],np.nan,[np.nan,np.inf]),
        ([-np.inf,-np.inf],np.nan,[np.nan,np.nan]),
        ([np.nan,1.],np.nan,[np.nan,np.nan])]])
def test_paired_nonfinite_moments_and_independent_mean_gradient(name,values,average,gradient,dtype):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x=torch.tensor(np.asarray(values,dtype=dtype),requires_grad=True)
        statistic,mean=getattr(torch,name)(x,correction=0)
        assert np.isnan(statistic.item())
        np.testing.assert_allclose(mean.item(),average,equal_nan=True)
        statistic.backward(retain_graph=True)
        expected=[np.nan,np.nan] if name=='std_mean' else gradient
        np.testing.assert_allclose(x.grad.numpy(),expected,equal_nan=True)
        x.grad=None;mean.backward();np.testing.assert_allclose(x.grad.numpy(),[.5,.5])
        _,j=torch.func.jvp(lambda z:getattr(torch,name)(z,correction=0),
                           (torch.tensor(np.asarray(values,dtype=dtype)),),(torch.ones(2,dtype=dtype),))
        assert np.isnan(j[0].item()) and j[1].item()==1
