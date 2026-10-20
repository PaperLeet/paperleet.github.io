"""Independent native contracts for moments, paired results and differentiation."""
import itertools
import warnings
import numpy as np
import pytest
import torch as native
from torchwasm import torch


def boundary_cases():
    for shape in [(), (1,), (3,), (2,3), (0,3), (2,0,3)]:
        for op, dim, correction, keep in itertools.product(
            ['var','std'], [None,0,-1,1,(),[],(0,1),(0,0),(0,-len(shape)),True,1.2],
            [0,.5,1,2,5,-1], [False,True]):
            yield shape,op,dim,correction,keep


def array(tensor): return tensor.detach().numpy()


def close(actual, expected, dtype=np.float64):
    tolerance = 3e-5 if dtype == np.float32 else 2e-12
    np.testing.assert_allclose(array(actual), array(expected), rtol=tolerance, atol=tolerance, equal_nan=True)
    assert actual.shape == tuple(expected.shape)
    assert actual.dtype == array(expected).dtype


@pytest.mark.parametrize('shape,op,dim,correction,keep', list(boundary_cases()))
def test_statistics_native_boundaries(shape,op,dim,correction,keep):
    data=np.linspace(.5,2.5,num=int(np.prod(shape,dtype=int)),dtype=np.float64).reshape(shape)
    x=torch.tensor(data,requires_grad=True);n=native.tensor(data,requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try: expected=getattr(n,op)(dim=dim,correction=correction,keepdim=keep)
        except (TypeError,IndexError,RuntimeError) as error:
            with pytest.raises(type(error)):getattr(x,op)(dim=dim,correction=correction,keepdim=keep)
            return
        actual=getattr(x,op)(dim=dim,correction=correction,keepdim=keep)
        close(actual,expected)
        seed=np.arange(actual.numel(),dtype=np.float64).reshape(actual.shape)+1
        actual.backward(torch.tensor(seed));expected.backward(native.tensor(seed))
        close(x.grad,n.grad)


def sample(layout,dtype):
    data=(np.arange(24,dtype=dtype)/7-1).reshape(2,3,4)
    if layout=='transpose':return data.transpose(2,0,1)
    if layout=='step':return data[...,::2]
    if layout=='broadcast':return np.broadcast_to(data[:1,:1,:],(2,3,4)).copy()
    return data


@pytest.mark.parametrize('op,layout,dtype,dim,correction,keep', list(itertools.product(
    ['var','std','var_mean','std_mean'],['contiguous','transpose','step','broadcast'],
    [np.float32,np.float64],[None,0,(-1,0),()],[0,.5,1],[False,True])))
def test_statistics_native_reverse_and_jvp(op,layout,dtype,dim,correction,keep):
    data=sample(layout,dtype);direction=np.cos(np.arange(data.size)).astype(dtype).reshape(data.shape)
    def call(lib,x):return getattr(lib,op)(x,dim=dim,correction=correction,keepdim=keep)
    x=torch.tensor(data,requires_grad=True);n=native.tensor(data,requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a=call(torch,x);b=call(native,n);aa=a if isinstance(a,tuple) else (a,);bb=b if isinstance(b,tuple) else (b,)
        loss=0;reference=0
        for index,(v,nv) in enumerate(zip(aa,bb)):
            close(v,nv,dtype);seed=(np.arange(v.numel(),dtype=dtype).reshape(v.shape)+1)/(index+1)
            loss=loss+(v*torch.tensor(seed)).sum();reference=reference+(nv*native.tensor(seed)).sum()
        loss.backward();reference.backward();close(x.grad,n.grad,dtype)
        pa,ta=torch.func.jvp(lambda z:call(torch,z),(torch.tensor(data),),(torch.tensor(direction),))
        pb,tb=native.func.jvp(lambda z:call(native,z),(native.tensor(data),),(native.tensor(direction),))
        for v,nv in zip(ta if isinstance(ta,tuple) else (ta,),tb if isinstance(tb,tuple) else (tb,)):close(v,nv,dtype)


@pytest.mark.parametrize('op,args,kwargs', list(itertools.product(['var','std','var_mean','std_mean'],
    [(),(False,),(True,),(None,False,True),(0,False),(False,True)],
    [{},{'dim':0},{'unbiased':False},{'correction':None},{'keepdim':True},
     {'unbiased':1},{'keepdim':1},{'correction':1j},{'unbiased':True,'correction':0}])))
def test_statistics_native_overloads(op,args,kwargs):
    x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64);n=native.tensor([[1.,2.],[3.,4.]],dtype=native.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:b=getattr(native,op)(n,*args,**kwargs)
        except (TypeError,IndexError,RuntimeError) as error:
            with pytest.raises(type(error)):getattr(torch,op)(x,*args,**kwargs)
            return
        a=getattr(torch,op)(x,*args,**kwargs)
        for v,nv in zip(a if isinstance(a,tuple) else (a,),b if isinstance(b,tuple) else (b,)):close(v,nv)


@pytest.mark.parametrize('op,mutate,selected',list(itertools.product(['var_mean','std_mean'],['input','first','mean'],[0,1])))
def test_statistics_saved_tensor_guards(op,mutate,selected):
    def run(lib):
        x=lib.tensor([1.,2.,4.],requires_grad=True);a,b=getattr(lib,op)(x)
        with lib.no_grad():{'input':x,'first':a,'mean':b}[mutate].detach().fill_(3)
        (a,b)[selected].backward()
        return x.grad
    try:b=run(native)
    except RuntimeError:
        with pytest.raises(RuntimeError):run(torch)
        return
    close(run(torch),b,np.float32)


@pytest.mark.parametrize('op', ['var','std','var_mean','std_mean'])
def test_statistics_reverse_over_jvp(op):
    data=np.array([[.2,.7,-.9],[1.1,-.4,.5]],dtype=np.float64);direction=np.array([[.3,-.2,.6],[.7,.5,-.1]])
    def run(lib):
        x=lib.tensor(data,requires_grad=True);t=lib.tensor(direction,requires_grad=True)
        p,j=lib.func.jvp(lambda z:getattr(lib,op)(z,dim=1,correction=.5),(x,),(t,))
        values=j if isinstance(j,tuple) else (j,)
        sum(v.sum() for v in values).backward()
        return x.grad,t.grad
    a,b=run(torch),run(native)
    for v,nv in zip(a,b):close(v,nv)


@pytest.mark.parametrize('op,correction',list(itertools.product(['var','std','var_mean','std_mean'],[0,1])))
def test_statistics_float32_large_offset_paired_rounding(op,correction):
    data=np.array([1e8,1e8,1e8+8],dtype=np.float32)
    x=torch.tensor(data,requires_grad=True);n=native.tensor(data,requires_grad=True)
    a=getattr(torch,op)(x,correction=correction);b=getattr(native,op)(n,correction=correction)
    aa=a if isinstance(a,tuple) else (a,);bb=b if isinstance(b,tuple) else (b,)
    for v,nv in zip(aa,bb):close(v,nv,np.float32)
    aa[0].backward();bb[0].backward();close(x.grad,n.grad,np.float32)


@pytest.mark.parametrize('op,shape,dim,correction',list(itertools.product(
    ['var','std','var_mean','std_mean'],[(),(0,3),(2,3)],[None,(),0],[float('nan'),float('inf'),float('-inf') ])))
def test_statistics_nonfinite_corrections(op,shape,dim,correction):
    data=np.arange(int(np.prod(shape,dtype=int)),dtype=np.float64).reshape(shape)
    x=torch.tensor(data,requires_grad=True);n=native.tensor(data,requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a=getattr(torch,op)(x,dim=dim,correction=correction);b=getattr(native,op)(n,dim=dim,correction=correction)
        aa=a if isinstance(a,tuple) else (a,);bb=b if isinstance(b,tuple) else (b,)
        for v,nv in zip(aa,bb):close(v,nv)
        aa[0].backward(torch.ones_like(aa[0]));bb[0].backward(native.ones_like(bb[0]));close(x.grad,n.grad)


@pytest.mark.parametrize('op,values,dtype,dim',list(itertools.product(
    ['var','std','var_mean','std_mean'],
    [[1.,np.inf],[np.inf,np.inf],[-np.inf,1.],[-np.inf,-np.inf],[np.inf,-np.inf],[np.nan,1.],[np.nan,np.nan],[2.,2.]],
    [np.float32,np.float64],[None,0,()])))
def test_statistics_nonfinite_inputs_and_constant_jvp(op,values,dtype,dim):
    data=np.asarray(values,dtype=dtype)
    x=torch.tensor(data,requires_grad=True);n=native.tensor(data,requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a=getattr(torch,op)(x,dim=dim,correction=0);b=getattr(native,op)(n,dim=dim,correction=0)
        aa=a if isinstance(a,tuple) else (a,);bb=b if isinstance(b,tuple) else (b,)
        for v,nv in zip(aa,bb):close(v,nv,dtype)
        aa[0].backward();bb[0].backward();close(x.grad,n.grad,dtype)
        _,j=torch.func.jvp(lambda z:getattr(torch,op)(z,dim=dim,correction=0),(torch.tensor(data),),(torch.ones_like(x),))
        _,nj=native.func.jvp(lambda z:getattr(native,op)(z,dim=dim,correction=0),(native.tensor(data),),(native.ones_like(n),))
        for v,nv in zip(j if isinstance(j,tuple) else (j,),nj if isinstance(nj,tuple) else (nj,)):close(v,nv,dtype)
