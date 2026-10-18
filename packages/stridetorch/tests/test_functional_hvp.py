"""Analytical HVP values and independent normalization chain rules."""
import numpy as np
import pytest
from stridetorch import as_torch, torch

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('kind',['sin','tanh','cubic','square'])
@pytest.mark.parametrize('strided',[False,True])
def test_hvp_analytic_elementwise_and_caller_state(dtype,kind,strided):
    data=np.linspace(-.7,.8,6,dtype=dtype).reshape(2,3)
    if strided:
        storage=np.empty(data.shape+(2,),dtype=dtype)
        storage[...,0]=data;storage[...,1]=-7;data=storage[...,0]
    direction=np.cos(np.arange(6)).reshape(2,3).astype(dtype)
    x=torch.from_numpy(data).requires_grad_(True)
    v=torch.tensor(direction,requires_grad=True)
    buffers=[];hooks=[]
    for item in [x,v]:
        item.grad=torch.ones_like(item)*3;buffers.append(item.grad)
        item.register_hook(lambda g:hooks.append(g))
    calls=[]
    def callback(a):
        calls.append(torch.is_grad_enabled())
        if kind=='cubic':return a.pow(3).sum()
        return getattr(a,kind)().sum()
    with torch.no_grad():
        output,actual=torch.autograd.functional.hvp(callback,x,v)
        assert not torch.is_grad_enabled()
    if kind=='sin':curvature=-np.sin(data)
    elif kind=='tanh':curvature=-2*np.tanh(data)*(1-np.tanh(data)**2)
    elif kind=='cubic':curvature=6*data
    else:curvature=np.full_like(data,2)
    np.testing.assert_allclose(actual.numpy(),curvature*direction,rtol=2e-6,atol=2e-7)
    assert actual.dtype==np.dtype(dtype) and actual.shape==x.shape
    assert not output.requires_grad and not actual.requires_grad
    assert calls==[True] and not hooks
    for item,buffer,original in zip([x,v],buffers,[data,direction]):
        assert item.grad is buffer and item.requires_grad
        np.testing.assert_array_equal(item.detach().numpy(),original)
        np.testing.assert_array_equal(item.grad.numpy(),np.full_like(original,3))

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('duplicate',[False,True])
def test_hvp_coupled_tuple_inputs_keep_distinct_derivatives(dtype,duplicate):
    a=np.array([.2,.7],dtype=dtype);b=a if duplicate else np.array([.4,.9],dtype=dtype)
    x=torch.tensor(a);y=x if duplicate else torch.tensor(b)
    va=np.array([.3,.8],dtype=dtype);vb=np.array([.6,-.2],dtype=dtype)
    output,values=torch.autograd.functional.hvp(lambda p,q:(p.pow(3)+p*q+q.sin()).sum(),
        (x,y),(torch.tensor(va),torch.tensor(vb)),strict=True)
    np.testing.assert_allclose(values[0].numpy(),6*a*va+vb,rtol=2e-6,atol=2e-7)
    np.testing.assert_allclose(values[1].numpy(),va-np.sin(b)*vb,rtol=2e-6,atol=2e-7)
    assert not output.requires_grad and x.grad is None and y.grad is None

def normalization_hvp(arrays,vectors,axes,broadcast,affine_axes):
    x,w,b=[a.astype(np.float64) for a in arrays]
    dx,dw,db=[a.astype(np.float64) for a in vectors]
    w=w.reshape(broadcast);b=b.reshape(broadcast);dw=dw.reshape(broadcast);db=db.reshape(broadcast)
    mean=lambda a:a.mean(axes,keepdims=True)
    c=x-mean(x);dc=dx-mean(dx);r=(mean(c*c)+1e-5)**-.5
    dr=-r**3*mean(c*dc);n=c*r;dn=dc*r+c*dr
    y=n*w+b;dy=dn*w+n*dw+db
    g=2*y*w;dg=2*dy*w+2*y*dw
    projected=g-mean(g)-n*mean(g*n)
    hx=dr*projected+r*(dg-mean(dg)-dn*mean(g*n)-n*mean(dg*n+g*dn))
    return hx,(2*dy*n+2*y*dn).sum(affine_axes),(2*dy).sum(affine_axes)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('kind',['layer','instance','batch'])
@pytest.mark.parametrize('strided',[False,True])
def test_hvp_normalization_matches_independent_analytic_chain_rule(dtype,kind,strided):
    rng=np.random.default_rng(7021)
    shape=(2,3) if kind=='layer' else (2,2,3)
    weight=(3,) if kind=='layer' else (2,)
    arrays=[(rng.normal(size=s)*.2).astype(dtype) for s in [shape,weight,weight]]
    vectors=[np.cos(np.arange(a.size)*.11+i).reshape(a.shape).astype(dtype) for i,a in enumerate(arrays)]
    if strided:
        for i,a in enumerate(arrays):
            storage=np.empty(a.shape+(2,),dtype=dtype);storage[...,0]=a;storage[...,1]=-7;arrays[i]=storage[...,0]
    inputs=tuple(torch.from_numpy(a) for a in arrays)
    directions=tuple(torch.tensor(a) for a in vectors)
    if kind=='layer':
        function=lambda a,w,b:torch.nn.functional.layer_norm(a,(3,),w,b)
        axes=(-1,);broadcast=(1,3);affine_axes=(0,)
    elif kind=='instance':
        function=lambda a,w,b:torch.nn.functional.instance_norm(a,weight=w,bias=b)
        axes=(2,);broadcast=(1,2,1);affine_axes=(0,2)
    else:
        function=lambda a,w,b:torch.nn.functional.batch_norm(a,None,None,w,b,training=True)
        axes=(0,2);broadcast=(1,2,1);affine_axes=(0,2)
    _,actual=torch.autograd.functional.hvp(lambda *args:function(*args).square().sum(),inputs,directions)
    expected=normalization_hvp(arrays,vectors,axes,broadcast,affine_axes)
    for value,reference in zip(actual,expected):
        assert value.dtype==np.dtype(dtype) and not value.requires_grad
        np.testing.assert_allclose(value.numpy(),reference,rtol=2e-5 if dtype==np.float32 else 1e-9,atol=3e-6 if dtype==np.float32 else 2e-11)

def test_hvp_ordinary_import_and_linear_strictness():
    with as_torch() as alias:
        from torch.autograd.functional import hvp
        assert hvp is alias.autograd.functional.hvp
        output,value=hvp(lambda a:a.sum(),alias.ones(2),alias.ones(2))
        assert output.item()==2 and value.tolist()==[0.,0.]
        with pytest.raises(RuntimeError):hvp(lambda a:a.sum(),alias.ones(2),alias.ones(2),strict=True)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('kind',['scalar_zero','singular_two','dense_two'])
@pytest.mark.parametrize('squared',[False,True])
def test_hvp_small_determinants_have_polynomial_curvature(dtype,kind,squared):
    data=np.array([[0.]],dtype=dtype) if kind=='scalar_zero' else np.array([[0.,0.],[0.,2.]] if kind=='singular_two' else [[2.,.3],[.4,1.7]],dtype=dtype)
    direction=np.full_like(data,.3) if data.size==1 else np.array([[.6,.2],[.3,.4]],dtype=dtype)
    if data.size==1:expected=2*direction if squared else np.zeros_like(data)
    else:
        a,b,c,d=data.ravel();va,vb,vc,vd=direction.ravel()
        determinant=a*d-b*c;cofactor=np.array([[d,-c],[-b,a]],dtype=dtype)
        second=np.array([[vd,-vc],[-vb,va]],dtype=dtype)
        first=(cofactor*direction).sum()
        expected=2*first*cofactor+2*determinant*second if squared else second
    _,actual=torch.autograd.functional.hvp(lambda x:torch.det(x).square() if squared else torch.det(x),torch.tensor(data),torch.tensor(direction))
    np.testing.assert_allclose(actual.numpy(),expected,rtol=2e-6,atol=2e-7)

@pytest.mark.parametrize('mode',['graph','seed','implicit','complex'])
def test_hvp_unsupported_arguments_reject_before_callback(mode):
    calls=[];x=torch.ones(2);v=torch.ones(2);kwargs={}
    if mode=='graph':kwargs['create_graph']=True
    elif mode=='seed':v=torch.ones(3)
    elif mode=='implicit':v=None
    else:x=torch.tensor([1+2j],dtype=torch.complex64);v=torch.ones_like(x)
    with pytest.raises(RuntimeError):torch.autograd.functional.hvp(lambda a:calls.append(a),x,v,**kwargs)
    assert not calls

@pytest.mark.parametrize('mode',['detach','force','no_grad'])
def test_hvp_explicit_numpy_disconnection_and_strict_mode(mode):
    def function(a):
        if mode=='detach':data=a.detach().numpy()
        elif mode=='force':data=a.numpy(force=True)
        else:
            with torch.no_grad():data=a.numpy()
        return torch.from_numpy(np.sin(data)).sum()
    x=torch.tensor([.2,.7]);v=torch.ones_like(x)
    output,value=torch.autograd.functional.hvp(function,x,v)
    np.testing.assert_allclose(output.numpy(),np.sin([.2,.7]).sum(),rtol=1e-6)
    assert value.tolist()==[0.,0.]
    with pytest.raises(RuntimeError):torch.autograd.functional.hvp(function,x,v,strict=True)

@pytest.mark.parametrize('conversion',[lambda a:a.numpy(),np.asarray])
def test_hvp_implicit_numpy_escape_restores_all_transform_state(conversion):
    x=torch.ones(2,requires_grad=True)
    with pytest.raises(RuntimeError):torch.autograd.functional.hvp(lambda a:torch.from_numpy(conversion(a)).sum(),x,torch.ones_like(x))
    assert x.numpy().tolist()==[1.,1.]
    output,gradient=torch.autograd.functional.vjp(lambda a:a.square().sum(),x)
    assert output.item()==2 and gradient.tolist()==[2.,2.]
    with pytest.raises(RuntimeError):torch.func.jvp(lambda a:torch.from_numpy(a.numpy(force=True)),(x,),(torch.ones_like(x),))
