"""Native average-pooling values, IEEE classes, linear tapes and CNN states."""
import numpy as np
import pytest
import torch
from torchwasm import torch as lite


@pytest.fixture(params=[(a,b) for a in ('scan','packed','auto') for b in ('loop','scatter')]+[('matrix','packed'),('block','block')],autouse=True)
def algorithms(request,monkeypatch):
    from torchwasm import _average_pooling
    monkeypatch.setattr(_average_pooling,'_ALGORITHM',request.param[0]);monkeypatch.setattr(_average_pooling,'_BACKWARD_ALGORITHM',request.param[1])


def data(x):return x.detach().numpy()


def close(a,b,dtype=np.float64):
    aa,bb=data(a),data(b)
    np.testing.assert_allclose(aa,bb,rtol=4e-5 if dtype==np.float32 else 2e-11,atol=3e-6 if dtype==np.float32 else 2e-12,equal_nan=True)
    np.testing.assert_array_equal(np.isnan(aa),np.isnan(bb));np.testing.assert_array_equal(np.isinf(aa),np.isinf(bb))


ADAPTIVE=[((2,3,5,7),(3,2)),((3,5,7),(8,9)),((1,2,4,6),(2,3)),((1,2,5,7),(None,3)),((0,2,3,4),(2,2)),((2,3,5,7),(1,1)),((2,3,4,6),(4,6))]
FIXED=[dict(kernel_size=2),dict(kernel_size=3,stride=1,padding=1),dict(kernel_size=(2,3),stride=(2,2),padding=(1,1),ceil_mode=True),dict(kernel_size=3,stride=2,padding=1,ceil_mode=True,count_include_pad=False),dict(kernel_size=3,stride=2,padding=1,divisor_override=-3)]


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('case',ADAPTIVE)
@pytest.mark.parametrize('layout',['contiguous','fortran','strided'])
def test_adaptive_forward_gradient_layouts_and_repeated_tapes(dtype,case,layout):
    shape,size=case;rng=np.random.default_rng(3903);a=rng.normal(size=shape).astype(dtype)
    if layout=='fortran':a=np.asfortranarray(a)
    if layout=='strided':a=np.repeat(a,2,axis=-1)[...,::2]
    rows=[];seed=None
    for api in (lite,torch):
        x=api.from_numpy(a).requires_grad_();y=api.nn.functional.adaptive_avg_pool2d(x,size)
        if seed is None:seed=rng.normal(size=y.shape).astype(dtype)
        y.backward(api.tensor(seed),retain_graph=True);y.backward(api.tensor(seed));rows.append((y,x.grad))
    for a,b in zip(*rows):close(a,b,dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('kw',FIXED)
@pytest.mark.parametrize('shape',[(2,3,5,7),(3,5,7),(0,3,5,7)])
def test_fixed_forward_and_ordered_gradient(dtype,kw,shape):
    rng=np.random.default_rng(3904);a=rng.normal(size=shape).astype(dtype);rows=[];seed=None
    for api in (lite,torch):
        x=api.from_numpy(a).requires_grad_();y=api.nn.functional.avg_pool2d(x,**kw)
        if seed is None:seed=rng.normal(size=y.shape).astype(dtype)
        y.backward(api.tensor(seed));rows.append((y,x.grad))
    for a,b in zip(*rows):close(a,b,dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf,'large',-0.])
@pytest.mark.parametrize('size',[(1,1),(2,2),(4,4)])
def test_ieee_locality_overflow_and_zero_signs(dtype,value,size):
    a=np.ones((1,1,4,4),dtype=dtype)
    if value=='large':a[:]=1e38 if dtype==np.float32 else 1e308
    elif value==0:a[:]=-0.
    else:a[...,0,0]=value
    rows=[api.nn.functional.adaptive_avg_pool2d(api.from_numpy(a),size) for api in (lite,torch)]
    close(*rows,dtype);np.testing.assert_array_equal(np.signbit(data(rows[0])),np.signbit(data(rows[1])))


@pytest.mark.parametrize('name',['adaptive','fixed','area'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('kind',['finite','nonfinite','zero'])
def test_jvp_and_reverse_over_forward(name,dtype,kind):
    rng=np.random.default_rng(3905);a=rng.normal(size=(1,2,5,7)).astype(dtype);v=rng.normal(size=a.shape).astype(dtype)
    if kind=='nonfinite':v[...,0,0]=np.inf;v[...,2,3]=np.nan
    elif kind=='zero':a[:]=-0.;v[:]=-0.
    rows=[]
    for api in (lite,torch):
        x=api.tensor(a,requires_grad=True);direction=api.tensor(v,requires_grad=True)
        if name=='adaptive':fn=lambda z:api.nn.functional.adaptive_avg_pool2d(z,(3,2))
        elif name=='area':fn=lambda z:api.nn.functional.interpolate(z,size=(3,2),mode='area')
        else:fn=lambda z:api.nn.functional.avg_pool2d(z,3,2,1,ceil_mode=True)
        p,t=api.func.jvp(fn,(x,),(direction,));t.sum().backward();assert x.grad is None
        rows.append((p,t,direction.grad))
    for a,b in zip(*rows):close(a,b,dtype)


@pytest.mark.parametrize('name',['adaptive','fixed'])
@pytest.mark.parametrize('target',['input','output','tangent','tangent_input','jvp_input'])
@pytest.mark.parametrize('global_size',[False,True])
def test_native_storage_versions_and_output_edits(name,target,global_size):
    outcomes=[]
    for api in (lite,torch):
        x=api.ones(1,1,4,4,dtype=api.float64,requires_grad=True);v=api.ones_like(x,requires_grad=True)
        fn=(lambda z:api.nn.functional.adaptive_avg_pool2d(z,1 if global_size else 2)) if name=='adaptive' else (lambda z:api.nn.functional.avg_pool2d(z,4 if global_size else 2))
        if target in ['tangent','tangent_input','jvp_input']:p,t=api.func.jvp(fn,(x,),(v,));y=t
        else:p=fn(x);y=p
        with api.no_grad():{'input':x,'jvp_input':x,'output':p,'tangent':t if target=='tangent' else p,'tangent_input':v}[target].detach().add_(1)
        try:y.sum().backward();outcomes.append('success')
        except RuntimeError:outcomes.append('RuntimeError')
    assert outcomes[0]==outcomes[1]


@pytest.mark.parametrize('shape,size',[((1,0,4,4),(2,2)),((0,2,4,4),(2,2)),((1,2,4,4),(0,2)),((1,0,4,4),(1,1))])
def test_native_empty_axis_behavior(shape,size):
    outcomes=[]
    for api in (lite,torch):
        x=api.empty(shape,requires_grad=True);y=api.nn.functional.adaptive_avg_pool2d(x,size)
        try:y.sum().backward();outcomes.append((tuple(y.shape),tuple(x.grad.shape)))
        except RuntimeError:outcomes.append((tuple(y.shape),'RuntimeError'))
    assert outcomes[0]==outcomes[1]


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64,np.int64])
def test_inference_dtypes_and_module_representations(dtype):
    a=np.arange(-12,12,dtype=dtype).reshape(1,1,4,6);rows=[]
    for api in (lite,torch):
        m=api.nn.AvgPool2d((2,3),stride=1,padding=(1,1),ceil_mode=True,count_include_pad=False,divisor_override=-3)
        rows.append((m(api.from_numpy(a)),repr(m)))
        assert not m.state_dict() and m.eval() is m
    if dtype==np.int64:np.testing.assert_array_equal(data(rows[0][0]),data(rows[1][0]))
    else:np.testing.assert_allclose(data(rows[0][0]),data(rows[1][0]),rtol=1e-3,atol=1e-3)
    assert rows[0][1]==rows[1][1]


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('head',['adaptive','fixed'])
def test_cnn_two_actual_sgd_steps(dtype,head):
    rng=np.random.default_rng(3906);a=rng.normal(size=(2,2,6,8)).astype(dtype);labels=rng.normal(size=(2,4)).astype(dtype);initial=None;rows=[]
    for api in (lite,torch):
        pool=api.nn.AdaptiveAvgPool2d((2,2)) if head=='adaptive' else api.nn.AvgPool2d((3,4))
        model=api.nn.Sequential(api.nn.Conv2d(2,3,3,padding=1),api.nn.ReLU(),pool,api.nn.Flatten(),api.nn.Linear(12,4)).to(dtype=api.float32 if dtype==np.float32 else api.float64)
        params=list(model.parameters())
        if initial is None:initial=[rng.normal(size=p.shape).astype(dtype)*.1 for p in params]
        with api.no_grad():
            for p,value in zip(params,initial):p.copy_(api.tensor(value))
        optimizer=api.optim.SGD(params,lr=.01,momentum=.6);steps=[]
        for _ in range(2):
            x=api.tensor(a,requires_grad=True);optimizer.zero_grad(set_to_none=True);y=model(x);loss=api.nn.functional.mse_loss(y,api.tensor(labels));loss.backward()
            steps.append([data(y).copy(),data(x.grad).copy(),*[data(p.grad).copy() for p in params]]);optimizer.step()
        rows.append((steps,[data(p).copy() for p in params]))
    for left,right in zip(rows[0][0],rows[1][0]):
        for a,b in zip(left,right):np.testing.assert_allclose(a,b,rtol=8e-5 if dtype==np.float32 else 2e-10,atol=2e-6 if dtype==np.float32 else 2e-12)
    for a,b in zip(rows[0][1],rows[1][1]):np.testing.assert_allclose(a,b,rtol=5e-5 if dtype==np.float32 else 2e-11,atol=2e-7 if dtype==np.float32 else 2e-12)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('seed',[3,8,13])
def test_native_upsampling_adjoint_cancellation_is_ordered(dtype,seed):
    direction=np.random.default_rng(seed).normal(size=(2,2,10,7)).astype(dtype);rows=[]
    for api in (lite,torch):
        x=api.ones(2,2,1,3,dtype=np.dtype(dtype) if api is lite else getattr(api,np.dtype(dtype).name),requires_grad=True)
        api.nn.functional.adaptive_avg_pool2d(x,(10,7)).backward(api.tensor(direction));rows.append(x.grad.numpy())
    np.testing.assert_array_equal(*rows)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_native_mixed_sign_irregular_forward_cancellation(dtype):
    a=np.tile(np.array([1e6,1.3,-1e6,1.3,0,0],dtype=dtype),(4,1)).reshape(1,1,4,6)
    rows=[api.nn.functional.adaptive_avg_pool2d(api.tensor(a),(3,2)).numpy() for api in (lite,torch)]
    np.testing.assert_array_equal(*rows)


@pytest.mark.parametrize('nonfinite',[np.inf,np.nan])
def test_native_upsampling_adjoint_nonfinite_locality(nonfinite):
    seed=np.ones((1,1,5,7));seed[...,0,0]=nonfinite;rows=[]
    for api in (lite,torch):
        x=api.ones(1,1,2,2,dtype=api.float64,requires_grad=True)
        api.nn.functional.adaptive_avg_pool2d(x,(5,7)).backward(api.tensor(seed));rows.append(x.grad.numpy())
    np.testing.assert_array_equal(*rows)
