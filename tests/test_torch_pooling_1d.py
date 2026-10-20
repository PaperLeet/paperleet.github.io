"""Native 1D max pooling: selection, metadata, linear tapes and bounded scans."""
import pickle
import numpy as np
import pytest
import torch
from torchwasm import torch as lite, UnsupportedOperation



@pytest.fixture(params=['scan', 'window', 'block', 'auto'], autouse=True)
def algorithm(request, monkeypatch):
    from torchwasm import _pooling
    monkeypatch.setattr(_pooling, '_ALGORITHM', request.param)


def data(x):return x.detach().numpy()


CASES=[((2,9),dict(kernel_size=2)),((1,2,9),dict(kernel_size=2)),
       ((0,2,9),dict(kernel_size=2)),((1,2,9),dict(kernel_size=3,stride=2,padding=1,ceil_mode=True)),
       ((1,2,9),dict(kernel_size=3,stride=1,padding=1,dilation=2)),
       ((2,9),dict(kernel_size=(2,),stride=[],ceil_mode=True)),((1,2,9),dict(kernel_size=1))]


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['contiguous','fortran','strided'])
@pytest.mark.parametrize('case',CASES)
@pytest.mark.parametrize('indices',[False,True])
def test_values_index_metadata_gradients_jvp_and_reverse_tangent(dtype,layout,case,indices):
    shape,kw=case;rng=np.random.default_rng(4201);a=rng.integers(-3,4,size=shape).astype(dtype)
    if layout=='fortran':a=np.asfortranarray(a)
    if layout=='strided':a=np.repeat(a,2,axis=-1)[...,::2]
    v=rng.normal(size=shape).astype(dtype);rows=[];seed=None
    for api in (lite,torch):
        x=api.from_numpy(a).requires_grad_();direction=api.tensor(v,requires_grad=True)
        fn=lambda z:api.nn.functional.max_pool1d(z,**kw,return_indices=indices)
        out=fn(x);y,i=out if indices else (out,None)
        if seed is None:seed=rng.normal(size=y.shape).astype(dtype)
        y.backward(api.tensor(seed),retain_graph=True);y.backward(api.tensor(seed));gradient=data(x.grad).copy();x.grad=None
        p,t=api.func.jvp(fn,(x,),(direction,))
        if indices:(p,pi),(t,ti)=p,t
        else:pi=ti=None
        t.backward(api.tensor(seed));assert x.grad is None
        rows.append(dict(arrays=[data(z) for z in (y,p,t,direction.grad)]+[gradient],index=None if i is None else data(i),
                         primal_index=None if pi is None else data(pi),tangent_index=None if ti is None else data(ti),
                         strides=y.stride(),index_strides=None if i is None else i.stride()))
    for left,right in zip(rows[0]['arrays'],rows[1]['arrays']):np.testing.assert_allclose(left,right,rtol=3e-6,atol=3e-7)
    assert rows[0]['strides']==rows[1]['strides']
    if indices:
        assert rows[0]['index_strides']==rows[1]['index_strides']
        for name in ('index','primal_index','tangent_index'):np.testing.assert_array_equal(rows[0][name],rows[1][name])


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64,np.int8,np.int16,np.int32,np.int64,np.uint8])
@pytest.mark.parametrize('indices',[False,True])
def test_native_inference_dtype_boundaries(dtype,indices):
    a=np.arange(18,dtype=dtype).reshape(1,2,9)
    if np.dtype(dtype).kind!='f' and not indices:
        with pytest.raises(NotImplementedError):torch.nn.functional.max_pool1d(torch.from_numpy(a),2)
        with pytest.raises(UnsupportedOperation):lite.nn.functional.max_pool1d(lite.from_numpy(a),2)
        return
    rows=[api.nn.functional.max_pool1d(api.from_numpy(a),2,return_indices=indices) for api in (lite,torch)]
    if not indices:rows=[(row,) for row in rows]
    for aa,bb in zip(*rows):np.testing.assert_array_equal(data(aa),data(bb))


@pytest.mark.parametrize('values',[[0.,-0.,-0.,0.],[-0.,0.,0.,-0.],[-np.inf]*4,[np.nan,1.,np.nan,0.],[np.inf,np.nan,np.inf,np.nan]])
@pytest.mark.parametrize('indices',[False,True])
def test_ieee_bitwise_selection(values,indices):
    a=np.asarray(values,np.float32).reshape(1,1,4)
    if np.isnan(a).any():a.view(np.uint32)[np.isnan(a)]=np.array([0x7fc00001,0x7fc00005],np.uint32)
    rows=[]
    for api in (lite,torch):
        result=api.nn.functional.max_pool1d(api.from_numpy(a),4,return_indices=indices)
        y,i=result if indices else (result,None)
        rows.append((data(y).view(np.uint32),None if i is None else data(i)))
    np.testing.assert_array_equal(rows[0][0],rows[1][0])
    if indices:np.testing.assert_array_equal(rows[0][1],rows[1][1])


@pytest.mark.parametrize('target',['input','indices','output','tangent','primal_output','tangent_output'])
def test_native_saved_versions(target):
    rows=[]
    for api in (lite,torch):
        x=api.arange(9,dtype=api.float64).reshape(1,1,9).requires_grad_();v=api.ones_like(x,requires_grad=True)
        if target in ('input','indices','output'):p,i=api.nn.functional.max_pool1d(x,3,1,return_indices=True);t=p
        else:(p,i),(t,ti)=api.func.jvp(lambda z:api.nn.functional.max_pool1d(z,3,1,return_indices=True),(x,),(v,))
        with api.no_grad():{'input':x,'indices':i,'output':p,'tangent':v,'primal_output':p,'tangent_output':t}[target].detach().add_(1)
        try:t.sum().backward();rows.append('success')
        except RuntimeError:rows.append('RuntimeError')
    assert rows[0]==rows[1]


def test_module_pickle_and_root_aliases():
    x=lite.arange(18,dtype=lite.float64).reshape(1,2,9)
    model=lite.nn.MaxPool1d(3,2,1,return_indices=True)
    assert not isinstance(model,lite.nn.MaxPool2d)
    assert str(model)==str(torch.nn.MaxPool1d(3,2,1,return_indices=True))
    for aa,bb in zip(model(x),pickle.loads(pickle.dumps(model))(x)):np.testing.assert_array_equal(data(aa),data(bb))
    for aa,bb in zip(model(x),lite.max_pool1d_with_indices(x,3,2,1)):np.testing.assert_array_equal(data(aa),data(bb))
    np.testing.assert_array_equal(data(lite.max_pool1d(x,3,2,1)),data(model(x)[0]))


@pytest.mark.parametrize('shape',[(0,2,3,9),(2,3,9),(1,0,3,9)])
def test_two_dimensional_pair_block_or_empty_metadata(shape):
    rng=np.random.default_rng(4202);a=rng.integers(-2,3,size=shape).astype(np.float64)
    if shape[-3]==0:
        for api in (lite,torch):
            with pytest.raises(RuntimeError):api.nn.functional.max_pool2d(api.from_numpy(a),(1,2))
        return
    rows=[api.nn.functional.max_pool2d(api.from_numpy(a),(1,2),return_indices=True) for api in (lite,torch)]
    for aa,bb in zip(*rows):
        np.testing.assert_array_equal(data(aa),data(bb));assert aa.stride()==bb.stride()


@pytest.mark.parametrize('target',['input','output','tangent','primal_output','tangent_output'])
def test_without_public_indices_saved_versions(target):
    outcomes=[]
    for api in (lite,torch):
        x=api.arange(9,dtype=api.float64).reshape(1,1,9).requires_grad_();v=api.ones_like(x,requires_grad=True)
        if target in ('input','output'):p=api.nn.functional.max_pool1d(x,3,1);t=p
        else:p,t=api.func.jvp(lambda z:api.nn.functional.max_pool1d(z,3,1),(x,),(v,))
        with api.no_grad():{'input':x,'output':p,'tangent':v,'primal_output':p,'tangent_output':t}[target].detach().add_(1)
        try:t.sum().backward();outcomes.append('success')
        except RuntimeError:outcomes.append('RuntimeError')
    assert outcomes[0]==outcomes[1]


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_native_convolution_maxpool_training_checkpoint_and_jvp(dtype):
    rng=np.random.default_rng(4206);a=rng.normal(size=(2,2,17)).astype(dtype);wanted=rng.normal(size=(2,2)).astype(dtype)
    initial=None;rows=[]
    for api in (lite,torch):
        def make():
            return api.nn.Sequential(api.nn.Conv1d(2,3,3,padding=1),api.nn.SiLU(),api.nn.MaxPool1d(2),
                                     api.nn.AdaptiveAvgPool1d(3),api.nn.Flatten(),api.nn.Linear(9,2)).to(dtype=getattr(api,np.dtype(dtype).name))
        model=make()
        if initial is None:initial={name:rng.normal(scale=.1,size=value.shape).astype(dtype) for name,value in model.state_dict().items()}
        model.load_state_dict({name:api.tensor(value) for name,value in initial.items()})
        opt=api.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.01)
        x=api.tensor(a,requires_grad=True);target=api.tensor(wanted);history=[]
        for _ in range(2):
            opt.zero_grad();x.grad=None;y=model(x);loss=(y-target).square().mean();loss.backward()
            history.extend([data(y).copy(),data(loss).copy(),data(x.grad).copy(),*[data(p.grad).copy() for p in model.parameters()]])
            opt.step()
        saved=pickle.loads(pickle.dumps((model.state_dict(),opt.state_dict())))
        restored=make();restored.load_state_dict(saved[0]);resumed=api.optim.SGD(restored.parameters(),lr=.01,momentum=.9,weight_decay=.01);resumed.load_state_dict(saved[1])
        resumed.zero_grad();(restored(x)-target).square().mean().backward();resumed.step()
        p,t=api.func.jvp(restored,(x,),(api.ones_like(x),))
        rows.append(history+[data(p),data(t),*[data(p) for p in restored.parameters()]])
    for aa,bb in zip(*rows):np.testing.assert_allclose(aa,bb,rtol=8e-5 if dtype==np.float32 else 2e-10,atol=3e-6 if dtype==np.float32 else 2e-11)


@pytest.mark.parametrize('kw',[
    dict(kernel_size=None),dict(kernel_size=0),dict(kernel_size=-1),dict(kernel_size=True),dict(kernel_size=2.),
    dict(kernel_size=[]),dict(kernel_size=[2,3]),dict(kernel_size=[True]),dict(kernel_size=2**63),dict(kernel_size=(np.int64(2),)),
    *[dict(kernel_size=3,stride=value) for value in ([],(),0,-1,True,2.,[2,3],2**63)],
    *[dict(kernel_size=3,padding=value) for value in (None,-1,2,True,1.5,[1,2],2**63)],
    *[dict(kernel_size=3,dilation=value) for value in (None,0,-1,True,1.5,[],[1,2],2**63)],
    *[dict(kernel_size=3,ceil_mode=value) for value in (1,None,np.bool_(True))],
    *[dict(kernel_size=3,return_indices=value) for value in (None,0,1,'yes')],
    dict(kernel_size=2,unexpected=True),
])
def test_native_parameter_outcomes(kw):
    rows=[]
    for api in (lite,torch):
        try:
            value=api.nn.functional.max_pool1d(api.ones(1,2,9),**kw)
            if isinstance(value,tuple):rows.append(('tuple',*[tuple(v.shape) for v in value]))
            else:rows.append(('value',tuple(value.shape)))
        except Exception as error:rows.append(('error',type(error).__name__))
    assert rows[0]==rows[1]


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('dimensions',[1,2])
def test_seeded_random_geometry_selection_and_linear_tapes(dtype,dimensions):
    rng=np.random.default_rng(4210+dimensions)
    for _ in range(90):
        spatial=tuple(int(v) for v in rng.integers(1,10,size=dimensions));channels=int(rng.integers(1,4))
        shape=(channels,*spatial) if rng.random()<.5 else (int(rng.integers(0,3)),channels,*spatial)
        kernel=tuple(int(v) for v in rng.integers(1,6,size=dimensions));step=tuple(int(v) for v in rng.integers(1,5,size=dimensions))
        pad=tuple(int(rng.integers(0,k//2+1)) for k in kernel);dilation=tuple(int(v) for v in rng.integers(1,4,size=dimensions));ceil=bool(rng.integers(0,2))
        out=[]
        for n,k,s,p,d in zip(spatial,kernel,step,pad,dilation):
            count=(n+2*p-d*(k-1)-1+(s-1 if ceil else 0))//s+1
            if ceil and (count-1)*s>=n+p:count-=1
            out.append(count)
        args=(kernel[0],step[0],pad[0],dilation[0],ceil) if dimensions==1 else (kernel,step,pad,dilation,ceil)
        name='max_pool'+str(dimensions)+'d'
        if min(out)<=0:
            for api in (lite,torch):
                with pytest.raises(RuntimeError):getattr(api.nn.functional,name)(api.ones(shape),*args)
            continue
        a=rng.integers(-3,4,size=shape).astype(dtype)
        if rng.random()<.5:a=np.asfortranarray(a)
        if rng.random()<.5:a=np.repeat(a,2,axis=-1)[...,::2]
        direction=rng.normal(size=shape).astype(dtype);seed=rng.normal(size=(*shape[:-dimensions],*out)).astype(dtype)
        no_sample=any(any(not any(0<=i*s-p+t*d<n for t in range(k)) for i in range(o)) for n,k,s,p,d,o in zip(spatial,kernel,step,pad,dilation,out))
        empty=np.prod(shape[:-dimensions])==0;rows=[]
        for api in (lite,torch):
            fn=lambda z:getattr(api.nn.functional,name)(z,*args,return_indices=True)
            y,index=fn(api.from_numpy(a));row=[data(y),data(index)]
            if no_sample and not empty:
                if api is lite:
                    with pytest.raises(UnsupportedOperation):fn(api.from_numpy(a).requires_grad_())
                rows.append(row);continue
            x=api.from_numpy(a).requires_grad_();v=api.tensor(direction,requires_grad=True)
            y,index=fn(x);y.backward(api.tensor(seed));grad=data(x.grad).copy();x.grad=None
            (p,pi),(t,ti)=api.func.jvp(fn,(x,),(v,));t.backward(api.tensor(seed));assert x.grad is None
            row.extend([grad,data(p),data(pi),data(t),data(ti),data(v.grad)]);rows.append(row)
        for aa,bb in zip(*rows):np.testing.assert_allclose(aa,bb,rtol=3e-5,atol=3e-6,equal_nan=True)


@pytest.mark.parametrize('dimensions',[1,2])
@pytest.mark.parametrize('kernel',[1024,1_000_000])
def test_native_huge_padding_values_indices_and_vjp(dimensions,kernel):
    rows=[]
    for api in (lite,torch):
        shape=(1,1,7) if dimensions==1 else (1,1,1,7)
        x=api.arange(7,dtype=api.float64).reshape(shape).requires_grad_()
        args=(kernel,1,kernel//2) if dimensions==1 else ((1,kernel),(1,1),(0,kernel//2))
        y,i=getattr(api.nn.functional,'max_pool'+str(dimensions)+'d')(x,*args,return_indices=True)
        y.sum().backward();rows.append((data(y),data(i),data(x.grad)))
    for aa,bb in zip(*rows):np.testing.assert_array_equal(aa,bb)
