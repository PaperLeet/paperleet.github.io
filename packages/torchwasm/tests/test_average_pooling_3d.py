"""Independent scalar-region sums and ordered volume-adjoint oracles."""
import pickle
import numpy as np
import pytest
from torchwasm import torch,as_torch,UnsupportedOperation


@pytest.fixture(params=[('scan','loop'),('packed','scatter'),('block','block'),('auto','auto'),('packed','inverse')],autouse=True)
def algorithm(request,monkeypatch):
    from torchwasm import _average_pooling3d
    monkeypatch.setattr(_average_pooling3d,'_ALGORITHM',request.param[0]);monkeypatch.setattr(_average_pooling3d,'_BACKWARD_ALGORITHM',request.param[1])


def regions(shape,kind):
    axes=[]
    for axis,length in enumerate(shape[-3:]):
        if kind in ('fixed','ceil','override'):
            k=2 if kind=='fixed' else 3;s=2 if kind!='override' else 1;p=0 if kind=='fixed' else 1;ceil=kind=='ceil'
            count=(length+2*p-k+(s-1 if ceil else 0))//s+1
            if ceil and (count-1)*s>=length+p:count-=1
            rows=[]
            for i in range(count):
                begin=i*s-p;end=min(begin+k,length+p);a=max(begin,0);b=min(end,length)
                rows.append((a,b,(b-a) if kind=='ceil' else end-begin))
        else:
            count={'adaptive':(3,2,4),'upsample':(7,8,9),'identity':shape[-3:]}[kind][axis]
            rows=[(i*length//count,((i+1)*length+count-1)//count,((i+1)*length+count-1)//count-i*length//count) for i in range(count)]
        axes.append(rows)
    return axes


def operation(api,x,kind):
    if kind=='fixed':return api.nn.functional.avg_pool3d(x,2)
    if kind=='ceil':return api.nn.functional.avg_pool3d(x,3,2,1,True,False)
    if kind=='override':return api.nn.functional.avg_pool3d(x,3,1,1,False,True,-7)
    return api.nn.functional.adaptive_avg_pool3d(x,{'adaptive':(3,2,4),'upsample':(7,8,9),'identity':x.shape[-3:]}[kind])


def oracle(a,axes,kind,seed=None):
    out=tuple(map(len,axes));adaptive=kind in ('adaptive','upsample','identity');acc=np.float32 if a.dtype==np.float16 else a.dtype
    result=np.zeros(a.shape if seed is not None else (*a.shape[:-3],*out),dtype=seed.dtype if seed is not None else a.dtype)
    for position in np.ndindex(out):
        rows=[axis[i] for axis,i in zip(axes,position)];region=tuple(slice(begin,end) for begin,end,count in rows)
        if seed is None:
            samples=a[(Ellipsis,)+region].reshape(*a.shape[:-3],int(np.prod([end-begin for begin,end,count in rows])));value=np.zeros(a.shape[:-3],dtype=acc)
            for j in range(samples.shape[-1]):value=np.add(value,samples[...,j],dtype=acc)
        else:value=seed[(Ellipsis,)+position].copy()
        if adaptive:
            for begin,end,count in rows:value=np.divide(value,np.dtype(acc).type(count),dtype=acc)
        else:value=np.divide(value,np.dtype(acc).type(-7 if kind=='override' else np.prod([row[2] for row in rows])),dtype=acc)
        if seed is None:result[(Ellipsis,)+position]=value
        else:result[(Ellipsis,)+region]+=value[...,None,None,None]
    return result


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(2,4,4,6),(1,2,4,5,6),(0,2,4,5,6)])
@pytest.mark.parametrize('layout',['contiguous','fortran','strided','channels_last'])
@pytest.mark.parametrize('kind',['fixed','ceil','override','adaptive','upsample','identity'])
def test_ordered_regions_vjp_jvp_and_reverse_tangent(dtype,shape,layout,kind):
    rng=np.random.default_rng(4501);a=rng.normal(size=shape).astype(dtype)
    if layout=='fortran':a=np.asfortranarray(a)
    if layout=='strided':a=np.repeat(a,2,axis=-1)[...,::2]
    if layout=='channels_last' and a.ndim==5:a=a.transpose(0,2,3,4,1).copy().transpose(0,4,1,2,3)
    before=a.copy();axes=regions(shape,kind);expected=oracle(a,axes,kind);x=torch.from_numpy(a).requires_grad_();fn=lambda z:operation(torch,z,kind)
    y=fn(x);np.testing.assert_allclose(y.numpy(),expected,rtol=3e-6,atol=3e-7)
    steps=[];stride=1
    for n in reversed(y.shape):steps.append(stride);stride*=max(1,n)
    assert y.stride()==tuple(reversed(steps));seed=rng.normal(size=y.shape).astype(dtype)
    if layout=='fortran':seed=np.asfortranarray(seed)
    if layout=='strided':seed=np.repeat(seed,2,axis=-1)[...,::2]
    y.backward(torch.tensor(seed));np.testing.assert_allclose(x.grad.numpy(),oracle(a,axes,kind,seed),rtol=3e-6,atol=3e-7)
    x.grad=None;v=rng.normal(size=shape).astype(dtype);direction=torch.tensor(v,requires_grad=True);primal,tangent=torch.func.jvp(fn,(x,),(direction,))
    np.testing.assert_allclose(primal.numpy(),expected,rtol=3e-6,atol=3e-7);np.testing.assert_allclose(tangent.numpy(),oracle(v,axes,kind),rtol=3e-6,atol=3e-7)
    tangent.backward(torch.tensor(seed));assert x.grad is None
    np.testing.assert_allclose(direction.grad.numpy(),oracle(a,axes,kind,seed),rtol=3e-6,atol=3e-7)
    np.testing.assert_array_equal(a,before);assert not np.shares_memory(a,y.numpy())


def test_fixed_order_and_global_mean_contract_preserve_zero():
    a=np.array([-0.,-0.,1e20,-1e20,1.,2.,3.,4.],dtype=np.float32).reshape(1,1,2,2,2)
    x=torch.from_numpy(a);assert torch.nn.functional.avg_pool3d(x,2).item()==1.25
    # The global specialization follows mean. Native SIMD and NumPy reductions
    # may differ on this ill-conditioned input; their bit patterns are not an
    # API guarantee. Fixed pooling must retain its ordered spatial sum.
    np.testing.assert_array_equal(torch.nn.functional.adaptive_avg_pool3d(x,1).numpy(),x.mean((-3,-2,-1),keepdim=True).numpy())
    y=torch.nn.functional.avg_pool3d(torch.full((1,1,2,2,2),-0.),2);assert not np.signbit(y.numpy()).any()


@pytest.mark.parametrize('adaptive',[False,True])
def test_empty_batches_avoid_coordinate_grids(monkeypatch,adaptive):
    x=torch.zeros(0,1,200,200,200,requires_grad=True)
    original=np.arange
    def empty_only(*args,**kwargs):
        value=original(*args,**kwargs);assert not value.size;return value
    monkeypatch.setattr(np,'arange',empty_only)
    fn=lambda z:torch.nn.functional.adaptive_avg_pool3d(z,(200,200,200)) if adaptive else torch.nn.functional.avg_pool3d(z,1)
    y=fn(x);assert y.shape==x.shape;y.sum().backward();assert x.grad.shape==x.shape
    _,tangent=torch.func.jvp(fn,(x,),(torch.ones_like(x),));assert tangent.shape==x.shape


def test_bounded_tiling_streaming_and_source_mutation(monkeypatch):
    from torchwasm import _average_pooling3d
    monkeypatch.setattr(_average_pooling3d,'_WORKSPACE_BYTES',256)
    a=np.arange(4*5*6,dtype=np.float64).reshape(1,1,4,5,6);x=torch.from_numpy(a).requires_grad_()
    y=torch.nn.functional.avg_pool3d(x,(4,5,6));np.testing.assert_array_equal(y.numpy(),59.5);y.sum().backward();np.testing.assert_allclose(x.grad.numpy(),1/120)
    y=torch.nn.functional.avg_pool3d(x,2);x.detach().add_(1)
    with pytest.raises(RuntimeError,match='modified|version|in-place'):y.sum().backward()


@pytest.mark.parametrize('shape',[(1,0,3,4,5),(1,2,0,4,5),(0,2,0,4,5)])
def test_global_empty_and_zero_output_boundaries(shape):
    x=torch.zeros(shape,requires_grad=True);y=torch.nn.functional.adaptive_avg_pool3d(x,1)
    assert y.shape==(*shape[:-3],1,1,1)
    if y.numel():assert np.isnan(y.numpy()).all()
    y.sum().backward();assert x.grad.shape==x.shape
    with pytest.raises(RuntimeError):torch.nn.functional.adaptive_avg_pool3d(x,2)
    z=torch.ones(1,2,3,4,5,requires_grad=True);out=torch.nn.functional.adaptive_avg_pool3d(z,(0,2,2));assert out.shape==(1,2,0,2,2)
    with pytest.raises(RuntimeError):out.sum().backward()


def test_dtype_rules_and_integer_truncation():
    x=torch.arange(-32,32,dtype=torch.int64).reshape(1,1,4,4,4)
    np.testing.assert_array_equal(torch.nn.functional.avg_pool3d(x,2).numpy(),[[[[[-21,-19],[-13,-11]],[[10,12],[18,20]]]]])
    half=x.to(dtype=torch.float16)
    with pytest.raises(UnsupportedOperation):torch.nn.functional.avg_pool3d(half,2)
    with pytest.raises(UnsupportedOperation):torch.nn.functional.adaptive_avg_pool3d(x,2)
    assert torch.nn.functional.adaptive_avg_pool3d(half,2).dtype==torch.float16
    with pytest.raises(RuntimeError):torch.nn.functional.avg_pool3d(torch.ones(1,1,1,1,1),2,1,1)


@pytest.mark.parametrize('layout',['contiguous','fortran','strided','channels_last'])
def test_global_output_layout_and_stateless_backward(layout):
    a=np.arange(2*3*4*5*6,dtype=np.float64).reshape(2,3,4,5,6)
    if layout=='fortran':a=np.asfortranarray(a)
    if layout=='strided':a=np.repeat(a,2,axis=-1)[...,::2]
    if layout=='channels_last':a=a.transpose(0,2,3,4,1).copy().transpose(0,4,1,2,3)
    x=torch.from_numpy(a).requires_grad_();fn=lambda z:torch.nn.functional.adaptive_avg_pool3d(z,1)
    y=fn(x);assert y.stride()==((3,1,3,3,3) if layout=='channels_last' else (3,1,1,1,1))
    _,tangent=torch.func.jvp(fn,(x,),(torch.ones_like(x),))
    np.testing.assert_array_equal(tangent.numpy(),1.)
    x.detach().add_(1);y.sum().backward();np.testing.assert_allclose(x.grad.numpy(),1/120)


@pytest.mark.parametrize('kwargs,error',[
    ({'kernel_size':0},RuntimeError),({'kernel_size':-2},RuntimeError),
    ({'kernel_size':True},TypeError),({'kernel_size':(2,2)},RuntimeError),
    ({'kernel_size':2,'stride':0},RuntimeError),
    ({'kernel_size':2,'padding':2},RuntimeError),
    ({'kernel_size':2,'ceil_mode':1},TypeError),
    ({'kernel_size':2,'count_include_pad':1},TypeError),
    ({'kernel_size':2,'divisor_override':0},RuntimeError),
    ({'kernel_size':2,'divisor_override':True},TypeError),
    ({'kernel_size':2,'divisor_override':2**63},OverflowError),
])
def test_fixed_argument_boundaries(kwargs,error):
    with pytest.raises(error):torch.nn.functional.avg_pool3d(torch.ones(1,2,4,5,6),**kwargs)


@pytest.mark.parametrize('output,error',[(True,TypeError),((2,2),RuntimeError),((2,-1,3),RuntimeError),((2,True,3),TypeError),((2,2.5,3),TypeError)])
def test_adaptive_argument_boundaries(output,error):
    with pytest.raises(error):torch.nn.functional.adaptive_avg_pool3d(torch.ones(1,2,4,5,6),output)


@pytest.mark.parametrize('kind',['ceil','adaptive','upsample'])
@pytest.mark.parametrize('nonfinite',['input','seed'])
def test_nonfinite_regions_and_strict_numpy_error_mode(kind,nonfinite):
    a=np.arange(2*4*5*6,dtype=np.float64).reshape(1,2,4,5,6)/31
    if nonfinite=='input':a.flat[0]=np.nan;a.flat[17]=np.inf;a.flat[-1]=-np.inf
    axes=regions(a.shape,kind);shape=(*a.shape[:-3],*(len(axis) for axis in axes));seed=np.ones(shape,dtype=a.dtype)
    if nonfinite=='seed':seed.flat[0]=np.nan;seed.flat[-1]=np.inf
    with np.errstate(all='ignore'):
        expected=oracle(a,axes,kind);gradient=oracle(a,axes,kind,seed)
    with np.errstate(all='raise'):
        x=torch.from_numpy(a).requires_grad_();y=operation(torch,x,kind);y.backward(torch.from_numpy(seed))
    np.testing.assert_allclose(y.numpy(),expected,equal_nan=True)
    np.testing.assert_allclose(x.grad.numpy(),gradient,equal_nan=True)


@pytest.mark.parametrize('strided',[False,True])
def test_readonly_plan_bounds_and_live_input_values(strided):
    from torchwasm import _average_pooling3d as engine
    engine._cached_geometry.cache_clear()
    for size in range(2,42):
        geometry=engine._geometry((1,2,8,8,8),(size,2,3))
        for array in (*geometry.start,*geometry.end):assert not array.flags.writeable
        if geometry.pack is not None:
            assert sum(array.nbytes for array in geometry.pack)<=128*1024
            assert all(not array.flags.writeable for array in geometry.pack)
    assert engine._cached_geometry.cache_info().currsize==32
    a=np.ones((1,2,8,8,8),dtype=np.float64)
    if strided:a=np.repeat(a,2,axis=-1)[...,::2]
    x=torch.from_numpy(a);first=torch.nn.functional.avg_pool3d(x,3,1,1).numpy().copy()
    a[...,0,0,0]=np.nan
    second=torch.nn.functional.avg_pool3d(x,3,1,1).numpy()
    assert np.isnan(second[...,:2,:2,:2]).all()
    mask=np.ones(second.shape,dtype=bool);mask[...,:2,:2,:2]=False
    np.testing.assert_array_equal(second[mask],first[mask])


def test_inverse_cache_bounds_and_live_seed_values(monkeypatch):
    from torchwasm import _average_pooling3d as engine
    monkeypatch.setattr(engine,'_BACKWARD_ALGORITHM','inverse')
    engine._cached_inverse_geometry.cache_clear()
    spatial=(8,8,8)
    for size in range(2,42):
        g=engine._geometry((1,2,*spatial),(size,2,3))
        inverse=engine._cached_inverse_geometry(spatial,tuple(a.tobytes() for a in g.start),tuple(a.tobytes() for a in g.end),512*1024,True)
        for array in (*inverse.start,*inverse.end):assert not array.flags.writeable
        if inverse.pack is not None:
            assert sum(a.nbytes for a in inverse.pack)<=128*1024
            assert all(not a.flags.writeable for a in inverse.pack)
    assert engine._cached_inverse_geometry.cache_info().currsize==32
    g=engine._geometry((1,2,*spatial),(7,8,9));seed=np.ones((1,2,7,8,9),dtype=np.float64)
    first=engine._inverse_backward(seed,(1,2,*spatial),g)
    seed[...,0,0,0]=np.nan
    second=engine._inverse_backward(seed,(1,2,*spatial),g)
    assert np.isnan(second[...,:2,:1,:1]).all()
    assert np.isfinite(first).all() and not np.shares_memory(seed,second)


def test_inverse_rejects_large_axis_before_coordinate_allocation(monkeypatch):
    from torchwasm import _average_pooling3d as engine
    shape=(1,1,1_000_000_000,1,1)
    g=engine._Geometry((np.array([0,500_000_000]),np.array([0]),np.array([0])),(np.array([500_000_000,1_000_000_000]),np.array([1]),np.array([1])),(2,1,1),None,True,None)
    seed=np.ones((1,1,2,1,1),dtype=np.float64)
    def forbidden(*args,**kwargs):raise AssertionError('Unbounded inverse coordinate allocation')
    monkeypatch.setattr(np,'arange',forbidden)
    assert engine._inverse_backward(seed,shape,g) is None


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_inverse_gradient_order_near_cancellation(monkeypatch,dtype):
    from torchwasm import _average_pooling3d as engine
    seed=np.resize(np.array([1e20,1.,-1e20,2.,3.,-0.],dtype=dtype),(1,1,5,5,5))
    x=torch.ones(1,1,2,2,2,dtype=dtype,requires_grad=True)
    monkeypatch.setattr(engine,'_BACKWARD_ALGORITHM','scatter')
    torch.nn.functional.adaptive_avg_pool3d(x,5).backward(torch.from_numpy(seed));expected=x.grad.numpy().copy();x.grad=None
    monkeypatch.setattr(engine,'_BACKWARD_ALGORITHM','inverse')
    torch.nn.functional.adaptive_avg_pool3d(x,5).backward(torch.from_numpy(seed))
    np.testing.assert_array_equal(x.grad.numpy(),expected)


def test_aliases_modules_checkpoint_and_convolution_training():
    with as_torch() as api:
        from torch.nn.modules.pooling import AvgPool3d,AdaptiveAvgPool3d
        x=api.arange(64,dtype=api.float64).reshape(1,1,4,4,4);module=AvgPool3d(2)
        assert not isinstance(module,api.nn.AvgPool2d)
        np.testing.assert_array_equal(module(x).numpy(),pickle.loads(pickle.dumps(module))(x).numpy())
        assert AdaptiveAvgPool3d((None,2,3))(x).shape==(1,1,4,2,3)
    def make():return torch.nn.Sequential(torch.nn.Conv3d(2,3,3,padding=1),torch.nn.SiLU(),torch.nn.AvgPool3d(2),torch.nn.AdaptiveAvgPool3d((1,1,1)),torch.nn.Flatten(),torch.nn.Linear(3,2)).double()
    torch.manual_seed(4502);m=make();data=torch.randn(2,2,4,4,4,dtype=torch.float64);o=torch.optim.SGD(m.parameters(),lr=.01,momentum=.9)
    o.zero_grad();m(data).square().mean().backward();o.step();state=pickle.loads(pickle.dumps((m.state_dict(),o.state_dict())))
    restored=make();restored.load_state_dict(state[0]);resume=torch.optim.SGD(restored.parameters(),lr=.01,momentum=.9);resume.load_state_dict(state[1])
    for model,opt in ((m,o),(restored,resume)):opt.zero_grad();model(data).square().mean().backward();opt.step()
    for a,b in zip(m.parameters(),restored.parameters()):np.testing.assert_array_equal(a.numpy(),b.numpy())
