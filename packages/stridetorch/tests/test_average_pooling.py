"""Independent region-local averaging, linear tapes and Module contracts."""
import pickle
import gc
import weakref
import numpy as np
import pytest
from stridetorch import UnsupportedOperation, as_torch, torch


@pytest.fixture(params=[(a,b) for a in ('scan','packed','auto') for b in ('loop','scatter')]+[('matrix','packed'),('block','block')], autouse=True)
def algorithm(request,monkeypatch):
    from stridetorch import _average_pooling
    monkeypatch.setattr(_average_pooling,'_ALGORITHM',request.param[0])
    monkeypatch.setattr(_average_pooling,'_BACKWARD_ALGORITHM',request.param[1])


def operation(name,x,size=(2,2)):
    if name=='adaptive':return torch.nn.functional.adaptive_avg_pool2d(x,size)
    if name=='area':return torch.nn.functional.interpolate(x,size=size,mode='area')
    return torch.nn.functional.avg_pool2d(x,2)


@pytest.mark.parametrize('name',['adaptive','area','fixed'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_known_blocks_values_routes_and_tangent(name,dtype):
    x=torch.arange(16,dtype=np.dtype(dtype)).reshape(1,1,4,4).requires_grad_()
    y=operation(name,x)
    np.testing.assert_allclose(y.numpy(),[[[[2.5,4.5],[10.5,12.5]]]])
    y.backward(torch.tensor([[[[1.,2.],[3.,4.]]]],dtype=np.dtype(dtype)),retain_graph=True)
    np.testing.assert_allclose(x.grad.numpy(),[[[[.25,.25,.5,.5],[.25,.25,.5,.5],[.75,.75,1.,1.],[.75,.75,1.,1.]]]])
    v=torch.ones_like(x,requires_grad=True)
    p,t=torch.func.jvp(lambda z:operation(name,z),(x,),(v,))
    np.testing.assert_allclose(p.numpy(),y.numpy());np.testing.assert_array_equal(t.numpy(),1)
    old=x.grad;t.sum().backward();assert x.grad is old
    np.testing.assert_array_equal(v.grad.numpy(),.25)


@pytest.mark.parametrize('value',[np.inf,np.nan])
@pytest.mark.parametrize('name',['adaptive','area','fixed'])
def test_nonfinite_values_stay_inside_their_region(name,value):
    a=np.ones((1,1,4,4));a[...,0,0]=value
    y=operation(name,torch.tensor(a))
    np.testing.assert_allclose(y.numpy(),[[[[value,1.],[1.,1.]]]],equal_nan=True)
    x=torch.tensor(a,requires_grad=True)
    y=operation(name,x);y.backward(torch.tensor([[[[np.inf,1.],[1.,1.]]]]))
    expected=np.full((1,1,4,4),.25);expected[...,:2,:2]=np.inf
    np.testing.assert_array_equal(x.grad.numpy(),expected)


@pytest.mark.parametrize('dtype,value',[(np.float32,1e38),(np.float64,1e308)])
def test_sum_precedes_scaling_and_identity_preserves_local_nonfinites(dtype,value):
    x=torch.full((1,1,4,4),value,dtype=np.dtype(dtype))
    np.testing.assert_array_equal(torch.nn.functional.adaptive_avg_pool2d(x,2).numpy(),np.inf)
    np.testing.assert_array_equal(torch.nn.functional.adaptive_avg_pool2d(x,1).numpy(),np.inf)
    np.testing.assert_array_equal(torch.nn.functional.adaptive_avg_pool2d(x,(4,4)).numpy(),x.numpy())


@pytest.mark.parametrize('include',[False,True])
@pytest.mark.parametrize('divisor',[None,2,-2])
def test_padding_ceil_and_divisor_are_distinct(include,divisor):
    x=torch.ones(1,1,4,4,dtype=torch.float64)
    y=torch.nn.functional.avg_pool2d(x,3,2,1,ceil_mode=True,count_include_pad=include,divisor_override=divisor)
    real=np.asarray([2.,3.,1.]);padding=np.asarray([3.,3.,2.])
    denominator=divisor if divisor is not None else padding[:,None]*padding if include else real[:,None]*real
    np.testing.assert_allclose(y.numpy()[0,0],real[:,None]*real/denominator)


def test_nonfinite_seed_does_not_poison_unrelated_windows():
    x=torch.ones(1,1,5,5,requires_grad=True);y=torch.nn.functional.avg_pool2d(x,3,1,1)
    seed=np.zeros(y.shape,dtype=np.float32);seed[...,2,2]=np.nan;y.backward(torch.tensor(seed))
    expected=np.zeros(x.shape,dtype=np.float32);expected[...,1:4,1:4]=np.nan
    np.testing.assert_allclose(x.grad.numpy(),expected,equal_nan=True)


@pytest.mark.parametrize('size',[(1,1),(2,2),(4,4)])
@pytest.mark.parametrize('target',['input','output','tangent'])
def test_selective_versions_and_global_stateless_path(size,target):
    x=torch.ones(1,1,4,4,requires_grad=True);v=torch.ones_like(x,requires_grad=True)
    if target=='tangent':p,y=torch.func.jvp(lambda z:torch.nn.functional.adaptive_avg_pool2d(z,size),(x,),(v,))
    else:y=torch.nn.functional.adaptive_avg_pool2d(x,size)
    with torch.no_grad():{'input':x,'output':y,'tangent':v}[target].detach().add_(1)
    if target!='output' and size!=(1,1):
        with pytest.raises(RuntimeError):y.sum().backward()
    else:y.sum().backward()


def test_zero_signs_empty_output_and_empty_channels():
    x=torch.full((1,1,2,2),-0.,dtype=torch.float64,requires_grad=True)
    y=torch.nn.functional.adaptive_avg_pool2d(x,(2,2));assert not np.signbit(y.numpy()).any()
    y.backward(torch.full_like(y,-0.));assert not np.signbit(x.grad.numpy()).any()
    for shape,size,backward_error in [((0,2,4,4),(2,2),False),((1,0,4,4),(2,2),True),((1,2,4,4),(0,2),True)]:
        x=torch.empty(shape,requires_grad=True);y=torch.nn.functional.adaptive_avg_pool2d(x,size)
        assert y.shape==(*shape[:-2],*size)
        if backward_error:
            with pytest.raises(RuntimeError):y.sum().backward()
        else:y.sum().backward();assert x.grad.shape==x.shape


def test_half_and_signed_integer_inference():
    x=torch.full((1,1,4,4),65504.,dtype=torch.float16)
    assert torch.nn.functional.adaptive_avg_pool2d(x,2).dtype==torch.float16
    np.testing.assert_array_equal(torch.nn.functional.adaptive_avg_pool2d(x,2).numpy(),65504.)
    a=torch.tensor([[[[-4,-3],[-2,0]]]],dtype=torch.int64)
    assert torch.nn.functional.avg_pool2d(a,2).item()==-2
    assert torch.nn.functional.avg_pool2d(a,2,divisor_override=-2).item()==4


def test_module_import_pickle_hooks_and_geometry_snapshot():
    with as_torch():
        from torch.nn.modules.pooling import AvgPool2d,AdaptiveAvgPool2d
        size=[2,2];model=AdaptiveAvgPool2d(size);x=torch.ones(1,1,4,4,requires_grad=True);y=model(x)
        size[:]=[3,3];y.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),.25)
        restored=pickle.loads(pickle.dumps(model));assert restored(x).shape==(1,1,3,3)
        assert not model.state_dict() and model.eval() is model
        assert AvgPool2d(2)(x).shape==(1,1,2,2)


def test_bounded_fallback_and_bicubic_forward_mode_boundary(monkeypatch):
    from stridetorch import _average_pooling
    monkeypatch.setattr(_average_pooling,'_WORKSPACE_BYTES',1)
    x=torch.ones(1,1,6,8,requires_grad=True);y=torch.nn.functional.adaptive_avg_pool2d(x,(3,4));y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(),.25)
    with pytest.raises(UnsupportedOperation):torch.func.jvp(lambda z:torch.nn.functional.interpolate(z,size=4,mode='bicubic'),(x,),(torch.ones_like(x),))


@pytest.mark.parametrize('dtype,large',[(np.float32,1e30),(np.float64,1e100)])
def test_ill_conditioned_finite_regions_keep_ordered_accumulation(dtype,large):
    block=np.asarray([[large,1.],[-large,1.]],dtype=dtype)
    x=torch.tensor(np.tile(block,(2,2)).reshape(1,1,4,4))
    np.testing.assert_array_equal(torch.nn.functional.adaptive_avg_pool2d(x,2).numpy(),.25)


def test_geometry_caches_are_bounded_and_do_not_own_tensors():
    from stridetorch import _average_pooling as pooling
    for n in range(70):torch.nn.functional.adaptive_avg_pool2d(torch.ones(1,1,5,7,requires_grad=True),(2+n%9,2+n//9)).sum().backward()
    assert pooling._cached_geometry.cache_info().currsize<=64
    assert pooling._binary_matrices.cache_info().currsize<=32
    assert pooling._cached_adjoint_geometry.cache_info().currsize<=64
    x=torch.ones(1,1,5,7,requires_grad=True);ref=weakref.ref(x)
    y=torch.nn.functional.adaptive_avg_pool2d(x,(2,3));y.sum().backward()
    del x,y;gc.collect();assert ref() is None


def test_cached_plans_recheck_live_numpy_values_and_gradient_seeds():
    a=np.ones((1,1,4,4));x=torch.from_numpy(a).requires_grad_()
    y=torch.nn.functional.adaptive_avg_pool2d(x,2)
    a[...,0,0]=np.inf
    np.testing.assert_array_equal(torch.nn.functional.adaptive_avg_pool2d(x,2).numpy(),[[[[np.inf,1.],[1.,1.]]]])
    raw=np.ones((1,1,2,2));seed=torch.from_numpy(raw)
    y.backward(seed,retain_graph=True);x.grad=None;raw[...,0,0]=np.inf;y.backward(seed)
    expected=np.full((1,1,4,4),.25);expected[...,:2,:2]=np.inf
    np.testing.assert_array_equal(x.grad.numpy(),expected)


def test_tiled_packing_uses_absolute_region_coordinates(monkeypatch):
    from stridetorch import _average_pooling as pooling
    x=torch.arange(640,dtype=torch.float64).reshape(1,2,16,20).requires_grad_()
    expected=torch.nn.functional.adaptive_avg_pool2d(x,(5,7)).numpy().copy()
    monkeypatch.setattr(pooling,'_WORKSPACE_BYTES',2048)
    value=torch.nn.functional.adaptive_avg_pool2d(x,(5,7))
    np.testing.assert_allclose(value.numpy(),expected,rtol=1e-12,atol=1e-12)
    value.sum().backward();assert np.isfinite(x.grad.numpy()).all()


def test_streaming_large_region_keeps_prefix_order_and_zero_signs(monkeypatch):
    from stridetorch import _average_pooling as pooling
    monkeypatch.setattr(pooling,'_WORKSPACE_BYTES',256)
    x=torch.ones(1,1,32,64,dtype=torch.float64,requires_grad=True)
    value=torch.nn.functional.adaptive_avg_pool2d(x,(1,2))
    np.testing.assert_array_equal(value.numpy(),1)
    value.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),1/1024)


@pytest.mark.parametrize('nonfinite',[np.inf,np.nan])
def test_chunked_finite_guard_checks_late_input_regions(monkeypatch,nonfinite):
    from stridetorch import _average_pooling as pooling
    monkeypatch.setattr(pooling,'_WORKSPACE_BYTES',4096)
    a=np.ones((1,1,64,64));a[...,40,40]=nonfinite
    value=torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(a),2)
    np.testing.assert_allclose(value.numpy(),[[[[1.,1.],[1.,nonfinite]]]],equal_nan=True)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('seed',[3,8,13])
def test_upsampling_adjoint_preserves_output_order_near_cancellation(dtype,seed):
    direction=np.random.default_rng(seed).normal(size=(2,2,10,7)).astype(dtype)
    x=torch.ones(2,2,1,3,dtype=np.dtype(dtype),requires_grad=True)
    torch.nn.functional.adaptive_avg_pool2d(x,(10,7)).backward(torch.tensor(direction))
    expected=np.zeros(x.shape,dtype=dtype)
    for y in range(10):
        for z in range(7):
            start=z*3//7;end=((z+1)*3+6)//7
            delta=np.divide(direction[...,y,z],dtype(end-start))
            expected[...,0,start:end]+=delta[...,None]
    np.testing.assert_array_equal(x.grad.numpy(),expected)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_mixed_sign_irregular_regions_preserve_small_residual(dtype):
    a=np.tile(np.array([1e6,1.3,-1e6,1.3,0,0],dtype=dtype),(4,1)).reshape(1,1,4,6)
    value=torch.nn.functional.adaptive_avg_pool2d(torch.tensor(a),(3,2))
    expected=np.empty((1,1,3,2),dtype=dtype)
    for y in range(3):
        first=y*4//3;last=((y+1)*4+2)//3
        for z in range(2):
            total=dtype(0)
            for iy in range(first,last):
                for ix in range(z*3,(z+1)*3):total=dtype(total+a[0,0,iy,ix])
            expected[...,y,z]=dtype(dtype(total/dtype(last-first))/dtype(3))
    np.testing.assert_array_equal(value.numpy(),expected)


@pytest.mark.parametrize('nonfinite',[np.inf,np.nan])
def test_upsampling_adjoint_nonfinite_seed_stays_in_its_input_region(nonfinite):
    x=torch.ones(1,1,2,2,dtype=torch.float64,requires_grad=True)
    seed=np.ones((1,1,5,7));seed[...,0,0]=nonfinite
    torch.nn.functional.adaptive_avg_pool2d(x,(5,7)).backward(torch.tensor(seed))
    expected=np.zeros(x.shape)
    for y in range(5):
        start_y=y*2//5;end_y=((y+1)*2+4)//5
        for z in range(7):
            start_x=z*2//7;end_x=((z+1)*2+6)//7
            delta=seed[...,y,z]/(end_y-start_y)/(end_x-start_x)
            expected[...,start_y:end_y,start_x:end_x]+=delta[...,None,None]
    np.testing.assert_array_equal(x.grad.numpy(),expected)


def test_inverse_gather_checks_workspace_before_constructing_indices(monkeypatch):
    from stridetorch import _average_pooling as pooling
    x=torch.ones(1,1,16,32,dtype=torch.float64,requires_grad=True)
    y=torch.nn.functional.adaptive_avg_pool2d(x,(3,5))
    def forbidden(*args):raise AssertionError('Oversized inverse geometry must not be constructed')
    monkeypatch.setattr(pooling,'_cached_adjoint_geometry',forbidden)
    monkeypatch.setattr(pooling,'_WORKSPACE_BYTES',128)
    y.sum().backward()
    expected=np.zeros(x.shape)
    for oy in range(3):
        sy=oy*16//3;ey=((oy+1)*16+2)//3
        for ox in range(5):
            sx=ox*32//5;ex=((ox+1)*32+4)//5
            expected[...,sy:ey,sx:ex]+=1./(ey-sy)/(ex-sx)
    np.testing.assert_array_equal(x.grad.numpy(),expected)
