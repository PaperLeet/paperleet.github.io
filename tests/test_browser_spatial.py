"""Independent native checks for spatial/token lookup and robust-alignment APIs."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from paperleet.browser import numpy_autograd as wasm


def compare(actual,expected,dtype):
    assert actual.dtype==expected.numpy().dtype
    np.testing.assert_allclose(actual.numpy(),expected.numpy(),atol=6e-6 if dtype==np.float32 else 3e-13,rtol=6e-6 if dtype==np.float32 else 3e-13)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('align',[False,True])
@pytest.mark.parametrize('shape,size',[((2,3,3,5),(7,9)),((1,2,5,7),(2,3)),((1,1,1,4),(3,7)),((1,1,4,1),(7,3)),((1,1,5,3),(1,1)),((0,2,3,4),(5,7)),((1,1,1,2048),(2048,1))])
def test_bicubic_values_and_analytical_vjp_match_native(dtype,align,shape,size):
    rng=np.random.RandomState(73)
    data=rng.randn(*shape[:-2],shape[-1],shape[-2]).astype(dtype).swapaxes(-1,-2)
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    actual,expected=wasm.interpolate(a,size=size,mode='bicubic',align_corners=align),F.interpolate(b,size=size,mode='bicubic',align_corners=align)
    weights=rng.randn(*actual.shape).astype(dtype)
    actual.backward(wasm.Tensor(weights));expected.backward(torch.tensor(weights))
    compare(actual,expected.detach(),dtype)
    if dtype==np.float32 and shape[-2:]==(1,2048):
        # The singleton-axis backward sums 2,048 float32 contributions in a
        # different order. Check against a higher-precision native VJP on the
        # identical rounded arrays instead of reproducing accumulation error.
        precise=torch.tensor(data.astype(np.float64),requires_grad=True)
        F.interpolate(precise,size=size,mode='bicubic',align_corners=align).backward(torch.tensor(weights.astype(np.float64)))
        compare(a.grad,precise.grad.to(torch.float32),dtype)
    else: compare(a.grad,b.grad,dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('mode',['bicubic','area'])
@pytest.mark.parametrize('recompute',[None,False,True])
def test_noninteger_scale_factors_use_native_coordinate_convention(dtype,mode,recompute):
    data=np.random.RandomState(79).randn(1,2,3,7).astype(dtype)
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    kwargs=dict(scale_factor=(1.7,.6),mode=mode,recompute_scale_factor=recompute)
    if mode=='bicubic':kwargs['align_corners']=False
    actual,expected=wasm.interpolate(a,**kwargs),F.interpolate(b,**kwargs)
    actual.square().sum().backward();expected.square().sum().backward()
    compare(actual,expected.detach(),dtype);compare(a.grad,b.grad,dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape,size',[((2,3,5,7),(3,2)),((3,5,7),(8,9)),((1,2,1,1),(3,4)),((1,2,5,7),(None,3)),((0,2,3,4),1)])
def test_adaptive_area_regions_and_broadcast_vjp_match_native(dtype,shape,size):
    rng=np.random.RandomState(83);data=rng.randn(*shape).astype(dtype)
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    actual,expected=wasm.adaptive_avg_pool2d(a,size),F.adaptive_avg_pool2d(b,size)
    weights=rng.randn(*actual.shape).astype(dtype)
    actual.backward(wasm.Tensor(weights));expected.backward(torch.tensor(weights))
    compare(actual,expected.detach(),dtype);compare(a.grad,b.grad,dtype)


def test_bicubic_keeps_native_overshoot_and_replicated_edges():
    data=np.array([[[[0.,1.],[0.,1.]]]])
    actual=wasm.interpolate(wasm.Tensor(data),size=6,mode='bicubic',align_corners=False)
    expected=F.interpolate(torch.tensor(data),size=6,mode='bicubic',align_corners=False)
    compare(actual,expected,np.float64)
    assert actual.numpy().min()<0 and actual.numpy().max()>1


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('padding',[None,0,-1])
@pytest.mark.parametrize('frequency',[False,True])
def test_embedding_repeated_indices_padding_and_frequency_gradients(dtype,padding,frequency):
    rng=np.random.RandomState(89);data=rng.randn(5,3).astype(dtype)
    indices=np.array([[0,2,2],[4,0,2]],dtype=np.int64).T
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    actual=wasm.embedding(wasm.Tensor(indices),a,padding_idx=padding,scale_grad_by_freq=frequency)
    expected=F.embedding(torch.tensor(indices),b,padding_idx=padding,scale_grad_by_freq=frequency)
    weights=rng.randn(*actual.shape).astype(dtype)
    actual.backward(wasm.Tensor(weights));expected.backward(torch.tensor(weights))
    compare(actual,expected.detach(),dtype);compare(a.grad,b.grad,dtype)
    # A padding row still contributes its actual stored value to the forward.
    np.testing.assert_array_equal(actual.numpy(),data[indices])


@pytest.mark.parametrize('indexshape,width',[((),2),((2,0),3),((2,3),0)])
def test_embedding_scalar_empty_and_zero_width(indexshape,width):
    indices=np.zeros(indexshape,dtype=np.int32);data=np.arange(5*width,dtype=np.float64).reshape(5,width)
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    actual,expected=wasm.embedding(wasm.Tensor(indices),a),F.embedding(torch.tensor(indices),b)
    actual.sum().backward();expected.sum().backward()
    compare(actual,expected.detach(),np.float64);compare(a.grad,b.grad,np.float64)


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int32,np.int64,np.bool_])
@pytest.mark.parametrize('dim',[None,0,1])
def test_unique_values_inverse_counts_and_unsorted_cpu_contract(dtype,dim):
    data=np.array([[3,1,3],[3,1,3],[2,2,1]],dtype=dtype).T
    for sorted in (False,True):
        actual=wasm.unique(wasm.Tensor(data),sorted=sorted,return_inverse=True,return_counts=True,dim=dim)
        expected=torch.unique(torch.tensor(data),sorted=sorted,return_inverse=True,return_counts=True,dim=dim)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a.numpy(),b.numpy())
        assert all(x.dtype==np.dtype('int64') for x in actual[1:])


def test_unique_nan_zero_empty_scalar_and_no_grad():
    for data in (np.array([np.nan,0.,-0.,np.nan,np.inf,-np.inf]),np.array([]),np.array(2.)):
        actual=wasm.unique(wasm.Tensor(data),return_inverse=True,return_counts=True)
        expected=torch.unique(torch.tensor(data),return_inverse=True,return_counts=True)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a.numpy(),b.numpy())
    x=wasm.Tensor([2.,1.,2.],requires_grad=True)
    with pytest.raises(wasm.BrowserUnsupported):wasm.unique(x)
    with wasm.no_grad():assert not wasm.unique(x).requires_grad


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_remainder_values_broadcast_gradients_and_reverse_scalar(dtype):
    data=np.array([[-3.7,-1.2],[.4,3.3]],dtype=dtype)
    divisor=np.array([1.7,-2.3],dtype=dtype)
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    c,d=wasm.Tensor(divisor,requires_grad=True),torch.tensor(divisor,requires_grad=True)
    actual,expected=a%c,b%d
    actual.sum().backward();expected.sum().backward()
    compare(actual,expected.detach(),dtype);compare(a.grad,b.grad,dtype);compare(c.grad,d.grad,dtype)
    compare(2.1%a,(2.1%b).detach(),dtype)


@pytest.mark.parametrize('dtype',[np.int32,np.int64])
def test_integer_remainder_preserves_native_scalar_promotion(dtype):
    data=np.array([-7,-2,0,3,7],dtype=dtype)
    a,b=wasm.Tensor(data),torch.tensor(data)
    for actual,expected in [(a%3,b%3),(3%a[a!=0],3%b[b!=0]),(a%wasm.Tensor(np.array(3,dtype=np.int64)),b%torch.tensor(3,dtype=torch.int64))]:
        compare(actual,expected,dtype)


def test_integer_sine_and_cosine_use_float32_before_float64_conversion():
    a,b=wasm.Tensor(np.arange(21,dtype=np.int64)),torch.arange(21)
    for operation in ('sin','cos'):
        actual,expected=getattr(a,operation)(),getattr(b,operation)()
        assert actual.dtype==np.dtype('float32')
        compare(actual,expected,np.float32)


def test_storage_addresses_distinguish_snapshots_from_aliases():
    x=wasm.Tensor(np.arange(12.).reshape(3,4))
    assert x.data_ptr()==x.detach().data_ptr()==x.transpose(0,1).data_ptr()
    assert x.data_ptr()!=x.clone().data_ptr() and x.data_ptr()!=x[1:].data_ptr()
    assert wasm.Tensor(np.empty((0,3))).data_ptr()==0


def test_unsupported_spatial_options_and_invalid_indices_do_not_silently_approximate():
    x=wasm.Tensor(np.ones((1,2,3,4)))
    for kwargs in ({'mode':'bilinear'},{'mode':'nearest'},{'mode':'bicubic','antialias':True}):
        with pytest.raises(wasm.BrowserUnsupported):wasm.interpolate(x,size=(5,7),**kwargs)
    with pytest.raises(ValueError):wasm.interpolate(x,size=4,scale_factor=2,mode='area')
    with pytest.raises(ValueError):wasm.interpolate(x,size=4,align_corners=False,mode='area')
    with pytest.raises(wasm.BrowserUnsupported):wasm.interpolate(x,size=(500000,500000),mode='bicubic')
    with pytest.raises(wasm.BrowserUnsupported):wasm.interpolate(x,size=(np.int64(2**62),np.int64(2**62)),mode='bicubic')
    for index in (-1,5):
        with pytest.raises(IndexError):wasm.embedding(wasm.Tensor([index]),wasm.Tensor(np.ones((5,3))))
    with pytest.raises(wasm.BrowserUnsupported):wasm.embedding(wasm.Tensor([1]),wasm.Tensor(np.ones((5,3))),sparse=True)
    with pytest.raises(RuntimeError):wasm.Tensor([1,2])%0
    with pytest.raises(wasm.BrowserUnsupported):wasm.Tensor([True])%True
