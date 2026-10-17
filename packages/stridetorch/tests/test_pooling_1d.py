"""Independent max-pool regions/scatter oracles; no native Torch dependency."""
import pickle
import numpy as np
import pytest
from stridetorch import torch, as_torch, UnsupportedOperation


@pytest.fixture(params=['scan','window','block','auto'],autouse=True)
def algorithm(request,monkeypatch):
    from stridetorch import _pooling
    monkeypatch.setattr(_pooling,'_ALGORITHM',request.param)


def oracle(a,k,s,p,d,ceil):
    length=a.shape[-1];count=(length+2*p-d*(k-1)-1+(s-1 if ceil else 0))//s+1
    if ceil and (count-1)*s>=length+p:count-=1
    value=np.empty((*a.shape[:-1],count),dtype=a.dtype);index=np.empty(value.shape,dtype=np.int64)
    for position in np.ndindex(a.shape[:-1]):
        for out in range(count):
            candidates=[out*s-p+offset*d for offset in range(k) if 0<=out*s-p+offset*d<length]
            samples=a[position][candidates]
            nan=np.flatnonzero(np.isnan(samples)) if a.dtype.kind=='f' else []
            selected=candidates[int(nan[-1]) if len(nan) else int(np.argmax(samples))]
            value[position+(out,)]=a[position+(selected,)];index[position+(out,)]=selected
    return value,index


def scatter(shape,index,seed):
    result=np.zeros(shape,dtype=seed.dtype)
    for position in np.ndindex(shape[:-1]):
        for out in range(index.shape[-1]):result[position+(int(index[position+(out,)]),)]+=seed[position+(out,)]
    return result


CASES=[((2,9),2,2,0,1,False),((1,2,9),2,2,0,1,False),((0,2,9),2,2,0,1,False),
       ((1,2,9),3,2,1,1,True),((1,2,9),3,1,1,2,False),((2,9),1,1,0,1,False)]


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['contiguous','fortran','strided'])
@pytest.mark.parametrize('case',CASES)
@pytest.mark.parametrize('with_indices',[False,True])
def test_region_values_indices_scatter_and_jvp(dtype,layout,case,with_indices):
    shape,k,s,p,d,ceil=case;rng=np.random.default_rng(4204);a=rng.integers(-3,4,size=shape).astype(dtype)
    if layout=='fortran':a=np.asfortranarray(a)
    if layout=='strided':a=np.repeat(a,2,axis=-1)[...,::2]
    before=a.copy();expected,index=oracle(a,k,s,p,d,ceil);x=torch.from_numpy(a).requires_grad_()
    fn=lambda z:torch.nn.functional.max_pool1d(z,k,s,p,d,ceil,with_indices)
    out=fn(x);y,i=out if with_indices else (out,None)
    np.testing.assert_array_equal(y.numpy(),expected)
    if with_indices:np.testing.assert_array_equal(i.numpy(),index)
    seed=rng.normal(size=y.shape).astype(dtype);y.backward(torch.tensor(seed))
    np.testing.assert_allclose(x.grad.numpy(),scatter(shape,index,seed),rtol=2e-6,atol=2e-7)
    x.grad=None;v=rng.normal(size=shape).astype(dtype);direction=torch.tensor(v,requires_grad=True)
    primal,tangent=torch.func.jvp(fn,(x,),(direction,))
    if with_indices:(primal,pi),(tangent,ti)=primal,tangent
    np.testing.assert_array_equal(primal.numpy(),expected)
    np.testing.assert_array_equal(tangent.numpy(),np.take_along_axis(v,index,axis=-1))
    tangent.backward(torch.tensor(seed));assert x.grad is None
    np.testing.assert_allclose(direction.grad.numpy(),scatter(shape,index,seed),rtol=2e-6,atol=2e-7)
    if with_indices:
        np.testing.assert_array_equal(pi.numpy(),index);np.testing.assert_array_equal(ti.numpy(),0)
    np.testing.assert_array_equal(a,before);assert not np.shares_memory(a,y.numpy())


@pytest.mark.parametrize('with_indices',[False,True])
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64,torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8])
def test_dtype_inference_boundaries(dtype,with_indices):
    x=torch.arange(8,dtype=dtype).reshape(1,1,8)
    if dtype.kind!='f' and not with_indices:
        with pytest.raises(UnsupportedOperation):torch.nn.functional.max_pool1d(x,2)
        return
    result=torch.nn.functional.max_pool1d(x,2,return_indices=with_indices)
    y,i=result if with_indices else (result,None)
    np.testing.assert_array_equal(y.numpy(),[[[1,3,5,7]]]);assert y.dtype==dtype
    if with_indices:
        np.testing.assert_array_equal(i.numpy(),[[[1,3,5,7]]]);assert i.dtype==torch.int64


def test_local_nonfinite_seed_and_first_tie_selection():
    x=torch.tensor([[[2.,2.,1.,1.,3.,3.,4.,4.]]],requires_grad=True)
    y,i=torch.nn.functional.max_pool1d(x,2,return_indices=True)
    np.testing.assert_array_equal(i.numpy(),[[[0,2,4,6]]])
    y.backward(torch.tensor([[[np.nan,np.inf,1.,-0.]]]))
    np.testing.assert_allclose(x.grad.numpy(),[[[np.nan,0.,np.inf,0.,1.,0.,0.,0.]]],equal_nan=True)
    assert not np.signbit(x.grad.numpy()[...,-2:]).any()


@pytest.mark.parametrize('shape',[(0,2,9),(2,9),(1,2,9)])
def test_contiguous_output_indices_and_input_owned_storage(shape):
    x=torch.zeros(shape);y,i=torch.nn.functional.max_pool1d(x,2,return_indices=True)
    for output in (y,i):
        assert output.is_contiguous();steps=[];stride=1
        for n in reversed(output.shape):steps.append(stride);stride*=max(1,n)
        assert output.stride()==tuple(reversed(steps))
    assert not np.shares_memory(x.numpy(),y.numpy())


def test_workspace_fallback_and_bounded_kernel(monkeypatch):
    from stridetorch import _pooling
    monkeypatch.setattr(_pooling,'_WINDOW_WORKSPACE_BYTES',1)
    x=torch.arange(9,dtype=torch.float64).reshape(1,1,9).requires_grad_()
    y,i=torch.nn.functional.max_pool1d(x,3,1,1,return_indices=True)
    np.testing.assert_array_equal(i.numpy(),[[[1,2,3,4,5,6,7,8,8]]]);y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(),[[[0,1,1,1,1,1,1,1,2]]])
    with pytest.raises(UnsupportedOperation):torch.nn.functional.max_pool1d(x,8_000_002,padding=4_000_001)


def test_aliases_pickle_and_checkpointed_convolution_training():
    with as_torch() as api:
        from torch.nn.modules.pooling import MaxPool1d
        from torch.nn.functional import max_pool1d_with_indices
        x=api.arange(9,dtype=api.float64).reshape(1,1,9)
        model=MaxPool1d(3,1,1,return_indices=True)
        assert isinstance(model,api.nn.Module) and not isinstance(model,api.nn.MaxPool2d)
        for aa,bb in zip(model(x),pickle.loads(pickle.dumps(model))(x)):np.testing.assert_array_equal(aa.numpy(),bb.numpy())
        for aa,bb in zip(api.max_pool1d_with_indices(x,3,1,1),max_pool1d_with_indices(x,3,1,1)):np.testing.assert_array_equal(aa.numpy(),bb.numpy())
    def make():
        return torch.nn.Sequential(torch.nn.Conv1d(2,3,3,padding=1),torch.nn.SiLU(),torch.nn.MaxPool1d(2),
                                   torch.nn.AdaptiveAvgPool1d(3),torch.nn.Flatten(),torch.nn.Linear(9,2)).double()
    rng=np.random.default_rng(4205);torch.manual_seed(4205);model=make();x=torch.tensor(rng.normal(size=(2,2,17)))
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    for _ in range(2):
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
    saved=pickle.loads(pickle.dumps((model.state_dict(),optimizer.state_dict())))
    restored=make();restored.load_state_dict(saved[0]);resume=torch.optim.SGD(restored.parameters(),lr=.01,momentum=.9);resume.load_state_dict(saved[1])
    for m,opt in ((model,optimizer),(restored,resume)):
        opt.zero_grad();m(x).square().mean().backward();opt.step()
    for aa,bb in zip(model.parameters(),restored.parameters()):np.testing.assert_array_equal(aa.numpy(),bb.numpy())
    direction=torch.tensor(rng.normal(size=x.shape));_,tangent=torch.func.jvp(restored,(x,),(direction,))
    epsilon=1e-5;numeric=(restored(x+epsilon*direction).numpy()-restored(x-epsilon*direction).numpy())/(2*epsilon)
    np.testing.assert_allclose(tangent.numpy(),numeric,rtol=1e-7,atol=1e-9)


@pytest.mark.parametrize('dimensions',[1,2])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('kernel',[1024,1_000_000])
@pytest.mark.parametrize('indices',[False,True])
def test_huge_padded_kernel_reaches_only_tiny_input(dimensions,dtype,kernel,indices):
    shape=(1,1,7) if dimensions==1 else (1,1,1,7)
    x=torch.arange(7,dtype=dtype).reshape(shape).requires_grad_()
    size=kernel if dimensions==1 else (1,kernel)
    step=1 if dimensions==1 else (1,1);pad=kernel//2 if dimensions==1 else (0,kernel//2)
    fn=lambda z:getattr(torch.nn.functional,'max_pool'+str(dimensions)+'d')(z,size,step,pad,return_indices=indices)
    result=fn(x);y,i=result if indices else (result,None)
    assert y.shape==(*shape[:-1],8);np.testing.assert_array_equal(y.numpy(),6.)
    if indices:np.testing.assert_array_equal(i.numpy(),6)
    y.sum().backward();expected=np.zeros(shape,dtype=dtype);expected[...,-1]=8
    np.testing.assert_array_equal(x.grad.numpy(),expected);x.grad=None
    v=torch.ones_like(x,requires_grad=True)
    primal,tangent=torch.func.jvp(lambda z:fn(z)[0] if indices else fn(z),(x,),(v,))
    np.testing.assert_array_equal(primal.numpy(),6.);np.testing.assert_array_equal(tangent.numpy(),1.)
    tangent.sum().backward();assert x.grad is None
    np.testing.assert_array_equal(v.grad.numpy(),expected)


def test_large_padding_keeps_last_nan_and_local_nonfinite_pullback():
    x=torch.tensor([[[np.nan,1.,2.,np.nan,4.,5.,6.]]],requires_grad=True)
    y,i=torch.nn.functional.max_pool1d(x,1_000_000,1,500_000,return_indices=True)
    assert np.isnan(y.numpy()).all();np.testing.assert_array_equal(i.numpy(),3)
    seed=np.ones(y.shape,dtype=np.float32);seed[...,0]=np.inf;y.backward(torch.tensor(seed))
    expected=np.zeros(x.shape,dtype=np.float32);expected[...,3]=np.inf
    np.testing.assert_array_equal(x.grad.numpy(),expected)
