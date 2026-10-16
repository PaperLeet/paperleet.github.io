"""Independent NumPy-only selection derivatives, compact storage and guards."""
import gc
import weakref
import numpy as np
import pytest
from stridetorch import UnsupportedOperation, torch


@pytest.mark.parametrize('name',['gather','sort','topk'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_indexing_directional_derivative_and_adjoint(name,dtype):
    data=np.array([[3.,1.,2.],[6.,4.,5.]],dtype=dtype)
    direction=np.array([[.2,.4,.8],[.3,.6,.9]],dtype=dtype)
    index=torch.tensor([[1,1],[0,2]],dtype=torch.int64)
    def f(x):return torch.gather(x,1,index) if name=='gather' else getattr(x,name)(**({'dim':1,'stable':True} if name=='sort' else {'k':2,'dim':1})).values
    p=torch.tensor(data,requires_grad=True);t=torch.tensor(direction,requires_grad=True)
    value,jvp=torch.func.jvp(f,(p,),(t,));seed=torch.ones_like(value)*.37
    vjp=torch.autograd.grad(value,p,grad_outputs=seed)[0]
    np.testing.assert_allclose((vjp.numpy()*direction).sum(),(seed.numpy()*jvp.numpy()).sum(),rtol=2e-6)
    h=1e-3 if dtype==torch.float32 else 1e-5
    numeric=(f(torch.tensor(data+h*direction)).numpy()-f(torch.tensor(data-h*direction)).numpy())/(2*h)
    np.testing.assert_allclose(jvp.numpy(),numeric,rtol=2e-3,atol=1e-4)
    jvp.backward(seed);assert p.grad is None
    np.testing.assert_allclose(t.grad.numpy(),vjp.numpy(),rtol=1e-6)


def test_half_duplicate_gradient_and_scalar_gather():
    x=torch.tensor(1.,dtype=torch.float16,requires_grad=True)
    y=torch.gather(x,0,torch.zeros(1000,dtype=torch.int32))
    y.backward(torch.full(y.shape,.1,dtype=torch.float16));assert x.grad.item()==100
    scalar=torch.tensor([2.],requires_grad=True)
    out=scalar.gather(0,torch.tensor(0));assert out.shape==()
    out.backward();np.testing.assert_array_equal(scalar.grad.numpy(),[1.])


@pytest.mark.parametrize('name',['gather','sort','topk'])
def test_indexing_does_not_retain_intermediate_values(name):
    leaf=torch.ones(2,4096,requires_grad=True);middle=leaf+1
    reference=weakref.ref(middle._data)
    result=middle.gather(1,torch.zeros(2,8,dtype=torch.int64)) if name=='gather' else middle.sort(1).values if name=='sort' else middle.topk(8,1).values
    del middle;gc.collect();assert reference() is None
    result.sum().backward();assert leaf.grad.shape==leaf.shape


def test_topk_indices_have_compact_owner_and_stable_result_type():
    from stridetorch._tensor import storage_owner
    x=torch.arange(16384,dtype=torch.float32).reshape(4,4096)
    for sorted in (True,False):
        pair=x.topk(8,dim=1,sorted=sorted)
        assert storage_owner(pair.indices._data).nbytes==4*8*8
        assert type(pair) is type(x.topk(2,dim=1))
        np.testing.assert_array_equal(np.sort(pair.values.numpy(),axis=1),x.numpy()[:,-8:])
    assert type(x.sort()) is type(x.sort())


@pytest.mark.parametrize('name',['gather','sort','topk'])
def test_raw_index_storage_mutations_are_seen_and_negative_indices_rejected(name):
    x=torch.tensor([3.,1.,2.],requires_grad=True);index=torch.tensor([0,1],dtype=torch.int64)
    if name=='gather':y=x.gather(0,index)
    else:y,index=x.sort() if name=='sort' else x.topk(2)
    index.numpy()[...]=0;y.sum().backward()
    expected=[float(index.numel()),0.,0.] if name=='gather' else [1.,0.,0.]
    np.testing.assert_array_equal(x.grad.numpy(),expected)
    x.grad=None
    if name=='gather':y=x.gather(0,index)
    else:y,index=x.sort() if name=='sort' else x.topk(2)
    index.numpy()[...]=-1
    with pytest.raises(RuntimeError):y.sum().backward()


def test_sparse_forward_without_recording_and_unsupported_modes_are_explicit():
    np.testing.assert_array_equal(torch.gather(torch.ones(3),0,torch.tensor([0]),sparse_grad=True).numpy(),[1.])
    with pytest.raises(UnsupportedOperation):torch.gather(torch.ones(3,requires_grad=True),0,torch.tensor([0]),sparse_grad=True)
    with pytest.raises(UnsupportedOperation):torch.gather(torch.ones(3),0,torch.tensor([0]),out=torch.ones(1))
    for name in ('argsort','sort','topk'):
        f=lambda x:getattr(torch,name)(x,**({'k':1} if name=='topk' else {}))
        p,t=torch.func.jvp(f,(torch.tensor([3.,1.,2.]),),(torch.ones(3),))
        if name=='argsort':np.testing.assert_array_equal(t.numpy(),[0,0,0])
        else:
            assert type(p) is type(t);np.testing.assert_array_equal(t.indices.numpy(),np.zeros(t.indices.shape,dtype=np.int64))
        with pytest.raises(RuntimeError):torch.func.jvp(f,(torch.ones(3),),(torch.ones(3),),strict=True)
