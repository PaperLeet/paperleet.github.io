"""Native indexing/selection, opmath gradients and transform/lifetime contracts."""
import numpy as np
import pytest
import torch as native
from stridetorch import UnsupportedOperation, torch


def assert_equal(actual,expected):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    np.testing.assert_allclose(actual,expected,rtol=2e-3 if expected.dtype==np.float16 else 3e-6,
                               atol=0 if expected.dtype==np.float16 else 1e-7,equal_nan=True)


GATHER=[((),(),0), ((),(3,),0), ((1,),(),0), ((2,3,4),(2,5,3),1),
        ((3,2,4),(2,2,3),0), ((2,3),(1,0,9),-1), ((0,3),(0,3),0)]


@pytest.mark.parametrize('dtype',['float16','float32','float64'])
@pytest.mark.parametrize('index_dtype',['int32','int64'])
@pytest.mark.parametrize('shape,index_shape,axis',GATHER)
@pytest.mark.parametrize('functional',[False,True])
def test_gather_values_and_duplicate_gradients(dtype,index_dtype,shape,index_shape,axis,functional):
    data=np.arange(int(np.prod(shape))).reshape(shape).astype(dtype)
    extent=shape[axis] if shape else 1
    indices=(np.arange(int(np.prod(index_shape)))%max(1,extent)).reshape(index_shape)
    results=[]
    for api in (native,torch):
        x=api.tensor(data,requires_grad=True);i=api.tensor(indices,dtype=getattr(api,index_dtype))
        y=api.gather(x,axis,i) if functional else x.gather(axis,i)
        y.backward(api.full(y.shape,.37,dtype=getattr(api,dtype)))
        results.append((y.detach().numpy(),x.grad.numpy()))
    for expected,actual in zip(*results):assert_equal(actual,expected)


@pytest.mark.parametrize('index_dtype',['int16','int64','float32','complex64'])
def test_empty_gather_bypasses_index_rank_and_dtype_but_not_dimensions(index_dtype):
    for api in (native,torch):
        x=api.ones(2,3,requires_grad=True);index=api.empty(1,0,9,dtype=getattr(api,index_dtype))
        y=api.gather(x,1,index);assert y.shape==(1,0,9)
        y.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),np.zeros((2,3)))
        with pytest.raises(IndexError):api.gather(x,5,index)


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('dtype',['float16','float32','float64'])
def test_half_scatter_add_opmath_and_strided_inputs(axis,dtype):
    results=[]
    for api in (native,torch):
        data=np.array([[.3,.7],[.2,.8]],dtype=dtype).T
        x=api.from_numpy(data).requires_grad_();shape=(1000,2) if axis==0 else (2,1000)
        y=api.gather(x,axis,api.zeros(shape,dtype=api.int64))
        y.backward(api.full(shape,.1,dtype=getattr(api,dtype)));results.append(x.grad.numpy())
    assert_equal(results[1],results[0])
    if dtype=='float16':assert results[0].max()==100


@pytest.mark.parametrize('dtype',['float16','float32','float64','int64','uint8'])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('k_mode',['empty','one','small','all'])
@pytest.mark.parametrize('largest',[False,True])
@pytest.mark.parametrize('sorted',[False,True])
def test_topk_partial_and_full_selection(dtype,axis,k_mode,largest,sorted):
    data=np.arange(4*129).reshape(4,129).astype(dtype)
    count={'empty':0,'one':1,'small':min(4,data.shape[axis]),'all':data.shape[axis]}[k_mode]
    outputs=[]
    for api in (native,torch):
        x=api.tensor(data,requires_grad=data.dtype.kind=='f');out=api.topk(x,count,dim=axis,largest=largest,sorted=sorted)
        np.testing.assert_array_equal(out.values.detach().numpy(),np.take_along_axis(data,out.indices.numpy(),axis))
        gradient=None
        if x.requires_grad:
            # Compare the selected set's VJP; sorted=False order is unspecified.
            out.values.sum().backward();gradient=x.grad.numpy()
        outputs.append((np.sort(out.values.detach().numpy(),axis=axis),gradient))
    for expected,actual in zip(*outputs):
        if expected is None:assert actual is None
        else:assert_equal(actual,expected)


@pytest.mark.parametrize('name',['sort','topk'])
@pytest.mark.parametrize('largest',[False,True])
def test_nan_ties_signed_infinities_and_seed_follow_returned_indices(name,largest):
    data=np.tile([2.,2.,0.,np.nan,np.inf,-np.inf,1.,np.nan],32).reshape(2,128)
    values=[]
    for api in (native,torch):
        x=api.tensor(data,requires_grad=True)
        out=x.sort(dim=1,descending=largest,stable=True) if name=='sort' else x.topk(8,dim=1,largest=largest)
        values.append(out.values.detach().numpy())
        seed=api.full(out.values.shape,np.inf,dtype=api.float64);out.values.backward(seed)
        expected=np.zeros_like(data);np.put_along_axis(expected,out.indices.numpy(),np.inf,axis=1)
        np.testing.assert_array_equal(x.grad.numpy(),expected)
    assert_equal(values[1],values[0])


@pytest.mark.parametrize('name',['gather','sort','topk'])
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('functional',[False,True])
def test_indexing_jvp_named_outputs_and_reverse_tangent(name,dtype,functional):
    results=[]
    for api in (native,torch):
        p=api.tensor([[3.,1.,2.],[6.,4.,5.]],dtype=getattr(api,dtype),requires_grad=True)
        t=api.tensor([[.2,.4,.8],[.3,.6,.9]],dtype=getattr(api,dtype),requires_grad=True)
        index=api.tensor([[1,1],[0,2]],dtype=api.int64)
        def fn(x):
            if name=='gather':return api.gather(x,1,index) if functional else x.gather(1,index)
            if name=='sort':return api.sort(x,dim=1,stable=True) if functional else x.sort(dim=1,stable=True)
            return api.topk(x,2,dim=1) if functional else x.topk(2,1)
        a,b=api.func.jvp(fn,(p,),(t,))
        if name!='gather':
            assert type(a) is type(b)
            np.testing.assert_array_equal(b.indices.numpy(),np.zeros(b.indices.shape,dtype=np.int64))
            a,b=a.values,b.values
        b.square().sum().backward();assert p.grad is None
        results.append((a.detach().numpy(),b.detach().numpy(),t.grad.numpy()))
    for expected,actual in zip(*results):assert_equal(actual,expected)


@pytest.mark.parametrize('name',['gather','sort','topk'])
@pytest.mark.parametrize('mutation',['input','output','indices'])
def test_saved_input_and_index_guards_match_the_native_rule(name,mutation):
    for api in (native,torch):
        x=api.tensor([3.,1.,2.],requires_grad=True);i=api.tensor([1,1],dtype=api.int64)
        if name=='gather':y=x.gather(0,i)
        else:y,i=x.sort() if name=='sort' else x.topk(2)
        with api.no_grad():(x if mutation=='input' else y if mutation=='output' else i).detach().fill_(0)
        if mutation=='indices' or mutation=='input' and name=='gather':
            with pytest.raises((RuntimeError,UnsupportedOperation)):y.sum().backward()
        else:y.sum().backward();assert x.grad is not None


@pytest.mark.parametrize('name',['gather','sort','topk'])
def test_retained_replay_and_model_conversion(name):
    results=[]
    for api in (native,torch):
        layer=api.nn.Linear(3,1,bias=False).double()
        with api.no_grad():layer.weight.copy_(api.tensor([[3.,1.,2.]],dtype=api.float64))
        x=layer.weight;i=api.tensor([[1,1]],dtype=api.int64)
        y=x.gather(1,i) if name=='gather' else x.sort(1).values if name=='sort' else x.topk(2,1).values
        layer.float();loss=y.sum();loss.backward(retain_graph=True);loss.backward()
        results.append(x.grad.numpy())
        with pytest.raises(RuntimeError):loss.backward()
    assert_equal(results[1],results[0])


@pytest.mark.parametrize('name',['argsort','sort','topk','gather'])
@pytest.mark.parametrize('dim',[None,True,1.5,5,-5])
def test_strict_axis_exception_classes(name,dim):
    errors=[]
    for api in (native,torch):
        x=api.ones(3)
        try:
            if name=='gather':x.gather(dim,api.tensor([0]))
            elif name=='topk':x.topk(1,dim=dim)
            else:getattr(x,name)(dim=dim)
        except Exception as e:errors.append(type(e))
        else:raise AssertionError('Invalid axis accepted')
    assert errors[0] is errors[1]


@pytest.mark.parametrize('index_dtype',['int16','uint64','float32','bool'])
def test_nonempty_gather_invalid_index_dtype(index_dtype):
    for api in (native,torch):
        with pytest.raises(RuntimeError):api.ones(3).gather(0,api.tensor([0],dtype=getattr(api,index_dtype)))


@pytest.mark.parametrize('name',['argsort','sort'])
@pytest.mark.parametrize('stable',[None,False,True,1])
@pytest.mark.parametrize('positional',[False,True])
def test_sort_stable_overload_requires_keyword_dimension(name,stable,positional):
    outputs=[]
    for api in (native,torch):
        x=api.tensor([3.,1.,2.])
        try:
            y=getattr(x,name)(0,stable=stable) if positional else getattr(x,name)(dim=0,stable=stable)
            outputs.append((y if name=='argsort' else y.indices).numpy())
        except Exception as error:outputs.append(type(error))
    if isinstance(outputs[0],type):assert outputs[0] is outputs[1]
    else:np.testing.assert_array_equal(outputs[0],outputs[1])
