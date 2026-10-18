"""Strict matrix API contracts and independent bilinear differential algebra."""
import pickle
import tracemalloc
import numpy as np
import pytest
from stridetorch import torch, as_torch, UnsupportedOperation, _tensor


def product_oracle(a,b,ta,tb,seed):
    a,b,ta,tb,seed=[x.astype('float64') for x in (a,b,ta,tb,seed)]
    prefix=a.shape[:-2];y=np.empty(prefix+(a.shape[-2],b.shape[-1]));dy=np.empty_like(y)
    ga=np.zeros_like(a);gb=np.zeros_like(b);ha=np.zeros_like(a);hb=np.zeros_like(b)
    for batch in np.ndindex(prefix):
        x,w,dx,dw,g=[z[batch] for z in (a,b,ta,tb,seed)]
        value=x@w;direction=dx@w+x@dw;y[batch]=value;dy[batch]=direction
        ga[batch]=g@w.T;gb[batch]=x.T@g
        ha[batch]=direction@w.T+value@dw.T;hb[batch]=dx.T@value+x.T@direction
    return y,ga,gb,dy,ha,hb,(y*y).sum()*.5


def operation(name,method):
    return (lambda input,mat2:getattr(input,name)(mat2=mat2)) if method else getattr(torch,name)


@pytest.mark.parametrize('name,shapes',[
    ('mm',((3,4),(4,2))),('mm',((0,4),(4,2))),('mm',((3,0),(0,2))),
    ('bmm',((2,3,4),(2,4,2))),('bmm',((0,3,4),(0,4,2))),('bmm',((2,3,0),(2,0,2))),
    ('mm',((128,64),(64,128))),('bmm',((4,64,64),(4,64,64)))])
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('layout',['C','F','stride'])
@pytest.mark.parametrize('method',[False,True])
@pytest.mark.parametrize('wasm',[False,True])
def test_matrix_api_values_vjp_jvp_hvp_and_layout(name,shapes,dtype,layout,method,wasm,monkeypatch):
    monkeypatch.setattr(_tensor,'_WASM_MATMUL',wasm)
    rng=np.random.default_rng(78011)
    arrays=[rng.integers(-8,9,size=s,dtype=np.int32).astype(dtype)/16 for s in [*shapes,*shapes]]
    if layout=='F':arrays=list(map(np.asfortranarray,arrays))
    if layout=='stride':arrays=[np.repeat(x,2,axis=-1)[...,::2] for x in arrays]
    a,b,ta,tb=arrays;shape=a.shape[:-2]+(a.shape[-2],b.shape[-1]);seed=rng.integers(-8,9,size=shape,dtype=np.int32).astype(dtype)/16
    expected=product_oracle(a,b,ta,tb,seed);x,w=[torch.from_numpy(z).requires_grad_() for z in (a,b)];dx,dw=map(torch.from_numpy,(ta,tb));fn=operation(name,method)
    y=fn(x,w);ga,gb=torch.autograd.grad(y,(x,w),torch.from_numpy(seed))
    primal,direction=torch.func.jvp(fn,(x,w),(dx,dw))
    loss,(ha,hb)=torch.autograd.functional.hvp(lambda x,w:fn(x,w).square().sum()*.5,(x,w),(dx,dw))
    tolerance=3e-5 if dtype=='float32' else 3e-11
    for actual,control in zip([y,ga,gb,direction,ha,hb,loss],expected):np.testing.assert_allclose(actual.detach().numpy(),control,rtol=tolerance,atol=tolerance)
    np.testing.assert_array_equal(primal.numpy(),y.numpy());assert y.is_contiguous()


@pytest.mark.parametrize('name',['mm','bmm'])
@pytest.mark.parametrize('dtype',['bool','uint8','int8','int16','int32','int64','uint16','uint32','uint64','float16','float32','float64','complex64','complex128'])
@pytest.mark.parametrize('dims',[(2,3,4,2),(0,3,4,2),(2,0,4,2),(2,3,0,2),(2,3,4,0)])
@pytest.mark.parametrize('method',[False,True])
def test_matrix_inference_dtype_and_empty_dispatch(name,dtype,dims,method):
    m,k,n,batch=dims;a=np.ones((m,k) if name=='mm' else (batch,m,k),dtype=dtype);b=np.ones((k,n) if name=='mm' else (batch,k,n),dtype=dtype)
    fn=operation(name,method);x,w=map(torch.from_numpy,(a,b))
    if a.size and b.size and (dtype=='bool' or dtype in ['uint16','uint32','uint64']):
        with pytest.raises(RuntimeError):fn(x,w)
    else:
        y=fn(x,w);assert y.dtype==a.dtype;np.testing.assert_array_equal(y.numpy(),a@b)


@pytest.mark.parametrize('name',['mm','bmm'])
@pytest.mark.parametrize('method',[False,True])
@pytest.mark.parametrize('bad',['input_type','other_type','rank','inner','dtype','broadcast','out','typo'])
def test_matrix_strict_arguments(name,method,bad):
    a=torch.ones(2,3) if name=='mm' else torch.ones(2,2,3)
    b=torch.ones(3,4) if name=='mm' else torch.ones(2,3,4)
    fn=operation(name,method)
    if bad=='input_type':
        with pytest.raises(TypeError):
            getattr(torch.Tensor,name)(a.numpy(),b) if method else fn(a.numpy(),b)
    elif bad=='other_type':
        with pytest.raises(TypeError):fn(a,b.numpy())
    elif bad=='rank':
        with pytest.raises(RuntimeError):fn(a.unsqueeze(0),b.unsqueeze(0))
    elif bad=='inner':
        with pytest.raises(RuntimeError):fn(a,b.transpose(-1,-2))
    elif bad=='dtype':
        with pytest.raises(RuntimeError):fn(a,b.double())
    elif bad=='broadcast':
        if name=='mm':
            with pytest.raises(RuntimeError):fn(a.unsqueeze(0),b)
        else:
            with pytest.raises(RuntimeError):fn(a,b[:1])
    elif bad=='out':
        if method:
            with pytest.raises(TypeError):getattr(a,name)(b,out=None)
        else:
            np.testing.assert_array_equal(getattr(torch,name)(a,b,out=None).numpy(),a.numpy()@b.numpy())
            with pytest.raises(UnsupportedOperation):getattr(torch,name)(a,b,out=torch.empty(0))
    else:
        with pytest.raises(TypeError):getattr(a,name)(batch2=b) if method else getattr(torch,name)(input=a,other=b)


@pytest.mark.parametrize('shapes',[((2,0,3),(2,3,4)),((2,3,0),(2,0,4)),((2,3,4),(2,4,0)),((0,3,4),(0,4,2))])
@pytest.mark.parametrize('dtypes',[('float32','float64'),('float64','float32')])
@pytest.mark.parametrize('method',[False,True])
def test_empty_bmm_mixed_dtype_keeps_values_and_all_ad_channels(shapes,dtypes,method):
    a,b=[torch.from_numpy(np.ones(shape,dtype=dtype)).requires_grad_() for shape,dtype in zip(shapes,dtypes)]
    fn=operation('bmm',method);y=fn(a,b);assert y.dtype==b.dtype
    ga,gb=torch.autograd.grad(y,(a,b),torch.ones_like(y));assert ga.dtype==a.dtype and gb.dtype==b.dtype
    primal,tangent=torch.func.jvp(fn,(a,b),(torch.ones_like(a),torch.ones_like(b)))
    for value in [y,ga,gb,primal,tangent]:np.testing.assert_array_equal(value.numpy(),np.zeros(value.shape,dtype=value.dtype))


@pytest.mark.parametrize('name',['mm','bmm'])
@pytest.mark.parametrize('active',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('mutated',['a','b','output'])
def test_matrix_saved_value_guards(name,active,mutated):
    a=torch.ones(2,3,requires_grad=active[0]) if name=='mm' else torch.ones(2,2,3,requires_grad=active[0])
    b=torch.ones(3,4,requires_grad=active[1]) if name=='mm' else torch.ones(2,3,4,requires_grad=active[1]);y=getattr(torch,name)(a,b)
    with torch.no_grad():{'a':a,'b':b,'output':y}[mutated].detach().add_(1)
    if mutated=='a' and active[1] or mutated=='b' and active[0]:
        with pytest.raises(RuntimeError,match='modified'):y.sum().backward()
    else:y.sum().backward()


@pytest.mark.parametrize('shapes',[((2,0,3),(2,3,4)),((2,3,0),(2,0,4)),((2,3,4),(2,4,0)),((0,3,4),(0,4,2))])
@pytest.mark.parametrize('dtypes',[('float32','float64'),('float64','float32')])
@pytest.mark.parametrize('active',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('mutated',['a','b','output'])
def test_empty_mixed_bmm_preserves_original_saved_storage(shapes,dtypes,active,mutated):
    a,b=[torch.from_numpy(np.ones(shape,dtype=dtype)).requires_grad_(enabled) for shape,dtype,enabled in zip(shapes,dtypes,active)]
    y=torch.bmm(a,b)
    with torch.no_grad():{'a':a,'b':b,'output':y}[mutated].detach().add_(1)
    if mutated=='a' and active[1] or mutated=='b' and active[0]:
        with pytest.raises(RuntimeError,match='modified'):y.sum().backward()
    else:y.sum().backward()


@pytest.mark.parametrize('name',['mm','bmm'])
def test_matrix_aliases_capabilities_and_pickle(name):
    fn=getattr(torch,name);model=torch.nn.Module();model.operator=fn;restored=pickle.loads(pickle.dumps(model));assert restored.operator is fn
    capability=torch._browser_capabilities['torch.'+name];assert capability.reverse=='analytic' and capability.forward and 'out' in capability.unsupported_keywords
    with as_torch():
        import torch as bound
        a=bound.ones(2,2) if name=='mm' else bound.ones(1,2,2)
        np.testing.assert_array_equal(getattr(bound,name)(input=a,mat2=a).numpy(),np.full(a.shape,2,dtype='float32'))


_DTYPES=['bool','uint8','int8','int16','int32','int64','uint16','uint32','uint64','float16','float32','float64','complex64','complex128']


@pytest.mark.parametrize('source_dtype',_DTYPES)
@pytest.mark.parametrize('result_dtype',_DTYPES)
@pytest.mark.parametrize('shapes',[((2,0,3),(2,3,4)),((2,3,0),(2,0,4)),((2,3,4),(2,4,0)),((0,3,4),(0,4,2))])
@pytest.mark.parametrize('method',[False,True])
def test_empty_bmm_uses_metadata_without_input_casts(source_dtype,result_dtype,shapes,method,monkeypatch):
    a=torch.from_numpy(np.ones(shapes[0],dtype=source_dtype));b=torch.from_numpy(np.ones(shapes[1],dtype=result_dtype))
    def unexpected(*args,**kwargs):raise AssertionError('Empty bmm must not convert source values')
    monkeypatch.setattr(a,'to',unexpected);monkeypatch.setattr(b,'to',unexpected)
    value=operation('bmm',method)(a,b);assert value.dtype==b.dtype
    np.testing.assert_array_equal(value.numpy(),np.zeros((shapes[0][0],shapes[0][1],shapes[1][2]),dtype=result_dtype))


def test_empty_mixed_bmm_does_not_allocate_a_source_sized_cast():
    a=torch.ones(2,256,256,dtype=torch.float64);b=torch.ones(2,256,0,dtype=torch.float32)
    tracing=tracemalloc.is_tracing()
    if not tracing:tracemalloc.start()
    try:
        baseline=tracemalloc.get_traced_memory()[0];tracemalloc.reset_peak();value=torch.bmm(a,b)
        peak=tracemalloc.get_traced_memory()[1]-baseline
        assert value.numel()==0 and peak<65536
    finally:
        if not tracing:tracemalloc.stop()
