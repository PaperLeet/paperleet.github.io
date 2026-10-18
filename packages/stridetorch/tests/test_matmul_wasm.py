"""Large WASM matmul values, derivatives, layout, and exceptional fallback."""
import numpy as np
import pytest

from stridetorch import torch, _matmul, _tensor


@pytest.fixture(autouse=True)
def wasm_matmul(monkeypatch):
    monkeypatch.setattr(_tensor, '_WASM_MATMUL', True)


def arrange(value, layout):
    if layout == 'fortran': return np.asfortranarray(value)
    if layout == 'stride':
        storage=np.empty((value.shape[0]*2,value.shape[1]*2),dtype=value.dtype)
        storage[::2,::2]=value
        return storage[::2,::2]
    if layout == 'broadcast': return np.broadcast_to(value[:1],value.shape)
    return value


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(128,64,128),(64,128,128),(64,256,64)])
@pytest.mark.parametrize('layout',['contiguous','fortran','stride','broadcast'])
def test_large_values_vjp_jvp_and_hvp(dtype,shape,layout,monkeypatch):
    rng=np.random.default_rng(76040);m,k,n=shape
    a,b,ta,tb=[arrange(rng.uniform(-1,1,size=dimensions).astype(dtype),layout)
               for dimensions in [(m,k),(k,n),(m,k),(k,n)]]
    seed=rng.uniform(-1,1,size=(m,n)).astype(dtype)
    aa,bb,va,vb=[torch.from_numpy(x).requires_grad_(i<2) for i,x in enumerate((a,b,ta,tb))]
    calls=[];original=np.einsum
    def counted(*args,**kwargs):
        calls.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(np,'einsum',counted)
    y=aa@bb;ga,gb=torch.autograd.grad(y,(aa,bb),torch.from_numpy(seed))
    primal,tangent=torch.func.jvp(lambda x,w:x@w,(aa,bb),(va,vb))
    loss,(ha,hb)=torch.autograd.functional.hvp(lambda x,w:(x@w).square().sum()*.5,(aa,bb),(va,vb))
    assert len(calls)>=6
    a,b,ta,tb,seed=[x.astype('float64') for x in (a,b,ta,tb,seed)]
    expected=a@b;direction=ta@b+a@tb
    controls=[(y,expected),(primal,expected),(ga,seed@b.T),(gb,a.T@seed),
              (tangent,direction),(ha,direction@b.T+expected@tb.T),
              (hb,ta.T@expected+a.T@direction),(loss,(expected**2).sum()*.5)]
    tolerance=3e-4 if dtype==np.float32 else 3e-11
    for index,(actual,reference) in enumerate(controls):
        # Composed float32 HVP reductions need a separate absolute budget.
        # Native Torch and the unchanged adapter also reach ~0.002 error
        # against float64 controls for the repeated-row broadcast fixture.
        absolute=2e-3 if dtype==np.float32 and index in (5,6) else tolerance
        np.testing.assert_allclose(actual.detach().numpy(),reference,atol=absolute,rtol=tolerance)
    assert y.is_contiguous() and y.stride()==(n,1)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('operand',[0,1])
@pytest.mark.parametrize('kind',['nan','inf','large','tiny','subnormal','small_normal'])
def test_exceptional_range_keeps_original_matmul(dtype,operand,kind,monkeypatch):
    a=np.ones((128,64),dtype=dtype);b=np.ones((64,128),dtype=dtype);limits=np.finfo(dtype)
    value={'nan':np.nan,'inf':np.inf,'large':limits.max/2,'tiny':limits.tiny,
           'subnormal':limits.smallest_subnormal,'small_normal':np.sqrt(limits.tiny)/2}[kind]
    [a,b][operand][0,0]=value
    def unexpected(*args,**kwargs):raise AssertionError('exceptional input reached einsum')
    monkeypatch.setattr(np,'einsum',unexpected)
    with np.errstate(all='ignore'):
        expected=a@b;actual=(torch.from_numpy(a)@torch.from_numpy(b)).numpy()
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('values',[(0.,0.),(-0.,1.),(1.,-0.),(-1.,0.),(-0.,-0.),(0.,-1.)])
def test_zero_signs_and_output_layout(dtype,values):
    a=np.full((128,64),values[0],dtype=dtype);b=np.full((64,128),values[1],dtype=dtype)
    expected=a@b;actual=_matmul.product(a,b)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))
    assert actual.flags.c_contiguous


@pytest.mark.parametrize('active',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('changed',['a','b','output'])
def test_large_route_retains_minimal_saved_value_guards(active,changed):
    a=torch.ones(128,64,requires_grad=active[0]);b=torch.ones(64,128,requires_grad=active[1]);y=a@b
    with torch.no_grad():{'a':a,'b':b,'output':y}[changed].detach().add_(1)
    guarded=changed=='a' and active[1] or changed=='b' and active[0]
    if guarded:
        with pytest.raises(RuntimeError,match='modified'):y.sum().backward()
    else:
        y.sum().backward()
        if active[0]:np.testing.assert_array_equal(a.grad.numpy(),np.full((128,64),128))
        if active[1]:np.testing.assert_array_equal(b.grad.numpy(),np.full((64,128),128))


@pytest.mark.parametrize('shape_a,shape_b',[
    ((4,64),(64,4)),((128,0),(0,128)),((2,128,64),(64,128)),((64,),(64,128))])
def test_ineligible_shapes_retain_the_original_operation(shape_a,shape_b,monkeypatch):
    def unexpected(*args,**kwargs):raise AssertionError('ineligible shape reached einsum')
    monkeypatch.setattr(np,'einsum',unexpected)
    a=np.ones(shape_a);b=np.ones(shape_b)
    np.testing.assert_array_equal(_matmul.product(a,b),a@b)
