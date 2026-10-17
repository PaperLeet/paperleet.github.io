"""Independent scalar distance/gradient oracles and bounded tape lifecycles."""
import gc
import io
import itertools
import math
import weakref

import numpy as np
import pytest

from stridetorch import UnsupportedOperation, torch


_ORDERS = [0.,.5,1.,1.5,2.,3.,float('inf')]


def scalar_distance(a,b,p):
    delta=[abs(float(x)-float(y)) for x,y in zip(a,b)]
    if p==0:return float(sum(x!=0 for x in delta))
    if math.isinf(p):return max(delta,default=0.)
    return math.fsum(x**p for x in delta)**(1/p)


def scalar_direction(a,b,p):
    distance=scalar_distance(a,b,p)
    result=[]
    for x,y in zip(a,b):
        d=float(x)-float(y);sign=1. if d>0 else -1. if d<0 else 0.
        if p==0 or not distance:result.append(0.)
        elif math.isinf(p):result.append(sign if abs(d)==distance else 0.)
        elif not d:result.append(0.)
        else:result.append(sign*abs(d)**(p-1)/distance**(p-1))
    return result


@pytest.fixture(params=[4096,8*1024*1024])
def workspace(request,monkeypatch):
    from stridetorch import _distance
    monkeypatch.setattr(_distance,'_WORKSPACE_BYTES',request.param)
    return request.param


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('p',_ORDERS)
def test_cdist_and_condensed_pdist_values_match_independent_scalar_oracle(dtype,p,workspace):
    data=[[0.,0.,0.],[1.,-1.,2.],[2.,0.,1.],[1.,-1.,2.]]
    other=[[1.,0.,1.],[0.,2.,-1.]]
    a=torch.tensor(data,dtype=dtype);b=torch.tensor(other,dtype=dtype)
    expected=[[scalar_distance(x,y,p) for y in other] for x in data]
    np.testing.assert_allclose(torch.cdist(a,b,p=p).numpy(),expected,rtol=3e-6,atol=3e-6)
    condensed=[scalar_distance(data[i],data[j],p) for i in range(4) for j in range(i+1,4)]
    np.testing.assert_allclose(torch.pdist(a,p=p).numpy(),condensed,rtol=3e-6,atol=3e-6)
    np.testing.assert_array_equal(torch.nn.functional.pdist(a,p=p).numpy(),torch.pdist(a,p=p).numpy())


@pytest.mark.parametrize('p',[.5,1.,1.5,2.,3.])
@pytest.mark.parametrize('kind',['cdist','pdist'])
def test_distance_pullbacks_match_independent_scalar_finite_differences(p,kind,workspace):
    data=np.array([[.2,-1.3,.7],[2.1,.4,-.6],[-1.7,-.2,-.9]])
    other=np.array([[-.5,.9,1.8],[1.3,-2.2,.1]])
    weights=np.sin(np.arange(6)+.3).reshape(3,2) if kind=='cdist' else np.array([.7,-1.2,.3])
    def objective(a,b):
        values=[[scalar_distance(x,y,p) for y in b] for x in a] if kind=='cdist' else [scalar_distance(a[i],a[j],p) for i in range(3) for j in range(i+1,3)]
        return float(np.sum(np.asarray(values)*weights))
    x=torch.tensor(data,dtype=torch.float64,requires_grad=True);y=torch.tensor(other,dtype=torch.float64,requires_grad=True)
    value=torch.cdist(x,y,p=p) if kind=='cdist' else torch.pdist(x,p=p)
    (value*torch.tensor(weights,dtype=torch.float64)).sum().backward()
    operands=[(data,x.grad.numpy(),0)] if kind=='pdist' else [(data,x.grad.numpy(),0),(other,y.grad.numpy(),1)]
    for original,gradient,which in operands:
        for index in np.ndindex(original.shape):
            plus,minus=original.copy(),original.copy();plus[index]+=1e-6;minus[index]-=1e-6
            fd=(objective(plus,other)-objective(minus,other))/(2e-6) if which==0 else (objective(data,plus)-objective(data,minus))/(2e-6)
            assert gradient[index]==pytest.approx(fd,rel=3e-7,abs=3e-8)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('p',_ORDERS)
def test_broadcast_gradient_accumulation_matches_independent_pair_loops(dtype,p,workspace):
    a=np.sin(np.arange(2*1*3*11)+.2).reshape(2,1,3,11)
    b=np.cos(np.arange(1*3*4*11)+.7).reshape(1,3,4,11)
    a=a.astype(dtype);b=b.astype(dtype)
    seed=np.sin(np.arange(2*3*3*4)+.4).reshape(2,3,3,4).astype(dtype)
    expected=np.empty(seed.shape);ga=np.zeros_like(a,dtype='float64');gb=np.zeros_like(b,dtype='float64')
    for i,j,k,l in itertools.product(range(2),range(3),range(3),range(4)):
        x,y=a[i,0,k],b[0,j,l];expected[i,j,k,l]=scalar_distance(x,y,p)
        direction=np.array(scalar_direction(x,y,p))*float(seed[i,j,k,l]);ga[i,0,k]+=direction;gb[0,j,l]-=direction
    x=torch.tensor(a,requires_grad=True);y=torch.tensor(b,requires_grad=True);value=torch.cdist(x,y,p=p)
    (value*torch.tensor(seed)).sum().backward()
    for actual,want in [(value.numpy(),expected),(x.grad.numpy(),ga),(y.grad.numpy(),gb)]:np.testing.assert_allclose(actual,want,rtol=3e-5,atol=8e-6)


@pytest.mark.parametrize('kind',['cdist','pdist'])
@pytest.mark.parametrize('p',[0.,1.,2.,float('inf')])
def test_nonfinite_component_and_infinity_norm_native_cpu_conventions(kind,p,workspace):
    x=torch.tensor([[float('nan'),1.],[0.,0.]],dtype=torch.float64,requires_grad=True)
    value=torch.cdist(x[:1],x[1:],p=p) if kind=='cdist' else torch.pdist(x,p=p)
    value.sum().backward()
    if p==0:
        assert np.isnan(value.item());np.testing.assert_array_equal(x.grad.numpy(),0)
    elif p==1 or p==float('inf') and kind=='cdist':
        assert value.item()==1 if p==float('inf') else np.isnan(value.item())
        np.testing.assert_array_equal(np.isnan(x.grad.numpy()),[[True,False],[True,False]])
        np.testing.assert_array_equal(x.grad.numpy()[:,1],[1,-1])
    else:assert np.isnan(value.item()) and np.isnan(x.grad.numpy()).all()


@pytest.mark.parametrize('kind',['cdist','pdist'])
def test_infinity_norm_ties_receive_full_signed_gradient(kind,workspace):
    x=torch.tensor([[1.,-1.],[0.,0.]],dtype=torch.float64,requires_grad=True)
    value=torch.cdist(x[:1],x[1:],p=float('inf')) if kind=='cdist' else torch.pdist(x,p=float('inf'))
    (value*3).sum().backward();np.testing.assert_array_equal(x.grad.numpy(),[[3.,-3.],[-3.,3.]])


@pytest.mark.parametrize('shape',[(0,3),(1,3),(3,0)])
@pytest.mark.parametrize('p',_ORDERS)
def test_pdist_empty_geometry_has_independent_zero_gradient_contract(shape,p,workspace):
    x=torch.ones(*shape,dtype=torch.float64,requires_grad=True);value=torch.pdist(x,p=p)
    assert value.shape==(shape[0]*(shape[0]-1)//2,)
    np.testing.assert_array_equal(value.numpy(),0);value.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),np.zeros(shape))


@pytest.mark.parametrize('kind',['cdist','pdist'])
@pytest.mark.parametrize('rows',[3,32])
def test_distance_saved_arrays_release_with_the_graph_and_do_not_keep_nonleaf_tensor(kind,rows):
    a=torch.tensor(np.sin(np.arange(rows*2)+.2).reshape(rows,2),dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[.3,-.1],[1.4,.6]],dtype=torch.float64,requires_grad=True)
    x=a*2;y=b*3;owner=weakref.ref(x);storage=weakref.ref(x._data)
    value=torch.cdist(x,y,p=1.5) if kind=='cdist' else torch.pdist(x,p=1.5)
    del x,y;gc.collect();assert owner() is None and storage() is not None
    value.sum().backward();gc.collect();assert storage() is None


@pytest.mark.parametrize('p',[0.,.5,1.,2.,3.,float('inf')])
def test_retained_distance_graph_recomputes_for_each_seed(p):
    x=torch.tensor([[.2,1.3],[2.,-.4]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[.3,-.1],[1.4,.6]],dtype=torch.float64,requires_grad=True);value=torch.cdist(x,y,p=p)
    first=torch.autograd.grad(value,(x,y),torch.ones_like(value),retain_graph=True)
    second=torch.autograd.grad(value,(x,y),torch.full_like(value,2.))
    for a,b in zip(first,second):np.testing.assert_allclose(b.numpy(),2*a.numpy(),rtol=1e-13,atol=1e-13)
    with pytest.raises(RuntimeError):value.sum().backward()


@pytest.mark.parametrize('target',['first','second','output'])
def test_cdist_mutation_guards_cover_both_inputs_and_saved_lengths(target):
    x=torch.ones(2,3,requires_grad=True);y=torch.zeros(3,3,requires_grad=True);value=torch.cdist(x,y)
    with torch.no_grad():{'first':x,'second':y,'output':value}[target].detach().add_(1.)
    with pytest.raises(UnsupportedOperation):value.sum().backward()


def test_previously_rejected_feature_workload_is_tiled_and_differentiable():
    x=torch.ones(128,1024,requires_grad=True);y=torch.zeros(129,1024,requires_grad=True)
    value=torch.cdist(x,y,compute_mode='donot_use_mm_for_euclid_dist');np.testing.assert_array_equal(value.numpy(),32.)
    value.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),129/32);np.testing.assert_array_equal(y.grad.numpy(),-4.)


def test_pairwise_output_and_work_budgets_reject_before_materializing_broadcast_views():
    small=torch.ones(1,1)
    with pytest.raises(UnsupportedOperation):torch.cdist(small.expand(4097,1),small.expand(4097,1))
    with pytest.raises(UnsupportedOperation):torch.cdist(small.expand(1000,1024),small.expand(1000,1024))
    with pytest.raises(UnsupportedOperation):torch.pdist(small.expand(1000,1026))
    for p in [-1.,float('-inf'),float('nan')]:
        with pytest.raises(RuntimeError):torch.cdist(small,small,p=p)
        with pytest.raises(RuntimeError):torch.pdist(small,p=p)
    for p in ['2',1j]:
        with pytest.raises(TypeError):torch.cdist(small,small,p=p)
    with pytest.raises(UnsupportedOperation):torch.cdist(small.long(),small.long())


def test_distance_training_checkpoint_and_continuation_are_reproducible():
    model=torch.nn.Linear(3,4,dtype=torch.float64);opt=torch.optim.AdamW(model.parameters(),lr=.01)
    x=torch.tensor(np.sin(np.arange(15)+.2).reshape(5,3),dtype=torch.float64)
    anchors=torch.tensor(np.cos(np.arange(12)+.4).reshape(3,4),dtype=torch.float64)
    def step(m,o):
        o.zero_grad();loss=torch.cdist(m(x),anchors,p=1.5).square().mean();loss.backward();o.step();return loss.item()
    step(model,opt);stream=io.BytesIO();torch.save({'model':model.state_dict(),'opt':opt.state_dict()},stream);stream.seek(0);saved=torch.load(stream)
    clone=torch.nn.Linear(3,4,dtype=torch.float64);newopt=torch.optim.AdamW(clone.parameters(),lr=.01);clone.load_state_dict(saved['model']);newopt.load_state_dict(saved['opt'])
    for _ in range(3):assert step(model,opt)==step(clone,newopt)
    for a,b in zip(model.parameters(),clone.parameters()):np.testing.assert_array_equal(a.numpy(),b.numpy())


@pytest.mark.parametrize('kind',['cdist','pdist'])
@pytest.mark.parametrize('p',[.5,1.5,2.,3.])
@pytest.mark.parametrize('seed',[float('inf'),float('nan')])
def test_zero_distance_vjp_is_zero_even_for_nonfinite_upstream(kind,p,seed):
    x=torch.ones(2,3,dtype=torch.float64,requires_grad=True)
    value=torch.cdist(x[:1],x[1:],p=p,compute_mode='donot_use_mm_for_euclid_dist') if kind=='cdist' else torch.pdist(x,p=p)
    value.backward(torch.full_like(value,seed));np.testing.assert_array_equal(x.grad.numpy(),0.)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('p',_ORDERS)
def test_condensed_medium_pdist_values_and_gradients_match_scalar_pair_loops(dtype,p,workspace):
    data=np.cos(np.arange(17*3)*.19+.2).reshape(17,3).astype(dtype)
    seed=np.sin(np.arange(17*16//2)+.3).astype(dtype)
    expected=[];gradient=np.zeros(data.shape,dtype='float64');index=0
    for i in range(17):
        for j in range(i+1,17):
            expected.append(scalar_distance(data[i],data[j],p))
            direction=np.asarray(scalar_direction(data[i],data[j],p))*float(seed[index])
            gradient[i]+=direction;gradient[j]-=direction;index+=1
    x=torch.tensor(data,requires_grad=True);value=torch.pdist(x,p=p)
    (value*torch.tensor(seed)).sum().backward()
    tolerance=(3e-5,8e-6) if dtype==torch.float32 else (3e-12,3e-12)
    np.testing.assert_allclose(value.numpy(),expected,rtol=tolerance[0],atol=tolerance[1])
    np.testing.assert_allclose(x.grad.numpy(),gradient,rtol=tolerance[0],atol=tolerance[1])
