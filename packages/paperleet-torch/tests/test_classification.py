"""Independent scalar classification oracles, finite differences and lifecycle."""
import copy
import math
import numpy as np
import pytest
from paperleet_torch import torch, as_torch, UnsupportedOperation


def scalar_oracle(raw,target,weight,smoothing,soft,reduction,nll=False):
    dim=0 if raw.ndim==1 else 1;c=raw.shape[dim];shape=raw.shape[:dim]+raw.shape[dim+1:]
    losses=np.zeros(shape,dtype=np.float64);gradient=np.zeros(raw.shape,dtype=np.float64)
    for position in np.ndindex(shape):
        key=position[:dim]+(slice(None),)+position[dim:];row=raw[key].astype(np.float64)
        if not soft and target[position]==-100:continue
        if nll:logp=row
        else:
            maximum=float(max(row));partition=math.log(sum(math.exp(float(a)-maximum) for a in row));logp=np.array([float(a)-maximum-partition for a in row])
        if soft:
            t=target[key].astype(np.float64)*(1-smoothing)+smoothing/c
            coeff=t*weight;losses[position]=-sum(float(a)*float(b) for a,b in zip(logp,coeff))
        else:
            label=int(target[position]);coeff=weight*smoothing/c;coeff[label]+=(1-smoothing)*weight[label]
            losses[position]=-sum(float(a)*float(b) for a,b in zip(logp,coeff))
        gradient[key]=-coeff if nll else np.exp(logp)*sum(float(a) for a in coeff)-coeff
    denominator=(math.prod(shape) if soft else sum(float(weight[int(target[p])]) for p in np.ndindex(shape) if target[p]!=-100)) if reduction=='mean' else 1
    return (losses if reduction=='none' else losses.sum()/denominator),gradient/denominator


@pytest.mark.parametrize('kind',['cross_entropy','nll_loss'])
@pytest.mark.parametrize('shape',[(4,),(3,4),(2,4,3),(2,4,2,3)])
@pytest.mark.parametrize('reduction',['none','sum','mean'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_class_index_losses_against_independent_scalar_values_and_gradients(kind,shape,reduction,dtype):
    rng=np.random.default_rng(1808);dim=0 if len(shape)==1 else 1;targetshape=shape[:dim]+shape[dim+1:]
    raw=rng.normal(size=shape).astype(dtype);labels=rng.integers(0,shape[dim],size=targetshape,dtype=np.int64)
    if labels.size>1:labels.reshape(-1)[0]=-100
    weights=rng.uniform(.2,2,size=shape[dim]).astype(dtype);smooth=.2 if kind=='cross_entropy' else 0.
    x=torch.from_numpy(np.asfortranarray(raw) if len(shape)>1 else raw).requires_grad_();kw=dict(reduction=reduction)
    if kind=='cross_entropy':kw['label_smoothing']=smooth
    out=getattr(torch.nn.functional,kind)(x,torch.tensor(labels),torch.tensor(weights),**kw);out.sum().backward()
    value,grad=scalar_oracle(raw,labels,weights,smooth,False,reduction,nll=kind=='nll_loss')
    for actual,expected in [(out,value),(x.grad,grad)]:np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-5 if dtype==torch.float32 else 8e-12,atol=2e-6 if dtype==torch.float32 else 1e-12)


@pytest.mark.parametrize('shape',[(4,),(2,4),(2,4,3)])
@pytest.mark.parametrize('reduction',['none','sum','mean'])
@pytest.mark.parametrize('smoothing',[0.,.25,1.])
def test_probability_targets_trainable_weights_against_scalar_math(shape,reduction,smoothing):
    rng=np.random.default_rng(1809);dim=0 if len(shape)==1 else 1
    arrays=[rng.normal(size=shape),rng.uniform(-.1,1,size=shape),rng.uniform(.1,2,size=shape[dim])]
    x,t,w=[torch.tensor(a,requires_grad=True) for a in arrays]
    y=torch.nn.functional.cross_entropy(x,t,w,reduction=reduction,label_smoothing=smoothing);y.sum().backward()
    value,grad=scalar_oracle(*arrays,smoothing,True,reduction);np.testing.assert_allclose(y.numpy(),value,atol=1e-12);np.testing.assert_allclose(x.grad.numpy(),grad,atol=2e-12)
    def objective(values):return np.sum(scalar_oracle(*values,smoothing,True,reduction)[0])
    for i,variable in enumerate([t,w],1):
        expected=np.empty_like(arrays[i])
        for index in np.ndindex(expected.shape):
            plus=[a.copy() for a in arrays];minus=[a.copy() for a in arrays];plus[i][index]+=1e-6;minus[i][index]-=1e-6
            expected[index]=(objective(plus)-objective(minus))/2e-6
        np.testing.assert_allclose(variable.grad.numpy(),expected,rtol=3e-6,atol=3e-6)


@pytest.mark.parametrize('soft',[False,True])
def test_first_order_gradcheck_and_jvp_reverse_finite_difference(soft):
    rng=np.random.default_rng(1810);arrays=[rng.normal(size=(2,3)),rng.uniform(.2,1,size=(2,3)),rng.uniform(.2,2,size=3)];labels=np.array([0,2],dtype=np.int64)
    arrays=[*arrays,*[rng.normal(size=a.shape) for a in arrays]]
    v=[torch.tensor(a,requires_grad=True) for a in arrays]
    fn=(lambda x,t,w:torch.nn.functional.cross_entropy(x,t,w,label_smoothing=.2)) if soft else (lambda x:torch.nn.functional.cross_entropy(x,torch.tensor(labels),torch.tensor(arrays[2]),label_smoothing=.2))
    assert torch.autograd.gradcheck(fn,tuple(v[:3]) if soft else (v[0],))
    p,t=torch.func.jvp(fn,tuple(v[:3]) if soft else (v[0],),tuple(v[3:]) if soft else (v[3],));(p+t).backward()
    def objective(a):
        x,y,w,dx,dy,dw=a;maximum=x.max(-1,keepdims=True);lp=x-maximum-np.log(np.exp(x-maximum).sum(-1,keepdims=True));probs=np.exp(lp);dlp=dx-(probs*dx).sum(-1,keepdims=True)
        if soft:
            effective=y*.8+.2/3;deffective=dy*.8
            primal=-(lp*effective*w).sum()/2;tangent=-(dlp*effective*w+lp*deffective*w+lp*effective*dw).sum()/2
        else:
            coeff=np.broadcast_to(w*.2/3,x.shape).copy();coeff[np.arange(2),labels]+=.8*w[labels]
            denominator=w[labels].sum();primal=-(lp*coeff).sum()/denominator;tangent=-(dlp*coeff).sum()/denominator
        return primal+tangent
    for i in (range(6) if soft else [0,3]):
        expected=np.empty_like(arrays[i])
        for index in np.ndindex(expected.shape):
            plus=[a.copy() for a in arrays];minus=[a.copy() for a in arrays];plus[i][index]+=1e-6;minus[i][index]-=1e-6
            expected[index]=(objective(plus)-objective(minus))/2e-6
        np.testing.assert_allclose(v[i].grad.numpy(),expected,rtol=3e-6,atol=3e-6)


def test_standard_aliases_loss_buffers_dtype_and_state_dictionary():
    with as_torch():
        from torch.nn import CrossEntropyLoss, NLLLoss
        from torch.nn.modules.loss import NLLLoss as Alias
        import torch.nn.functional as F
        assert Alias is NLLLoss
        weight=torch.tensor([.2,.7,1.4],dtype=torch.float64);x=torch.tensor([[.2,.3,.7],[.5,-.1,.9]],dtype=torch.float64);labels=torch.tensor([0,2])
        for kind,fn in [(CrossEntropyLoss,F.cross_entropy),(NLLLoss,F.nll_loss)]:
            model=kind(weight=weight);np.testing.assert_allclose(model(x,labels).numpy(),fn(x,labels,weight).numpy(),atol=1e-12)
            assert list(model.state_dict())==['weight'] and list(model.parameters())==[]
            state=copy.deepcopy(model.state_dict());model.float();assert model.weight.dtype==torch.float32;model.double();model.load_state_dict(state)
            np.testing.assert_allclose(model.weight.numpy(),weight.numpy(),atol=1e-12)


def test_train_checkpoint_resume_with_weighted_smoothed_objective():
    rng=np.random.default_rng(1811)
    def new():return torch.nn.Sequential(torch.nn.Linear(5,7,dtype=torch.float64),torch.nn.SiLU(),torch.nn.Linear(7,3,dtype=torch.float64))
    a=new();b=new();b.load_state_dict(copy.deepcopy(a.state_dict()));oa=torch.optim.Adam(a.parameters(),lr=.004);ob=torch.optim.Adam(b.parameters(),lr=.004)
    loss=torch.nn.CrossEntropyLoss(weight=torch.tensor([.3,1.5,.8],dtype=torch.float64),label_smoothing=.2)
    for step in range(6):
        x=torch.tensor(rng.normal(size=(6,5)));labels=torch.tensor(rng.integers(0,3,size=6,dtype=np.int64))
        for model,opt in [(a,oa),(b,ob)]:opt.zero_grad();loss(model(x),labels).backward();opt.step()
        for p,q in zip(a.parameters(),b.parameters()):np.testing.assert_allclose(p.numpy(),q.numpy(),atol=1e-12)
        if step==2:
            state=copy.deepcopy(b.state_dict());os=copy.deepcopy(ob.state_dict());b=new();b.load_state_dict(state);ob=torch.optim.Adam(b.parameters(),lr=99);ob.load_state_dict(os)


@pytest.mark.parametrize('kind',['cross_entropy','nll_loss'])
@pytest.mark.parametrize('reduction',['none','sum','mean'])
def test_all_ignored_values_gradients_and_nan_mean_jvp(kind,reduction):
    x=torch.ones(2,3,requires_grad=True);labels=torch.tensor([-100,-100]);fn=lambda z:getattr(torch.nn.functional,kind)(z,labels,reduction=reduction)
    p,t=torch.func.jvp(fn,(x,),(torch.ones_like(x),));p.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),np.zeros((2,3)))
    if reduction=='mean':assert math.isnan(p.item()) and math.isnan(t.item())
    else:assert np.all(p.numpy()==0) and np.all(t.numpy()==0)


@pytest.mark.parametrize('kind',['cross_entropy','nll_loss'])
def test_class_weight_differentiation_boundary_and_no_grad(kind):
    x=torch.ones(2,3,requires_grad=True);w=torch.ones(3,requires_grad=True);labels=torch.tensor([0,2]);fn=getattr(torch.nn.functional,kind)
    with pytest.raises(RuntimeError):fn(x,labels,w)
    with torch.no_grad():assert not fn(x,labels,w).requires_grad
    with pytest.raises(UnsupportedOperation):torch.func.jvp(lambda z:fn(x,labels,z),(w.detach(),),(torch.ones_like(w),))


def test_probability_weight_only_pullback_does_not_retain_unused_tensor_objects():
    import gc,weakref
    x=torch.ones(2,3);target=torch.full((2,3),1/3);weight=torch.ones(3,requires_grad=True)
    refs=[weakref.ref(x),weakref.ref(target)];value=torch.nn.functional.cross_entropy(x,target,weight);del x,target;gc.collect()
    assert all(ref() is None for ref in refs);value.backward();assert weight.grad is not None


@pytest.mark.parametrize('kind',['cross_entropy','nll_loss'])
def test_saved_target_weight_versions_and_output_mutation(kind):
    x=torch.ones(2,3,requires_grad=True);target=torch.tensor([0,2]);weight=torch.ones(3);out=getattr(torch.nn.functional,kind)(x,target,weight)
    with torch.no_grad():out.detach().add_(1)
    out.backward();assert x.grad is not None
    x.grad=None;out=getattr(torch.nn.functional,kind)(x,target,weight)
    with torch.no_grad():target.add_(1)
    with pytest.raises(RuntimeError):out.backward()


@pytest.mark.parametrize('kwargs',[{'reduction':'bad'},{'label_smoothing':1.2},{'ignore_index':0}])
def test_probability_invalid_options(kwargs):
    with pytest.raises((RuntimeError,ValueError)):torch.nn.functional.cross_entropy(torch.ones(2,3),torch.ones(2,3),**kwargs)


def test_finite_unweighted_cache_guards_labels_and_uses_pre_mutation_input_values():
    raw=np.array([[.2,-.4,.7],[.4,1.1,-.9]]);labels=np.array([0,2],dtype=np.int64)
    expected=scalar_oracle(raw,labels,np.ones(3),0.,False,'mean')[1]
    for changed in ['input','target','output']:
        x=torch.tensor(raw,requires_grad=True);target=torch.tensor(labels);out=torch.nn.functional.cross_entropy(x,target)
        value=dict(input=x,target=target,output=out)[changed].detach()
        with torch.no_grad():value.add_(1 if changed!='input' else torch.tensor([[1.,.3,-.2],[-.2,.1,.7]]))
        if changed=='target':
            with pytest.raises(RuntimeError):out.backward()
        else:out.backward();np.testing.assert_allclose(x.grad.numpy(),expected,atol=1e-12)


def test_fast_matrix_index_validation_preserves_bounds_dtype_shape_and_options():
    x=torch.ones(2,3)
    for labels in [[3,0],[-1,0],[0,99]]:
        with pytest.raises(IndexError):torch.nn.functional.cross_entropy(x,torch.tensor(labels))
    with pytest.raises(RuntimeError):torch.nn.functional.cross_entropy(x,torch.tensor([0,1],dtype=torch.int32))
    with pytest.raises(RuntimeError):torch.nn.functional.cross_entropy(x,torch.tensor([0]))
    with pytest.raises(ValueError):torch.nn.functional.cross_entropy(x,torch.tensor([0,1]),reduction='bad')
    assert math.isfinite(torch.nn.functional.cross_entropy(x,torch.tensor([-1,0]),ignore_index=-1).item())
