"""Native-reference weighted spatial CE/NLL and differentiable probability labels."""
import copy
import numpy as np
import pytest
import torch
from stridetorch import torch as lite


def close(a,b,dtype=np.float64):
    a=a.detach().numpy() if hasattr(a,'detach') else np.asarray(a)
    b=b.detach().numpy() if hasattr(b,'detach') else np.asarray(b)
    np.testing.assert_allclose(a,b,rtol=8e-6 if dtype==np.float32 else 1e-11,atol=2e-6 if dtype==np.float32 else 1e-12,equal_nan=True)


@pytest.mark.parametrize('kind',['cross_entropy','nll_loss'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(5,),(4,5),(2,5,3),(2,5,2,3),(2,5,2,3,2),(0,5),(2,5,0)])
@pytest.mark.parametrize('reduction',['none','sum','mean'])
@pytest.mark.parametrize('weighted',[False,True])
def test_index_values_and_active_pullback_all_ranks_layouts(kind,dtype,shape,reduction,weighted):
    rng=np.random.default_rng(1802);dim=0 if len(shape)==1 else 1;targetshape=shape[:dim]+shape[dim+1:]
    raw=rng.normal(size=shape).astype(dtype);raw=np.asfortranarray(raw) if len(shape)>1 else raw
    labels=rng.integers(0,shape[dim],size=targetshape,dtype=np.int64)
    if labels.size>1:labels.reshape(-1)[0]=-100
    weight=rng.uniform(.1,2,size=shape[dim]).astype(dtype) if weighted else None
    records=[]
    for api in [lite,torch]:
        # Native 2.14.1 macOS NLL2D backward rejects Fortran grad storage.
        # Compare our original layout with a contiguous native value oracle.
        oracle_raw=np.ascontiguousarray(raw) if api is torch and kind=='nll_loss' and len(shape)==4 else raw
        x=api.from_numpy(oracle_raw).requires_grad_();y=api.tensor(labels);w=api.tensor(weight) if weighted else None
        z=getattr(api.nn.functional,kind)(x,y,w,reduction=reduction)
        seed=rng.uniform(.2,1.2,size=z.shape).astype(dtype) if api is lite else seed
        for step in range(2):z.backward(api.tensor(seed),retain_graph=step==0)
        records.append([z,x.grad])
    for a,b in zip(*records):close(a,b,dtype)
    assert records[0][0].dtype==np.dtype(dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(5,),(3,5),(2,5,3),(2,5,2,3),(0,5),(2,5,0)])
@pytest.mark.parametrize('smoothing',[-.2,.15,1.])
@pytest.mark.parametrize('reduction',['none','sum','mean'])
def test_smoothed_weighted_index_values_and_gradients(dtype,shape,smoothing,reduction):
    rng=np.random.default_rng(1803);dim=0 if len(shape)==1 else 1;targetshape=shape[:dim]+shape[dim+1:]
    raw=rng.normal(size=shape).astype(dtype);labels=rng.integers(0,shape[dim],size=targetshape,dtype=np.int64)
    if labels.size>1:labels.reshape(-1)[-1]=-100
    weight=rng.uniform(.1,2,size=shape[dim]).astype(dtype);records=[]
    for api in [lite,torch]:
        x=api.tensor(raw,requires_grad=True);z=api.nn.functional.cross_entropy(x,api.tensor(labels),api.tensor(weight),label_smoothing=smoothing,reduction=reduction)
        z.sum().backward();records.append([z,x.grad])
    for a,b in zip(*records):close(a,b,dtype)


@pytest.mark.parametrize('shape',[(5,),(2,5),(2,5,3),(2,5,2,3)])
@pytest.mark.parametrize('reduction',['none','sum','mean'])
@pytest.mark.parametrize('smoothing',[0.,.2,1.])
@pytest.mark.parametrize('dtypes',[(np.float32,np.float32,np.float32),(np.float32,np.float64,np.float32),(np.float64,np.float32,np.float64),(np.float32,np.float32,np.float64)])
def test_probability_targets_and_weights_all_gradients_with_mixed_dtypes(shape,reduction,smoothing,dtypes):
    rng=np.random.default_rng(1804);dim=0 if len(shape)==1 else 1
    # Probability targets are deliberately not normalized; native permits this.
    arrays=[rng.normal(size=shape).astype(dtypes[0]),rng.uniform(-.1,1,size=shape).astype(dtypes[1]),rng.uniform(.1,2,size=shape[dim]).astype(dtypes[2])]
    records=[]
    for api in [lite,torch]:
        v=[api.tensor(a,requires_grad=True) for a in arrays];z=api.nn.functional.cross_entropy(*v,label_smoothing=smoothing,reduction=reduction);z.sum().backward();records.append([z,*[x.grad for x in v]])
    for a,b in zip(*records):close(a,b,np.float32 if np.float32 in dtypes else np.float64)
    assert records[0][0].dtype==np.result_type(*dtypes)


@pytest.mark.parametrize('kind',['cross_entropy','nll_loss'])
@pytest.mark.parametrize('shape',[(4,),(2,4),(2,4,3),(2,4,2,3)])
@pytest.mark.parametrize('reduction',['none','sum','mean'])
def test_index_jvp_and_reverse_over_forward(kind,shape,reduction):
    rng=np.random.default_rng(1805);dim=0 if len(shape)==1 else 1;arrays=[rng.normal(size=shape),rng.normal(size=shape)];labels=rng.integers(0,shape[dim],size=shape[:dim]+shape[dim+1:],dtype=np.int64);weights=rng.uniform(.3,2,size=shape[dim]);records=[]
    for api in [lite,torch]:
        x,dx=[api.tensor(a,requires_grad=True) for a in arrays];kwargs=dict(reduction=reduction)
        if kind=='cross_entropy':kwargs['label_smoothing']=.2
        fn=lambda z:getattr(api.nn.functional,kind)(z,api.tensor(labels),api.tensor(weights),**kwargs)
        p,t=api.func.jvp(fn,(x,),(dx,));(p.square().sum()+t.square().sum()).backward();records.append([p,t,x.grad,dx.grad])
    for a,b in zip(*records):close(a,b)


@pytest.mark.parametrize('shape',[(4,),(2,4),(2,4,3)])
@pytest.mark.parametrize('active',['all','input','target','weight'])
@pytest.mark.parametrize('smoothing',[0.,.2,1.])
def test_probability_jvp_all_inputs_and_reverse_tapes(shape,active,smoothing):
    rng=np.random.default_rng(1806);dim=0 if len(shape)==1 else 1;arrays=[rng.normal(size=shape),rng.uniform(.1,1,size=shape),rng.uniform(.3,2,size=shape[dim])];arrays +=[rng.normal(size=a.shape) for a in arrays];records=[]
    for api in [lite,torch]:
        v=[api.tensor(a,requires_grad=True) for a in arrays]
        if active=='all':fn=lambda x,y,w:api.nn.functional.cross_entropy(x,y,w,label_smoothing=smoothing);p,t=api.func.jvp(fn,tuple(v[:3]),tuple(v[3:]))
        else:
            i=['input','target','weight'].index(active)
            def fn(a):
                args=v[:3].copy();args[i]=a;return api.nn.functional.cross_entropy(*args,label_smoothing=smoothing)
            p,t=api.func.jvp(fn,(v[i],),(v[i+3],))
        (p.square().sum()+t.square().sum()).backward();records.append([p,t,*[x.grad for x in v]])
    for a,b in zip(*records):
        if a is None or b is None:assert a is b
        else:close(a,b)


@pytest.mark.parametrize('kind',['cross_entropy','nll_loss'])
@pytest.mark.parametrize('changed',['input','target','weight','output'])
@pytest.mark.parametrize('smoothing',[0.,.2])
def test_index_saved_versions_follow_native(kind,changed,smoothing):
    results=[]
    for api in [lite,torch]:
        x=api.ones(2,3,requires_grad=True);y=api.tensor([0,2]);w=api.ones(3);kwargs={'label_smoothing':smoothing} if kind=='cross_entropy' else {}
        out=getattr(api.nn.functional,kind)(x,y,w,**kwargs);target=dict(input=x,target=y,weight=w,output=out)[changed].detach()
        with api.no_grad():target.add_(1)
        try:out.backward();results.append('ok')
        except RuntimeError:results.append('RuntimeError')
    assert results[0]==results[1]


@pytest.mark.parametrize('smoothing',[0.,.2])
@pytest.mark.parametrize('needs',[(True,False,False),(False,True,False),(False,False,True),(True,True,True)])
@pytest.mark.parametrize('changed',['input','target','weight','output'])
def test_probability_saved_versions_depend_on_active_operands(smoothing,needs,changed):
    results=[]
    for api in [lite,torch]:
        x=api.ones(2,3,requires_grad=needs[0]);y=api.full((2,3),1/3,requires_grad=needs[1]);w=api.ones(3,requires_grad=needs[2])
        out=api.nn.functional.cross_entropy(x,y,w,label_smoothing=smoothing);target=dict(input=x,target=y,weight=w,output=out)[changed].detach()
        with api.no_grad():target.add_(1)
        try:out.backward();results.append('ok')
        except RuntimeError:results.append('RuntimeError')
    assert results[0]==results[1]


@pytest.mark.parametrize('kind',['cross_entropy','nll_loss'])
@pytest.mark.parametrize('reduction',['none','sum','mean'])
@pytest.mark.parametrize('weights',[[0.,1.,2.],[0.,0.,0.],[-1.,1.,2.],[1.,-1.,2.]])
@pytest.mark.parametrize('smoothing',[0.,.2,1.])
def test_zero_negative_weight_denominators_ignored_rows_and_nonfinite_classes(kind,reduction,weights,smoothing):
    records=[]
    for api in [lite,torch]:
        x=api.tensor([[.2,-.3,.7],[.8,.4,-.2],[.7,-.2,.4]],dtype=api.float64,requires_grad=True);labels=api.tensor([0,1,-100]);w=api.tensor(weights,dtype=api.float64)
        kw=dict(reduction=reduction)
        if kind=='cross_entropy':kw['label_smoothing']=smoothing
        z=getattr(api.nn.functional,kind)(x,labels,w,**kw);z.sum().backward();records.append([z,x.grad])
    for a,b in zip(*records):close(a,b)


@pytest.mark.parametrize('kind',['cross_entropy','nll_loss'])
@pytest.mark.parametrize('smoothing',[0.,.2])
def test_nonfinite_logits_match_native_values_and_gradient_classes(kind,smoothing):
    raw=np.array([[np.nan,0.,1.],[np.inf,0.,1.],[-np.inf,0.,1.],[-np.inf]*3]);labels=np.array([-100,0,1,2]);records=[]
    for api in [lite,torch]:
        x=api.tensor(raw,requires_grad=True);kw=dict(reduction='none')
        if kind=='cross_entropy':kw['label_smoothing']=smoothing
        z=getattr(api.nn.functional,kind)(x,api.tensor(labels),**kw);z.sum().backward();records.append([z,x.grad])
    for a,b in zip(*records):close(a,b)


@pytest.mark.parametrize('shape',[(0,3),(2,3,0),(2,0),(2,0,3)])
@pytest.mark.parametrize('soft',[False,True])
@pytest.mark.parametrize('reduction',['none','sum','mean'])
def test_empty_axes_and_all_ignored_targets(shape,soft,reduction):
    records=[]
    for api in [lite,torch]:
        x=api.ones(shape,dtype=api.float64,requires_grad=True);y=api.zeros(shape,dtype=api.float64,requires_grad=True) if soft else api.full((shape[0],*shape[2:]),-100,dtype=api.int64)
        if not soft and shape[1]==0 and len(shape)>2:
            with pytest.raises(RuntimeError):api.nn.functional.cross_entropy(x,y,label_smoothing=.2,reduction=reduction)
            continue
        z=api.nn.functional.cross_entropy(x,y,label_smoothing=.2,reduction=reduction);z.sum().backward();records.append([z,x.grad,*([y.grad] if soft else [])])
    for a,b in zip(*records):close(a,b)


def test_complete_weighted_smoothed_classifier_adam_checkpoint_resume():
    rng=np.random.default_rng(1807);models=[api.nn.Sequential(api.nn.Linear(6,8,dtype=api.float64),api.nn.SiLU(),api.nn.Linear(8,4,dtype=api.float64)) for api in [lite,torch]]
    for pair in zip(*[list(m.parameters()) for m in models]):
        raw=rng.normal(size=pair[0].shape)*.2
        for api,p in zip([lite,torch],pair):
            with api.no_grad():p.copy_(api.tensor(raw))
    losses=[api.nn.CrossEntropyLoss(weight=api.tensor([.3,1.2,2.,.6],dtype=api.float64),label_smoothing=.15) for api in [lite,torch]]
    optimizers=[api.optim.Adam(m.parameters(),lr=.005) for api,m in zip([lite,torch],models)]
    for step in range(8):
        raw=rng.normal(size=(7,6));labels=rng.integers(0,4,size=7,dtype=np.int64);labels[0]=-100;values=[]
        for api,m,loss,opt in zip([lite,torch],models,losses,optimizers):
            opt.zero_grad();value=loss(m(api.tensor(raw)),api.tensor(labels));value.backward();opt.step();values.append(value)
        close(*values)
        for p,q in zip(models[0].parameters(),models[1].parameters()):close(p,q)
        if step==3:
            for api,i in [(lite,0),(torch,1)]:
                state=copy.deepcopy(models[i].state_dict());lstate=copy.deepcopy(losses[i].state_dict());ostate=copy.deepcopy(optimizers[i].state_dict())
                models[i]=api.nn.Sequential(api.nn.Linear(6,8,dtype=api.float64),api.nn.SiLU(),api.nn.Linear(8,4,dtype=api.float64));models[i].load_state_dict(state)
                losses[i]=api.nn.CrossEntropyLoss(weight=api.ones(4,dtype=api.float64),label_smoothing=.15);losses[i].load_state_dict(lstate)
                optimizers[i]=api.optim.Adam(models[i].parameters(),lr=99);optimizers[i].load_state_dict(ostate)


@pytest.mark.parametrize('changed',['input','target','output'])
def test_finite_unweighted_cache_keeps_native_selective_guards(changed):
    records=[]
    for api in [lite,torch]:
        x=api.tensor([[.2,-.4,.7],[.4,1.1,-.9]],dtype=api.float64,requires_grad=True);labels=api.tensor([0,2]);out=api.nn.functional.cross_entropy(x,labels)
        value=dict(input=x,target=labels,output=out)[changed].detach()
        with api.no_grad():value.add_(1 if changed!='input' else api.tensor([[1.,.3,-.2],[-.2,.1,.7]],dtype=api.float64))
        if changed=='target':
            with pytest.raises(RuntimeError):out.backward()
        else:out.backward();records.append(x.grad)
    if records:close(*records)


@pytest.mark.parametrize('shape',[(4,),(2,4),(2,4,3),(0,4)])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_finite_unweighted_cache_jvp_and_reverse_tape(shape,dtype):
    rng=np.random.default_rng(1814);dim=0 if len(shape)==1 else 1;arrays=[rng.normal(size=shape).astype(dtype),rng.normal(size=shape).astype(dtype)];labels=rng.integers(0,shape[dim],size=shape[:dim]+shape[dim+1:],dtype=np.int64);records=[]
    for api in [lite,torch]:
        x,dx=[api.tensor(a,requires_grad=True) for a in arrays];p,t=api.func.jvp(lambda z:api.nn.functional.cross_entropy(z,api.tensor(labels)),(x,),(dx,));(p+t).backward();records.append([p,t,x.grad,dx.grad])
    for a,b in zip(*records):close(a,b,dtype)
