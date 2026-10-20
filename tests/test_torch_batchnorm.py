"""Native-reference batch normalization, running state and reverse/forward tapes."""
import copy
import numpy as np
import pytest
import torch
from torchwasm import torch as light, UnsupportedOperation


def close(a,b,dtype):
    actual=a.numpy() if hasattr(a,'numpy') else np.asarray(a)
    expected=b.detach().numpy() if hasattr(b,'detach') else np.asarray(b)
    tolerance=6e-6 if dtype==np.float32 else 3e-12
    np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(5,3),(2,3,4),(2,3,2,4),(2,3,2,2,3)])
@pytest.mark.parametrize('training',[False,True])
@pytest.mark.parametrize('affine',[False,True])
@pytest.mark.parametrize('layout',['contiguous','fortran','stride'])
def test_function_values_gradients_running_stats_and_retained_seeds(dtype,shape,training,affine,layout):
    rng=np.random.default_rng(1604)
    raw=rng.normal(size=shape).astype(dtype)
    if layout=='fortran':raw=np.asfortranarray(raw)
    elif layout=='stride':raw=np.repeat(raw,2,axis=-1)[...,::2]
    arrays=[raw,rng.normal(size=3).astype(dtype),rng.normal(size=3).astype(dtype)]
    a=[light.from_numpy(value).requires_grad_() for value in arrays]
    b=[torch.from_numpy(value).requires_grad_() for value in arrays]
    mean=rng.normal(size=3).astype(dtype);variance=rng.uniform(.2,2,size=3).astype(dtype)
    am,av=light.tensor(mean),light.tensor(variance);bm,bv=torch.tensor(mean),torch.tensor(variance)
    y=light.nn.functional.batch_norm(a[0],am,av,*a[1:] if affine else (None,None),training=training)
    yy=torch.nn.functional.batch_norm(b[0],bm,bv,*b[1:] if affine else (None,None),training=training)
    close(y,yy,dtype);close(am,bm,dtype);close(av,bv,dtype)
    assert y.dtype==np.dtype(dtype) and y.shape==shape
    for i in range(2):
        seed=rng.normal(size=shape).astype(dtype)
        y.backward(light.tensor(seed),retain_graph=i==0);yy.backward(torch.tensor(seed),retain_graph=i==0)
        for x,z in zip(a if affine else a[:1],b if affine else b[:1]):close(x.grad,z.grad,dtype)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('dimension,shape',[(1,(4,3)),(1,(2,3,4)),(2,(2,3,2,4)),(3,(2,3,2,2,3))])
@pytest.mark.parametrize('momentum',[None,.1,0.,1.4])
@pytest.mark.parametrize('tracking',[False,True])
def test_module_train_eval_state_roundtrip_and_cumulative_average(dtype,dimension,shape,momentum,tracking):
    rng=np.random.default_rng(1605);kwargs=dict(momentum=momentum,track_running_stats=tracking)
    a=getattr(light.nn,f'BatchNorm{dimension}d')(3,dtype=np.dtype(dtype),**kwargs)
    b=getattr(torch.nn,f'BatchNorm{dimension}d')(3,dtype=torch.float32 if dtype==np.float32 else torch.float64,**kwargs)
    for step in range(3):
        raw=(rng.normal(size=shape)*(.5+step)+step).astype(dtype)
        x=light.tensor(raw,requires_grad=True);xx=torch.tensor(raw,requires_grad=True)
        y,yy=a(x),b(xx);close(y,yy,dtype)
        y.square().sum().backward();yy.square().sum().backward();close(x.grad,xx.grad,dtype)
        if tracking:
            close(a.running_mean,b.running_mean,dtype);close(a.running_var,b.running_var,dtype)
            assert a.num_batches_tracked.item()==b.num_batches_tracked.item()
    a.eval();b.eval();raw=rng.normal(size=shape).astype(dtype);close(a(light.tensor(raw)),b(torch.tensor(raw)),dtype)
    snapshot=copy.deepcopy(a.state_dict());rebuilt=getattr(light.nn,f'BatchNorm{dimension}d')(3,dtype=np.dtype(dtype),**kwargs).eval()
    rebuilt.load_state_dict(snapshot)
    close(rebuilt(light.tensor(raw)),a(light.tensor(raw)),dtype)
    assert snapshot._metadata['']['version']==2
    a.reset_parameters();b.reset_parameters()
    for x,z in zip(a.parameters(),b.parameters()):close(x,z,dtype)
    if tracking:close(a.running_var,b.running_var,dtype)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('training',[False,True])
@pytest.mark.parametrize('dual',[0,1,2,'all'])
@pytest.mark.parametrize('shape',[(5,3),(2,3,4),(2,3,2,4),(2,3,2,2,3)])
def test_jvp_and_native_reverse_tape(dtype,training,dual,shape):
    rng=np.random.default_rng(1606);raw=[rng.normal(size=shape).astype(dtype),rng.normal(size=3).astype(dtype),rng.normal(size=3).astype(dtype)]
    directions=[rng.normal(size=value.shape).astype(dtype) for value in raw]
    a=[light.tensor(value,requires_grad=True) for value in raw+directions];b=[torch.tensor(value,requires_grad=True) for value in raw+directions]
    am,av=light.zeros(3,dtype=np.dtype(dtype)),light.ones(3,dtype=np.dtype(dtype));bm,bv=torch.zeros(3,dtype=b[0].dtype),torch.ones(3,dtype=b[0].dtype)
    active=list(range(3)) if dual=='all' else [dual]
    def invoke(values,owner):
        args=owner[:3].copy()
        for index,value in zip(active,values):args[index]=value
        if owner is a:return light.nn.functional.batch_norm(args[0],am,av,*args[1:],training=training)
        return torch.nn.functional.batch_norm(args[0],bm,bv,*args[1:],training=training)
    p,t=light.func.jvp(lambda *values:invoke(values,a),tuple(a[i] for i in active),tuple(a[3+i] for i in active))
    pp,tt=torch.func.jvp(lambda *values:invoke(values,b),tuple(b[i] for i in active),tuple(b[3+i] for i in active))
    close(p,pp,dtype);close(t,tt,dtype);close(am,bm,dtype);close(av,bv,dtype)
    seed=rng.normal(size=p.shape).astype(dtype)
    (p.square()+t*light.tensor(seed)).sum().backward();(pp.square()+tt*torch.tensor(seed)).sum().backward()
    for index in [0,1,2,*[3+i for i in active]]:close(a[index].grad,b[index].grad,dtype)


@pytest.mark.parametrize('training',[False,True])
@pytest.mark.parametrize('active',[(True,False,False),(False,True,False),(False,False,True)])
@pytest.mark.parametrize('changed',['input','weight','bias','mean','variance','output'])
def test_saved_versions_follow_native_even_for_frozen_operands(training,active,changed):
    statuses=[]
    for api,operation in [(light,light.nn.functional.batch_norm),(torch,torch.nn.functional.batch_norm)]:
        x=api.ones(4,3,requires_grad=active[0]);w=api.ones(3,requires_grad=active[1]);b=api.zeros(3,requires_grad=active[2])
        mean=api.zeros(3);variance=api.ones(3)
        value=operation(x,mean,variance,w,b,training=training);loss=value.sum()
        target=dict(input=x,weight=w,bias=b,mean=mean,variance=variance,output=value)[changed]
        # Direct differentiable non-leaf writes remain an existing library
        # boundary; a detached alias exercises the same saved storage counter.
        if changed=='output':target=target.detach()
        with api.no_grad():target.add_(1.)
        try:loss.backward()
        except (RuntimeError,UnsupportedOperation):statuses.append('guarded')
        else:statuses.append('allowed')
    assert statuses[0]==statuses[1],(active,changed,statuses)

@pytest.mark.parametrize('shape',[(0,3),(2,3,0),(0,3,2,4),(2,3,0,4),(2,3,2,0,3)])
@pytest.mark.parametrize('training',[False,True])
def test_empty_input_keeps_statistics_and_has_zero_parameter_gradients(shape,training):
    for api,operation in [(light,light.nn.functional.batch_norm),(torch,torch.nn.functional.batch_norm)]:
        x=api.ones(shape,requires_grad=True);w=api.ones(3,requires_grad=True);b=api.zeros(3,requires_grad=True)
        mean=api.zeros(3);variance=api.ones(3)
        value=operation(x,mean,variance,w,b,training=training);value.sum().backward()
        assert tuple(value.shape)==shape and tuple(x.grad.shape)==shape
        assert w.grad.tolist()==b.grad.tolist()==[0.,0.,0.]
        assert mean.tolist()==[0.,0.,0.] and variance.tolist()==[1.,1.,1.]

@pytest.mark.parametrize('first_training',[False,True])
def test_running_updates_do_not_invalidate_graph_and_eval_backward_reads_current_buffers(first_training):
    rng=np.random.default_rng(1607);raw=rng.normal(size=(5,3));later=rng.normal(size=(7,3))*4+2;seed=rng.normal(size=raw.shape)
    a=light.nn.BatchNorm1d(3,dtype=np.float64).train(first_training)
    b=torch.nn.BatchNorm1d(3,dtype=torch.float64).train(first_training)
    x=light.tensor(raw,requires_grad=True);xx=torch.tensor(raw,requires_grad=True)
    value,other=a(x),b(xx)
    av=[a.running_mean._storage.version,a.running_var._storage.version]
    bv=[b.running_mean._version,b.running_var._version]
    a.train();b.train();a(light.tensor(later));b(torch.tensor(later))
    assert av==[a.running_mean._storage.version,a.running_var._storage.version]
    assert bv==[b.running_mean._version,b.running_var._version]
    value.backward(light.tensor(seed));other.backward(torch.tensor(seed))
    for u,v in [(x.grad,xx.grad),(a.weight.grad,b.weight.grad),(a.bias.grad,b.bias.grad)]:np.testing.assert_allclose(u.numpy(),v.numpy(),rtol=3e-12,atol=3e-12)

@pytest.mark.parametrize('missing_version',[1,2,None])
def test_old_checkpoint_fills_counter_but_modern_checkpoint_reports_missing_counter(missing_version):
    outcomes=[]
    for api,kind in [(light,light.nn.BatchNorm1d),(torch,torch.nn.BatchNorm1d)]:
        model=kind(3);state=copy.deepcopy(model.state_dict());del state['num_batches_tracked']
        if missing_version is None:del state._metadata['']['version']
        else:state._metadata['']['version']=missing_version
        try:model.load_state_dict(state)
        except RuntimeError:outcomes.append('missing')
        else:outcomes.append(model.num_batches_tracked.item())
    assert outcomes[0]==outcomes[1]

@pytest.mark.parametrize('tracked',[False,True])
@pytest.mark.parametrize('training',[False,True])
def test_mode_toggles_and_captured_counter_jvp_match_native(tracked,training):
    a=light.nn.BatchNorm1d(3,track_running_stats=tracked).train(training)
    b=torch.nn.BatchNorm1d(3,track_running_stats=tracked).train(training)
    statuses=[]
    for api,model in [(light,a),(torch,b)]:
        before=None if model.num_batches_tracked is None else model.num_batches_tracked.item()
        try:api.func.jvp(model,(api.ones(4,3),),(api.ones(4,3),))
        except (RuntimeError,UnsupportedOperation):statuses.append('rejected')
        else:statuses.append('allowed')
        after=None if model.num_batches_tracked is None else model.num_batches_tracked.item()
        assert before==after
    assert statuses[0]==statuses[1]

@pytest.mark.parametrize('shape,eps',[( (1,3),1e-5),((1,3,1),1e-5),((2,3),0.),((2,3),-1e-5),((2,0),1e-5)])
def test_training_failures_keep_native_counter_increment(shape,eps):
    outcomes=[]
    for kind,api in [(light.nn.BatchNorm1d,light),(torch.nn.BatchNorm1d,torch)]:
        model=kind(shape[1],eps=eps)
        try:model(api.ones(shape))
        except (ValueError,IndexError):outcomes.append(model.num_batches_tracked.item())
        else:outcomes.append('unexpected success')
    assert outcomes[0]==outcomes[1]==1

@pytest.mark.parametrize('tracking',[False,True])
def test_switching_tracking_after_construction_retains_native_buffer_mode_semantics(tracking):
    rng=np.random.default_rng(1608);raw=rng.normal(size=(5,3)).astype(np.float32)
    a=light.nn.BatchNorm1d(3,track_running_stats=tracking)
    b=torch.nn.BatchNorm1d(3,track_running_stats=tracking)
    a.track_running_stats=b.track_running_stats=not tracking
    for training in [True,False,True]:
        a.train(training);b.train(training)
        np.testing.assert_allclose(a(light.tensor(raw)).numpy(),b(torch.tensor(raw)).detach().numpy(),rtol=5e-6,atol=5e-6)


@pytest.mark.parametrize('training',[False,True])
@pytest.mark.parametrize('grad_enabled',[False,True])
@pytest.mark.parametrize('statistic',['mean','variance'])
def test_running_statistics_require_grad_only_rejects_active_reverse_recording(training,grad_enabled,statistic):
    outcomes=[]
    for api,operation in [(light,light.nn.functional.batch_norm),(torch,torch.nn.functional.batch_norm)]:
        mean=api.zeros(3,requires_grad=statistic=='mean');variance=api.ones(3,requires_grad=statistic=='variance')
        with api.set_grad_enabled(grad_enabled):
            try:operation(api.ones(4,3),mean,variance,training=training)
            except RuntimeError:outcomes.append('rejected')
            else:outcomes.append('allowed')
    assert outcomes[0]==outcomes[1]


@pytest.mark.parametrize('resume',[False,True])
def test_complete_batchnorm_mlp_adam_training_and_checkpoint_continuation_match_native(resume):
    rng=np.random.default_rng(1614)
    def model(api,dtype):
        return api.nn.Sequential(api.nn.Linear(3,5,bias=False,dtype=dtype),api.nn.BatchNorm1d(5,dtype=dtype),api.nn.SiLU(),api.nn.Linear(5,2,dtype=dtype))
    a=model(light,light.float64);b=model(torch,torch.float64)
    with light.no_grad(),torch.no_grad():
        for left,right in zip(a.parameters(),b.parameters()):
            raw=rng.normal(size=left.shape)*.2
            left[...] = light.tensor(raw);right.copy_(torch.tensor(raw))
    first=light.optim.Adam(a.parameters(),lr=.01);second=torch.optim.Adam(b.parameters(),lr=.01)
    for step in range(10):
        raw=rng.normal(size=(16,3));target=np.stack([raw[:,0]-.2*raw[:,1],raw[:,2]**2],axis=-1)
        first.zero_grad();second.zero_grad()
        out=a(light.tensor(raw));other=b(torch.tensor(raw))
        loss=(out-light.tensor(target)).square().mean();native=(other-torch.tensor(target)).square().mean()
        loss.backward();native.backward()
        np.testing.assert_allclose(out.numpy(),other.detach().numpy(),rtol=3e-11,atol=3e-11)
        for left,right in zip(a.parameters(),b.parameters()):np.testing.assert_allclose(left.grad.numpy(),right.grad.numpy(),rtol=3e-11,atol=3e-11)
        first.step();second.step()
        for left,right in zip(a.state_dict().values(),b.state_dict().values()):np.testing.assert_allclose(left.numpy(),right.numpy(),rtol=3e-11,atol=3e-11)
        if resume and step==3:
            state,optimizer=copy.deepcopy(a.state_dict()),copy.deepcopy(first.state_dict())
            native_state,native_optimizer=copy.deepcopy(b.state_dict()),copy.deepcopy(second.state_dict())
            a=model(light,light.float64);b=model(torch,torch.float64)
            a.load_state_dict(state);b.load_state_dict(native_state)
            first=light.optim.Adam(a.parameters(),lr=.01);second=torch.optim.Adam(b.parameters(),lr=.01)
            first.load_state_dict(optimizer);second.load_state_dict(native_optimizer)
    a.eval();b.eval();raw=rng.normal(size=(7,3))
    np.testing.assert_allclose(a(light.tensor(raw)).numpy(),b(torch.tensor(raw)).detach().numpy(),rtol=3e-11,atol=3e-11)


@pytest.mark.parametrize('training',[False,True])
@pytest.mark.parametrize('which',['mean','variance'])
@pytest.mark.parametrize('strict',[False,True])
def test_running_statistic_forward_tangents_are_nondifferentiable_like_native(training,which,strict):
    outcomes=[]
    for api in [light,torch]:
        x=api.tensor(np.arange(12,dtype=np.float64).reshape(4,3),requires_grad=True)
        mean=api.zeros(3,dtype=x.dtype);variance=api.ones(3,dtype=x.dtype)
        value=mean if which=='mean' else variance
        operation=lambda stat:api.nn.functional.batch_norm(x,stat if which=='mean' else mean,stat if which=='variance' else variance,training=training)
        try:
            primal,tangent=api.func.jvp(operation,(value,),(api.ones_like(value),),strict=strict)
        except (RuntimeError,UnsupportedOperation):outcomes.append('rejected')
        else:
            outcomes.append('allowed')
            np.testing.assert_array_equal(tangent.detach().numpy(),np.zeros((4,3)))
            assert primal.requires_grad and not tangent.requires_grad
    assert outcomes[0]==outcomes[1]
