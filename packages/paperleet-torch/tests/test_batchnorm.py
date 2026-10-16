"""Independent NumPy/scalar batch statistics, tapes and module-state workflows."""
import copy
import math
import numpy as np
import pytest
from paperleet_torch import as_torch, torch, UnsupportedOperation


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('shape',[(5,3),(2,3,4),(2,3,2,4),(2,3,2,2,3)])
@pytest.mark.parametrize('training',[False,True])
def test_channel_normalization_all_gradients_and_running_statistics_against_numpy(dtype,shape,training):
    rng=np.random.default_rng(1610)
    raw=rng.normal(size=shape).astype(dtype);weight=rng.normal(size=3).astype(dtype);bias=rng.normal(size=3).astype(dtype)
    mean=rng.normal(size=3).astype(dtype);variance=rng.uniform(.2,2,size=3).astype(dtype);seed=rng.normal(size=shape).astype(dtype)
    x=torch.tensor(raw,requires_grad=True);w=torch.tensor(weight,requires_grad=True);b=torch.tensor(bias,requires_grad=True)
    rm,rv=torch.tensor(mean),torch.tensor(variance)
    versions=[rm._storage.version,rv._storage.version]
    axes=(0,*range(2,len(shape)));broadcast=(1,3,*([1]*(len(shape)-2)));count=math.prod(shape)//3
    data=raw.astype(np.float64);g=seed.astype(np.float64);scale=weight.astype(np.float64).reshape(broadcast)
    center=data-data.mean(axis=axes,keepdims=True) if training else data-mean.reshape(broadcast)
    var=(center**2).mean(axis=axes,keepdims=True) if training else variance.reshape(broadcast)
    inverse=1/np.sqrt(var+1e-5);normal=center*inverse;scaled=g*scale
    derivative=inverse*(scaled-scaled.mean(axis=axes,keepdims=True)-normal*(scaled*normal).mean(axis=axes,keepdims=True)) if training else scaled*inverse
    output=torch.nn.functional.batch_norm(x,rm,rv,w,b,training=training,momentum=.25)
    output.backward(torch.tensor(seed))
    tolerance=6e-6 if dtype==torch.float32 else 3e-12
    for actual,expected in [(output,normal*scale+bias.reshape(broadcast)),(x.grad,derivative),(w.grad,(g*normal).sum(axis=axes)),(b.grad,g.sum(axis=axes))]:
        np.testing.assert_allclose(actual.numpy(),expected,rtol=tolerance,atol=tolerance)
    expected_mean=.75*mean+.25*data.mean(axis=axes) if training else mean
    expected_var=.75*variance+.25*var.reshape(3)*count/(count-1) if training else variance
    np.testing.assert_allclose(rm.numpy(),expected_mean,rtol=tolerance,atol=tolerance)
    np.testing.assert_allclose(rv.numpy(),expected_var,rtol=tolerance,atol=tolerance)
    assert [rm._storage.version,rv._storage.version]==versions
    assert output.dtype==dtype and output.shape==shape
    np.testing.assert_array_equal(x.numpy(),raw)


@pytest.mark.parametrize('kind,shape', [('BatchNorm1d',(4,3)),('BatchNorm2d',(2,3,2,4)),('BatchNorm3d',(2,3,2,2,3))])
@pytest.mark.parametrize('tracking',[False,True])
@pytest.mark.parametrize('affine',[False,True])
def test_module_cumulative_statistics_checkpoint_dtype_and_reset(kind,shape,tracking,affine):
    rng=np.random.default_rng(1611);constructor=getattr(torch.nn,kind)
    model=constructor(3,momentum=None,affine=affine,track_running_stats=tracking,dtype=torch.float64)
    axes=(0,*range(2,len(shape)));count=math.prod(shape)//3;means=[];variances=[]
    for step in range(3):
        raw=rng.normal(size=shape)+step
        output=model(torch.tensor(raw))
        means.append(raw.mean(axis=axes));variances.append(raw.var(axis=axes)*count/(count-1))
        np.testing.assert_allclose(output.numpy().mean(axis=axes),np.zeros(3),atol=1e-12)
    if tracking:
        np.testing.assert_allclose(model.running_mean.numpy(),np.mean(means,axis=0),atol=1e-12)
        np.testing.assert_allclose(model.running_var.numpy(),np.mean(variances,axis=0),atol=1e-12)
        assert model.num_batches_tracked.item()==3
    else:assert model.running_mean is model.running_var is model.num_batches_tracked is None
    state=copy.deepcopy(model.state_dict());assert state._metadata['']['version']==2
    expected_keys=(['weight','bias'] if affine else [])+(['running_mean','running_var','num_batches_tracked'] if tracking else [])
    assert list(state)==expected_keys
    restored=constructor(3,momentum=None,affine=affine,track_running_stats=tracking,dtype=torch.float64).eval()
    restored.load_state_dict(state);model.eval();raw=torch.tensor(rng.normal(size=shape))
    np.testing.assert_allclose(restored(raw).numpy(),model(raw).numpy(),atol=1e-12)
    restored.float()
    if tracking:
        assert restored.running_mean.dtype==restored.running_var.dtype==torch.float32
        assert restored.num_batches_tracked.dtype==torch.int64
    restored.reset_parameters()
    if affine:assert restored.weight.tolist()==[1.,1.,1.] and restored.bias.tolist()==[0.,0.,0.]
    if tracking:assert restored.num_batches_tracked.item()==0 and restored.running_var.tolist()==[1.,1.,1.]


def test_ordinary_torch_aliases_and_batchnorm_base_type():
    with as_torch():
        from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
        from torch.nn.modules.batchnorm import _BatchNorm
        import torch.nn.functional as F
        for kind in [BatchNorm1d,BatchNorm2d,BatchNorm3d]:assert isinstance(kind(3),_BatchNorm)
        value=F.batch_norm(torch.tensor([[1.,2.],[3.,6.]],dtype=torch.float64),None,None,training=True)
        np.testing.assert_allclose(value.numpy().mean(0),[0.,0.],atol=1e-12)


@pytest.mark.parametrize('training',[False,True])
def test_batchnorm_first_order_finite_difference_gradcheck(training):
    x=torch.tensor([[.3,-.4],[.2,.8],[-.6,.7]],dtype=torch.float64,requires_grad=True)
    w=torch.tensor([.7,1.1],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([-.1,.2],dtype=torch.float64,requires_grad=True)
    mean=torch.tensor([.2,-.1],dtype=torch.float64);variance=torch.tensor([.6,1.3],dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda a,c,d:torch.nn.functional.batch_norm(a,None if training else mean,None if training else variance,c,d,training=training),(x,w,b))


@pytest.mark.parametrize('training',[False,True])
def test_reverse_over_forward_against_numpy_finite_differences_with_native_saved_statistics(training):
    rng=np.random.default_rng(1612)
    arrays=[rng.normal(size=(3,2)),rng.normal(size=2),rng.normal(size=2),rng.normal(size=(3,2)),rng.normal(size=2),rng.normal(size=2)]
    mean=rng.normal(size=2);variance=rng.uniform(.4,2,size=2)
    fixed_mean=arrays[0].mean(0) if training else mean
    fixed_var=arrays[0].var(0) if training else variance
    inverse=1/np.sqrt(fixed_var+1e-5)
    seeds=[rng.normal(size=(3,2)),rng.normal(size=(3,2))]
    values=[torch.tensor(array,requires_grad=True) for array in arrays]
    fn=lambda x,w,b:torch.nn.functional.batch_norm(x,None if training else torch.tensor(mean),None if training else torch.tensor(variance),w,b,training=training)
    primal,tangent=torch.func.jvp(fn,tuple(values[:3]),tuple(values[3:]))
    (primal*torch.tensor(seeds[0])+tangent*torch.tensor(seeds[1])).sum().backward()
    def objective(raw):
        x,w,b,dx,dw,db=raw
        normalized=(x-x.mean(0))/np.sqrt(x.var(0)+1e-5) if training else (x-mean)/np.sqrt(variance+1e-5)
        center=x-fixed_mean;projected=dx-dx.mean(0) if training else dx
        j=projected*inverse-center*inverse**3*(projected*center).mean(0) if training else projected*inverse
        t=j*w+center*inverse*dw+db
        return float(np.sum((normalized*w+b)*seeds[0]+t*seeds[1]))
    for position,value in enumerate(values):
        expected=np.empty_like(arrays[position])
        for index in np.ndindex(expected.shape):
            plus=[array.copy() for array in arrays];minus=[array.copy() for array in arrays]
            plus[position][index]+=1e-6;minus[position][index]-=1e-6
            expected[index]=(objective(plus)-objective(minus))/2e-6
        np.testing.assert_allclose(value.grad.numpy(),expected,rtol=2e-6,atol=2e-6)


@pytest.mark.parametrize('first_training',[False,True])
def test_later_training_updates_preserve_graph_and_native_eval_buffer_semantics(first_training):
    rng=np.random.default_rng(1613);raw=rng.normal(size=(5,3));seed=rng.normal(size=raw.shape)
    model=torch.nn.BatchNorm1d(3,dtype=torch.float64).train(first_training)
    x=torch.tensor(raw,requires_grad=True);value=model(x)
    original_mean=raw.mean(0) if first_training else model.running_mean.numpy().copy()
    original_var=raw.var(0) if first_training else model.running_var.numpy().copy()
    model.train();model(torch.tensor(rng.normal(size=(7,3))*3+2))
    value.backward(torch.tensor(seed))
    mean=original_mean if first_training else model.running_mean.numpy()
    variance=original_var if first_training else model.running_var.numpy()
    inverse=1/np.sqrt(variance+model.eps);normal=(raw-mean)*inverse
    dx=inverse*(seed-seed.mean(0)-normal*(seed*normal).mean(0)) if first_training else seed*inverse
    np.testing.assert_allclose(x.grad.numpy(),dx,atol=1e-12)
    np.testing.assert_allclose(model.weight.grad.numpy(),(seed*normal).sum(0),atol=1e-12)


@pytest.mark.parametrize('changed',['input','weight','bias','mean','variance','output'])
def test_saved_storage_guards_exclude_bias_and_output(changed):
    x=torch.ones(4,3,requires_grad=True);w=torch.ones(3,requires_grad=True);b=torch.zeros(3,requires_grad=True)
    mean=torch.zeros(3);variance=torch.ones(3)
    value=torch.nn.functional.batch_norm(x,mean,variance,w,b,training=True);loss=value.sum()
    with torch.no_grad():dict(input=x,weight=w,bias=b,mean=mean,variance=variance,output=value)[changed].detach().add_(1.)
    if changed in ['bias','output']:loss.backward();assert b.grad.tolist()==[4.,4.,4.]
    else:
        with pytest.raises(UnsupportedOperation,match='modified in-place'):loss.backward()


@pytest.mark.parametrize('shape',[(0,3),(2,3,0),(0,3,2,4),(2,3,0,4),(2,3,2,0,3)])
@pytest.mark.parametrize('training',[False,True])
def test_empty_input_has_empty_input_grad_zero_affine_grads_and_unchanged_statistics(shape,training):
    x=torch.ones(shape,requires_grad=True);w=torch.ones(3,requires_grad=True);b=torch.zeros(3,requires_grad=True)
    mean=torch.zeros(3);variance=torch.ones(3)
    value=torch.nn.functional.batch_norm(x,mean,variance,w,b,training=training);value.sum().backward()
    assert value.shape==x.grad.shape==shape
    assert w.grad.tolist()==b.grad.tolist()==[0.,0.,0.]
    assert mean.tolist()==[0.,0.,0.] and variance.tolist()==[1.,1.,1.]


def test_failed_training_counts_attempt_and_jvp_counter_rejection_is_side_effect_free():
    model=torch.nn.BatchNorm1d(3)
    with pytest.raises(ValueError):model(torch.ones(1,3))
    assert model.num_batches_tracked.item()==1
    with pytest.raises(UnsupportedOperation):torch.func.jvp(model,(torch.ones(4,3),),(torch.ones(4,3),))
    assert model.num_batches_tracked.item()==1
    assert model.running_mean.tolist()==[0.,0.,0.]


@pytest.mark.parametrize('dtype',[torch.float16,torch.complex64])
def test_unsupported_numerical_kinds_fail_explicitly(dtype):
    with pytest.raises(UnsupportedOperation):torch.nn.BatchNorm1d(3,dtype=dtype)
    with pytest.raises(UnsupportedOperation):torch.nn.functional.batch_norm(torch.ones(4,3,dtype=dtype),None,None,training=True)


@pytest.mark.parametrize('layout',['contiguous','fortran','stride'])
def test_float32_ill_conditioned_channels_match_high_precision_centered_math(layout):
    small=np.array([[0,.001],[.001,.002],[.002,.004],[.004,.008]],dtype=np.float32)
    raw=(10000+small).astype(np.float32)
    if layout=='fortran':raw=np.asfortranarray(raw)
    elif layout=='stride':raw=np.repeat(raw,2,axis=0)[::2]
    # Include a well-conditioned channel so selective recomputation cannot
    # accidentally exchange channel/batch axes or alter the input allocation.
    raw=np.concatenate([raw,np.array([[1.],[-2.],[3.],[-4.]],dtype=np.float32)],axis=1)
    if layout=='fortran':raw=np.asfortranarray(raw)
    elif layout=='stride':raw=np.repeat(raw,2,axis=1)[:,::2]
    before=raw.copy();x=torch.from_numpy(raw).requires_grad_()
    weight=torch.tensor([.7,1.3,-.8],requires_grad=True);bias=torch.tensor([.1,-.2,.3],requires_grad=True)
    mean,variance=torch.zeros(3),torch.ones(3)
    value=torch.nn.functional.batch_norm(x,mean,variance,weight,bias,training=True,momentum=.25)
    data=raw.astype(np.float64);center=data-data.mean(0);var=data.var(0);inverse=1/np.sqrt(var+1e-5)
    normalized=center*inverse;scale=weight.numpy().astype(np.float64)
    seed=np.arange(12,dtype=np.float32).reshape(4,3)+1
    value.backward(torch.tensor(seed))
    scaled=seed.astype(np.float64)*scale
    dx=inverse*(scaled-scaled.mean(0)-normalized*(scaled*normalized).mean(0))
    for actual,expected in [(value,normalized*scale+bias.numpy()),(x.grad,dx),(weight.grad,(seed*normalized).sum(0)),(bias.grad,seed.sum(0))]:
        np.testing.assert_allclose(actual.numpy(),expected,rtol=6e-6,atol=6e-6)
    np.testing.assert_allclose(mean.numpy(),.25*data.mean(0),rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(variance.numpy(),.75+.25*var*4/3,rtol=1e-7,atol=1e-7)
    np.testing.assert_array_equal(raw,before)


def test_float32_inverse_avoids_extra_intermediate_rounding_and_keeps_overflow_classes():
    x=torch.tensor([[-1e-35],[1e-35]],requires_grad=True)
    value=torch.nn.functional.batch_norm(x,None,None,training=True)
    expected=np.float32(1/np.sqrt(1e-5))
    np.testing.assert_array_equal(value.numpy(),x.numpy()*expected)
    value.backward(torch.tensor([[1.],[3.]]))
    np.testing.assert_array_equal(x.grad.numpy(),np.array([[-expected],[expected]],dtype=np.float32))
    huge=torch.tensor([[-1e20],[1e20]])
    np.testing.assert_array_equal(torch.nn.functional.batch_norm(huge,None,None,training=True).numpy(),np.zeros((2,1),dtype=np.float32))
    assert np.isnan(torch.nn.functional.batch_norm(torch.full((4,1),3e38),None,None,training=True).numpy()).all()


def test_complete_training_checkpoint_restores_optimizer_and_batch_counter():
    def model():
        return torch.nn.Sequential(torch.nn.Linear(3,5,bias=False,dtype=torch.float64),torch.nn.BatchNorm1d(5,dtype=torch.float64),torch.nn.SiLU(),torch.nn.Linear(5,2,dtype=torch.float64))
    torch.manual_seed(1615);first=model();optimizer=torch.optim.Adam(first.parameters(),lr=.01)
    rng=np.random.default_rng(1615)
    def step(owner,update,raw,target):
        update.zero_grad();loss=(owner(torch.tensor(raw))-torch.tensor(target)).square().mean()
        loss.backward();update.step();return float(loss)
    for _ in range(3):step(first,optimizer,rng.normal(size=(12,3)),rng.normal(size=(12,2)))
    state=copy.deepcopy(first.state_dict());optim_state=copy.deepcopy(optimizer.state_dict())
    second=model();second.load_state_dict(state)
    resumed=torch.optim.Adam(second.parameters(),lr=.01);resumed.load_state_dict(optim_state)
    assert first[1].num_batches_tracked.item()==second[1].num_batches_tracked.item()==3
    for _ in range(4):
        raw,target=rng.normal(size=(12,3)),rng.normal(size=(12,2))
        np.testing.assert_allclose(step(first,optimizer,raw,target),step(second,resumed,raw,target),rtol=1e-13,atol=1e-13)
        for left,right in zip(first.state_dict().values(),second.state_dict().values()):np.testing.assert_allclose(left.numpy(),right.numpy(),rtol=1e-13,atol=1e-13)
    assert first[1].num_batches_tracked.item()==second[1].num_batches_tracked.item()==7
    first.eval();second.eval();raw=torch.tensor(rng.normal(size=(8,3)))
    np.testing.assert_allclose(first(raw).numpy(),second(raw).numpy(),rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('training',[False,True])
@pytest.mark.parametrize('which',['mean','variance'])
@pytest.mark.parametrize('strict',[False,True])
def test_running_statistics_stop_forward_tangents_without_inventing_reverse_edges(training,which,strict):
    x=torch.tensor(np.arange(12,dtype=np.float64).reshape(4,3),requires_grad=True)
    mean=torch.zeros(3,dtype=torch.float64);variance=torch.ones(3,dtype=torch.float64)
    value=mean if which=='mean' else variance
    operation=lambda stat:torch.nn.functional.batch_norm(x,stat if which=='mean' else mean,stat if which=='variance' else variance,training=training)
    if strict:
        with pytest.raises((RuntimeError,UnsupportedOperation)):torch.func.jvp(operation,(value,),(torch.ones_like(value),),strict=True)
    else:
        primal,tangent=torch.func.jvp(operation,(value,),(torch.ones_like(value),))
        np.testing.assert_array_equal(tangent.numpy(),np.zeros((4,3)))
        assert primal.requires_grad and not tangent.requires_grad
        primal.sum().backward();assert x.grad is not None
        assert value.grad is None
