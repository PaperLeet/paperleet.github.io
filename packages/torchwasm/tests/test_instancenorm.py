"""Independent spatial normalization, state snapshots and mixed AD workflows."""
import copy
import io
import math
import numpy as np
import pytest
from torchwasm import as_torch, torch, UnsupportedOperation


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape', [(2,3,7), (2,3,3,4), (2,3,2,3,4)])
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
@pytest.mark.parametrize('affine', [False, True])
def test_values_vjp_and_running_stats_against_independent_numpy(dtype, shape, training, layout, affine):
    rng = np.random.default_rng(4711)
    raw = rng.normal(size=shape).astype(dtype)
    if layout == 'F': raw = np.asfortranarray(raw)
    if layout == 'strided':
        base = np.empty((*shape[:-1], shape[-1]*2), dtype=dtype)
        base[...,::2] = raw; raw = base[...,::2]
    scale, shift = rng.normal(size=(2,3)).astype(dtype)
    mean = rng.normal(size=3).astype(dtype); variance = rng.uniform(.4,2,size=3).astype(dtype)
    seed = rng.normal(size=shape).astype(dtype)
    x = torch.from_numpy(raw).requires_grad_()
    w = torch.tensor(scale, requires_grad=True) if affine else None
    b = torch.tensor(shift, requires_grad=True) if affine else None
    rm, rv = torch.tensor(mean), torch.tensor(variance)
    axes = tuple(range(2,len(shape))); broadcast = (1,3,*([1]*(len(shape)-2)))
    data = raw.astype(np.float64); g = seed.astype(np.float64)
    centers = data - (data.mean(axis=axes,keepdims=True) if training else mean.reshape(broadcast))
    var = (centers**2).mean(axis=axes,keepdims=True) if training else variance.reshape(broadcast)
    inv = 1/np.sqrt(var+1e-5); normal = centers*inv
    scaled = g*(scale.reshape(broadcast) if affine else 1)
    dx = inv*(scaled-scaled.mean(axis=axes,keepdims=True)-normal*(scaled*normal).mean(axis=axes,keepdims=True)) if training else scaled*inv
    value = torch.nn.functional.instance_norm(x,rm,rv,w,b,training,.25)
    value.backward(torch.tensor(seed))
    tol = 8e-6 if dtype == torch.float32 else 3e-12
    np.testing.assert_allclose(value.numpy(),normal*scale.reshape(broadcast)+shift.reshape(broadcast) if affine else normal,rtol=tol,atol=tol)
    np.testing.assert_allclose(x.grad.numpy(),dx,rtol=tol,atol=tol)
    if affine:
        np.testing.assert_allclose(w.grad.numpy(),(g*normal).sum(axis=(0,*axes)),rtol=tol,atol=tol)
        np.testing.assert_allclose(b.grad.numpy(),g.sum(axis=(0,*axes)),rtol=tol,atol=tol)
    spatial = math.prod(shape[2:])
    expected_mean = .75*mean+.25*data.mean(axis=axes).mean(0) if training else mean
    expected_var = .75*variance+.25*var.reshape(shape[:2]).mean(0)*spatial/(spatial-1) if training else variance
    np.testing.assert_allclose(rm.numpy(),expected_mean,rtol=tol,atol=tol)
    np.testing.assert_allclose(rv.numpy(),expected_var,rtol=tol,atol=tol)
    assert [rm._version,rv._version] == [int(training)]*2
    assert value.is_contiguous() and value.dtype == dtype


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('mutated', ['input', 'weight', 'bias', 'mean', 'variance', 'output'])
@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
def test_native_snapshot_and_storage_mutation_semantics(training, mutated, layout):
    raw = np.arange(24,dtype=np.float64).reshape(2,3,4)/7
    if layout == 'F': raw = np.asfortranarray(raw)
    if layout == 'strided': raw = np.repeat(raw,2,axis=-1)[...,::2]
    x = torch.from_numpy(raw).requires_grad_()
    w = torch.tensor([.7,1.2,-.4],dtype=torch.float64,requires_grad=True)
    b = torch.tensor([.2,.4,-.3],dtype=torch.float64,requires_grad=True)
    rm = torch.tensor([.1,.3,-.2],dtype=torch.float64); rv = torch.tensor([.4,1.2,.7],dtype=torch.float64)
    seed = torch.tensor(np.sin(np.arange(24)).reshape(2,3,4))
    y = torch.nn.functional.instance_norm(x,rm,rv,w,b,training)
    expected = torch.autograd.grad(y,(x,w,b),seed,retain_graph=True)
    target = {'input':x,'weight':w,'bias':b,'mean':rm,'variance':rv,'output':y}[mutated]
    with torch.no_grad(): target.detach().add_(1)
    if mutated == 'input' and layout == 'C':
        with pytest.raises(RuntimeError,match='modified|version'): y.backward(seed)
    else:
        y.backward(seed)
        for actual, reference in zip((x.grad,w.grad,b.grad),expected):
            np.testing.assert_allclose(actual.numpy(),reference.numpy(),atol=1e-12)


@pytest.mark.parametrize('training', [False, True])
def test_reverse_over_forward_saved_statistic_convention(training):
    rng = np.random.default_rng(4712)
    arrays = [rng.normal(size=(2,2,3)),rng.normal(size=2),rng.normal(size=2),rng.normal(size=(2,2,3)),rng.normal(size=2),rng.normal(size=2)]
    mean = rng.normal(size=2); variance = rng.uniform(.3,2,size=2)
    fixed_mean = arrays[0].mean(-1,keepdims=True) if training else mean.reshape(1,2,1)
    fixed_var = arrays[0].var(-1,keepdims=True) if training else variance.reshape(1,2,1)
    inverse = 1/np.sqrt(fixed_var+1e-5)
    seeds = [rng.normal(size=(2,2,3)),rng.normal(size=(2,2,3))]
    values = [torch.tensor(a,requires_grad=True) for a in arrays]
    def fn(x,w,b):
        return torch.nn.functional.instance_norm(x,torch.tensor(mean),torch.tensor(variance),w,b,training)
    primal,tangent = torch.func.jvp(fn,tuple(values[:3]),tuple(values[3:]))
    (primal*torch.tensor(seeds[0])+tangent*torch.tensor(seeds[1])).sum().backward()
    def objective(raw):
        x,w,b,dx,dw,db = raw
        normal = (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5) if training else (x-fixed_mean)*inverse
        center = x-fixed_mean; projected = dx-dx.mean(-1,keepdims=True) if training else dx
        j = projected*inverse-center*inverse**3*(projected*center).mean(-1,keepdims=True) if training else projected*inverse
        t = j*w.reshape(1,2,1)+center*inverse*dw.reshape(1,2,1)+db.reshape(1,2,1)
        return float(np.sum((normal*w.reshape(1,2,1)+b.reshape(1,2,1))*seeds[0]+t*seeds[1]))
    for position,value in enumerate(values):
        expected = np.empty_like(arrays[position])
        for index in np.ndindex(expected.shape):
            plus = [a.copy() for a in arrays]; minus = [a.copy() for a in arrays]
            plus[position][index] += 1e-6; minus[position][index] -= 1e-6
            expected[index] = (objective(plus)-objective(minus))/2e-6
        np.testing.assert_allclose(value.grad.numpy(),expected,rtol=3e-6,atol=3e-6)


@pytest.mark.parametrize('kind,shape', [('InstanceNorm1d',(2,3,5)),('InstanceNorm2d',(2,3,2,4)),('InstanceNorm3d',(2,3,2,2,3))])
@pytest.mark.parametrize('unbatched', [False,True])
@pytest.mark.parametrize('tracking', [False,True])
@pytest.mark.parametrize('affine,bias', [(False,True),(True,True),(True,False)])
def test_module_defaults_state_counter_checkpoint_and_reset(kind,shape,unbatched,tracking,affine,bias):
    cls = getattr(torch.nn,kind)
    model = cls(3,affine=affine,bias=bias,track_running_stats=tracking,momentum=None,dtype=torch.float64)
    raw = torch.tensor(np.arange(math.prod(shape)).reshape(shape)/19)
    if unbatched: raw = raw[0]
    output = model(raw)
    if tracking:
        assert model.running_mean.tolist() == [0.,0.,0.]
        assert model.running_var.tolist() == [1.,1.,1.]
        assert model.num_batches_tracked.item() == 0
        assert model.running_mean._version == 2
    state = copy.deepcopy(model.state_dict())
    assert state._metadata['']['version'] == 2
    assert list(state) == (['weight']+(['bias'] if bias else []) if affine else [])+(['running_mean','running_var','num_batches_tracked'] if tracking else [])
    restored = cls(3,affine=affine,bias=bias,track_running_stats=tracking,momentum=None,dtype=torch.float64)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(output.numpy(),restored(raw).numpy())
    model.eval(); restored.eval()
    np.testing.assert_array_equal(model(raw).numpy(),restored(raw).numpy())
    restored.float();restored.reset_parameters()
    if affine:
        assert restored.weight.tolist() == [1.,1.,1.]
        if bias: assert restored.bias.tolist() == [0.,0.,0.]
    if tracking: assert restored.running_mean.dtype == torch.float32 and restored.num_batches_tracked.dtype == torch.int64


@pytest.mark.parametrize('shape', [(0,3,4),(0,3,0),(2,3,0),(2,0,4)])
@pytest.mark.parametrize('training', [False,True])
def test_empty_inputs_and_state_versions(shape,training):
    x = torch.empty(shape,dtype=torch.float64,requires_grad=True)
    rm = torch.zeros(shape[1],dtype=torch.float64);rv = torch.ones(shape[1],dtype=torch.float64)
    y = torch.nn.functional.instance_norm(x,rm,rv,use_input_stats=training)
    y.sum().backward()
    assert y.shape == shape and x.grad.shape == shape
    assert [rm._version,rv._version] == [int(training)]*2
    if training and shape[0] == 0: assert np.isnan(rm.numpy()).all() and np.isnan(rv.numpy()).all()
    else: assert rm.tolist() == [0.]*shape[1] and rv.tolist() == [1.]*shape[1]


@pytest.mark.parametrize('training', [False,True])
def test_buffer_dual_tangents_ignored_and_tracked_training_jvp_allowed(training):
    x = torch.tensor(np.arange(24,dtype=np.float64).reshape(2,3,4))
    rm = torch.zeros(3,dtype=torch.float64);rv = torch.ones(3,dtype=torch.float64)
    _,t = torch.func.jvp(lambda m,v:torch.nn.functional.instance_norm(x,m,v,use_input_stats=training),(rm,rv),(torch.ones_like(rm),torch.ones_like(rv)))
    np.testing.assert_array_equal(t.numpy(),np.zeros(x.shape))
    model = torch.nn.InstanceNorm1d(3,affine=True,track_running_stats=True,dtype=torch.float64)
    model.train(training)
    _,t = torch.func.jvp(model,(x,),(torch.ones_like(x),))
    assert model.num_batches_tracked.item() == 0
    assert model.running_mean._version == 1+int(training)
    np.testing.assert_allclose(t.numpy(),0 if training else 1/np.sqrt(1+1e-5),atol=1e-12)


def test_import_aliases_root_spatial_singleton_and_batchnorm_bias_option():
    with as_torch():
        from torch.nn.modules.instancenorm import _InstanceNorm, InstanceNorm1d
        from torch.nn.modules.batchnorm import _NormBase, _BatchNorm
        import torch.nn.functional as F
        model = InstanceNorm1d(3)
        assert isinstance(model,_InstanceNorm) and isinstance(model,_NormBase) and not isinstance(model,_BatchNorm)
        x = torch.ones(2,3,1,dtype=torch.float64,requires_grad=True)
        with pytest.raises(ValueError,match='spatial'): F.instance_norm(x)
        rm = torch.zeros(3,dtype=torch.float64);rv = torch.ones(3,dtype=torch.float64)
        output = torch.instance_norm(x,None,None,rm,rv,True,.1,1e-5,False)
        assert output.tolist() == [[[0.],[0.],[0.]],[[0.],[0.],[0.]]]
        assert np.isnan(rv.numpy()).all()
        output.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),np.zeros(x.shape))
        layer = torch.nn.BatchNorm1d(3,bias=False)
        assert layer.bias is None and list(layer.state_dict()) == ['weight','running_mean','running_var','num_batches_tracked']
        layer.reset_parameters(); assert 'bias=False' in repr(layer)


@pytest.mark.parametrize('eps', [0.,-1.,float('nan'),float('inf')])
def test_constant_input_epsilon_boundaries(eps):
    x = torch.ones(2,3,4,dtype=torch.float64,requires_grad=True)
    with np.errstate(all='ignore'):
        output = torch.nn.functional.instance_norm(x,eps=eps)
        output.sum().backward()
    if eps in (0.,float('inf')):
        np.testing.assert_array_equal(output.numpy(),np.zeros(x.shape))
        np.testing.assert_array_equal(x.grad.numpy(),np.zeros(x.shape))
    else: assert np.isnan(output.numpy()).all()


def test_feature_warnings_invalid_shapes_and_legacy_state():
    x = torch.ones(2,3,4)
    with pytest.warns(UserWarning,match='num_features'): torch.nn.InstanceNorm1d(4)(x)
    with pytest.raises(ValueError): torch.nn.InstanceNorm1d(4,affine=True)(x)
    with pytest.raises(ValueError): torch.nn.InstanceNorm1d(3)(torch.ones(3))
    with pytest.raises(RuntimeError): torch.nn.functional.instance_norm(x,use_input_stats=False)
    with pytest.raises(ValueError): torch.nn.functional.instance_norm(x,torch.zeros(3),None)
    with pytest.raises(IndexError): torch.nn.functional.instance_norm(torch.empty(0,3,4),weight=torch.ones(3))
    with pytest.raises(IndexError): torch.nn.functional.instance_norm(torch.empty(2,0,4),bias=torch.empty(0))
    with pytest.raises(UnsupportedOperation): torch.nn.functional.instance_norm(torch.ones(2,3,4,dtype=torch.float16))
    with pytest.raises(RuntimeError,match='running stats'): torch.nn.InstanceNorm1d(3).load_state_dict({'running_mean':torch.zeros(3),'running_var':torch.ones(3)})
    model = torch.nn.InstanceNorm1d(3,track_running_stats=True)
    model.load_state_dict({'running_mean':torch.zeros(3),'running_var':torch.ones(3)})
    assert model.num_batches_tracked.item() == 0


def test_convolution_instance_norm_sgd_checkpoint_resume():
    def make(): return torch.nn.Sequential(torch.nn.Conv2d(2,3,3,padding=1,dtype=torch.float64),torch.nn.InstanceNorm2d(3,affine=True,track_running_stats=True,dtype=torch.float64),torch.nn.GELU())
    torch.manual_seed(4713);model = make();optimizer = torch.optim.SGD(model.parameters(),lr=.03,momentum=.7)
    x = torch.tensor(np.sin(np.arange(80)).reshape(2,2,4,5));target = torch.tensor(np.cos(np.arange(120)).reshape(2,3,4,5))
    def step(m,o):
        o.zero_grad();loss = (m(x)-target).square().mean();loss.backward();o.step();return loss.item()
    step(model,optimizer)
    buffer = io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},buffer);buffer.seek(0)
    state = torch.load(buffer,weights_only=True)
    restored = make();restored.load_state_dict(state['model']);other = torch.optim.SGD(restored.parameters(),lr=.03,momentum=.7);other.load_state_dict(state['optimizer'])
    np.testing.assert_allclose(step(model,optimizer),step(restored,other),atol=1e-12)
    for left,right in zip(model.parameters(),restored.parameters()): np.testing.assert_allclose(left.numpy(),right.numpy(),atol=1e-12)
    assert model[1].num_batches_tracked.item() == 0


@pytest.mark.parametrize('training', [False,True])
@pytest.mark.parametrize('mask', range(8))
def test_each_active_gradient_subset_and_retained_backward(training,mask):
    raw = np.arange(24,dtype=np.float64).reshape(2,3,4)/13
    values = [torch.tensor(raw,requires_grad=bool(mask&1)),torch.tensor([.4,.8,-.6],requires_grad=bool(mask&2),dtype=torch.float64),torch.tensor([.1,-.3,.2],requires_grad=bool(mask&4),dtype=torch.float64)]
    mean,variance = torch.zeros(3,dtype=torch.float64),torch.ones(3,dtype=torch.float64)
    y = torch.nn.functional.instance_norm(values[0],mean,variance,*values[1:],use_input_stats=training)
    assert y.requires_grad == bool(mask)
    if not mask: return
    seed = torch.tensor(np.sin(np.arange(24)).reshape(2,3,4))
    active = [v for v in values if v.requires_grad]
    gradients = torch.autograd.grad(y,active,seed,retain_graph=True)
    y.backward(seed)
    for value,g in zip(active,gradients): np.testing.assert_array_equal(value.grad.numpy(),g.numpy())


def test_shared_running_buffers_read_original_values_before_copy():
    x = torch.tensor(np.arange(24,dtype=np.float64).reshape(2,3,4))
    shared = torch.ones(3,dtype=torch.float64)
    torch.nn.functional.instance_norm(x,shared,shared,momentum=.25)
    np.testing.assert_allclose(shared.numpy(),np.full(3,.75+.25*5/3),atol=1e-12)
    assert shared._version == 2


@pytest.mark.parametrize('training', [False,True])
def test_first_order_finite_difference_gradcheck(training):
    x = torch.tensor(np.sin(np.arange(24)).reshape(2,3,4),requires_grad=True)
    w = torch.tensor([.7,-.2,.9],requires_grad=True,dtype=torch.float64)
    b = torch.tensor([.2,-.1,.4],requires_grad=True,dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda a,c,d:torch.nn.functional.instance_norm(a,torch.zeros(3,dtype=torch.float64),torch.ones(3,dtype=torch.float64),c,d,training),(x,w,b))
