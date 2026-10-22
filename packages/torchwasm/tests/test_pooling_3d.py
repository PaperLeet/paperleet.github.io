"""Independent ordered-region, scatter and directional-derivative oracles."""
import itertools
import pickle
import numpy as np
import pytest
from torchwasm import torch, as_torch, UnsupportedOperation


@pytest.fixture(params=['scan', 'window', 'block', 'auto', 'separable'], autouse=True)
def algorithm(request, monkeypatch):
    from torchwasm import _pooling, _pooling3d
    monkeypatch.setattr(_pooling, '_ALGORITHM', 'auto' if request.param == 'separable' else request.param)
    monkeypatch.setattr(_pooling3d, '_SEPARABLE_MODE', 'force' if request.param == 'separable' else 'auto')


def oracle(a, k, s, p, d, ceil):
    out = []
    for size, kernel, step, pad, dil in zip(a.shape[-3:], k, s, p, d):
        count = (size + 2*pad - dil*(kernel-1) - 1 + (step-1 if ceil else 0)) // step + 1
        if ceil and (count-1)*step >= size+pad: count -= 1
        out.append(count)
    values = np.empty((*a.shape[:-3], *out), dtype=a.dtype)
    indices = np.empty(values.shape, dtype=np.int64)
    for prefix in np.ndindex(a.shape[:-3]):
        for dest in np.ndindex(tuple(out)):
            axes = [[o*step-pad+t*dil for t in range(kernel)
                     if 0 <= o*step-pad+t*dil < size]
                    for o, step, pad, dil, kernel, size in zip(dest, s, p, d, k, a.shape[-3:])]
            candidates = list(itertools.product(*axes))
            samples = np.array([a[prefix+coord] for coord in candidates], dtype=a.dtype)
            nan = np.flatnonzero(np.isnan(samples)) if a.dtype.kind == 'f' else []
            selected = candidates[int(nan[-1]) if len(nan) else int(np.argmax(samples))]
            values[prefix+dest] = a[prefix+selected]
            indices[prefix+dest] = np.ravel_multi_index(selected, a.shape[-3:])
    return values, indices


def scatter(shape, indices, seed):
    result = np.zeros(shape, dtype=seed.dtype)
    for prefix in np.ndindex(shape[:-3]):
        for dest in np.ndindex(indices.shape[-3:]):
            coord = np.unravel_index(int(indices[prefix+dest]), shape[-3:])
            result[prefix+coord] += seed[prefix+dest]
    return result


CASES = [((2,4,6,8),(2,2,2),(2,2,2),(0,0,0),(1,1,1),False),
         ((1,2,4,5,7),(2,3,2),(1,2,2),(1,1,0),(1,1,1),True),
         ((1,2,5,6,7),(2,2,2),(1,1,2),(0,1,1),(2,2,2),False),
         ((0,2,4,6,8),(2,2,2),(2,2,2),(0,0,0),(1,1,1),False),
         ((2,3,4,6,8),(1,2,2),(1,2,2),(0,0,0),(1,1,1),False),
         ((1,3,1,2,4),(1,1,2),(1,1,2),(0,0,0),(1,1,1),False)]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('layout', ['contiguous', 'fortran', 'strided', 'channels_last'])
@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('with_indices', [False, True])
def test_values_indices_backward_jvp_and_reverse_tangent(dtype, layout, case, with_indices):
    shape,k,s,p,d,ceil = case;rng = np.random.default_rng(4301)
    a = rng.integers(-3,4,size=shape).astype(dtype)
    if layout == 'fortran': a = np.asfortranarray(a)
    if layout == 'strided': a = np.repeat(a,2,axis=-1)[...,::2]
    if layout == 'channels_last' and a.ndim == 5:
        a = a.transpose(0,2,3,4,1).copy().transpose(0,4,1,2,3)
    expected,index = oracle(a,k,s,p,d,ceil);before = a.copy()
    x = torch.from_numpy(a).requires_grad_()
    fn = lambda z:torch.nn.functional.max_pool3d(z,k,s,p,d,ceil,with_indices)
    result = fn(x);y,i = result if with_indices else (result,None)
    np.testing.assert_array_equal(y.numpy(),expected)
    if with_indices: np.testing.assert_array_equal(i.numpy(),index)
    seed = rng.normal(size=y.shape).astype(dtype)
    y.backward(torch.tensor(seed));np.testing.assert_allclose(x.grad.numpy(),scatter(shape,index,seed),rtol=2e-6,atol=2e-7)
    x.grad = None;v = rng.normal(size=shape).astype(dtype);direction = torch.tensor(v,requires_grad=True)
    primal,tangent = torch.func.jvp(fn,(x,),(direction,))
    if with_indices: (primal,pi),(tangent,ti) = primal,tangent
    np.testing.assert_array_equal(primal.numpy(),expected)
    planes = int(np.prod(shape[:-3]));points = int(np.prod(index.shape[-3:]))
    expected_tangent = np.take_along_axis(v.reshape(planes,-1) if planes else v.reshape(0,int(np.prod(shape[-3:]))),
                                           index.reshape(planes,points),axis=1).reshape(y.shape)
    np.testing.assert_array_equal(tangent.numpy(),expected_tangent)
    tangent.backward(torch.tensor(seed));assert x.grad is None
    np.testing.assert_allclose(direction.grad.numpy(),scatter(shape,index,seed),rtol=2e-6,atol=2e-7)
    if with_indices:
        np.testing.assert_array_equal(pi.numpy(),index);np.testing.assert_array_equal(ti.numpy(),0)
    np.testing.assert_array_equal(a,before);assert not np.shares_memory(a,y.numpy())


@pytest.mark.parametrize('dtype', [np.float16,np.float32,np.float64,np.int8,np.int16,np.int32,np.int64,np.uint8])
@pytest.mark.parametrize('with_indices', [False,True])
def test_dtype_inference_and_minimum_padding(dtype, with_indices):
    minimum = -np.inf if np.dtype(dtype).kind == 'f' else np.iinfo(dtype).min
    a = np.full((1,2,3,3,3),minimum,dtype=dtype)
    x = torch.from_numpy(a)
    result = torch.nn.functional.max_pool3d(x,3,1,1,return_indices=with_indices)
    y,i = result if with_indices else (result,None)
    expected,index = oracle(a,(3,3,3),(1,1,1),(1,1,1),(1,1,1),False)
    np.testing.assert_array_equal(y.numpy(),expected);assert y.dtype == np.dtype(dtype)
    if with_indices: np.testing.assert_array_equal(i.numpy(),index);assert i.dtype == torch.int64


def test_last_nan_first_finite_and_signed_zero():
    a = np.zeros((1,1,2,2,4),dtype=np.float64);a[...,0] = -0.
    a[0,0,0,0,2] = np.nan;a[0,0,1,1,3] = np.nan
    y,i = torch.nn.functional.max_pool3d(torch.tensor(a),2,return_indices=True)
    np.testing.assert_array_equal(i.numpy(),[[[[[0,15]]]]]);assert np.signbit(y.numpy()[0,0,0,0,0])
    assert np.isnan(y.numpy()[0,0,0,0,1])


def test_channels_last_output_indices_tangent_and_mutation_guard():
    a = np.arange(2*3*4*6*8,dtype=np.float64).reshape(2,3,4,6,8)
    a = a.transpose(0,2,3,4,1).copy().transpose(0,4,1,2,3)
    x = torch.from_numpy(a).requires_grad_()
    y,i = torch.nn.functional.max_pool3d(x,2,return_indices=True)
    assert y.stride() == i.stride() == (72,1,36,12,3)
    primal,tangent = torch.func.jvp(lambda z:torch.max_pool3d(z,2),(x,),(torch.ones_like(x),))
    assert primal.stride() == tangent.stride() == y.stride()
    i[0,0,0,0,0] = 0
    with pytest.raises(RuntimeError,match='modified|version|inplace|in-place'): y.sum().backward()


@pytest.mark.parametrize('shape,primal_stride,tangent_stride', [
    ((2,1,4,6,8),(24,1,12,4,1),(24,24,12,4,1)),
    ((1,3,2,2,2),(3,1,3,3,3),(3,1,1,1,1)),
    ((2,3,4,6,8),(72,1,36,12,3),(72,1,36,12,3))])
def test_ambiguous_channels_last_jvp_strides(shape,primal_stride,tangent_stride):
    a = np.zeros(shape,dtype=np.float64).transpose(0,2,3,4,1).copy().transpose(0,4,1,2,3)
    x = torch.from_numpy(a)
    primal,tangent = torch.func.jvp(lambda z:torch.max_pool3d(z,2),(x,),(torch.ones(shape,dtype=torch.float64),))
    assert primal.stride() == primal_stride;assert tangent.stride() == tangent_stride


def test_alias_pickle_checkpoint_and_convolution_training():
    with as_torch() as api:
        from torch.nn.modules.pooling import MaxPool3d
        from torch.nn.functional import max_pool3d_with_indices
        layer = MaxPool3d(2,return_indices=True);x = api.arange(64,dtype=api.float64).reshape(1,1,4,4,4)
        assert isinstance(layer,api.nn.Module) and not isinstance(layer,api.nn.MaxPool2d)
        for a,b in zip(layer(x),pickle.loads(pickle.dumps(layer))(x)): np.testing.assert_array_equal(a.numpy(),b.numpy())
        for a,b in zip(layer(x),max_pool3d_with_indices(x,2)): np.testing.assert_array_equal(a.numpy(),b.numpy())
    def make():
        return torch.nn.Sequential(torch.nn.Conv3d(2,3,3,padding=1),torch.nn.SiLU(),torch.nn.MaxPool3d(2),
                                   torch.nn.Flatten(),torch.nn.Linear(24,2)).double()
    torch.manual_seed(4302);model = make();rng = np.random.default_rng(4302)
    x = torch.tensor(rng.normal(size=(2,2,4,4,4)));opt = torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    for _ in range(2):
        opt.zero_grad();model(x).square().mean().backward();opt.step()
    state = pickle.loads(pickle.dumps((model.state_dict(),opt.state_dict())))
    restored = make();restored.load_state_dict(state[0]);resume = torch.optim.SGD(restored.parameters(),lr=.01,momentum=.9);resume.load_state_dict(state[1])
    for m,o in ((model,opt),(restored,resume)):
        o.zero_grad();m(x).square().mean().backward();o.step()
    for a,b in zip(model.parameters(),restored.parameters()): np.testing.assert_array_equal(a.numpy(),b.numpy())
    v = torch.tensor(rng.normal(size=x.shape));_,tangent = torch.func.jvp(restored,(x,),(v,))
    epsilon = 1e-5;numeric = (restored(x+epsilon*v).numpy()-restored(x-epsilon*v).numpy())/(2*epsilon)
    np.testing.assert_allclose(tangent.numpy(),numeric,rtol=1e-7,atol=1e-9)


@pytest.mark.parametrize('kwargs,exception', [({'kernel_size':0},RuntimeError),({'kernel_size':(2,2)},RuntimeError),
    ({'kernel_size':True},TypeError),({'kernel_size':2,'stride':0},RuntimeError),
    ({'kernel_size':2,'padding':2},RuntimeError),({'kernel_size':2,'dilation':0},RuntimeError),
    ({'kernel_size':2,'ceil_mode':1},TypeError),({'kernel_size':2**32+1},RuntimeError),
    ({'kernel_size':8_000_002,'padding':4_000_001},UnsupportedOperation)])
def test_invalid_and_bounded_geometry(kwargs,exception):
    with pytest.raises(exception): torch.nn.functional.max_pool3d(torch.zeros(1,1,2,2,2),**kwargs)


def test_sparse_large_kernel_and_workspace_fallback(monkeypatch):
    from torchwasm import _pooling
    monkeypatch.setattr(_pooling,'_WINDOW_WORKSPACE_BYTES',1)
    x = torch.arange(7,dtype=torch.float64).reshape(1,1,1,1,7).requires_grad_()
    y,i = torch.nn.functional.max_pool3d(x,(1,1,1_000_000),1,(0,0,500_000),return_indices=True)
    np.testing.assert_array_equal(y.numpy(),6);np.testing.assert_array_equal(i.numpy(),6)
    y.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),[[[[[0,0,0,0,0,0,8]]]]])


def test_empty_sampling_inference_and_explicit_derivative_boundary():
    x = torch.zeros(1,1,1,1,1)
    y,i = torch.nn.functional.max_pool3d(x,2,1,1,2,return_indices=True)
    assert y.shape == (1,1,1,1,1);np.testing.assert_array_equal(y.numpy(),-np.inf);np.testing.assert_array_equal(i.numpy(),3)
    with pytest.raises(UnsupportedOperation): torch.nn.functional.max_pool3d(x.requires_grad_(),2,1,1,2)


def test_empty_batch_does_not_materialize_spatial_coordinates(monkeypatch):
    x = torch.zeros(0,1,200,200,200,requires_grad=True)
    arange = np.arange
    def only_empty_ranges(*args,**kwargs):
        result = arange(*args,**kwargs)
        assert not result.size, 'Empty batches must not allocate spatial coordinate grids'
        return result
    monkeypatch.setattr(np,'arange',only_empty_ranges)
    y,i = torch.nn.functional.max_pool3d(x,1,return_indices=True)
    assert y.shape == i.shape == x.shape and y.numel() == 0
    y.sum().backward();assert x.grad.shape == x.shape
    _,tangent = torch.func.jvp(lambda z:torch.max_pool3d(z,1),(x,),(torch.ones_like(x),))
    assert tangent.shape == x.shape


@pytest.mark.parametrize('shape,k,s', [((1,1,3,4,4),(3,2,2),(1,2,2)),
    ((2,3,2,2,2),(2,2,2),(2,2,2))])
def test_canonical_singleton_output_index_and_tangent_strides(shape,k,s):
    x = torch.arange(int(np.prod(shape)),dtype=torch.float64).reshape(shape)
    y,i = torch.nn.functional.max_pool3d(x,k,s,return_indices=True)
    steps=[];step=1
    for n in reversed(y.shape):steps.append(step);step*=max(1,n)
    expected=tuple(reversed(steps));assert y.stride()==i.stride()==expected
    _,tangent=torch.func.jvp(lambda z:torch.max_pool3d(z,k,s),(x,),(torch.ones_like(x),))
    assert tangent.stride()==expected


def test_separable_uses_bounded_scan_stages_when_packing_does_not_fit(monkeypatch):
    from torchwasm import _pooling,_pooling3d
    monkeypatch.setattr(_pooling,'_ALGORITHM','auto');monkeypatch.setattr(_pooling3d,'_SEPARABLE_MODE','force')
    a=np.arange(2*6*8*10,dtype=np.float64).reshape(1,2,6,8,10);x=torch.from_numpy(a).requires_grad_()
    geometry=_pooling3d._geometry(x,3,1,1,1,False)
    monkeypatch.setattr(_pooling,'_WINDOW_WORKSPACE_BYTES',_pooling3d._separable_reserve(a,geometry,True)+1)
    original=_pooling3d._separable;calls=[]
    def tracked(*args):calls.append(True);return original(*args)
    def unexpected(*args):raise AssertionError('Packed windows must respect remaining workspace')
    monkeypatch.setattr(_pooling3d,'_separable',tracked);monkeypatch.setattr(_pooling,'_window',unexpected)
    y,i=torch.nn.functional.max_pool3d(x,3,1,1,return_indices=True)
    expected,index=oracle(a,(3,3,3),(1,1,1),(1,1,1),(1,1,1),False)
    assert calls==[True];np.testing.assert_array_equal(y.numpy(),expected);np.testing.assert_array_equal(i.numpy(),index)
    y.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),scatter(a.shape,index,np.ones(y.shape)))


def test_separable_declines_oversized_intermediate_without_allocating(monkeypatch):
    from torchwasm import _pooling,_pooling3d
    monkeypatch.setattr(_pooling3d,'_SEPARABLE_MODE','force')
    x=torch.zeros(1,1,1,1,1).expand(1,1,3_000_000,1,1)
    geometry=_pooling3d._geometry(x,(3_000_000,1,1),None,0,1,False)
    assert _pooling3d._select_compute(x.numpy(),geometry,True,_pooling3d._scan) is _pooling3d._scan


def test_small_index_tapes_keep_the_original_route(monkeypatch):
    from torchwasm import _pooling3d
    monkeypatch.setattr(_pooling3d,'_SEPARABLE_MODE','auto')
    x=torch.ones(2,3,4,4,4,dtype=torch.float64)
    geometry=_pooling3d._geometry(x,3,1,1,1,False)
    assert _pooling3d._select_compute(x.numpy(),geometry,True,_pooling3d._window) is _pooling3d._window
    assert _pooling3d._select_compute(x.numpy(),geometry,False,_pooling3d._window) is _pooling3d._separable
