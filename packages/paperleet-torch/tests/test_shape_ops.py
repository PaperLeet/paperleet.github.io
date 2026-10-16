"""Independent coordinate maps, finite derivatives and shape lifecycle."""
import copy
import itertools
import numpy as np
import pytest
from paperleet_torch import torch, UnsupportedOperation


@pytest.mark.parametrize('axes',list(itertools.permutations(range(3))))
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_permutation_coordinates_and_adjoint(axes,dtype):
    a=np.arange(24,dtype=dtype).reshape(2,3,4)/17;x=torch.from_numpy(a).requires_grad_();y=x.permute(axes)
    seed=np.arange(24,dtype=dtype).reshape(y.shape)/13;expected=np.empty(y.shape,dtype=dtype);adjoint=np.empty(a.shape,dtype=dtype)
    for dst in itertools.product(*(range(n) for n in y.shape)):
        src=[0]*3
        for k,axis in enumerate(axes):src[axis]=dst[k]
        expected[dst]=a[tuple(src)];adjoint[tuple(src)]=seed[dst]
    np.testing.assert_array_equal(y.numpy(),expected)
    y.backward(torch.from_numpy(seed));np.testing.assert_array_equal(x.grad.numpy(),adjoint)
    assert y._storage is x._storage


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_broadcast_coordinates_and_adjoint(dtype):
    a=np.array([[[.2],[.5],[-.1]]],dtype=dtype);x=torch.from_numpy(a).requires_grad_();y=x.expand(4,3,2)
    seed=np.arange(24,dtype=dtype).reshape(4,3,2)/17;expected=np.empty((4,3,2),dtype=dtype);dx=np.zeros_like(a)
    for i,j,k in itertools.product(range(4),range(3),range(2)):
        expected[i,j,k]=a[0,j,0];dx[0,j,0]+=seed[i,j,k]
    np.testing.assert_array_equal(y.numpy(),expected);y.backward(torch.from_numpy(seed));np.testing.assert_allclose(x.grad.numpy(),dx,rtol=2e-6,atol=1e-7)
    assert y.stride()==(0,1,0)


def chain(x):return x.unsqueeze(0).expand(4,2,1,3).squeeze(2).permute(1,0,2).flatten(1)


def test_composed_shape_jvp_reverse_and_finite_difference():
    a=np.linspace(-.4,.6,6).reshape(2,1,3);b=np.linspace(.1,.8,6).reshape(2,1,3)
    x=torch.from_numpy(a).requires_grad_();t=torch.from_numpy(b).requires_grad_()
    p,q=torch.func.jvp(lambda z:chain(z).square(),(x,),(t,))
    expected=np.tile(a[:,0,:],(1,4));tangent=np.tile(b[:,0,:],(1,4))
    np.testing.assert_allclose(p.numpy(),expected**2,rtol=1e-13,atol=1e-13);np.testing.assert_allclose(q.numpy(),2*expected*tangent,rtol=1e-13,atol=1e-13)
    gx,gt=torch.autograd.grad(q.sum(),(x,t));np.testing.assert_allclose(gx.numpy(),8*b,rtol=1e-13,atol=1e-13);np.testing.assert_allclose(gt.numpy(),8*a,rtol=1e-13,atol=1e-13)
    epsilon=1e-6
    numerical=(np.tile((a+epsilon*b)[:,0,:],(1,4))**2-np.tile((a-epsilon*b)[:,0,:],(1,4))**2)/(2*epsilon)
    np.testing.assert_allclose(q.numpy(),numerical,rtol=1e-8,atol=1e-9)


@pytest.mark.parametrize('shape',[(0,),(2,0,3),(0,1,3),()])
def test_scalar_and_empty_shapes(shape):
    x=torch.from_numpy(np.zeros(shape,dtype=np.float64)).requires_grad_()
    assert x.flatten().shape==(x.numel(),)
    y=x.unsqueeze(-1).squeeze(-1);assert y.shape==x.shape
    y.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),np.ones(shape))
    p,t=torch.func.jvp(lambda z:z.permute(tuple(reversed(range(z.ndim)))),(x,),(torch.ones_like(x),))
    assert p.shape==t.shape==tuple(reversed(shape))


def test_flatten_identity_squeeze_view_and_empty_tuple():
    x=torch.ones(2,3,requires_grad=True)
    assert x.flatten(0,0) is x
    assert x.squeeze(0) is not x and x.squeeze(()) is not x
    assert x.squeeze(0)._storage is x._storage
    assert torch.tensor(2.).permute(()).shape==()


@pytest.mark.parametrize('call',[lambda x:x.permute(2,1),lambda x:x.permute(0,0),lambda x:x.flatten(1,0),lambda x:x.flatten(0,2),lambda x:x.expand(-1,2,3),lambda x:x.squeeze((0,0))])
def test_invalid_axes_are_rejected(call):
    with pytest.raises((RuntimeError,IndexError)):call(torch.ones(2,3))


@pytest.mark.parametrize('value',[True,False,1.5,None,'0'])
def test_noninteger_dimensions_rejected(value):
    x=torch.ones(2,3)
    for fn in [lambda:x.unsqueeze(value),lambda:x.transpose(value,0),lambda:x.flatten(value),lambda:x.squeeze(value)]:
        with pytest.raises(TypeError):fn()


def test_imported_singleton_strides_and_expansion():
    base=np.arange(6.,dtype=np.float64).reshape(2,3);a=np.lib.stride_tricks.as_strided(base,shape=(2,1,3),strides=(24,0,8));x=torch.from_numpy(a)
    assert x.unsqueeze(0).stride()==(6,3,0,1)
    assert x.expand(2,1,3).stride()==(3,0,1)
    assert x.expand(4,2,1,3).stride()==(0,3,0,1)
    assert torch.tensor(2.).expand(1).stride()==(0,)


def test_view_alias_guards_and_checkpoint_copy():
    x=torch.arange(6,dtype=torch.float64).reshape(2,1,3).requires_grad_();loss=x.square().sum();y=x.squeeze(1).transpose(0,1).detach()
    copied=copy.deepcopy([x.detach(),y]);copied[1][0,0]=-1.;assert copied[0][0,0,0].item()==-1. and x[0,0,0].item()==0.
    y[0,0]=10.;assert x._version==y._version
    with pytest.raises(UnsupportedOperation):loss.backward()


def test_checkpoint_resume_through_flatten_module():
    torch.manual_seed(73)
    model=torch.nn.Sequential(torch.nn.Linear(3,4,dtype=torch.float64),torch.nn.Tanh(),torch.nn.Flatten(1,2),torch.nn.Linear(8,2,dtype=torch.float64))
    data=torch.from_numpy(np.arange(12.,dtype=np.float64).reshape(2,2,3)/17);target=torch.from_numpy(np.array([[.2,-.1],[.4,.3]]))
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    def step(m,o):
        o.zero_grad();loss=(m(data)-target).square().mean();loss.backward();o.step();return loss.item()
    for _ in range(3):step(model,opt)
    snapshot=copy.deepcopy((model.state_dict(),opt.state_dict()));expected=[step(model,opt) for _ in range(4)]
    restored=copy.deepcopy(model);newopt=torch.optim.AdamW(restored.parameters(),lr=.01);restored.load_state_dict(snapshot[0]);newopt.load_state_dict(snapshot[1]);actual=[step(restored,newopt) for _ in range(4)]
    np.testing.assert_array_equal(actual,expected)
    for a,b in zip(model.parameters(),restored.parameters()):np.testing.assert_array_equal(a.numpy(),b.numpy())


def test_operator_signature_sentinel_has_stable_repr():
    text=torch._browser_capabilities['torch.squeeze'].signature
    assert '<unspecified>' in text and '0x' not in text
    assert torch._browser_capabilities['torch.transpose'].forward
    assert torch._browser_capabilities['torch.broadcast_to'].forward


def test_scalar_T_is_a_distinct_shared_view():
    from paperleet_torch import _state
    old=getattr(_state,'_scalar_t_warning_emitted',False);_state._scalar_t_warning_emitted=False
    try:
        x=torch.tensor(2.,requires_grad=True)
        with pytest.warns(UserWarning):y=x.T
        assert y is not x and y._storage is x._storage
        y.backward();assert x.grad.item()==1.
    finally:_state._scalar_t_warning_emitted=old


def test_no_grad_views_remember_tensor_base_and_copy_boundaries():
    x=torch.from_numpy(np.arange(6.,dtype=np.float64).reshape(2,3)).requires_grad_()
    with torch.no_grad():y=x.transpose(0,1).unsqueeze(0)
    assert y.requires_grad and y.is_leaf and y._view_base is x
    out=(y*3).sum();assert out.requires_grad
    out.backward();assert y.grad is None and x.grad is None
    with pytest.raises(RuntimeError):y.backward(torch.ones_like(y))
    root=torch.arange(6,dtype=torch.float64);view=root.reshape(2,3).requires_grad_()
    with torch.no_grad():z=view.transpose(0,1)
    assert not z.requires_grad and z._view_base is root
    with torch.no_grad():packed=x.T.flatten()
    assert not packed.requires_grad and packed._view_base is None


def test_explicit_no_grad_view_promotion_creates_a_leaf_gradient():
    root=torch.arange(3,dtype=torch.float64).requires_grad_()
    with torch.no_grad():view=root.view(3)
    view.requires_grad_();view.square().sum().backward()
    np.testing.assert_array_equal(view.grad.numpy(),np.array([0.,2.,4.]));assert root.grad is None
    with torch.no_grad():untracked=root.view(3)
    untracked.requires_grad_(False);assert untracked.requires_grad
    untracked.sum().backward();assert untracked.grad is None


@pytest.mark.parametrize('recording',[False,True])
def test_graphless_view_base_flags_change_without_a_gradient_edge(recording):
    base=torch.arange(6,dtype=torch.float64)
    with torch.set_grad_enabled(recording):view=base.view(2,3)
    assert not view.requires_grad and view.is_leaf
    base.requires_grad_();assert view.requires_grad and view.is_leaf
    base.requires_grad_(False);assert not view.requires_grad
    base.requires_grad_()
    out=view.square().sum()
    assert torch.autograd.grad(out,(base,view),allow_unused=True)==(None,None)
    assert base.grad is None and view.grad is None
    view.requires_grad_()
    base.requires_grad_(False);assert view.requires_grad
    view.square().sum().backward()
    np.testing.assert_array_equal(view.grad.numpy(),2*np.arange(6.).reshape(2,3))
    assert base.grad is None


def test_stopped_view_joint_vjp_keeps_live_input_gradients_and_hooks():
    base=torch.from_numpy(np.arange(6.,dtype=np.float64).reshape(2,3)).requires_grad_()
    with torch.no_grad():view=base.transpose(0,1)
    weight=torch.from_numpy(np.array([[.2,.3],[-.1,.4]])).requires_grad_()
    calls=[];view.register_hook(lambda grad:calls.append(grad))
    out=(view@weight).sum()
    gw,gb,gv=torch.autograd.grad(out,(weight,base,view),allow_unused=True)
    expected=np.repeat(np.arange(6.).reshape(2,3).sum(axis=1)[:,None],2,axis=1)
    np.testing.assert_array_equal(gw.numpy(),expected)
    assert gb is None and gv is None and calls==[]


def test_singleton_expand_metadata_allows_checked_copy_without_duplicate_writes():
    base=torch.arange(12,dtype=torch.float64).reshape(3,4)
    for shape in [(1,4),(1,1,4),(1,1,1,4)]:
        target=base[:1].expand(shape)
        assert target.stride()==(4,)*(len(shape)-1)+(1,)
        assert not target.numpy().flags.writeable
        before=base._version
        target.copy_(torch.full(shape,2.,dtype=torch.float64))
        assert base._version==before+1
        np.testing.assert_array_equal(base.numpy()[0],np.full(4,2.))
    before=base._version
    with pytest.raises(RuntimeError,match='overlapping'):
        base[:1].expand(2,4).copy_(torch.zeros(2,4,dtype=torch.float64))
    assert base._version==before
