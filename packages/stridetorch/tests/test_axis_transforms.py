"""Independent matrix/axis view workflows and exact permutation adjoints."""
import gc
import itertools
import warnings
import weakref
import numpy as np
import pytest
from stridetorch import UnsupportedOperation,torch


@pytest.mark.parametrize('dtype',['float16','float32','float64'])
@pytest.mark.parametrize('functional',[False,True])
@pytest.mark.parametrize('name',['movedim','moveaxis','swapaxes','swapdims'])
def test_numpy_axis_values_and_adjoint_training(dtype,functional,name):
    data=np.arange(24,dtype=dtype).reshape(2,3,4).transpose(1,0,2)/16
    x=torch.from_numpy(data.copy(order='K')).requires_grad_()
    args=([0,2],[2,0]) if name in ('movedim','moveaxis') else (0,-1)
    y=getattr(torch,name)(x,*args) if functional else getattr(x,name)(*args)
    expected=np.moveaxis(data,*args) if name in ('movedim','moveaxis') else np.swapaxes(data,*args)
    np.testing.assert_array_equal(y.detach().numpy(),expected)
    assert y._storage is x._storage
    seed=np.arange(y.numel(),dtype=dtype).reshape(y.shape)/8+.5
    y.backward(torch.from_numpy(seed));gradient=x.grad.numpy().copy()
    expected_gradient=np.moveaxis(seed,args[1],args[0]) if name in ('movedim','moveaxis') else np.swapaxes(seed,*args)
    np.testing.assert_array_equal(gradient,expected_gradient)
    np.testing.assert_allclose(np.sum(data*gradient),np.sum(expected*seed),rtol=2e-3)
    with torch.no_grad():x.add_(x.grad,alpha=-.125)
    np.testing.assert_array_equal(x.detach().numpy(),data-gradient*.125)


@pytest.mark.parametrize('shape',[(),(3,),(2,3),(2,3,4)])
def test_t_and_batch_matrix_transpose_views_and_identity(shape):
    x=torch.arange(int(np.prod(shape)),dtype=torch.float32).reshape(shape)
    if len(shape)<=2:
        y=x.t();np.testing.assert_array_equal(y.numpy(),x.numpy().T);assert y is not x and y._storage is x._storage
        np.testing.assert_array_equal(torch.t(x).numpy(),y.numpy())
    else:
        with pytest.raises(RuntimeError):x.t()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if not shape:assert x.mT is x
        elif len(shape)==1:
            with pytest.raises(RuntimeError):x.mT
        else:
            y=x.mT;np.testing.assert_array_equal(y.numpy(),np.swapaxes(x.numpy(),-1,-2));assert y._storage is x._storage


@pytest.mark.parametrize('name',['movedim','moveaxis','t','mT','swapaxes','swapdims'])
def test_numpy_jvp_with_reverse_over_the_tangent(name):
    shape=(2,3) if name=='t' else (2,3,4)
    a=np.arange(np.prod(shape),dtype='float64').reshape(shape)/8
    b=np.asfortranarray(a+.3);x=torch.from_numpy(a).requires_grad_();t=torch.from_numpy(b).requires_grad_()
    def function(z):
        if name=='mT':return z.mT
        return getattr(z,name)(*(() if name=='t' else ((0,-1) if name.startswith('swap') else (0,-1))))
    p,q=torch.func.jvp(function,(x,),(t,));expected=function(torch.from_numpy(b)).numpy()
    np.testing.assert_array_equal(q.numpy(),expected)
    q.square().sum().backward();assert x.grad is None
    np.testing.assert_array_equal(t.grad.numpy(),2*b)


def test_axis_validation_and_mutating_views_guard_saved_values():
    x=torch.ones(2,3,requires_grad=True);loss=x.square().sum();view=x.swapaxes(axis0=0,axis1=1)
    with torch.no_grad():view.detach().add_(1)
    assert view._version==x._version
    with pytest.raises(UnsupportedOperation):loss.backward()
    for args in [([0,0],[0,1]),([0,1],[1]),([0,1],[0,0])]:
        with pytest.raises(RuntimeError):x.movedim(*args)
    for args in [(0,[0]),([0],0),(True,0),(None,0)]:
        with pytest.raises(TypeError):x.movedim(*args)


def test_metadata_transform_tape_releases_intermediate_values():
    leaf=torch.ones(2,3,4,requires_grad=True);middle=leaf*2;ref=weakref.ref(middle._data)
    output=middle.movedim([0,2],[2,0]).sum();del middle;gc.collect();assert ref() is None
    output.backward();np.testing.assert_array_equal(leaf.grad.numpy(),np.full(leaf.shape,2.))


def test_inverse_axis_cache_is_bounded_and_does_not_hold_public_argument_payloads():
    from stridetorch import _layout
    _layout.inverse_permutation.cache_clear()
    x=torch.ones(1,1,1,1,1,requires_grad=True)
    for axes in itertools.permutations(range(5)):
        x.permute(axes).sum().backward();x.grad=None
    assert _layout.inverse_permutation.cache_info().maxsize==32
    assert _layout.inverse_permutation.cache_info().currsize==32
    class Axis(int):pass
    axis=Axis(0);axis.payload=np.zeros(1024);reference=weakref.ref(axis.payload)
    output=x.permute(axis,1,2,3,4).sum();output.backward();del axis;gc.collect()
    assert reference() is None


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_reused_axis_patterns_accept_new_shapes_and_new_backward_seeds(dtype):
    for shape in [(2,3,4),(3,2,5),(1,3,2)]:
        x=torch.from_numpy(np.arange(np.prod(shape),dtype=dtype).reshape(shape)).requires_grad_()
        y=x.permute(1,2,0)
        for scale in (1.,-.5):
            seed=(np.arange(y.numel(),dtype=dtype).reshape(y.shape)+1)*scale
            gradient=torch.autograd.grad(y,x,grad_outputs=torch.from_numpy(seed),retain_graph=True)[0]
            np.testing.assert_array_equal(gradient.numpy(),np.transpose(seed,(2,0,1)))
            assert x.grad is None


def test_transpose_aliases_reject_nonnative_scalar_axis_forms():
    class Axis:
        def __index__(self):return 0
    x=torch.ones(2,3)
    for method in (x.transpose,x.swapaxes,x.swapdims):
        for axis in (Axis(),np.array(0),torch.tensor(True),torch.tensor([0])):
            with pytest.raises(TypeError):method(axis,0)
        with pytest.raises(TypeError):method(99,1.5)
        with pytest.raises(ValueError):method(99,2**63)
