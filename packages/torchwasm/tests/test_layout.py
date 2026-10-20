"""Independent storage-offset formulas, scalar derivatives and layout lifecycle."""
import copy
import gc
import itertools
import weakref
import numpy as np
import pytest
from torchwasm import torch, UnsupportedOperation


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int16,np.int64,np.complex64,np.bool_])
@pytest.mark.parametrize('shape',[(24,),(2,12),(2,3,4),(1,2,3,4)])
def test_view_coordinates_follow_original_storage(dtype,shape):
    raw=np.arange(48).astype(dtype);source=raw[::2];x=torch.from_numpy(source);y=x.view(shape)
    assert y.stride()==tuple(2*int(np.prod(shape[axis+1:])) for axis in range(len(shape)))
    for index in itertools.product(*(range(size) for size in shape)):
        offset=np.ravel_multi_index(index,shape)*2
        assert y[index].item()==raw[offset].item()
    assert y.data_ptr()==x.data_ptr() and y._storage is x._storage
    assert y.element_size()==np.dtype(dtype).itemsize


@pytest.mark.parametrize('shape',[(0,),(0,3),(2,0,4)])
@pytest.mark.parametrize('factory',['empty','zeros','ones','rand','randn'])
def test_empty_factory_metadata_and_reshape(shape,factory):
    x=getattr(torch,factory)(shape);expected=[];step=1
    for size in reversed(shape):expected.append(step);step*=max(size,1)
    assert x.stride()==tuple(reversed(expected))
    assert x.view(shape).stride()==x.stride()
    assert x.view(-1).shape==(0,) and x.view(-1).stride()==(1,)
    with pytest.raises(RuntimeError):x.view(0,-1)


def test_imported_empty_strides_are_preserved():
    source=np.empty((0,3),dtype=np.float64);x=torch.from_numpy(source)
    assert x.stride()==(0,0) and x.view(x.shape).stride()==(0,0)
    assert x.view(3,0).stride()==(1,1)
    assert x.view(3,0)._storage is x._storage


@pytest.mark.parametrize('layout',['step','transposed','singleton','broadcast'])
def test_scalar_reverse_oracle(layout):
    if layout=='step':x=torch.from_numpy(np.arange(24.,dtype=np.float64)[::2]);shape=(3,4)
    elif layout=='transposed':x=torch.from_numpy(np.arange(12.,dtype=np.float64).reshape(3,4).T);shape=(4,3,1)
    elif layout=='singleton':x=torch.from_numpy(np.arange(12.,dtype=np.float64).reshape(3,4)[:,None,:]);shape=(3,4)
    else:x=torch.tensor([2.],dtype=torch.float64).expand(3,4);shape=(12,)
    x=x.detach().requires_grad_();seed=np.linspace(-.4,.6,12).reshape(shape)
    y=(x.view(shape)*torch.from_numpy(seed)).sum();y.backward()
    np.testing.assert_array_equal(x.grad.numpy(),seed.reshape(x.shape))
    # Nonoverlapping leaf perturbations retain an independent scalar oracle.
    if layout!='broadcast':
        actual=torch.autograd.grad((x.view(shape).square()*torch.from_numpy(seed)).sum(),x)[0]
        expected=2*x.numpy()*seed.reshape(x.shape)
        np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('primal_step',[1,2])
@pytest.mark.parametrize('tangent_fortran',[False,True])
def test_jvp_and_reverse_have_independent_quadratic_derivative(primal_step,tangent_fortran):
    a=np.arange(12*primal_step,dtype=np.float64).reshape(3,4*primal_step)[:,::primal_step]/7
    if primal_step==2:a=np.repeat(np.arange(12.,dtype=np.float64).reshape(3,4)/7,2,axis=1)[:,::2]
    b=np.linspace(.1,.8,12).reshape(3,4)
    if tangent_fortran:b=np.asfortranarray(b)
    x=torch.from_numpy(a).requires_grad_();t=torch.from_numpy(b).requires_grad_()
    p,q=torch.func.jvp(lambda z:z.view(2,6).square(),(x,),(t,))
    np.testing.assert_allclose(p.numpy(),a.reshape(2,6)**2,rtol=1e-13)
    np.testing.assert_allclose(q.numpy(),(2*a*b).reshape(2,6),rtol=1e-13)
    gx,gt=torch.autograd.grad(q.sum(),(x,t))
    np.testing.assert_allclose(gx.numpy(),2*b,rtol=1e-13)
    np.testing.assert_allclose(gt.numpy(),2*a,rtol=1e-13)


def test_detached_view_mutation_uses_same_saved_version():
    x=torch.arange(12,dtype=torch.float64).requires_grad_();loss=x.square().sum()
    alias=x.view(3,4).detach();alias[1,2]=100.
    assert x[6].item()==100. and alias._version==x._version==1
    with pytest.raises(UnsupportedOperation):loss.backward()


def test_checkpoint_copy_preserves_view_storage_without_original_alias():
    root=torch.arange(24,dtype=torch.float64);a=root.view(2,3,4);b=a[:,1,:].detach()
    copied=copy.deepcopy([a.detach(),b]);ca,cb=copied
    assert ca.stride()==a.stride() and cb.stride()==b.stride()
    cb[0,0]=-1.;assert ca[0,1,0].item()==-1. and root[4].item()==4.
    assert ca._storage is cb._storage and ca._storage is not root._storage


def test_metadata_view_lifetime_keeps_array_and_releases_it():
    data=np.arange(12.,dtype=np.float64);ref=weakref.ref(data)
    x=torch.from_numpy(data);view=x.view(3,4);del data,x;gc.collect()
    assert ref() is not None
    del view;gc.collect();assert ref() is None


def test_view_rejects_nonwhole_element_strides():
    raw=np.arange(16,dtype=np.uint8);array=np.ndarray((3,),dtype=np.float32,buffer=raw,strides=(1,))
    x=torch.Tensor(array)
    with pytest.raises(UnsupportedOperation):x.stride()
    with pytest.raises(UnsupportedOperation):x.view(3)


def test_jvp_view_as_uses_only_other_shape():
    x=torch.arange(6,dtype=torch.float64).requires_grad_();t=torch.ones(6,dtype=torch.float64).requires_grad_();other=torch.zeros(2,3,dtype=torch.float64).requires_grad_()
    p,q=torch.func.jvp(lambda a:a.view_as(other),(x,),(t,))
    assert p.shape==q.shape==(2,3)
    grads=torch.autograd.grad(q.sum(),(x,t,other),allow_unused=True)
    assert grads[0] is None and grads[2] is None
    np.testing.assert_array_equal(grads[1].numpy(),np.ones(6))


def test_readonly_view_and_broadcast_chunk_rejection():
    a=np.arange(12.,dtype=np.float64).reshape(3,4);a.setflags(write=False)
    with pytest.warns(UserWarning):
        from torchwasm import _state
        old=_state._numpy_readonly_warning_emitted;_state._numpy_readonly_warning_emitted=False
        try:x=torch.from_numpy(a)
        finally:_state._numpy_readonly_warning_emitted=old
    assert not x.view(2,6).numpy().flags.writeable
    varying=torch.arange(4,dtype=torch.float64).expand(3,4)
    with pytest.raises(RuntimeError):varying.view(-1)
    packed=varying.reshape(-1);np.testing.assert_array_equal(packed.numpy(),np.tile(np.arange(4.),3))


def test_empty_randperm_layout_does_not_consume_rng():
    with torch.random.fork_rng():
        torch.manual_seed(71);expected=torch.randperm(9).tolist()
        torch.manual_seed(71);empty=torch.randperm(0)
        assert empty.stride()==(1,) and empty.shape==(0,)
        assert torch.randperm(9).tolist()==expected
