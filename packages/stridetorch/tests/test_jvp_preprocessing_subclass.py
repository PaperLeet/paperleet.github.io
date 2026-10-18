"""Functional JVP retains caller-defined Tensor detach dispatch on private leaves."""
import numpy as np
import pytest
from stridetorch import torch


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('graph',[False,True])
@pytest.mark.parametrize('gx,gv',[(False,False),(False,True),(True,False),(True,True)])
@pytest.mark.parametrize('carrier',['input','vector'])
@pytest.mark.parametrize('strict',[False,True])
@pytest.mark.parametrize('keep_subclass',[False,True])
def test_functional_jvp_subclass_detach_dispatch(dtype,graph,gx,gv,carrier,strict,keep_subclass):
    calls=[]
    class Spy(torch.Tensor):
        def detach(self):
            calls.append('detach')
            return Spy(self._data) if keep_subclass else super().detach()
    x=(Spy if carrier=='input' else torch.Tensor)(np.array([2.,3.],dtype=dtype),requires_grad=gx)
    v=(Spy if carrier=='vector' else torch.Tensor)(np.array([.5,.25],dtype=dtype),requires_grad=gv)
    x.grad=torch.full_like(x,17.);v.grad=torch.full_like(v,19.)
    buffers=(x.grad,v.grad)
    y,d=torch.autograd.functional.jvp(lambda a:a.square(),x,v,create_graph=graph,strict=strict)
    detached=not(graph and (gx if carrier=='input' else gv))
    expected_calls=int(detached)
    if carrier=='vector' and detached and strict and keep_subclass:expected_calls+=1
    assert calls==['detach']*expected_calls
    assert x.requires_grad==gx and v.requires_grad==gv
    assert x.grad is buffers[0] and v.grad is buffers[1]
    assert y.requires_grad==d.requires_grad==graph
    np.testing.assert_array_equal(y.detach().numpy(),[4.,9.])
    np.testing.assert_array_equal(d.detach().numpy(),[2.,1.5])
    if graph:
        targets=tuple(a for a in (x,v) if a.requires_grad)
        if targets:
            grads=torch.autograd.grad(d.sum(),targets,allow_unused=True)
            if gx:np.testing.assert_array_equal(grads[0].numpy(),[1.,.5])
            if gv:np.testing.assert_array_equal(grads[-1].numpy(),[4.,6.])
