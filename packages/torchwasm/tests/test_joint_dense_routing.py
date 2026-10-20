"""Repeated edges, mixed dtypes and hooks through an all-active joint rule."""
import numpy as np
import pytest
from torchwasm import torch
from torchwasm._autograd import joint_result

@pytest.mark.parametrize('selective',[False,True])
@pytest.mark.parametrize('reverse_dtypes',[False,True])
def test_shared_active_inputs_preserve_casting_hooks_and_repeated_backward(selective,reverse_dtypes):
    dx,dy=('float32','float64') if reverse_dtypes else ('float64','float32')
    raw=np.array([2.,3.,-4.]);x=torch.tensor(raw,dtype=getattr(torch,dx),requires_grad=True);y=torch.tensor([4.,-5.,6.],dtype=getattr(torch,dy),requires_grad=True)
    calls=[];seen=[]
    def backward(seed,needs):
        calls.append(needs);return seed*raw,seed*raw,seed*3
    value=joint_result(x.numpy()**2+3*y.numpy(),(x,x,y),backward,prune_inputs=selective);value.retain_grad();value.register_hook(lambda g:g*.5)
    x.register_hook(lambda g:seen.append((str(g.dtype),g.numpy().copy())))
    output=value+value
    for index,seed in enumerate([np.array([.5,-1.,2.]),np.array([-2.,3.,.5])]):
        x.grad=y.grad=None;value.grad=None;output.backward(torch.tensor(seed,dtype=output.dtype),retain_graph=index==0)
        np.testing.assert_array_equal(x.grad.numpy(),(2*raw*seed).astype(dx));np.testing.assert_array_equal(y.grad.numpy(),(3*seed).astype(dy));np.testing.assert_array_equal(value.grad.numpy(),seed.astype(value.numpy().dtype))
        assert seen[-1][0]==str(x.dtype);np.testing.assert_array_equal(seen[-1][1],(2*raw*seed).astype(dx))
    assert calls==[(True,True,True)]*2
