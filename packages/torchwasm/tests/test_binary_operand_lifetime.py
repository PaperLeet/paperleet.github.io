"""Regression checks for values a binary pullback does and does not need."""
import gc
import weakref
import numpy as np
import pytest
from torchwasm import torch

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('kind',['scalar_chain','tensor_left','tensor_right','divide_left'])
def test_unused_nonleaf_values_release_before_backward(dtype,kind):
 x=torch.tensor([[1.],[2.],[3.]],dtype=dtype,requires_grad=True)
 intermediate=x+.25;reference=weakref.ref(intermediate._data)
 constant=torch.tensor([[2.,4.]],dtype=dtype)
 if kind=='scalar_chain':
  value=intermediate
  for factor in [2.,.5]*8:value=value*factor
  expected=np.ones((3,1),dtype=np.float64)
 elif kind=='tensor_left':value=intermediate*constant;expected=np.full((3,1),6.)
 elif kind=='tensor_right':value=constant*intermediate;expected=np.full((3,1),6.)
 else:value=intermediate/constant;expected=np.full((3,1),.75)
 loss=value.sum();del intermediate,value;gc.collect();assert reference() is None
 gradient,=torch.autograd.grad(loss,x);np.testing.assert_array_equal(gradient.numpy(),expected);assert x.grad is None

@pytest.mark.parametrize('operation',['multiply','divide'])
def test_required_operand_values_remain_live_until_graph_is_consumed(operation):
 x=torch.tensor([3.,4.],dtype=torch.float64,requires_grad=True);y=torch.tensor([1.,2.],dtype=torch.float64,requires_grad=True)
 a=x+1.;b=y+2.;references=[weakref.ref(a._data),weakref.ref(b._data)]
 value=a*b if operation=='multiply' else a/b;loss=value.sum();del a,b,value;gc.collect();assert all(ref() is not None for ref in references)
 dx,dy=torch.autograd.grad(loss,(x,y));gc.collect();assert all(ref() is None for ref in references)
 if operation=='multiply':np.testing.assert_array_equal(dx.numpy(),[3.,4.]);np.testing.assert_array_equal(dy.numpy(),[4.,5.])
 else:np.testing.assert_allclose(dx.numpy(),[1/3,1/4]);np.testing.assert_allclose(dy.numpy(),[-4/9,-5/16])
