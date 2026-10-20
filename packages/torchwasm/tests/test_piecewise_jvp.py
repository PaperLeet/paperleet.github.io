"""Pinned production Torch2.6 inclusive clamp boundaries and abs's explicit zero reverse edge."""
import numpy as np
import pytest
from torchwasm import torch

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('case',['abs','clamp','clamp_min','clamp_max'])
@pytest.mark.parametrize('gx',[False,True])
@pytest.mark.parametrize('gv',[False,True])
def test_piecewise_primal_jvp_both_reverse_channels_and_hvp(dtype,case,gx,gv):
    data=np.array([-.2,-.1,0.,.1,.2],dtype=dtype);direction=np.array([.3,.5,.7,.9,1.1],dtype=dtype)
    x=torch.tensor(data,requires_grad=gx);v=torch.tensor(direction,requires_grad=gv)
    x.grad=torch.ones_like(x)*3;v.grad=torch.ones_like(v)*4;buffers=(x.grad,v.grad)
    def function(a):
        if case=='abs':return a.abs()
        if case=='clamp':return a.clamp(-.1,.1)
        return getattr(a,case)(-.1 if case=='clamp_min' else .1)
    if case=='abs':expected=np.abs(data);slope=np.sign(data)
    elif case=='clamp':expected=np.clip(data,-.1,.1);slope=((data>=np.asarray(-.1,dtype=dtype))&(data<=np.asarray(.1,dtype=dtype))).astype(dtype)
    elif case=='clamp_min':expected=np.maximum(data,-.1);slope=(data>=np.asarray(-.1,dtype=dtype)).astype(dtype)
    else:expected=np.minimum(data,.1);slope=(data<=np.asarray(.1,dtype=dtype)).astype(dtype)
    primal,tangent=torch.func.jvp(function,(x,),(v,))
    np.testing.assert_array_equal(primal.numpy(),expected)
    np.testing.assert_allclose(tangent.numpy(),slope*direction,rtol=1e-6)
    assert primal.requires_grad==gx and tangent.requires_grad==(gv or (case=='abs' and gx))
    inputs=tuple(a for a in [x,v] if a.requires_grad)
    if tangent.requires_grad:
        derivatives=torch.autograd.grad(tangent,inputs,torch.ones_like(tangent),allow_unused=True)
        by_id=dict(zip([id(a) for a in inputs],derivatives))
        if gx:
            if case=='abs':
                assert by_id[id(x)] is not None
                np.testing.assert_array_equal(by_id[id(x)].numpy(),np.zeros_like(data))
            else:assert by_id[id(x)] is None
        if gv:np.testing.assert_array_equal(by_id[id(v)].numpy(),slope)
    raw=function(x)
    if gx:np.testing.assert_array_equal(torch.autograd.grad(raw,x,torch.ones_like(raw))[0].numpy(),slope)
    _,hvp=torch.autograd.functional.hvp(lambda a:function(a).square().sum(),x,v)
    np.testing.assert_allclose(hvp.numpy(),2*slope*slope*direction,rtol=1e-6)
    with pytest.raises(RuntimeError):torch.autograd.functional.hvp(lambda a:function(a).sum(),x,v,strict=True)
    assert x.grad is buffers[0] and v.grad is buffers[1]
    assert x.grad.tolist()==[3.]*5 and v.grad.tolist()==[4.]*5
