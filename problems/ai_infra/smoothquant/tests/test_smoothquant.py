import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.ai_infra.smoothquant.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Scales align with input channels and invalid migration strength is rejected."""
    x=torch.ones(3,2,dtype=torch.float64); w=torch.ones(2,4,dtype=torch.float64)
    assert impl.smooth_scales(x,w).shape==(2,)
    with pytest.raises(ValueError): impl.smooth_scales(x,w,1.1)
    with pytest.raises(ValueError): impl.smooth_transform(x,w,torch.zeros(2,dtype=torch.float64))

@pytest.mark.t1
def test_hand_scales_and_equivalence(impl):
    """Activation maxima [8,2] and weight maxima [2,8] produce scales [2,.5] at alpha=.5."""
    x=torch.tensor([[8.,1.],[2.,-2.]],dtype=torch.float64); w=torch.tensor([[1.,-2.],[8.,3.]],dtype=torch.float64)
    s=impl.smooth_scales(x,w); close(s,torch.tensor([2.,.5],dtype=torch.float64))
    a,b=impl.smooth_transform(x,w,s); close(a@b,x@w,atol=1e-12,rtol=1e-12)

@pytest.mark.t1
def test_alpha_endpoints(impl):
    """Alpha zero normalizes weights and alpha one normalizes activation maxima."""
    x=torch.tensor([[4.,2.],[1.,-8.]],dtype=torch.float64); w=torch.tensor([[2.,1.],[4.,2.]],dtype=torch.float64)
    close(impl.smooth_scales(x,w,0),torch.tensor([.5,.25],dtype=torch.float64))
    close(impl.smooth_scales(x,w,1),torch.tensor([4.,8.],dtype=torch.float64))

@pytest.mark.t2
def test_zero_channels_and_offline_scales(impl):
    """Zero channels stay finite and offline calibration creates no gradient graph."""
    x=torch.zeros(3,2,dtype=torch.float64,requires_grad=True); w=torch.ones(2,2,dtype=torch.float64,requires_grad=True)
    s=impl.smooth_scales(x,w,.5,1e-4); close(s,torch.full((2,),1e-4,dtype=torch.float64)); assert not s.requires_grad

@pytest.mark.t2
def test_gradient_cancellation(impl):
    """Reciprocal scaling preserves d(XW)/dX and d(XW)/dW, with zero derivative in scales."""
    x=torch.randn(3,2,dtype=torch.float64,requires_grad=True); w=torch.randn(2,4,dtype=torch.float64,requires_grad=True); s=torch.tensor([3.,.2],dtype=torch.float64,requires_grad=True)
    a,b=impl.smooth_transform(x,w,s); gx,gw,gs=torch.autograd.grad((a@b).sum(),(x,w,s))
    close(gx,torch.ones(3,4,dtype=torch.float64)@w.detach().T); close(gw,x.detach().T@torch.ones(3,4,dtype=torch.float64)); close(gs,torch.zeros_like(s),atol=1e-12,rtol=1e-12)

@pytest.mark.t3
def test_int8_outlier_calibration(impl,artifact):
    """Balanced channel smoothing reduces simulated INT8 error on the specified activation outlier fixture."""
    r=run(impl.smooth_scales,impl.smooth_transform); artifact('smoothquant_calibration',r)
    assert r['metrics']['equivalence_error']<1e-12
    assert r['metrics']['balanced_mse']<r['metrics']['unsmoothed_mse']*.2
