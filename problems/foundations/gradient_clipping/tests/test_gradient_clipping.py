import math
import pytest
import torch


def norm_oracle(gradients):
    return math.sqrt(math.fsum(float(v)**2 for g in gradients for v in g.reshape(-1).tolist()))


@pytest.mark.t0
def test_contract(impl):
    """Return new gradient tensors and the original scalar norm, preserving all types/shapes."""
    for dtype in (torch.float32,torch.float64):
        gradients=[torch.tensor([3.],dtype=dtype),torch.tensor([[4.]],dtype=dtype)]
        out,norm=impl.clip_gradients(gradients,2.5)
        assert isinstance(out,list) and len(out)==2 and norm.shape==() and norm.dtype==dtype
        for g,v in zip(gradients,out):assert g.shape==v.shape and g.dtype==v.dtype and g.device==v.device
        torch.testing.assert_close(norm,torch.tensor(5.,dtype=dtype))
        torch.testing.assert_close(out[0],torch.tensor([3*2.5/(5+1e-6)],dtype=dtype))


@pytest.mark.t1
def test_independent_global_norm_oracle(impl):
    """Match a scalar global norm and a single scale across differently shaped parameters."""
    for dtype in (torch.float32,torch.float64):
        grads=[torch.tensor([[1.,-3.],[2.,4.]],dtype=dtype),torch.tensor([-5.,2.,1.],dtype=dtype),torch.tensor(.5,dtype=dtype)]
        original=norm_oracle(grads)
        for maximum in (0.,2.,100.):
            out,norm=impl.clip_gradients(grads,maximum,.03)
            torch.testing.assert_close(norm,torch.tensor(original,dtype=dtype))
            scale=min(1.,maximum/(original+.03))
            for g,v in zip(grads,out):torch.testing.assert_close(v,g*scale)


@pytest.mark.t2
def test_zero_empty_and_exact_threshold(impl):
    """Zero and empty gradients stay finite; epsilon is applied at the declared location."""
    for grads in ([torch.zeros(2),torch.empty(0,3)],[torch.empty(0),torch.empty(0,2)]):
        out,norm=impl.clip_gradients(grads,1.)
        torch.testing.assert_close(norm,torch.tensor(0.))
        for g,v in zip(grads,out):torch.testing.assert_close(g,v)
    out,norm=impl.clip_gradients([torch.tensor([3.,4.],dtype=torch.float64)],5.,.1)
    torch.testing.assert_close(out[0],torch.tensor([3.,4.],dtype=norm.dtype)*(5/5.1))
    out,_=impl.clip_gradients([torch.tensor([2.,-3.])],0.)
    torch.testing.assert_close(out[0],torch.zeros(2))


@pytest.mark.t2
def test_large_and_tiny_norms(impl):
    """Finite representable norms survive overflow/underflow in naive squares."""
    for dtype, magnitude in ((torch.float32, 1e20), (torch.float64, 1e200),
                             (torch.float32, 1e-30), (torch.float64, 1e-200)):
        g = torch.tensor([3*magnitude, 4*magnitude], dtype=dtype)
        out, norm = impl.clip_gradients([g], 2.5*magnitude, magnitude/100)
        torch.testing.assert_close(norm / magnitude, torch.tensor(5., dtype=dtype))
        torch.testing.assert_close(out[0] / magnitude,
                                   torch.tensor([3., 4.], dtype=dtype) * (2.5/5.01))


@pytest.mark.t2
def test_direction_and_below_limit_invariants(impl):
    """Global clipping preserves direction and does not amplify small gradients."""
    grads=[torch.tensor([3.,-4.],dtype=torch.float64),torch.tensor([12.],dtype=torch.float64)]
    out,norm=impl.clip_gradients(grads,2.,.01)
    torch.testing.assert_close(norm,torch.tensor(13.,dtype=norm.dtype))
    torch.testing.assert_close(out[0]/grads[0],torch.full((2,),2/13.01,dtype=norm.dtype))
    torch.testing.assert_close(out[1]/grads[1],torch.tensor([2/13.01],dtype=norm.dtype))
    assert norm_oracle(out)<=2.
    unchanged,_=impl.clip_gradients(grads,20.)
    for g,v in zip(grads,unchanged):torch.testing.assert_close(v,g)


@pytest.mark.t2
def test_input_preservation_and_partition_invariance(impl):
    """The norm is independent of parameter partitioning and inputs remain untouched."""
    g=torch.arange(12,dtype=torch.float64).reshape(2,6)[:,::2]-3
    grads=[g,torch.tensor([2.,-1.],dtype=g.dtype)];before=[v.clone() for v in grads];ids=[id(v) for v in grads]
    out,norm=impl.clip_gradients(grads,1.5)
    flat=torch.cat([v.reshape(-1) for v in grads]);merged,total=impl.clip_gradients([flat],1.5)
    torch.testing.assert_close(norm,total)
    torch.testing.assert_close(torch.cat([v.reshape(-1) for v in out]),merged[0])
    for v,b in zip(grads,before):torch.testing.assert_close(v,b)
    assert [id(v) for v in grads]==ids


@pytest.mark.t3
def test_gradient_spike_experiment(impl,artifact):
    """Measure bounded learner updates while sweeping the size of a gradient spike."""
    from problems.foundations.gradient_clipping.experiment import run
    artifact('toy_experiment',run(impl.clip_gradients))
