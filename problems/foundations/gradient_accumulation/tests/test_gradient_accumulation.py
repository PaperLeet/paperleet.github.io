from paperleet.common.asserts import assert_close_diagnostic as close
import pytest
import torch


@pytest.mark.t0
def test_contract(impl):
    """A single microbatch is returned in the same shape, dtype and device."""
    for dtype in (torch.float32,torch.float64):
        x=torch.tensor([[1.,-2.],[3.,4.]],dtype=dtype)
        out=impl.accumulate_gradients([x],[3])
        assert out.shape==x.shape and out.dtype==dtype and out.device==x.device
        close(out,x)


@pytest.mark.t1
def test_full_batch_autograd_oracle(impl):
    """Unequal microbatch mean gradients must reproduce full-batch autograd."""
    for dtype in (torch.float32,torch.float64):
        x=torch.tensor([[1.,2.],[3.,-1.],[.5,2.],[-1.,3.],[2.,1.],[4.,-.5]],dtype=dtype)
        y=torch.tensor([1.,-2.,.5,3.,-1.,2.],dtype=dtype)
        w=torch.tensor([.2,-.3],dtype=dtype,requires_grad=True)
        ((x@w-y).square().mean()).backward(); expected=w.grad.clone()
        gradients=[]
        for start,end in [(0,1),(1,4),(4,6)]:
            local=w.detach().clone().requires_grad_()
            ((x[start:end]@local-y[start:end]).square().mean()).backward()
            gradients.append(local.grad)
        close(impl.accumulate_gradients(gradients,[1,3,2]),expected)


@pytest.mark.t2
def test_zero_counts_scalars_and_empty_parameters(impl):
    """Empty microbatches contribute no gradient or denominator weight."""
    out=impl.accumulate_gradients([torch.tensor(999.),torch.tensor(2.),torch.tensor(-1.)],[0,1,3])
    close(out,torch.tensor(-.25))
    empty=impl.accumulate_gradients([torch.empty(0,2),torch.empty(0,2)],[2,3])
    assert empty.shape==(0,2)
    out=impl.accumulate_gradients([torch.tensor([8.]),torch.tensor([999.])],[7,0])
    close(out,torch.tensor([8.]))


@pytest.mark.t2
def test_regrouping_and_count_scale_invariance(impl):
    """Splitting a batch and scaling all counts preserve the effective gradient."""
    a=torch.tensor([1.,-3.],dtype=torch.float64);b=torch.tensor([5.,2.],dtype=a.dtype)
    expected=torch.tensor([4.,.75],dtype=a.dtype)
    close(impl.accumulate_gradients([a,b],[1,3]),expected)
    close(impl.accumulate_gradients([a,b],[5,15]),expected)
    close(impl.accumulate_gradients([a,b,b],[1,1,2]),expected)
    close(impl.accumulate_gradients([b,a],[3,1]),expected)


@pytest.mark.t2
def test_preserves_input_tensors_and_list(impl):
    """Strided gradients remain unchanged and the routine does not retain accumulated state."""
    gradients=[torch.arange(12,dtype=torch.float64).reshape(2,6)[:,::2],torch.ones(2,3,dtype=torch.float64)]
    before=[g.clone() for g in gradients]; identities=[id(g) for g in gradients];counts=[2,5]
    first=impl.accumulate_gradients(gradients,counts)
    close(first,(before[0]*2+before[1]*5)/7)
    second=impl.accumulate_gradients(gradients,counts);close(first,second)
    for g,b in zip(gradients,before):close(g,b)
    assert [id(g) for g in gradients]==identities and counts==[2,5]


@pytest.mark.t3
def test_unequal_microbatch_training_experiment(impl,artifact):
    """Learner accumulation drives real updates and matches the full-batch trajectory."""
    from problems.foundations.gradient_accumulation.experiment import run
    artifact('toy_experiment',run(impl.accumulate_gradients))
