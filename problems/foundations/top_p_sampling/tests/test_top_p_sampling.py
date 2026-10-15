import math
import pytest
import torch


def oracle(x, threshold, temperature):
    result=[]; v=x.shape[-1]
    for row in x.reshape(-1,v).tolist():
        # Scalar double arithmetic is independent of the tensor implementation.
        scaled=[value/temperature for value in row] if temperature>=1. else row
        m=max(scaled)
        weights=[math.exp(value-m if temperature>=1. else (value-m)/temperature) for value in scaled]
        total=sum(weights)
        order=sorted(range(v),key=lambda i:(-weights[i]/total,i))
        keep=[];mass=0.
        for i in order:
            keep.append(i);mass+=weights[i]/total
            if mass>=threshold and threshold!=1.:break
        retained=sum(weights[i] for i in keep)
        result.append([weights[i]/retained if i in keep else 0. for i in range(v)])
    return torch.tensor(result,dtype=x.dtype).reshape(x.shape)


@pytest.mark.t0
def test_contract(impl):
    """Candidate probabilities keep shape, dtype, device and normalized rows."""
    for dtype in (torch.float32,torch.float64):
        x=torch.zeros(2,3,4,dtype=dtype)
        out=impl.top_p_probs(x,.5)
        assert out.shape==x.shape and out.dtype==dtype and out.device==x.device
        torch.testing.assert_close(out,torch.tensor([.5,.5,0.,0.],dtype=dtype).expand_as(x))


@pytest.mark.t1
def test_independent_scalar_oracle(impl):
    """Compare each row to scalar stable sorting and normalized exponential weights."""
    for dtype in (torch.float32,torch.float64):
        x=torch.tensor([[[1.,-2.,4.,.5],[0.,3.,-1.,2.]],[[4.,1.,0.,2.],[-3.,-.5,2.,1.]]],dtype=dtype)
        for threshold in (.17,.61,.93,1.):
            for temperature in (.4,1.,2.5):
                torch.testing.assert_close(impl.top_p_probs(x,threshold,temperature),oracle(x,threshold,temperature))


@pytest.mark.t2
def test_extreme_and_empty_boundaries(impl):
    """Handle large finite logits, tiny temperature, one token and empty leading axes."""

    for dtype in (torch.float32,torch.float64):
        x=torch.tensor([10000.,-10000.,0.],dtype=dtype)
        out=impl.top_p_probs(x,.1,.01)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out,torch.tensor([1.,0.,0.],dtype=dtype))
        one=impl.top_p_probs(torch.tensor([7.],dtype=dtype),1.)
        torch.testing.assert_close(one,torch.ones(1,dtype=dtype))
        empty=impl.top_p_probs(torch.empty(0,3,dtype=dtype),.1)
        assert empty.shape==(0,3) and empty.dtype==dtype


@pytest.mark.t2
def test_full_finite_scalar_range(impl):
    """Python scalar parameters must not underflow when logits are float32."""
    cases = [
        (torch.zeros(4), 1e-60, 1., [1., 0., 0., 0.]),
        (torch.zeros(4, dtype=torch.float64), 5e-324, 1., [1., 0., 0., 0.]),
        (torch.tensor([1., 0., -1.]), .9, 1e-50, [1., 0., 0.]),
        (torch.tensor([1., 0., -1.], dtype=torch.float64), .9, 5e-324, [1., 0., 0.]),
        (torch.tensor([-2e38, 2e38]), 1., 2e38, [.119202922, .880797078]),
        (torch.tensor([-1e308, 1e308], dtype=torch.float64), 1., 1e308, [.119202922, .880797078]),
    ]
    for x, p, temperature, expected in cases:
        out = impl.top_p_probs(x, p, temperature)
        assert torch.isfinite(out).all(), (x, p, temperature)
        torch.testing.assert_close(out, torch.tensor(expected, dtype=x.dtype))


@pytest.mark.t2
def test_ties_threshold_and_full_distribution(impl):
    """Honor exact tie order, boundary equality and the unfiltered limiting case."""

    x=torch.zeros(4,dtype=torch.float64)
    expected=torch.tensor([.5,.5,0.,0.],dtype=x.dtype)
    torch.testing.assert_close(impl.top_p_probs(x,.5),expected)
    # Float32 input still uses double precision for the scalar threshold.
    torch.testing.assert_close(impl.top_p_probs(torch.zeros(4), .5000000001),
                               torch.tensor([1/3, 1/3, 1/3, 0.]))
    # Temperature can round unequal scores to equal working probabilities.
    torch.testing.assert_close(impl.top_p_probs(torch.tensor([0., 1.]), .5, 1e308),
                               torch.tensor([1., 0.]))
    x=torch.tensor([[2.,1.,2.,2.]],dtype=torch.float64)
    torch.testing.assert_close(impl.top_p_probs(x,.6),oracle(x,.6,1.))
    x=torch.tensor([0.,1.,-1.,3.],dtype=torch.float64)
    for temperature in (.25,2.,5.):
        torch.testing.assert_close(impl.top_p_probs(x,1.,temperature),torch.softmax(x/temperature,-1))
    x=torch.tensor([.1,.5,.15,.25],dtype=torch.float64).log()
    torch.testing.assert_close(impl.top_p_probs(x,.6),torch.tensor([0.,2/3,0.,1/3],dtype=x.dtype))


@pytest.mark.t2
def test_preservation_offsets_and_strides(impl):
    """Rows are independent, inputs unchanged, and logit offsets do not affect output."""
    x=torch.tensor([[2.,9.,-1.,8.,3.,7.,0.,6.],[1.,0.,4.,0.,-2.,0.,3.,0.]],dtype=torch.float64)[:,::2]
    before=x.clone(); out=impl.top_p_probs(x,.7,.8)
    torch.testing.assert_close(out,oracle(x,.7,.8))
    torch.testing.assert_close(impl.top_p_probs(x+torch.tensor([[100.],[-30.]],dtype=x.dtype),.7,.8),out)
    torch.testing.assert_close(x,before)
    assert (out>=0).all();torch.testing.assert_close(out.sum(-1),torch.ones(2,dtype=x.dtype))


@pytest.mark.t3
def test_candidate_distribution_experiment(impl,artifact):
    """Measure learner candidate probabilities and actual seeded categorical draws."""
    from problems.foundations.top_p_sampling.experiment import run
    artifact('toy_experiment',run(impl.top_p_probs))
