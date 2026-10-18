import math
import pytest
import torch
from paperleet.common.asserts import assert_close_diagnostic as close


def oracle(logits, p, temperature, uniforms):
    """Scalar softmax, shortest-prefix support and inverse-CDF, independent of helper."""
    rows=logits.reshape(-1,logits.shape[-1]).tolist()
    result=[]
    for row,u in zip(rows,uniforms.reshape(-1).tolist()):
        scaled=[(float(v)-max(row))/temperature for v in row]
        weights=[math.exp(v) for v in scaled]
        total=math.fsum(weights); probabilities=[v/total for v in weights]
        order=sorted(range(len(row)),key=lambda i:(-probabilities[i],i))
        keep=[]; mass=0.
        for i in order:
            keep.append(i); mass+=probabilities[i]
            if p<1. and mass>=p: break
        weights=[probabilities[i] if i in keep else 0. for i in range(len(row))]
        cdf=[]; mass=0.
        for weight in weights: mass+=weight; cdf.append(mass)
        result.append(next(i for i,value in enumerate(cdf) if value/cdf[-1]>u))
    return torch.tensor(result,dtype=torch.int64).reshape(logits.shape[:-1])


def seeded(seed=42):
    return torch.Generator().manual_seed(seed)


@pytest.mark.t0
def test_contract_and_shapes(impl):
    """One token per row, with vocabulary IDs and no remaining vocabulary axis."""
    for dtype in (torch.float32,torch.float64):
        for shape in ((4,),(2,3,4),(0,4),(2,0,4)):
            logits=torch.zeros(shape,dtype=dtype)
            got=impl.sample_top_p(logits,.5,1.,seeded())
            assert got.shape==logits.shape[:-1] and got.dtype==torch.int64 and got.device==logits.device
            assert ((got>=0)&(got<2)).all()


@pytest.mark.t1
def test_scalar_inverse_cdf_oracle(impl):
    """Hand-independent nucleus formation and vocabulary-order inverse CDF match exactly."""
    base=torch.tensor([[.1,.5,.15,.25],[.7,.05,.2,.05],[.2,.2,.2,.4]],dtype=torch.float64).log()
    logits=base.unsqueeze(0).expand(71,3,4)
    for p,temp in ((.01,1.),(.6,1.),(.85,.4),(1.,2.)):
        u=torch.rand(logits.shape[:-1],dtype=torch.float64,generator=seeded(17))
        got=impl.sample_top_p(logits,p,temp,seeded(17))
        close(got,oracle(logits,p,temp,u))


@pytest.mark.t2
def test_generator_stream_and_global_isolation(impl):
    """The supplied RNG is neither reseeded nor replaced, and exactly one draw is used per row."""
    logits=torch.tensor([[0.,1.,-.5]]).expand(32,3)
    expected=seeded(51);actual=seeded(51)
    u=torch.rand((32,),dtype=torch.float64,generator=expected)
    torch.manual_seed(19)
    global_expected=torch.rand(7,dtype=torch.float64)
    torch.manual_seed(19)
    got=impl.sample_top_p(logits,1.,1.,actual)
    close(got,oracle(logits,1.,1.,u))
    close(torch.rand(7,dtype=torch.float64),global_expected)
    close(torch.rand(11,dtype=torch.float64,generator=actual),torch.rand(11,dtype=torch.float64,generator=expected))
    a=impl.sample_top_p(logits,1.,1.,actual)
    u2=torch.rand((32,),dtype=torch.float64,generator=expected)
    close(a,oracle(logits,1.,1.,u2))


@pytest.mark.t2
def test_nucleus_boundaries_and_original_ids(impl):
    """Exact equal-mass thresholds retain the shortest prefix; zeros never become tokens."""
    for p,max_id in ((.25,0),(.5,1),(1.,3)):
        got=impl.sample_top_p(torch.zeros(512,4,dtype=torch.float64),p,1.,seeded())
        assert got.min()==0 and got.max()==max_id
    for dtype in (torch.float32,torch.float64):
        logits=torch.tensor([-1000.,0.,-1000.,1.],dtype=dtype).expand(2048,4)
        draws=impl.sample_top_p(logits,1.,1.,seeded())
        assert ((draws==1)|(draws==3)).all()
        close(impl.sample_top_p(logits,.01,1.,seeded()),torch.full((2048,),3,dtype=torch.int64))
        close(impl.sample_top_p(torch.ones(3,1,dtype=dtype),1.,1.,seeded()),torch.zeros(3,dtype=torch.int64))


@pytest.mark.t2
def test_strides_temperature_and_input_preservation(impl):
    """Noncontiguous batches and shifted logits preserve independent row draws and inputs."""
    x=torch.arange(36,dtype=torch.float64).reshape(3,12)[:,::3]/7
    before=x.clone()
    u=torch.rand((3,),dtype=torch.float64,generator=seeded(8))
    close(impl.sample_top_p(x,.8,.6,seeded(8)),oracle(x,.8,.6,u))
    close(impl.sample_top_p(x+100,.8,.6,seeded(8)),oracle(x,.8,.6,u))
    close(x,before)
    close(impl.sample_top_p(torch.tensor([[0.,-1.]],dtype=torch.float32),1.,1e-100,seeded()),torch.tensor([0]))
    empty=torch.empty(0,4,dtype=torch.float64);a=seeded();b=seeded()
    impl.sample_top_p(empty,.5,1.,a)
    close(torch.rand(3,generator=a),torch.rand(3,generator=b))


@pytest.mark.t2
def test_supplied_cdf_interval_endpoints(impl,monkeypatch):
    """Half-open intervals skip zero mass at u=0 and select the next token at an exact CDF boundary."""
    def helper(logits,p,temperature):
        return torch.tensor([[0.,.5,0.,.5]],dtype=torch.float64).expand(3,4)
    def uniforms(*args,**kwargs):
        return torch.tensor([0.,.5,1.-2**-52],dtype=torch.float64)
    monkeypatch.setattr(impl,'nucleus_probs',helper)
    monkeypatch.setattr(torch,'rand',uniforms)
    close(impl.sample_top_p(torch.zeros(3,4),1.,1.,seeded()),torch.tensor([1,3,3]))
    # A representable tail-rounding gap must not yield an out-of-vocabulary ID.
    monkeypatch.setattr(impl, 'nucleus_probs', lambda *args: torch.tensor([[0.,.5,0.,.5-2**-53]],dtype=torch.float64))
    monkeypatch.setattr(torch, 'rand', lambda *args, **kwargs: torch.tensor([1.-2**-53],dtype=torch.float64))
    close(impl.sample_top_p(torch.zeros(1,4),1.,1.,seeded()),torch.tensor([3]))


@pytest.mark.t3
def test_actual_sampling_experiment(impl,artifact):
    """Learner-generated tokens, not provided sampling code, obey the target distribution."""
    from problems.foundations.top_p_draw.experiment import run
    artifact('toy_experiment',run(impl.sample_top_p))
