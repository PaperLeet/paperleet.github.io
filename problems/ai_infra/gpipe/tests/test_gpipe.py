import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.ai_infra.gpipe.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Positive dimensions and matching regression shapes define valid work."""
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError): impl.pipeline_schedule(bad, 2)
        with pytest.raises(ValueError): impl.microbatch_mse(torch.ones(3,2), torch.ones(3), torch.ones(2), bad)
    with pytest.raises(ValueError): impl.microbatch_mse(torch.ones(3,2), torch.ones(2), torch.ones(2), 1)
    out=impl.microbatch_mse(torch.ones(3,2,dtype=torch.float64),torch.ones(3,dtype=torch.float64),torch.ones(2,dtype=torch.float64),2)
    assert out.shape == () and out.dtype == torch.float64

@pytest.mark.t1
def test_hand_schedule(impl):
    """Two stages and two microbatches fill for three slots, then drain for three."""
    events=impl.pipeline_schedule(2,2)
    assert events == [('F',0,0,0),('F',0,1,1),('F',1,0,1),('F',1,1,2),('B',1,1,3),('B',0,1,4),('B',1,0,4),('B',0,0,5)]

@pytest.mark.t1
def test_unequal_last_microbatch(impl):
    """Squared residuals 1,4,9 give 14/3 even when chunk sizes are 2 and 1."""
    x=torch.tensor([[1.],[2.],[3.]],dtype=torch.float64); y=torch.zeros(3,dtype=torch.float64)
    close(impl.microbatch_mse(x,y,torch.ones(1,dtype=torch.float64),2),torch.tensor(14/3,dtype=torch.float64))

@pytest.mark.t2
def test_dependencies_and_resource_exclusion(impl):
    """Every stage runs at most one event per slot and respects both data dependencies."""
    for k,m in [(1,1),(1,4),(4,1),(3,5)]:
        events=impl.pipeline_schedule(k,m); times={(p,s,i):t for p,s,i,t in events}
        assert len(times)==2*k*m and len({(s,t) for _,s,_,t in events})==2*k*m
        assert max(t for _,_,_,t in events)+1==2*(k+m-1)
        assert min(t for p,_,_,t in events if p=='B')>max(t for p,_,_,t in events if p=='F')
        for s in range(k-1):
            for i in range(m):
                assert times['F',s+1,i]>=times['F',s,i]+1
                assert times['B',s,i]>=times['B',s+1,i]+1

@pytest.mark.t2
def test_full_batch_gradient_and_input_immutability(impl):
    """The analytic gradient is 2 X^T(Xw-y)/N for every microbatch size."""
    x=torch.arange(21,dtype=torch.float64).reshape(7,3)/20; y=torch.linspace(-1,1,7,dtype=torch.float64)
    for size in (1,2,7,20):
        w=torch.tensor([.2,-.4,.3],dtype=torch.float64,requires_grad=True); before=x.clone()
        loss=impl.microbatch_mse(x,y,w,size)
        grad,=torch.autograd.grad(loss,(w,))
        close(grad,2*x.T@(x@w.detach()-y)/7,atol=1e-12,rtol=1e-12); close(x,before)

@pytest.mark.t3
def test_synchronous_training_and_bubbles(impl,artifact):
    """An uneven 11-sample batch must train identically; more microbatches amortize fill/drain bubbles."""
    data=run(impl.pipeline_schedule,impl.microbatch_mse); artifact('pipeline_training',data)
    assert data['metrics']['trajectory_error']<1e-12
    assert data['metrics']['final_loss']<data['metrics']['initial_loss']*.1
    assert data['metrics']['utilization_last']>data['metrics']['utilization_first']
