import pytest
from problems.ai_infra.orca.experiment import run

@pytest.mark.t0
def test_validation_and_empty(impl):
    """An empty request stream has no iterations and invalid token counts are rejected."""
    assert impl.iteration_schedule([],[],1)==([],[])
    for a,n,c in [([0],[0],1),([0],[1],0),([-1],[2],1),([0,1],[1],1),([True],[1],1)]:
        with pytest.raises(ValueError): impl.iteration_schedule(a,n,c)

@pytest.mark.t1
def test_admit_without_waiting_for_slow_peer(impl):
    """Request 2 enters at t=1 after request 1 finishes, while request 0 continues."""
    assert impl.iteration_schedule([0,0,0],[3,1,1],2)==([(0,1),(0,2),(0,)],[3,1,2])

@pytest.mark.t1
def test_idle_arrivals_and_stable_order(impl):
    """Idle slots and unsorted arrivals retain absolute time and original result order."""
    assert impl.iteration_schedule([3,1,1],[1,2,1],1)==([(),(1,),(1,),(2,),(0,)],[5,3,4])

@pytest.mark.t2
def test_token_conservation_and_capacity(impl):
    """Every request gets exactly its required token iterations and never executes before arrival."""
    arrivals=[2,0,1,0,4]; lengths=[3,5,2,1,3]; before=(list(arrivals),list(lengths))
    batches,ends=impl.iteration_schedule(arrivals,lengths,3)
    assert (arrivals,lengths)==before
    for t,batch in enumerate(batches):
        assert len(batch)<=3 and len(set(batch))==len(batch)
        assert all(arrivals[i]<=t for i in batch)
    for i,n in enumerate(lengths):
        slots=[t for t,b in enumerate(batches) if i in b]
        assert len(slots)==n and ends[i]==slots[-1]+1 and slots==list(range(slots[0],ends[i]))

@pytest.mark.t2
def test_time_translation(impl):
    """Shifting every arrival shifts every completion but leaves the nonidle schedule unchanged."""
    batches,ends=impl.iteration_schedule([0,1,3],[4,1,2],2)
    shifted,late=impl.iteration_schedule([5,6,8],[4,1,2],2)
    assert shifted==[()]*5+batches and late==[t+5 for t in ends]

@pytest.mark.t3
def test_dynamic_batch_replay(impl,artifact):
    """Refilling short-request slots improves mean latency on the specified mixed-length trace."""
    r=run(impl.iteration_schedule); artifact('iteration_replay',r)
    assert r['metrics']['dynamic_mean_latency']<r['metrics']['static_mean_latency']
    assert r['metrics']['tokens_executed']==11
