import pytest
from problems.ai_infra.distserve.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Empty traces are valid; unsorted arrivals and undefined per-output-token latencies are not."""
    assert impl.disaggregated_trace([],[],[],[],[])==([],[])
    assert impl.slo_goodput([],[],[],[],1,1)==0
    with pytest.raises(ValueError): impl.disaggregated_trace([1,0],[1,1],[1,1],[0,0],[2,2])
    with pytest.raises(ValueError): impl.disaggregated_trace([0],[1],[1],[0],[1])
    with pytest.raises(ValueError): impl.slo_goodput([0],[1],[1],[2],2,2)

@pytest.mark.t1
def test_independent_workers_and_transfer(impl):
    """Prefill overlaps earlier decode; KV transfer delays only the decode handoff."""
    first,last=impl.disaggregated_trace([0,0],[2,2],[3,1],[1,2],[3,4])
    assert first==[2,4] and last==[9,12]

@pytest.mark.t1
def test_both_slos_and_boundary(impl):
    """Only jointly satisfied TTFT and TPOT count, including threshold equality."""
    # TTFT=[2,4,6], TPOT=[1,3,1]: only the first meets both limits.
    assert impl.slo_goodput([0,0,0],[2,4,6],[4,10,8],[3,3,3],4,1)==pytest.approx(.1)

@pytest.mark.t2
def test_idle_and_time_translation(impl):
    """An idle worker waits for arrivals; translating the trace preserves all latencies and goodput."""
    a=[0,10]; args=([2,1],[1,2],[0,1],[3,2]); f,e=impl.disaggregated_trace(a,*args)
    shifted,ends=impl.disaggregated_trace([7,17],*args)
    assert shifted==[v+7 for v in f] and ends==[v+7 for v in e]
    assert impl.slo_goodput(a,f,e,[3,2],3,4)==impl.slo_goodput([7,17],shifted,ends,[3,2],3,4)

@pytest.mark.t2
def test_transfer_monotonicity(impl):
    """More KV transfer latency cannot produce earlier last tokens, while first tokens are unchanged."""
    args=([0,1,3],[2,1,2],[1,2,1]); f,e=impl.disaggregated_trace(*args,[0,0,0],[3,2,4]); f2,e2=impl.disaggregated_trace(*args,[3,3,3],[3,2,4])
    assert f==f2 and all(a<=b for a,b in zip(e,e2))

@pytest.mark.t3
def test_disaggregation_transfer_sweep(impl,artifact):
    """Goodput reflects both independent queues and growing KV-transfer delays under fixed SLOs."""
    r=run(impl.disaggregated_trace,impl.slo_goodput); artifact('disaggregated_replay',r)
    assert r['metrics']['first_token_last_request']==12
    assert r['metrics']['zero_transfer_goodput']>r['metrics']['long_transfer_goodput']
