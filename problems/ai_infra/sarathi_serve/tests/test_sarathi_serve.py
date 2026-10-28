import pytest
from problems.ai_infra.sarathi_serve.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Token budgets fit all active decodes and request lengths remain positive."""
    for args in [([0,1],[3],1,2),([0,0],[3],4,2),([],[0],4,2),([],[3],4,0)]:
        with pytest.raises(ValueError): impl.hybrid_batch(*args)
    assert impl.hybrid_batch([],[],0,1)=={'decodes':[],'prefills':[],'remaining':[]}

@pytest.mark.t1
def test_decode_first_and_ongoing_chunk(impl):
    """Two decodes reserve tokens and a long prefill gets one four-token chunk."""
    assert impl.hybrid_batch([10,11],[13,2],8,4)=={'decodes':[10,11],'prefills':[(0,4)],'remaining':[9,2]}

@pytest.mark.t1
def test_admit_next_after_short_prefill_completes(impl):
    """The remaining budget may admit the next request once the head completes."""
    assert impl.hybrid_batch([7],[2,7,1],6,4)=={'decodes':[7],'prefills':[(0,2),(1,3)],'remaining':[0,4,1]}

@pytest.mark.t2
def test_budget_conservation_and_no_mutation(impl):
    """Every scheduled token consumes capacity once and leaves inputs unchanged."""
    for budget in range(2,11):
        decodes=[3,4]; prefills=[1,9,3]; r=impl.hybrid_batch(decodes,prefills,budget,3)
        assert decodes==[3,4] and prefills==[1,9,3] and r['decodes']==decodes
        assert 2+sum(c for _,c in r['prefills'])<=budget
        assert sum(prefills)==sum(r['remaining'])+sum(c for _,c in r['prefills'])
        assert all(0<c<=3 for _,c in r['prefills'])

@pytest.mark.t2
def test_full_decode_budget(impl):
    """Prefill waits intact when ongoing decode uses the entire iteration budget."""
    r=impl.hybrid_batch([2,5],[8,3],2,4)
    assert r['decodes']==[2,5] and r['prefills']==[] and r['remaining']==[8,3]

@pytest.mark.t3
def test_multiround_chunk_replay(impl,artifact):
    """The real planner finishes all prompt tokens while preserving every decode iteration."""
    r=run(impl.hybrid_batch); artifact('hybrid_replay',r)
    assert r['metrics']['prefill_tokens']==25 and r['metrics']['decode_stalls']==0
    assert r['metrics']['maximum_batch_tokens']<=8 and r['metrics']['maximum_chunk']<=4
