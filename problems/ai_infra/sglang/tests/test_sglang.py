import pytest
from problems.ai_infra.sglang.experiment import run

@pytest.mark.t0
def test_contract_and_immutability(impl):
    """Cache paths are prefix-closed and each token is a nonnegative integer."""
    for cache,prompt,cap in [({(1,2):0},(1,2),3),({},(),3),({},(1,-1),3),({},(1,2),1)]:
        with pytest.raises(ValueError): impl.radix_cache_step(cache,prompt,cap,1)
    original={(1,):0}; result,hit=impl.radix_cache_step(original,(1,2),2,1)
    assert original=={(1,):0} and result=={(1,):1,(1,2):1} and hit==1

@pytest.mark.t1
def test_longest_prefix_and_shared_branch(impl):
    """A differing third token reuses exactly the first two nodes."""
    cache={(1,):0,(1,2):0,(1,2,3):0}
    updated,hit=impl.radix_cache_step(cache,(1,2,4),4,1)
    assert hit==2 and updated=={(1,):1,(1,2):1,(1,2,3):0,(1,2,4):1}
    assert impl.radix_cache_step(updated,(9,2,4),7,2)[1]==0

@pytest.mark.t1
def test_leaf_lru_and_pinned_path(impl):
    """Old internal ancestors survive while their leaves are present; active paths cannot be evicted."""
    cache={(1,):0,(1,2):3,(8,):1,(8,9):1}
    result,hit=impl.radix_cache_step(cache,(7,),4,4,pinned=((8,9),))
    assert result=={(1,):0,(8,):1,(8,9):1,(7,):4} and hit==0

@pytest.mark.t2
def test_pin_exhaustion_is_atomic(impl):
    """If every leaf is in use, allocation fails without editing live cache state."""
    cache={(1,):0,(1,2):0}; original=dict(cache)
    with pytest.raises(ValueError): impl.radix_cache_step(cache,(7,),2,1,pinned=((1,2),))
    assert cache==original

@pytest.mark.t2
def test_timestamp_tie_and_repeated_prompt(impl):
    """Deterministic LRU ties use token order, and repeated prompts need zero new nodes."""
    result,_=impl.radix_cache_step({(3,):0,(2,):0},(4,),2,1)
    assert set(result)=={(3,),(4,)}
    result,hit=impl.radix_cache_step(result,(4,),2,2)
    assert hit==1 and result[(4,)]==2 and len(result)==2

@pytest.mark.t3
def test_cache_replay(impl,artifact):
    """The shared-prefix trace saves actual prefill tokens without exceeding node capacity."""
    r=run(impl.radix_cache_step); artifact('radix_replay',r)
    assert r['metrics']['cached_prefill_tokens']<r['metrics']['uncached_prefill_tokens']
    assert r['metrics']['maximum_nodes']<=12 and r['metrics']['reused_tokens']>0
