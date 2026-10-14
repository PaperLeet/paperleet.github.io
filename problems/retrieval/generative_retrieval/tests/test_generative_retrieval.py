import itertools
import math
import pytest
import torch
from problems.retrieval.generative_retrieval.experiment import run


def scalar_search(callback,length,width):
    """Independent scalar normalization and explicit sorted-prefix pruning."""
    states={():0.}
    for _ in range(length):
        expanded={}
        for prefix,total in states.items():
            values=callback(prefix).tolist();peak=max(values)
            denominator=peak+math.log(sum(math.exp(v-peak) for v in values))
            for token,value in enumerate(values): expanded[prefix+(token,)]=total+value-denominator
        keys=sorted(expanded,key=lambda tokens:(-expanded[tokens],tokens))[:width]
        states={tokens:expanded[tokens] for tokens in keys}
    return list(states.items())


def equal_beams(actual,expected):
    assert isinstance(actual,list) and len(actual)==len(expected)
    for (tokens,score),(wanted,value) in zip(actual,expected):
        assert isinstance(tokens,tuple) and tokens==wanted
        assert isinstance(score,float) and score==pytest.approx(value,abs=1e-11,rel=1e-11)


@pytest.mark.t0
def test_identity_no_mutation_and_empty_lookup(impl):
    """Length zero makes no callback calls; wide beams retain all unique complete tuples."""
    def forbidden(prefix): raise AssertionError('No callback at length zero')
    equal_beams(impl.beam_search(forbidden,0,3),[((),0.)])
    logits=torch.tensor([0.,-1.],dtype=torch.float64);before=logits.clone()
    output=impl.beam_search(lambda prefix:logits,2,20)
    equal_beams(output,scalar_search(lambda prefix:logits,2,20))
    assert torch.equal(logits,before) and len(output)==4
    assert impl.lookup_candidates(output,torch.empty((0,2),dtype=torch.long),3)==[]
    assert impl.lookup_candidates([],torch.tensor([[0,0]],dtype=torch.long),3)==[]
    assert impl.lookup_candidates([((0,0),-1.)],torch.tensor([[0,0]],dtype=torch.long),0)==[]


@pytest.mark.t1
def test_two_step_hand_probability(impl):
    """A: .6*.51=.306, B: .4*.9=.36; B0 wins despite a smaller first probability."""
    table={():[.6,.4],(0,):[.51,.49],(1,):[.9,.1]}
    callback=lambda prefix:torch.tensor([math.log(p) for p in table[prefix]],dtype=torch.float64)
    equal_beams(impl.beam_search(callback,2,2),[((1,0),math.log(.36)),((0,0),math.log(.306))])


@pytest.mark.t2
def test_scalar_oracle_prefix_shifts_and_large_logits(impl):
    """Full-vocabulary conditional normalization removes distinct additive shifts at every prefix."""
    def callback(prefix):
        key=sum((i+2)*(t+1) for i,t in enumerate(prefix))
        return torch.tensor([math.sin(key+t*.9)+t*.2 for t in range(3)],dtype=torch.float64)
    def shifted(prefix): return callback(prefix)+(1000. if prefix and prefix[-1]==1 else -500.)
    for width in (1,2,5,27):
        expected=scalar_search(callback,3,width)
        equal_beams(impl.beam_search(callback,3,width),expected)
        equal_beams(impl.beam_search(shifted,3,width),expected)


@pytest.mark.t2
def test_global_expansion_and_lexicographic_ties(impl):
    """Expand all retained parents; equal scores sort by complete token tuples."""
    table={():[.55,.45],(0,):[.5,.5],(1,):[.99,.01]}
    callback=lambda p:torch.tensor([math.log(v) for v in table[p]],dtype=torch.float64)
    equal_beams(impl.beam_search(callback,2,2),[((1,0),math.log(.4455)),((0,0),math.log(.275))])
    output=impl.beam_search(lambda p:torch.zeros(3,dtype=torch.float64),3,4)
    equal_beams(output,[(p,-3*math.log(3)) for p in list(itertools.product(range(3),repeat=3))[:4]])
    equal_beams(impl.beam_search(lambda p:torch.tensor([17.],dtype=torch.float64),4,10),[((0,0,0,0),0.)])


@pytest.mark.t2
def test_invalid_and_duplicate_ids_filtered_before_k(impl):
    """Unknown complete IDs and repeated items consume no slots; original log-scores survive."""
    ids=torch.tensor([[2,1],[1,0],[0,0]],dtype=torch.long);before=ids.clone()
    beams=[((9,9),-.1),((1,0),-.3),((1,0),-.4),((2,1),-.7),((2,),-.8),((0,0),-1.2)]
    snapshot=list(beams)
    assert impl.lookup_candidates(beams,ids,2)==[(1,-.3),(0,-.7)]
    assert impl.lookup_candidates(beams,ids,10)==[(1,-.3),(0,-.7),(2,-1.2)]
    assert beams==snapshot and torch.equal(ids,before)
    assert impl.lookup_candidates([((9,9),0.)],ids,2)==[]


@pytest.mark.t2
def test_catalog_order_changes_indices_not_ids(impl):
    """Catalog row indices change under permutation, while retrieved IDs keep their ranked order."""
    ids=torch.tensor([[3,1,0],[0,2,0],[3,1,1]],dtype=torch.long)
    beams=[((3,1,1),-.2),((3,1,0),-.4),((0,2,0),-2.)]
    changed=ids[torch.tensor([2,0,1],dtype=torch.long)]
    for k in (1,2,3,10):
        a,b=impl.lookup_candidates(beams,ids,k),impl.lookup_candidates(beams,changed,k)
        assert [(tuple(ids[i].tolist()),s) for i,s in a]==[(tuple(changed[i].tolist()),s) for i,s in b]


@pytest.mark.t2
def test_wide_beam_equals_exhaustive_scoring(impl):
    """Whole-tree beam equals independent exhaustive sequence scoring, without catalog renormalization."""
    def callback(prefix): return torch.tensor([1.3+sum(prefix)*.2,-.8-len(prefix)*.1],dtype=torch.float64)
    expected=[]
    for tokens in itertools.product(range(2),repeat=4):
        score=0.
        for depth,token in enumerate(tokens):
            values=callback(tokens[:depth]).tolist();peak=max(values)
            score+=values[token]-peak-math.log(sum(math.exp(v-peak) for v in values))
        expected.append((tokens,score))
    expected.sort(key=lambda item:(-item[1],item[0]))
    equal_beams(impl.beam_search(callback,4,16),expected)


@pytest.mark.t3
def test_generate_then_filter_toy_catalog(impl,artifact):
    """Learner beams are checked against exact 625-ID model ranking; invalid IDs are filtered afterward."""
    result=run(impl.beam_search,impl.lookup_candidates);artifact('generative_retrieval_experiment',result)
    metrics=result['metrics']
    assert metrics['maximum_log_score_error']<1e-11
    assert metrics['exhaustive_top3_overlap']==1. and metrics['exhaustive_valid_count']==3
    assert metrics['full_space_invalid_ids']==617 and metrics['greedy_top3_overlap']<1.
