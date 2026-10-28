import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.ai_infra.awq.experiment import run

def scalar_quant(rows,bits,size):
    result=[]; maxq=2**bits-1
    for row in rows:
        out=[]
        for start in range(0,len(row),size):
            block=row[start:start+size]; step=max((max(block)-min(block))/maxq,1e-5); zero=max(0,min(maxq,round(-min(block)/step)))
            out.extend((max(0,min(maxq,round(v/step)+zero))-zero)*step for v in block)
        result.append(out)
    return result

@pytest.mark.t0
def test_contract(impl):
    """Group width divides the input dimension and the candidate grid is nonempty."""
    w=torch.ones(2,4,dtype=torch.float64); x=torch.ones(3,4,dtype=torch.float64)
    for bits,size in [(1,2),(4,3),(True,2)]:
        with pytest.raises(ValueError): impl.group_quantize(w,bits,size)
    with pytest.raises(ValueError): impl.awq_search(x,w,3,2,[])
    assert impl.group_quantize(w*0,3,2).dtype==w.dtype

@pytest.mark.t1
def test_scalar_group_quantizer(impl):
    """Independent scalar min/max, zero-point and ties-to-even operations verify each group."""
    w=torch.tensor([[-1.,-.2,.4,1.2],[.7,-.4,0.,0.]],dtype=torch.float64)
    close(impl.group_quantize(w,3,2),w.new_tensor(scalar_quant(w.tolist(),3,2)),atol=1e-12,rtol=1e-12)

@pytest.mark.t1
def test_single_candidate_mean_statistics(impl):
    """Mean absolute activations [2,1] imply sqrt(2) relative scaling at alpha .5."""
    x=torch.tensor([[4.,0.],[0.,2.]],dtype=torch.float64); w=torch.tensor([[-.4,.7],[.2,-.9]],dtype=torch.float64)
    s,q,alpha,loss=impl.awq_search(x,w,3,2,[.5]); raw=[math.sqrt(2),1.]; denom=math.sqrt(max(raw)*min(raw)); expected=x.new_tensor([v/denom for v in raw])
    close(s,expected); actual_q=x.new_tensor(scalar_quant((w*expected[None,:]).tolist(),3,2))/expected[None,:]
    close(q,actual_q); assert alpha==.5 and loss==pytest.approx(float(((x@q.T-x@w.T)**2).mean()))

@pytest.mark.t2
def test_grid_minimum_independent_oracle(impl):
    """The chosen exponent minimizes calibration output MSE rather than tensor-space weight error."""
    x=torch.tensor([[8.,.2,-.5,1.],[0.,.4,.1,-2.],[2.,-.3,.8,.2]],dtype=torch.float64); w=torch.tensor([[-.2,.8,-.7,.3],[.1,-.3,.6,-.8]],dtype=torch.float64); alphas=[0.,.25,.5,.75,1.]; scores=[]
    stats=[sum(abs(row[j]) for row in x.tolist())/len(x) for j in range(4)]
    for alpha in alphas:
        raw=[max(a**alpha,1e-4) for a in stats]; norm=math.sqrt(max(raw)*min(raw)); scale=[a/norm for a in raw]
        scaled=[[row[j]*scale[j] for j in range(4)] for row in w.tolist()]; quant=scalar_quant(scaled,3,4); candidate=[[row[j]/scale[j] for j in range(4)] for row in quant]
        errors=[sum(row[j]*(candidate[o][j]-w[o,j].item()) for j in range(4))**2 for row in x.tolist() for o in range(2)]; scores.append(sum(errors)/len(errors))
    s,q,alpha,error=impl.awq_search(x,w,3,4,alphas); best=min(range(len(scores)),key=lambda i:scores[i])
    assert alpha==alphas[best] and error==pytest.approx(scores[best],abs=1e-12)

@pytest.mark.t2
def test_ties_zero_channels_and_no_mutation(impl):
    """Zero activations have zero output error, so the first candidate wins and calibration stays detached."""
    x=torch.zeros(3,4,dtype=torch.float64,requires_grad=True); w=torch.randn(2,4,dtype=torch.float64,requires_grad=True); before=w.detach().clone()
    s,q,a,e=impl.awq_search(x,w,3,2,[.6,0.]); assert a==.6 and e==0 and bool(torch.isfinite(s).all())
    assert not s.requires_grad and not q.requires_grad; close(w,before)

@pytest.mark.t3
def test_awq_calibration_sweep(impl,artifact):
    """The searched calibration error is no worse than the included alpha-zero quantization baseline."""
    r=run(impl.group_quantize,impl.awq_search); artifact('awq_calibration',r)
    assert r['metrics']['best_mse']<=r['metrics']['baseline_mse']+1e-12
    assert r['metrics']['reported_error_difference']<1e-12 and r['metrics']['baseline_mse']>0
