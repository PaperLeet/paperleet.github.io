import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.ai_infra.flash_attention_2.experiment import run

def oracle(q,k,v,causal):
    outputs=[]; logs=[]
    for i,row in enumerate(q.tolist()):
        scores=[sum(a*b for a,b in zip(row,key))/math.sqrt(len(row)) for key in k.tolist()[:i+1 if causal else len(k)]]
        m=max(scores); exps=[math.exp(s-m) for s in scores]; total=sum(exps)
        outputs.append([sum(exps[j]*v[j,d].item() for j in range(len(exps)))/total for d in range(v.shape[1])]); logs.append(m+math.log(total))
    return q.new_tensor(outputs),q.new_tensor(logs)

@pytest.mark.t0
def test_shapes_and_validation(impl):
    """Square attention allows a different value width but requires positive block dimensions."""
    q=torch.ones(3,2,dtype=torch.float64); v=torch.ones(3,4,dtype=torch.float64)
    o,l=impl.flash2_forward(q,q,v,2,2); assert o.shape==(3,4) and l.shape==(3,) and o.dtype==q.dtype
    with pytest.raises(ValueError): impl.flash2_forward(q,q,v,0,2)
    with pytest.raises(ValueError): impl.flash2_forward(q,q[:2],v,2,2)

@pytest.mark.t1
def test_scalar_dense_and_tail_tiles(impl):
    """Independent scalar softmax verifies causal boundaries, scale, LSE and uneven tiles."""
    q=torch.sin(torch.arange(14,dtype=torch.float64).reshape(7,2)); k=q*2-.3; v=torch.cos(torch.arange(21,dtype=torch.float64).reshape(7,3))
    for causal in (False,True):
        expected=oracle(q,k,v,causal)
        for qb,kb in [(1,1),(3,2),(20,20)]:
            for a,b in zip(impl.flash2_forward(q,k,v,qb,kb,causal),expected): close(a,b,atol=1e-12,rtol=1e-12)

@pytest.mark.t1
def test_changing_running_max(impl):
    """A late larger key requires rescaling all accumulated mass and values."""
    q=torch.tensor([[1.],[1.],[1.]],dtype=torch.float64); k=torch.tensor([[-100.],[0.],[100.]],dtype=torch.float64); v=torch.tensor([[100.],[3.],[-2.]],dtype=torch.float64)
    for a,b in zip(impl.flash2_forward(q,k,v,2,1),oracle(q,k,v,False)): close(a,b,atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_causal_suffix_cannot_change_prefix(impl):
    """Future keys and values are invisible to early causal query rows."""
    q=torch.randn(5,2,dtype=torch.float64); k=torch.randn(5,2,dtype=torch.float64); v=torch.randn(5,3,dtype=torch.float64)
    before=impl.flash2_forward(q,k,v,2,3,True)[0]; k[3:]+=100; v[3:]-=90
    close(impl.flash2_forward(q,k,v,3,2,True)[0][:3],before[:3],atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_qkv_and_lse_gradients(impl):
    """Finite differences of the scalar oracle validate output and saved-LSE gradient paths."""
    tensors=[torch.randn(3,2,dtype=torch.float64).requires_grad_() for _ in range(3)]
    out,lse=impl.flash2_forward(*tensors,2,1,True); grads=torch.autograd.grad(out.sum()+.2*lse.sum(),tensors)
    for index in range(3):
        expected=torch.zeros_like(tensors[index])
        for j in range(6):
            plus=[x.detach().clone() for x in tensors]; minus=[x.detach().clone() for x in tensors]
            plus[index].reshape(-1)[j]+=1e-6; minus[index].reshape(-1)[j]-=1e-6
            a,b=oracle(*plus,True); c,d=oracle(*minus,True)
            expected.reshape(-1)[j]=(a.sum()+.2*b.sum()-c.sum()-.2*d.sum())/2e-6
        close(grads[index],expected,atol=1e-8,rtol=1e-6)

@pytest.mark.t3
def test_streaming_block_sweep(impl,artifact):
    """All tile choices compute the same attention and normalizer across complete sequences."""
    r=run(impl.flash2_forward); artifact('streaming_attention',r)
    assert r['metrics']['maximum_output_error']<1e-12 and r['metrics']['maximum_lse_error']<1e-12
