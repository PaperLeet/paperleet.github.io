from paperleet.common.asserts import assert_close_diagnostic as close
import math
import pytest
import torch


def scalar_oracle(x,y,r,a):
    terms=[]
    for row,ids,response,attention in zip(x.tolist(),y.tolist(),r.tolist(),a.tolist()):
        for j in range(1,len(ids)):
            if response[j] and attention[j]:
                m=max(row[j-1]); terms.append(math.log(sum(math.exp(v-m) for v in row[j-1]))-(row[j-1][ids[j]]-m))
    return sum(terms)/len(terms) if terms else 0.


@pytest.mark.t0
def test_contract(impl):
    """Return the masked scalar in the original logits dtype and device."""
    for dtype in (torch.float32,torch.float64):
        x=torch.zeros(1,3,2,dtype=dtype); y=torch.tensor([[0,1,0]])
        r=torch.tensor([[False,False,True]]); a=torch.ones(1,3,dtype=torch.bool)
        result=impl.sft_loss(x,y,r,a)
        assert result.shape==() and result.dtype==dtype and result.device==x.device
        close(result,torch.tensor(math.log(2),dtype=dtype))


@pytest.mark.t1
def test_independent_valid_token_oracle(impl):
    """Use unequal response lengths and padded responses to expose incorrect reductions."""
    for dtype in (torch.float32,torch.float64):
        x=torch.tensor([[[3.,0.,-1.],[1.,-2.,4.],[2.,3.,-1.],[-2.,0.,1.],[9.,-9.,0.]],
                        [[-1.,2.,3.],[4.,-1.,0.],[1.,0.,2.],[-3.,1.,5.],[1.,2.,3.]]],dtype=dtype)
        y=torch.tensor([[0,2,1,0,2],[1,0,2,1,0]])
        r=torch.tensor([[False,False,True,True,True],[False,True,False,True,True]])
        a=torch.tensor([[True,True,True,True,True],[True,True,True,False,False]])
        close(impl.sft_loss(x,y,r,a),torch.tensor(scalar_oracle(x,y,r,a),dtype=dtype))


@pytest.mark.t2
def test_empty_selection_and_extreme_logits(impl):
    """All-masked and one-token inputs retain a zero-gradient graph; extreme logits stay finite."""
    for length in (1,4):
        x=torch.ones(2,length,3,dtype=torch.float64,requires_grad=True)
        y=torch.zeros(2,length,dtype=torch.long); mask=torch.zeros(2,length,dtype=torch.bool)
        loss=impl.sft_loss(x,y,mask,~mask);loss.backward()
        close(loss,torch.tensor(0.,dtype=x.dtype));close(x.grad,torch.zeros_like(x))
    x=torch.tensor([[[10000.,-10000.],[0.,0.],[8.,9.]]],dtype=torch.float64)
    y=torch.tensor([[0,1,0]]);r=torch.tensor([[False,True,False]])
    close(impl.sft_loss(x,y,r,torch.ones_like(r)),torch.tensor(20000.,dtype=x.dtype))


@pytest.mark.t2
def test_large_common_offset_preserves_masked_loss_and_gradient(impl):
    """Center scores before reduction, retaining exactly the selected gradient."""
    for dtype in (torch.float32, torch.float64):
        x = torch.full((1, 3, 2), 1e20, dtype=dtype, requires_grad=True)
        y = torch.tensor([[0, 1, 0]])
        response = torch.tensor([[False, True, False]])
        loss = impl.sft_loss(x, y, response, torch.ones_like(response))
        close(loss, torch.tensor(math.log(2), dtype=dtype))
        loss.backward()
        close(x.grad, torch.tensor([[[.5, -.5], [0., 0.], [0., 0.]]], dtype=dtype))


@pytest.mark.t2
def test_masked_analytic_gradients(impl):
    """The response mask belongs to target positions; excluded predictions have zero gradients."""
    x=(torch.arange(24,dtype=torch.float64).reshape(2,4,3)/10).requires_grad_()
    y=torch.tensor([[0,1,2,1],[2,0,1,0]])
    r=torch.tensor([[True,False,True,True],[False,True,True,True]])
    a=torch.tensor([[True,True,True,True],[False,True,False,False]])
    impl.sft_loss(x,y,r,a).backward(); expected=torch.zeros_like(x)
    for b,j in [(0,2),(0,3),(1,1)]:
        expected[b,j-1]=torch.softmax(x.detach()[b,j-1],-1)/3
        expected[b,j-1,int(y[b,j])]-=1/3
    close(x.grad,expected)


@pytest.mark.t2
def test_no_mutation_and_masked_target_invariance(impl):
    """Masked target IDs, first targets and final logits do not affect the objective."""
    x=torch.arange(48,dtype=torch.float64).reshape(2,4,6)[...,::2]/8
    y=torch.tensor([[0,1,2,0],[2,0,1,2]])
    r=torch.tensor([[False,False,True,True],[False,True,False,False]]);a=torch.ones_like(r)
    originals=[v.clone() for v in (x,y,r,a)]; loss=impl.sft_loss(x,y,r,a)
    changed=y.clone(); changed[~r]=(changed[~r]+1)%3
    close(impl.sft_loss(x,changed,r,a),loss)
    shifted=x+torch.arange(8,dtype=x.dtype).reshape(2,4,1)*100
    close(impl.sft_loss(shifted,y,r,a),loss)
    close(impl.sft_loss(x.repeat(2,1,1),y.repeat(2,1),r.repeat(2,1),a.repeat(2,1)),loss)
    for current,before in zip((x,y,r,a),originals):close(current,before)


@pytest.mark.t3
def test_prompt_difficulty_experiment(impl,artifact):
    """Changing excluded prompt predictions must not change measured response loss."""
    from problems.foundations.sft_loss_mask.experiment import run
    artifact('toy_experiment',run(impl.sft_loss))
