import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from ..toy import fixture,scalar_assignments,scalar_loss
from ..experiment import run


@pytest.mark.t0
def test_shapes_dtype_strides_detachment_and_no_mutation(impl):
    """Sinkhorn produces independent stopped assignments and swapped prediction preserves floating scalars."""
    for dtype in (torch.float32,torch.float64):
        a,b=(v[:,::2] for v in fixture(dtype));a.requires_grad_();before=a.detach().clone()
        q=impl.sinkhorn_assignments(a,.2,3)
        assert q.shape==a.shape and q.dtype==dtype and q.device==a.device and not q.requires_grad
        close(q.sum(1),torch.ones(6,dtype=dtype),atol=2e-6,rtol=2e-6)
        loss=impl.swapped_prediction_loss(a,b,q,q,.3)
        assert loss.shape==() and loss.dtype==dtype
        q[0,0]+=1;close(a,before)


@pytest.mark.t0
def test_invalid_contract(impl):
    """Reject empty/nonfinite scores, invalid iterations and probability rows that are not normalized."""
    a,b=fixture();q=torch.ones_like(a)/3
    for values in (a[:0],a[:,:0],a.flatten(),a.long(),a*float('inf')):
        with pytest.raises(ValueError):impl.sinkhorn_assignments(values)
    for n in (0,-1,True,2.5):
        with pytest.raises(ValueError):impl.sinkhorn_assignments(a,iterations=n)
    for t in (0,-1,True,float('nan'),torch.tensor(.1)):
        with pytest.raises(ValueError):impl.sinkhorn_assignments(a,t)
        with pytest.raises(ValueError):impl.swapped_prediction_loss(a,b,q,q,t)
    for bad in (q*2,-q,q.float(),q[:1]):
        with pytest.raises(ValueError):impl.swapped_prediction_loss(a,b,bad,q)


@pytest.mark.t1
def test_finite_iteration_order_and_scalar_swapped_objective(impl):
    """Official row-then-column Sinkhorn has a defined finite-iteration result, followed by crossed CE."""
    a,b=fixture()
    for n in (1,3,9):
        for eps in (.1,.4):
            q1=impl.sinkhorn_assignments(a,eps,n);q2=impl.sinkhorn_assignments(b,eps,n)
            close(q1,torch.tensor(scalar_assignments(a.tolist(),eps,n),dtype=a.dtype),atol=1e-12,rtol=1e-11)
            close(q2,torch.tensor(scalar_assignments(b.tolist(),eps,n),dtype=a.dtype),atol=1e-12,rtol=1e-11)
            expected=scalar_loss(a.tolist(),b.tolist(),q1.tolist(),q2.tolist(),.37)
            close(impl.swapped_prediction_loss(a,b,q1,q2,.37),torch.tensor(expected,dtype=a.dtype),atol=1e-12,rtol=1e-11)


@pytest.mark.t2
def test_balanced_marginals_uniform_and_single_prototype(impl):
    """Final sample normalization is exact; repeated iterations approach equal prototype mass."""
    a,_=fixture();q=impl.sinkhorn_assignments(a,.4,60)
    close(q.sum(1),torch.ones(6,dtype=a.dtype),atol=1e-12,rtol=0)
    close(q.sum(0),torch.full((3,),2.,dtype=a.dtype),atol=1e-10,rtol=0)
    uniform=impl.sinkhorn_assignments(torch.zeros((4,3),dtype=a.dtype),.1,1)
    close(uniform,torch.full((4,3),1/3,dtype=a.dtype),atol=1e-12,rtol=0)
    one=impl.sinkhorn_assignments(a[:,:1],.2,3);close(one,torch.ones_like(one),atol=1e-12,rtol=0)
    close(impl.swapped_prediction_loss(a[:,:1],a[:,:1],one,one),torch.tensor(0.,dtype=a.dtype),atol=1e-12,rtol=0)


@pytest.mark.t2
def test_relabeling_common_shift_and_large_scores(impl):
    """Balanced assignments are equivariant to sample/prototype relabeling and stable under large global offsets."""
    a,_=fixture();base=impl.sinkhorn_assignments(a,.2,5)
    rows=torch.tensor([3,0,5,1,4,2]);cols=torch.tensor([2,0,1])
    close(impl.sinkhorn_assignments(a[rows][:,cols],.2,5),base[rows][:,cols],atol=1e-12,rtol=1e-11)
    close(impl.sinkhorn_assignments(a+10000,.2,5),base,atol=2e-11,rtol=2e-10)
    extreme=torch.tensor([[10000.,-10000.,0.],[-10000.,10000.,0.]],dtype=a.dtype)
    q=impl.sinkhorn_assignments(extreme,.1,10)
    assert bool(torch.isfinite(q).all()) and bool((q>=0).all())
    close(q.sum(1),torch.ones(2,dtype=a.dtype),atol=1e-10,rtol=0)


@pytest.mark.t2
def test_score_gradients_match_independent_softmax_and_assignments_are_stopped(impl):
    """Each score derivative is (softmax-q_other)/(2*B*temperature); assignment targets remain detached."""
    a,b=fixture();qa=scalar_assignments(a.tolist(),.2,3);qb=scalar_assignments(b.tolist(),.2,3)
    values=(a.clone().requires_grad_(),b.clone().requires_grad_(),torch.tensor(qa,dtype=a.dtype,requires_grad=True),torch.tensor(qb,dtype=a.dtype,requires_grad=True))
    grads=torch.autograd.grad(impl.swapped_prediction_loss(*values,.4),values,allow_unused=True)
    assert grads[0] is not None and grads[1] is not None and grads[2] is None and grads[3] is None
    for grad,scores,target in ((grads[0],a,qb),(grads[1],b,qa)):
        expected=[]
        for row,q in zip(scores.tolist(),target):
            peak=max(row);weights=[math.exp((v-peak)/.4) for v in row];total=sum(weights)
            expected.append([(w/total-prob)/(2*len(scores)*.4) for w,prob in zip(weights,q)])
        close(grad,torch.tensor(expected,dtype=a.dtype),atol=1e-12,rtol=1e-10)
    parents=tuple(torch.stack((v,torch.zeros_like(v)),dim=-1).reshape(6,6).requires_grad_() for v in (a,b))
    strided_loss=impl.swapped_prediction_loss(parents[0][:,::2],parents[1][:,::2],values[2].detach(),values[3].detach(),.4)
    assert strided_loss.requires_grad, 'strided scores must retain their gradient paths'
    parent_gradients=torch.autograd.grad(strided_loss,parents)
    for actual,expected in zip(parent_gradients,grads[:2]):
        close(actual[:,::2],expected,atol=1e-12,rtol=1e-11)
        close(actual[:,1::2],torch.zeros_like(expected),atol=0,rtol=0)


@pytest.mark.t3
def test_real_swapped_prediction_optimization(impl,artifact):
    """Eighty actual Adam updates fit balanced crossed soft labels rather than an arbitrary hard class."""
    result=run(impl.sinkhorn_assignments,impl.swapped_prediction_loss);artifact('swav_training',result);m=result['metrics']
    assert m['updates']==80 and m['trainable_parameters']==36
    assert m['assignment_error']<1e-11 and m['scalar_error']<1e-11 and m['maximum_balance_error']<1e-8
    assert m['assignments_detached'] and m['final_kl']<.002
    assert m['final_loss']<m['initial_loss']-.1
