import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from ..toy import fixture, scalar_nce
from ..experiment import run


@pytest.mark.t0
def test_shape_dtype_preservation_and_invalid_inputs(impl):
    """A fixed bank supplies float scores and valid int64 sampled IDs without in-place edits."""
    q,m,ids=fixture()
    for dtype in (torch.float32,torch.float64):
        x,bank=q.to(dtype),m.to(dtype);old=bank.clone()
        loss=impl.instance_nce_loss(x,bank,ids)
        assert loss.shape==() and loss.dtype==dtype
        out=impl.update_memory(bank,torch.tensor([2,0]),x)
        assert out.shape==bank.shape and out.dtype==dtype and not out.requires_grad
        close(bank,old)
    for bad in (ids.float(),ids[:1],ids+10,ids[:,:1]):
        with pytest.raises(ValueError):impl.instance_nce_loss(q,m,bad)
    for t,z in ((0.,0.),(-1.,0.),(.2,float('inf'))):
        with pytest.raises(ValueError):impl.instance_nce_loss(q,m,ids,t,z)
    with pytest.raises(ValueError):impl.update_memory(m,torch.tensor([1,1]),q)
    with pytest.raises(ValueError):impl.update_memory(m,torch.tensor([0,1]),q,1.1)


@pytest.mark.t1
def test_equal_model_and_noise_closed_form(impl):
    """When p_model=K/N, each binary term is log2, giving (K+1)log2 rather than log(K+1)."""
    q=torch.zeros(2,3,dtype=torch.float64);bank=torch.ones(5,3,dtype=q.dtype)
    ids=torch.tensor([[2,1,4],[0,0,0]])
    loss=impl.instance_nce_loss(q,bank,ids,1.,-math.log(2/5))
    close(loss,torch.tensor(3*math.log(2),dtype=q.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_independent_binary_nce_oracle_and_repeated_noise(impl,ref):
    """Original binary NCE sums sampled noise terms, including repeated draws, with a fixed partition Z."""
    q,m,ids=fixture()
    for t,z in ((.2,0.),(.7,1.8),(2.,-2.)):
        actual=impl.instance_nce_loss(q,m,ids,t,z)
        expected=scalar_nce(q.tolist(),m.tolist(),ids.tolist(),t,z)
        close(actual,torch.tensor(expected,dtype=q.dtype),atol=1e-12,rtol=1e-12)
        close(actual,ref.instance_nce_loss(q,m,ids,t,z),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_query_gradient_finite_difference_and_frozen_bank(impl):
    """Only the current query receives NCE gradients; memory lookup is a fixed historical target."""
    q,m,ids=fixture();q.requires_grad_();m.requires_grad_()
    gq,gm=torch.autograd.grad(impl.instance_nce_loss(q,m,ids,.7,1.1),(q,m),allow_unused=True)
    assert gm is None or float(gm.abs().max())==0.,'persistent memory must be detached'
    rows=q.detach().tolist();bank=m.detach().tolist();expected=[]
    for i,row in enumerate(rows):
        grad=[]
        for j,old in enumerate(row):
            row[j]=old+1e-6;plus=scalar_nce(rows,bank,ids.tolist(),.7,1.1)
            row[j]=old-1e-6;minus=scalar_nce(rows,bank,ids.tolist(),.7,1.1)
            row[j]=old;grad.append((plus-minus)/2e-6)
        expected.append(grad)
    close(gq,torch.tensor(expected,dtype=q.dtype),atol=1e-9,rtol=2e-7)
    parent=torch.stack([q.detach(),torch.zeros_like(q)],dim=2).reshape(2,4).requires_grad_()
    gp=torch.autograd.grad(impl.instance_nce_loss(parent[:,::2],m.detach(),ids,.7,1.1),parent)[0]
    close(gp[:,::2],gq,atol=1e-12,rtol=1e-11);close(gp[:,1::2],torch.zeros_like(gq),atol=0,rtol=0)


@pytest.mark.t2
def test_indexed_normalized_ema_endpoints_and_no_mutation(impl):
    """Each unique instance ID stores normalize(m old+(1-m)new), while untouched bank rows stay fixed."""
    q,m,_=fixture();old=m.clone();ids=torch.tensor([2,0])
    m.requires_grad_();q.requires_grad_()
    for momentum in (0.,.8,1.):
        got=impl.update_memory(m,ids,q,momentum)
        expected=old.tolist()
        for row,index in zip(q.detach().tolist(),ids.tolist()):
            mixed=[momentum*a+(1-momentum)*b for a,b in zip(old[index].tolist(),row)]
            norm=math.sqrt(sum(a*a for a in mixed));expected[index]=[a/norm for a in mixed]
        close(got,torch.tensor(expected,dtype=m.dtype),atol=1e-12,rtol=1e-12)
        assert not got.requires_grad
    close(m.detach(),old,atol=0,rtol=0)
    close(impl.update_memory(old,torch.tensor([2]),-old[2:3],.5)[2],torch.zeros(2,dtype=m.dtype),atol=0,rtol=0)
    # A nonzero sub-floor mixture must divide by the declared epsilon, not its own norm.
    small_bank=torch.tensor([[1.,0.],[0.,2.],[3.,0.]],dtype=m.dtype)
    small_features=torch.tensor([[.001,.002]],dtype=m.dtype)
    got=impl.update_memory(small_bank,torch.tensor([2]),small_features,0.,.1)
    expected=small_bank.tolist()
    row=small_features[0].tolist();denominator=max(math.sqrt(sum(x*x for x in row)),.1)
    expected[2]=[x/denominator for x in row]
    close(got,torch.tensor(expected,dtype=m.dtype),atol=1e-12,rtol=1e-12)
    # Momentum one still normalizes selected rows and must return separate storage.
    frozen=small_bank.clone()
    endpoint=impl.update_memory(small_bank,torch.tensor([0,1]),q.detach(),1.)
    close(endpoint,torch.tensor([[1.,0.],[0.,1.],[3.,0.]],dtype=m.dtype),atol=1e-12,rtol=1e-12)
    endpoint[0,0]+=4.;endpoint[2,0]+=7.
    close(small_bank,frozen,atol=0,rtol=0)




@pytest.mark.t2
def test_bank_identity_permutation_and_batch_mean(impl):
    """Renaming persistent instance IDs preserves NCE; duplicating a query batch preserves its mean."""
    q,m,ids=fixture();p=[2,0,3,1]
    mapped=torch.tensor([[p.index(i) for i in row] for row in ids.tolist()])
    base=impl.instance_nce_loss(q,m,ids,.4,.7)
    close(impl.instance_nce_loss(q,m[p],mapped,.4,.7),base,atol=1e-12,rtol=1e-12)
    close(impl.instance_nce_loss(torch.cat([q,q]),m,torch.cat([ids,ids]),.4,.7),base,atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_partition_constant_is_not_softmax_shift_invariance(impl):
    """Binary data-versus-noise classification depends on calibrated likelihood p=exp(score)/Z."""
    q,m,ids=fixture()
    a=impl.instance_nce_loss(q,m,ids,.7,0.);b=impl.instance_nce_loss(q,m,ids,.7,3.)
    assert abs(float(a-b))>.1
    for z in (0.,3.):
        close(impl.instance_nce_loss(q,m,ids,.7,z),torch.tensor(scalar_nce(q.tolist(),m.tolist(),ids.tolist(),.7,z),dtype=q.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_extreme_binary_scores_are_finite(impl):
    """Log-domain softplus remains finite when unnormalized likelihoods would overflow."""
    q,m,ids=fixture();q=(q*1e5).requires_grad_()
    loss=impl.instance_nce_loss(q,m,ids,.1,-100.)
    assert bool(torch.isfinite(loss))
    close(loss,torch.tensor(scalar_nce(q.detach().tolist(),m.tolist(),ids.tolist(),.1,-100.),dtype=q.dtype),atol=1e-8,rtol=1e-12)
    assert bool(torch.isfinite(torch.autograd.grad(loss,q)[0]).all())


@pytest.mark.t3
def test_actual_nce_and_memory_training(impl,artifact):
    """Eighty query updates and indexed EMA refreshes must lower binary NCE and align each instance."""
    result=run(impl.instance_nce_loss,impl.update_memory);artifact('instance_training',result);m=result['metrics']
    assert m['updates']==80 and m['trainable_parameters']==36
    assert m['final_loss']<.8 and m['final_loss']<.4*m['initial_loss']
    assert m['final_alignment']>.99 and m['final_alignment']>m['initial_alignment']
    assert m['bank_norm_error']<1e-12 and not m['bank_requires_grad']
