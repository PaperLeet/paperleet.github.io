import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from ..toy import scalar_loss, fixture
from ..experiment import run


@pytest.mark.t0
def test_shape_dtype_strides_and_no_mutation(impl):
    """A scalar loss preserves float32/64, accepts strided inputs and leaves both views intact."""
    for dtype in (torch.float32,torch.float64):
        a,b=fixture(dtype);a=a[:,::2];b=b[:,::2];old_a,old_b=a.clone(),b.clone()
        loss=impl.nt_xent_loss(a,b)
        assert loss.shape==() and loss.dtype==dtype and loss.device==a.device
        close(loss,torch.tensor(scalar_loss(a.tolist(),b.tolist(),.5),dtype=dtype),atol=2e-6,rtol=2e-6)
        close(a,old_a);close(b,old_b)


@pytest.mark.t0
def test_invalid_input_contract(impl):
    """Reject empty/mismatched views, nonfloating values and invalid temperatures."""
    a,b=fixture()
    for x,y in [(a,b[:2]),(a.flatten(),b.flatten()),(a[:0],b[:0]),(a[:,:0],b[:,:0]),
                (a.to(torch.int64),b.to(torch.int64)),(a,b.float()),
                (a*float('nan'),b),(a,b*float('inf'))]:
        with pytest.raises(ValueError):impl.nt_xent_loss(x,y)
    for temperature in (0,-1,float('nan'),float('inf'),True,'0.2',torch.tensor(.2)):
        with pytest.raises(ValueError):impl.nt_xent_loss(a,b,temperature)


@pytest.mark.t1
def test_orthogonal_pairs_collapse_and_single_pair(impl):
    """Two orthogonal pairs expose same-view negatives and reduction; one pair has zero loss/gradient."""
    a=torch.eye(2,dtype=torch.float64)
    for temperature in (.1,.5,2.):
        expected=math.log1p(2*math.exp(-1/temperature))
        close(impl.nt_xent_loss(a,a,temperature),torch.tensor(expected,dtype=a.dtype),atol=1e-12,rtol=1e-10)
    collapse=torch.tensor([[1.,2.,-1.]]*4,dtype=a.dtype)
    close(impl.nt_xent_loss(collapse,collapse,.3),torch.tensor(math.log(7),dtype=a.dtype),atol=1e-12,rtol=1e-12)
    x=torch.tensor([[1.,-2.]],dtype=a.dtype,requires_grad=True)
    y=torch.tensor([[2.,3.]],dtype=a.dtype,requires_grad=True)
    loss=impl.nt_xent_loss(x,y,.01);close(loss,torch.tensor(0.,dtype=a.dtype),atol=1e-12,rtol=0)
    dx,dy=torch.autograd.grad(loss,(x,y));close(dx,torch.zeros_like(x),atol=1e-12,rtol=0);close(dy,torch.zeros_like(y),atol=1e-12,rtol=0)


@pytest.mark.t2
def test_asymmetric_scalar_oracle_across_temperatures(impl):
    """Independent scalar denominator contains the positive and every other non-self example."""
    a,b=fixture()
    for count in (2,3,6):
        for temperature in (.03,.2,1.,4.):
            expected=scalar_loss(a[:count].tolist(),b[:count].tolist(),temperature)
            close(impl.nt_xent_loss(a[:count],b[:count],temperature),torch.tensor(expected,dtype=a.dtype),atol=1e-11,rtol=1e-11)


@pytest.mark.t2
def test_gradients_of_both_views_match_independent_finite_differences(impl):
    """Both branches and appearances as candidates must receive the complete derivative."""
    a,b=fixture();a=a[:3,:3].clone().requires_grad_();b=b[:3,:3].clone().requires_grad_()
    da,db=torch.autograd.grad(impl.nt_xent_loss(a,b,.37),(a,b))
    eps=1e-6;aa,bb=a.detach().tolist(),b.detach().tolist()
    for rows,other,grad,first in [(aa,bb,da,True),(bb,aa,db,False)]:
        numeric=[]
        for i,row in enumerate(rows):
            values=[]
            for j,old in enumerate(row):
                row[j]=old+eps;plus=scalar_loss(rows,other,.37) if first else scalar_loss(other,rows,.37)
                row[j]=old-eps;minus=scalar_loss(rows,other,.37) if first else scalar_loss(other,rows,.37)
                row[j]=old;values.append((plus-minus)/(2*eps))
            numeric.append(values)
        close(grad,torch.tensor(numeric,dtype=a.dtype),atol=2e-8,rtol=2e-6)


@pytest.mark.t2
def test_paired_permutation_view_swap_and_orthogonal_invariance(impl):
    """Changing instance order, view order or a common orthogonal feature basis preserves loss."""
    a,b=fixture();base=impl.nt_xent_loss(a,b,.4);order=torch.tensor([3,0,5,1,4,2])
    close(impl.nt_xent_loss(a[order],b[order],.4),base,atol=1e-12,rtol=1e-12)
    close(impl.nt_xent_loss(b,a,.4),base,atol=1e-12,rtol=1e-12)
    feature_order=torch.tensor([7,2,0,4,1,6,3,5]);signs=torch.tensor([1.,-1.,1.,-1.,1.,1.,-1.,1.],dtype=a.dtype)
    close(impl.nt_xent_loss(a[:,feature_order]*signs,b[:,feature_order]*signs,.4),base,atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_per_row_positive_scale_invariance(impl):
    """Cosine normalization removes independent positive row scales away from the epsilon clamp."""
    a,b=fixture();scales=torch.tensor([[.01],[3.],[1e3],[.4],[20.],[2.]],dtype=a.dtype)
    close(impl.nt_xent_loss(a*scales,b/scales,.3),impl.nt_xent_loss(a,b,.3),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_zero_near_zero_rows_and_stable_extreme_logits(impl):
    """Explicit epsilon convention includes zero rows; stable log-sum-exp handles small temperatures."""
    a=torch.tensor([[0.,0.],[1e-14,-2e-14],[3.,-4.]],dtype=torch.float64)
    b=torch.tensor([[2.,1.],[-2e-14,1e-14],[1.,3.]],dtype=a.dtype)
    close(impl.nt_xent_loss(a,b,.2),torch.tensor(scalar_loss(a.tolist(),b.tolist(),.2),dtype=a.dtype),atol=1e-12,rtol=1e-12)
    a,b=fixture();a=a*1e10;b=b*1e10
    for temperature in (1e-4,1000.):
        actual=impl.nt_xent_loss(a,b,temperature)
        assert bool(torch.isfinite(actual))
        close(actual,torch.tensor(scalar_loss(a.tolist(),b.tolist(),temperature),dtype=a.dtype),atol=1e-9,rtol=1e-11)


@pytest.mark.t3
def test_actual_paired_table_training_separates_instances(impl,artifact):
    """120 actual Adam updates learn 6 paired identities using 96 trainable scalars."""
    result=run(impl.nt_xent_loss);artifact('simclr_training',result);m=result['metrics']
    assert m['updates']==120 and m['trainable_parameters']==96 and m['pairs']==6
    assert m['maximum_scalar_error']<1e-11
    assert m['final_training_accuracy']==1 and m['final_training_accuracy']>m['initial_training_accuracy']
    assert m['final_loss']<.15*m['initial_loss'] and m['final_minimum_margin']>.5
