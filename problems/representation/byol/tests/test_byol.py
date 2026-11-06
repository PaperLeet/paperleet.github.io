import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from ..toy import fixture, scalar_loss
from ..experiment import run


@pytest.mark.t0
def test_shape_dtype_strides_and_input_preservation(impl):
    """Paired regression preserves float dtype, device and strided inputs; EMA accepts parameter shapes."""
    for dtype in (torch.float32,torch.float64):
        values=tuple(v[:,::2] for v in fixture(dtype)); before=tuple(v.clone() for v in values)
        loss=impl.byol_loss(*values)
        assert loss.shape==() and loss.dtype==dtype and loss.device==values[0].device
        close(loss,torch.tensor(scalar_loss(*(v.tolist() for v in values)),dtype=dtype),atol=2e-6,rtol=2e-6)
        for x,y in zip(values,before):close(x,y)
        a,b=values[0].reshape(2,3,2),values[1].reshape(2,3,2)
        close(impl.ema_update(a,b,.7),.7*b+.3*a)


@pytest.mark.t0
def test_validation_contract(impl):
    """Reject mismatched/empty/nonfinite representations and invalid scalar controls."""
    a,b,c,d=fixture()
    for x in (a[:0],a[:,:0],a.flatten(),a.float(),a.long(),a*float('nan')):
        with pytest.raises(ValueError):impl.byol_loss(x,b,c,d)
    for eps in (0,-1,True,float('inf'),torch.tensor(.1)):
        with pytest.raises(ValueError):impl.byol_loss(a,b,c,d,eps)
    for m in (-.1,1.1,True,float('nan'),torch.tensor(.5)):
        with pytest.raises(ValueError):impl.ema_update(a,b,m)
    with pytest.raises(ValueError):impl.ema_update(a,b[:1])


@pytest.mark.t1
def test_direction_sum_crossing_and_squared_normalization(impl):
    """Official BYOL sums both directional squared distances; opposite unit vectors give total eight."""
    a,b,c,d=fixture()
    close(impl.byol_loss(a,a,a,a),torch.tensor(0.,dtype=a.dtype),atol=1e-12,rtol=0)
    close(impl.byol_loss(a,a,-a,-a),torch.tensor(8.,dtype=a.dtype),atol=1e-12,rtol=0)
    close(impl.byol_loss(a,b,c,d),torch.tensor(scalar_loss(*(v.tolist() for v in (a,b,c,d))),dtype=a.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_independent_prediction_finite_differences_and_stopped_targets(impl):
    """The two online predictions are differentiable while both target projections are stopped."""
    values=tuple(v[:2,:2].clone().requires_grad_() for v in fixture())
    loss=impl.byol_loss(*values);assert loss.requires_grad
    gradients=torch.autograd.grad(loss,values,allow_unused=True)
    assert gradients[0] is not None and gradients[1] is not None
    assert gradients[2] is None and gradients[3] is None
    rows=[v.detach().tolist() for v in values]
    for branch in (0,1):
        numeric=[]
        for row in rows[branch]:
            out=[]
            for j,old in enumerate(row):
                row[j]=old+1e-6;plus=scalar_loss(*rows)
                row[j]=old-1e-6;minus=scalar_loss(*rows)
                row[j]=old;out.append((plus-minus)/2e-6)
            numeric.append(out)
        close(gradients[branch],torch.tensor(numeric,dtype=values[0].dtype),atol=3e-8,rtol=2e-6)
    parents=tuple(torch.stack((v.detach(),torch.zeros_like(v)),dim=-1).reshape(2,4).requires_grad_() for v in values[:2])
    strided_loss=impl.byol_loss(parents[0][:,::2],parents[1][:,::2],values[2].detach(),values[3].detach())
    assert strided_loss.requires_grad, 'strided predictions must retain their gradient paths'
    parent_gradients=torch.autograd.grad(strided_loss,parents)
    for actual,expected in zip(parent_gradients,gradients[:2]):
        close(actual[:,::2],expected,atol=1e-12,rtol=1e-11)
        close(actual[:,1::2],torch.zeros_like(expected),atol=0,rtol=0)


@pytest.mark.t2
def test_invariances_and_zero_floor(impl):
    """Paired permutation and view swapping preserve the sum; the official floor applies to squared norms."""
    a,b,c,d=fixture();base=impl.byol_loss(a,b,c,d);order=torch.tensor([2,5,0,1,4,3])
    close(impl.byol_loss(b,a,d,c),base,atol=1e-12,rtol=1e-12)
    close(impl.byol_loss(*(v[order] for v in (a,b,c,d))),base,atol=1e-12,rtol=1e-12)
    scales=torch.tensor([[.1],[3.],[.4],[8.],[.2],[5.]],dtype=a.dtype)
    close(impl.byol_loss(a*scales,b/scales,c/scales,d*scales),base,atol=1e-12,rtol=1e-12)
    x=torch.tensor([[0.,0.],[1e-8,-2e-8]],dtype=a.dtype,requires_grad=True)
    y=torch.tensor([[1.,0.],[3e-8,1e-8]],dtype=a.dtype)
    loss=impl.byol_loss(x,y,y,x)
    close(loss,torch.tensor(scalar_loss(*(v.detach().tolist() for v in (x,y,y,x))),dtype=a.dtype),atol=1e-12,rtol=1e-12)
    assert bool(torch.isfinite(torch.autograd.grad(loss,x)[0]).all())


@pytest.mark.t2
def test_ema_endpoints_direction_detachment_and_independent_storage(impl):
    """EMA weights the old target by momentum and never exposes gradients or aliases, even at endpoints."""
    a,b,_,_=fixture();a.requires_grad_();b.requires_grad_()
    for m in (0.,.2,.9,1.):
        out=impl.ema_update(a,b,m)
        close(out,m*b.detach()+(1-m)*a.detach(),atol=1e-12,rtol=1e-12)
        assert not out.requires_grad
        before_a,before_b=a.detach().clone(),b.detach().clone()
        out[0,0]+=1
        close(a,before_a);close(b,before_b)


@pytest.mark.t3
def test_actual_prediction_updates_with_moving_ema_targets(impl,artifact):
    """Eighty actual SGD-and-EMA updates reduce bootstrapped regression with detached moving targets."""
    result=run(impl.byol_loss,impl.ema_update);artifact('byol_training',result);m=result['metrics']
    assert m['updates']==80 and m['online_parameters']==45
    assert m['scalar_error']<1e-11 and m['ema_error']<1e-12
    assert m['targets_detached'] and m['target_movement']>.1
    assert m['final_loss']<.05 and m['final_loss']<.05*m['initial_loss']
