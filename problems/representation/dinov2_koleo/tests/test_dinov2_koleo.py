import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from ..toy import fixture, scalar_koleo
from ..experiment import run


@pytest.mark.t0
def test_shapes_dtypes_invalid_inputs_and_no_mutation(impl):
    """KoLeo needs at least two finite vectors and returns one floating scalar and int64 neighbors."""
    for dtype in (torch.float32,torch.float64):
        x=fixture().to(dtype);old=x.clone();loss=impl.koleo_loss(x)
        assert loss.shape==() and loss.dtype==dtype
        z=x/x.norm(dim=1,keepdim=True);ids=impl.nearest_neighbors(z)
        assert ids.shape==(4,) and ids.dtype==torch.int64 and not ids.requires_grad
        close(x,old,atol=0,rtol=0)
    for bad in (x[:1],x[:,:0],x.long(),x*float('nan')):
        with pytest.raises(ValueError):impl.koleo_loss(bad)
    with pytest.raises(ValueError):impl.koleo_loss(x,0.)


@pytest.mark.t1
def test_nonself_and_deterministic_ties(impl):
    """The nearest neighbor is never the anchor, even for duplicate or antipodal vectors."""
    cases=[([[1.,0.],[-1.,0.]],[1,0]),([[1.,0.],[1.,0.],[0.,1.]],[1,0,0]),([[0.,0.],[0.,0.]],[1,0])]
    for rows,expected in cases:
        assert impl.nearest_neighbors(torch.tensor(rows,dtype=torch.float64)).tolist()==expected


@pytest.mark.t2
def test_scalar_oracle_and_official_epsilon_placements(impl,ref):
    """DINOv2 uses normalize, nearest-neighbor PairwiseDistance(eps=1e-8), then log(distance+eps)."""
    x=fixture()
    for eps in (1e-8,.03,.2):
        got=impl.koleo_loss(x,eps)
        close(got,torch.tensor(scalar_koleo(x.tolist(),eps),dtype=x.dtype),atol=1e-12,rtol=1e-12)
        close(got,ref.koleo_loss(x,eps),atol=1e-12,rtol=1e-12)
    zeros=torch.zeros(2,3,dtype=x.dtype)
    close(impl.koleo_loss(zeros,.1),torch.tensor(-math.log(1e-8*math.sqrt(3)+.1),dtype=x.dtype),atol=1e-12,rtol=1e-12)
    tiny=torch.tensor([[.02,0.],[0.,.03],[-.04,.01]],dtype=x.dtype)
    close(impl.koleo_loss(tiny,.1),torch.tensor(scalar_koleo(tiny.tolist(),.1),dtype=x.dtype),atol=1e-12,rtol=1e-12)



@pytest.mark.t2
def test_both_endpoint_gradients_by_finite_differences(impl):
    """The discrete neighbor choice is locally fixed, but both gathered endpoint vectors receive gradients."""
    x=fixture().requires_grad_();eps=.03
    gradient=torch.autograd.grad(impl.koleo_loss(x,eps),x)[0]
    rows=x.detach().tolist();expected=[]
    for i,row in enumerate(rows):
        result=[]
        for j,old in enumerate(row):
            row[j]=old+1e-6;plus=scalar_koleo(rows,eps)
            row[j]=old-1e-6;minus=scalar_koleo(rows,eps)
            row[j]=old;result.append((plus-minus)/2e-6)
        expected.append(result)
    close(gradient,torch.tensor(expected,dtype=x.dtype),atol=2e-9,rtol=2e-7)
    parent=torch.stack([x.detach(),torch.zeros_like(x)],dim=2).reshape(4,6).requires_grad_()
    gp=torch.autograd.grad(impl.koleo_loss(parent[:,::2],eps),parent)[0]
    close(gp[:,::2],gradient,atol=1e-12,rtol=1e-11);close(gp[:,1::2],torch.zeros_like(gradient),atol=0,rtol=0)
    # Unique neighbors below the norm floor test the linear normalization branch.
    tiny=torch.tensor([[.02,0.],[0.,.03],[-.04,.01]],dtype=x.dtype,requires_grad=True)
    actual=torch.autograd.grad(impl.koleo_loss(tiny,.1),tiny)[0]
    rows=tiny.detach().tolist();numeric=[]
    for row in rows:
        values=[]
        for j,old in enumerate(row):
            row[j]=old+1e-7;plus=scalar_koleo(rows,.1)
            row[j]=old-1e-7;minus=scalar_koleo(rows,.1)
            row[j]=old;values.append((plus-minus)/2e-7)
        numeric.append(values)
    close(actual,torch.tensor(numeric,dtype=x.dtype),atol=1e-8,rtol=2e-7)



@pytest.mark.t2
def test_positive_scale_and_batch_permutation_invariance(impl):
    """L2 normalization removes positive per-row scales; unique-neighbor batch relabeling changes no distances."""
    x=fixture();loss=impl.koleo_loss(x,.03)
    close(impl.koleo_loss(x*torch.tensor([[2.],[.4],[3.],[.7]],dtype=x.dtype),.03),loss,atol=1e-12,rtol=1e-12)
    close(impl.koleo_loss(x[[2,0,3,1]],.03),loss,atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_neighbor_cosine_matches_euclidean_unit_distance(impl):
    """For unit vectors, maximizing cosine minimizes squared Euclidean distance."""
    x=fixture();unit=x/x.norm(dim=1,keepdim=True);rows=unit.tolist()
    expected=[min([j for j in range(4) if j!=i],key=lambda j:sum((a-b)**2 for a,b in zip(row,rows[j]))) for i,row in enumerate(rows)]
    assert impl.nearest_neighbors(unit).tolist()==expected
    assert all(i!=j for i,j in enumerate(expected))


@pytest.mark.t2
def test_duplicate_vectors_finite_and_spread_preference(impl):
    """The epsilon convention keeps coincident rows finite; separated unit points have lower KoLeo."""
    collapsed=torch.tensor([[1.,0.],[1.,0.],[1.,0.]],dtype=torch.float64,requires_grad=True)
    loss=impl.koleo_loss(collapsed,.03);assert bool(torch.isfinite(loss))
    assert bool(torch.isfinite(torch.autograd.grad(loss,collapsed)[0]).all())
    spread=torch.tensor([[1.,0.],[-.5,math.sqrt(3)/2],[-.5,-math.sqrt(3)/2]],dtype=collapsed.dtype)
    assert float(impl.koleo_loss(spread,.03))<float(loss)-2.


@pytest.mark.t3
def test_actual_sphere_spreading_optimization(impl,artifact):
    """One hundred SGD steps on six submitted-loss vectors must increase measured nearest-neighbor spacing."""
    result=run(impl.koleo_loss);artifact('koleo_training',result);m=result['metrics']
    assert m['updates']==100 and m['trainable_parameters']==18
    assert m['final_loss']<0. and m['final_loss']<m['initial_loss']-1.
    assert m['final_nearest_distance']>1. and m['final_nearest_distance']>5*m['initial_nearest_distance']
