import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from ..toy import fixture, scalar_assign, scalar_ce
from ..experiment import run


@pytest.mark.t0
def test_shapes_dtypes_strides_and_no_mutation(impl):
    """Whitened rows retain their layout, and hard cluster indices are int64 and detached."""
    for dtype in (torch.float32, torch.float64):
        x=fixture(dtype)[:, ::1]; old=x.clone()
        parent=torch.stack([x,torch.ones_like(x)],dim=2).reshape(9,4)
        z=impl.normalize_features(parent[:,::2])
        assert z.shape==x.shape and z.dtype==dtype and z.device==x.device
        ids=impl.assign_clusters(z,z[[0,3,6]])
        assert ids.dtype==torch.int64 and ids.shape==(9,) and not ids.requires_grad
        close(x,old)
        close(z.norm(dim=1),torch.ones(9,dtype=dtype),atol=2e-6,rtol=2e-6)


@pytest.mark.t0
def test_invalid_inputs(impl):
    """Every declared axis must be nonempty; labels must index the supplied cluster columns."""
    x=fixture()
    for bad in (x[:0],x.flatten(),x.long(),x*float('nan')):
        with pytest.raises(ValueError): impl.normalize_features(bad)
    with pytest.raises(ValueError): impl.normalize_features(x,0.)
    with pytest.raises(ValueError): impl.assign_clusters(x,torch.ones(2,3,dtype=x.dtype))
    for y in (torch.tensor([0.,1.]),torch.tensor([0,3]),torch.tensor([-1,0])):
        with pytest.raises(ValueError): impl.pseudo_label_loss(torch.ones(2,3,dtype=x.dtype),y)


@pytest.mark.t1
def test_zero_rows_and_hand_calculated_cases(impl):
    """Row normalization fixes zeros; Euclidean ties select the first center; uniform CE is log K."""
    x=torch.tensor([[3.,4.],[0.,0.]],dtype=torch.float64)
    close(impl.normalize_features(x),torch.tensor([[.6,.8],[0.,0.]],dtype=x.dtype),atol=1e-12,rtol=1e-12)
    centers=torch.tensor([[-1.,0.],[1.,0.]],dtype=x.dtype)
    assert impl.assign_clusters(x,centers).tolist()==[1,0]
    close(impl.pseudo_label_loss(torch.zeros(3,4,dtype=x.dtype),torch.tensor([3,1,0])),torch.tensor(math.log(4),dtype=x.dtype))


@pytest.mark.t2
def test_euclidean_centers_and_scalar_oracles(impl,ref):
    """K-means centers are arithmetic means, not forced unit vectors or cosine prototypes."""
    x=torch.tensor([[1.,0.],[.6,.8],[-1.,0.]],dtype=torch.float64)
    c=torch.tensor([[.9,.1],[3.,0.],[-.5,.2]],dtype=x.dtype)
    actual=impl.assign_clusters(x,c)
    assert actual.tolist()==scalar_assign(x.tolist(),c.tolist())
    assert actual.tolist()==ref.assign_clusters(x,c).tolist()
    logits=torch.tensor([[1.2,-.3,.4],[-.1,2.,.8],[3.,1.,-.9]],dtype=x.dtype)
    y=torch.tensor([2,0,1])
    close(impl.pseudo_label_loss(logits,y),torch.tensor(scalar_ce(logits.tolist(),y.tolist()),dtype=x.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_normalization_analytic_and_strided_gradient(impl):
    """For nonzero rows, normalization has VJP (g-z(z dot g))/||x||; unused storage gets zero."""
    parent=torch.tensor([[3.,7.,4.,8.],[5.,9.,-12.,1.]],dtype=torch.float64,requires_grad=True)
    x=parent[:,::2]
    g=torch.tensor([[.7,-.2],[.3,.8]],dtype=parent.dtype)
    z=impl.normalize_features(x)
    assert z.requires_grad
    grad=torch.autograd.grad((z*g).sum(),parent)[0]
    rows=[]
    for row,weight in zip(x.detach().tolist(),g.tolist()):
        r=math.sqrt(sum(a*a for a in row));unit=[a/r for a in row]
        dot=sum(a*b for a,b in zip(unit,weight))
        rows.append([(weight[0]-unit[0]*dot)/r,0.,(weight[1]-unit[1]*dot)/r,0.])
    close(grad,torch.tensor(rows,dtype=parent.dtype),atol=1e-12,rtol=1e-10)
    zero=torch.zeros(1,2,dtype=parent.dtype,requires_grad=True)
    close(torch.autograd.grad(impl.normalize_features(zero,.2).sum(),zero)[0],torch.full((1,2),5.,dtype=parent.dtype))
    tiny_parent=torch.tensor([[.03,9.,.04,8.,0.,7.],[0.,6.,0.,5.,.02,4.]],dtype=parent.dtype,requires_grad=True)
    tiny=tiny_parent[:,::2];weight=torch.tensor([[.2,-.5,.7],[.8,.3,-.2]],dtype=parent.dtype)
    normalized=impl.normalize_features(tiny,.1)
    expected=[]
    for row in tiny.detach().tolist():
        denominator=max(math.sqrt(sum(x*x for x in row)),.1)
        expected.append([x/denominator for x in row])
    close(normalized,torch.tensor(expected,dtype=parent.dtype),atol=1e-12,rtol=1e-12)
    gp=torch.autograd.grad((normalized*weight).sum(),tiny_parent)[0]
    want=torch.zeros_like(tiny_parent);want[:,::2]=weight/.1
    close(gp,want,atol=1e-12,rtol=1e-12)



@pytest.mark.t2
def test_pseudo_label_gradient_and_large_scores(impl):
    """Hard labels have no gradient; each logit receives (softmax-onehot)/B even at large offsets."""
    x=torch.tensor([[10000.,9999.,9998.],[-10000.,-9999.,-10002.]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([2,0]);loss=impl.pseudo_label_loss(x,y)
    expected=[]
    for row,target in zip(x.detach().tolist(),y.tolist()):
        weights=[math.exp(a-max(row)) for a in row]
        expected.append([(v/sum(weights)-(j==target))/2 for j,v in enumerate(weights)])
    close(torch.autograd.grad(loss,x)[0],torch.tensor(expected,dtype=x.dtype),atol=1e-11,rtol=1e-10)
    close(loss,torch.tensor(scalar_ce(x.detach().tolist(),y.tolist()),dtype=x.dtype),atol=1e-11,rtol=1e-12)


@pytest.mark.t2
def test_cluster_relabeling_and_row_scale(impl):
    """Permutation of centers relabels assignments, while positive row scaling vanishes before clustering."""
    x=fixture();z=impl.normalize_features(x);c=z[[0,3,6]]
    ids=impl.assign_clusters(z,c);p=[2,0,1]
    assert impl.assign_clusters(z,c[p]).tolist()==[p.index(v) for v in ids.tolist()]
    close(impl.normalize_features(x*torch.arange(1,10,dtype=x.dtype)[:,None]),z,atol=1e-12,rtol=1e-12)
    logits=z@c.T
    close(impl.pseudo_label_loss(logits[:,p],torch.tensor([p.index(v) for v in ids.tolist()])),impl.pseudo_label_loss(logits,ids))


@pytest.mark.t2
def test_batch_duplication_and_row_offsets(impl):
    """Mean CE is invariant to batch duplication and independent row offsets."""
    x=torch.tensor([[1.,2.,-1.],[.3,-.2,2.]],dtype=torch.float64);y=torch.tensor([0,2])
    loss=impl.pseudo_label_loss(x,y)
    close(impl.pseudo_label_loss(torch.cat([x,x]),torch.cat([y,y])),loss,atol=1e-12,rtol=1e-12)
    close(impl.pseudo_label_loss(x+torch.tensor([[50.],[-90.]],dtype=x.dtype),y),loss,atol=1e-12,rtol=1e-12)


@pytest.mark.t3
def test_alternating_toy_encoder_training(impl,artifact):
    """Actual alternating cluster refresh and 90 encoder/head updates must learn all three pseudo-classes."""
    result=run(impl.normalize_features,impl.assign_clusters,impl.pseudo_label_loss);artifact('deepcluster_training',result)
    m=result['metrics']
    assert m['updates']==90 and m['cluster_refreshes']==4 and m['trainable_parameters']==10
    assert m['occupied_clusters']==3 and m['encoder_motion']>.01
    assert m['final_accuracy']==1. and m['final_accuracy']>m['initial_accuracy']
    assert m['final_loss']<.1 and m['final_loss']<.1*m['initial_loss']
