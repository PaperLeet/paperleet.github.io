import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.retrieval.bpr.toy import scalar_loss
from problems.retrieval.bpr.experiment import run


@pytest.mark.t0
def test_mask_contract_empty_axes_order_and_validation(impl):
    """Repeated users retain input order, empty axes work, and input history stays unchanged."""
    for users,items in ((3,5),(0,5),(3,0)):
        history = torch.zeros(users,items,dtype=torch.bool)
        ids = torch.tensor([2,0,2] if users else [],dtype=torch.long)
        before = history.clone();result = impl.eligible_negatives(history,ids)
        assert result.shape==(len(ids),items) and result.dtype==torch.bool and result.device==history.device
        close(history,before)
    for history,ids in ((torch.zeros(2,3),torch.tensor([0])),(torch.zeros(2,3,dtype=torch.bool),torch.tensor([2])),
                        (torch.zeros(2,3,dtype=torch.bool),torch.tensor([-1])),(torch.zeros(2,3,dtype=torch.bool),torch.tensor([[0]]))):
        with pytest.raises(ValueError):impl.eligible_negatives(history,ids)


@pytest.mark.t0
def test_loss_contract_precision_strides_and_supplied_validation(impl):
    """The scalar loss preserves both precisions, strided tables and input bytes."""
    for dtype in (torch.float32,torch.float64):
        users,items=torch.randn(3,6,dtype=dtype)[:,::2],torch.randn(5,6,dtype=dtype)[:,::2]
        before=users.clone(),items.clone();triples=torch.tensor([[0,1,2],[0,1,3],[2,4,0]])
        result=impl.bpr_mf_loss(users,items,triples,.02)
        assert result.shape==() and result.dtype==dtype and result.device==users.device
        close(users,before[0]);close(items,before[1])
    users,items=torch.zeros(2,3),torch.zeros(4,3)
    for triples in (torch.zeros(1,3),torch.tensor([[2,1,0]]),torch.tensor([[0,1,4]]),torch.tensor([[0,1,1]]),torch.tensor([[0,1]])):
        with pytest.raises(ValueError):impl.bpr_mf_loss(users,items,triples)
    for l2 in (-1.,math.inf,math.nan):
        with pytest.raises(ValueError):impl.bpr_mf_loss(users,items,torch.tensor([[0,1,2]]),l2)
    with pytest.raises(ValueError):impl.bpr_mf_loss(users,torch.zeros(4,2),torch.tensor([[0,1,2]]))
    with pytest.raises(ValueError):impl.bpr_mf_loss(users,items.double(),torch.tensor([[0,1,2]]))


@pytest.mark.t1
def test_hand_computed_history_and_global_prior(impl):
    """Exclude all observed items; use a signed dot-product difference and both full table norms."""
    seen=torch.tensor([[True,False,True,False],[False,True,False,True]])
    close(impl.eligible_negatives(seen,torch.tensor([1,0,1])),torch.tensor([[True,False,True,False],[False,True,False,True],[True,False,True,False]]))
    users=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
    items=torch.tensor([[2.,0.],[0.,1.],[1.,1.]],dtype=users.dtype)
    # user0: positive item2 score3, negative item0 score2; norm total=30+7=37.
    wanted=math.log1p(math.exp(-1))+.1*37
    close(impl.bpr_mf_loss(users,items,torch.tensor([[0,2,0]]),.1),torch.tensor(wanted,dtype=users.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_independent_scalar_history_for_all_users_and_item_columns(impl):
    """Every column follows its own user's history, including all-seen and unseen users."""
    rows=[[True,False,True,False],[False,True,False,True],[True]*4,[False]*4]
    ids=[3,1,0,3,2]
    expected=[[not rows[u][i] for i in range(4)] for u in ids]
    close(impl.eligible_negatives(torch.tensor(rows),torch.tensor(ids)),torch.tensor(expected))


@pytest.mark.t2
def test_scalar_objective_and_duplicate_index_gradients_by_finite_differences(impl):
    """Independent scalar likelihoods check both factor-table gradients and repeated row accumulation."""
    users=torch.randn(3,3,dtype=torch.float64,requires_grad=True)
    items=torch.randn(5,3,dtype=torch.float64,requires_grad=True)
    triples=torch.tensor([[0,1,2],[0,1,3],[1,3,1],[1,3,1]])
    pairs=triples.tolist();us,vs=users.detach().tolist(),items.detach().tolist()
    loss=impl.bpr_mf_loss(users,items,triples,.03)
    close(loss,torch.tensor(scalar_loss(us,vs,pairs,.03),dtype=users.dtype),atol=1e-12,rtol=1e-12)
    loss.backward()
    assert users.grad is not None and items.grad is not None,'Both factor tables require gradients'
    for values,gradient in ((us,users.grad),(vs,items.grad)):
        expected=torch.zeros_like(gradient)
        for row,features in enumerate(values):
            for d,original in enumerate(features):
                features[d]=original+1e-5;above=scalar_loss(us,vs,pairs,.03)
                features[d]=original-1e-5;below=scalar_loss(us,vs,pairs,.03)
                features[d]=original;expected[row,d]=(above-below)/2e-5
        close(gradient,expected,atol=2e-8,rtol=2e-6)


@pytest.mark.t2
def test_global_prior_changes_unselected_rows_once_not_per_occurrence(impl):
    """All table rows receive exactly 2*l2*parameter, including absent users/items and duplicates."""
    users=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
    items=torch.randn(4,2,dtype=torch.float64,requires_grad=True)
    triples=torch.tensor([[0,1,2],[0,1,2],[0,1,2]])
    regular=impl.bpr_mf_loss(users,items,triples,.07)
    base=impl.bpr_mf_loss(users,items,triples,0.)
    gu,gi=torch.autograd.grad(regular-base,(users,items))
    close(gu,.14*users,atol=1e-12,rtol=1e-12);close(gi,.14*items,atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_mean_batch_replication_and_unregularized_item_translation(impl):
    """Replicating triples leaves the mean objective unchanged; a common item shift cancels in score differences."""
    users,items=torch.randn(3,3,dtype=torch.float64),torch.randn(5,3,dtype=torch.float64)
    triples=torch.tensor([[0,1,2],[1,2,4],[2,4,0]])
    for l2 in (0.,.02):
        close(impl.bpr_mf_loss(users,items,torch.cat([triples,triples]),l2),impl.bpr_mf_loss(users,items,triples,l2),atol=1e-12,rtol=1e-12)
    close(impl.bpr_mf_loss(users,items+torch.tensor([.3,-.7,.2],dtype=users.dtype),triples),impl.bpr_mf_loss(users,items,triples),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_extreme_signed_margins_and_both_item_directions_remain_stable(impl):
    """Margins +/-2000 are finite; positive/negative item gradients have opposing analytically known signs."""
    users=torch.ones(1,1,dtype=torch.float64,requires_grad=True)
    items=torch.tensor([[1000.],[-1000.]],dtype=users.dtype,requires_grad=True)
    triples=torch.tensor([[0,0,1],[0,1,0]])
    loss=impl.bpr_mf_loss(users,items,triples)
    close(loss,torch.tensor(1000.,dtype=users.dtype),atol=1e-12,rtol=1e-12)
    loss.backward();assert users.grad is not None and items.grad is not None
    close(users.grad,torch.tensor([[1000.]],dtype=users.dtype));close(items.grad,torch.tensor([[.5],[-.5]],dtype=users.dtype))


@pytest.mark.t2
def test_empty_triples_keep_the_global_prior_and_both_zero_gradient_graphs(impl):
    """No likelihood observations means zero data loss, with the full prior retained and both graphs intact."""
    for sizes in ((2,3),(0,0)):
        for l2 in (0.,.1):
            users=torch.ones(sizes[0],2,dtype=torch.float64,requires_grad=True)
            items=torch.ones(sizes[1],2,dtype=users.dtype,requires_grad=True)
            loss=impl.bpr_mf_loss(users,items,torch.empty(0,3,dtype=torch.long),l2)
            close(loss,torch.tensor(2*sum(sizes)*l2,dtype=users.dtype));loss.backward()
            assert users.grad is not None and items.grad is not None
            close(users.grad,2*l2*users);close(items.grad,2*l2*items)


@pytest.mark.t3
def test_actual_factor_training_exceeds_any_item_only_score_order(impl,artifact):
    """120 real updates learn all 24 conflicting user preferences; any global item order is bounded at 1/2."""
    result=run(impl.eligible_negatives,impl.bpr_mf_loss);artifact('bpr_mf_training',result);m=result['metrics']
    assert m['updates']==120 and m['trainable_parameters']==27 and m['triples']==24 and m['eligibility_errors']==0
    assert m['item_only_accuracy_upper_bound']==.5
    assert m['final_pair_accuracy']==1 and m['final_training_recall_at_2']==1
    assert m['final_pair_accuracy']>m['initial_pair_accuracy']
    assert m['final_loss']<.3*m['initial_loss'] and m['minimum_final_margin']>1.5
    assert m['maximum_scalar_error']<1e-12
