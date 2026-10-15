import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.retrieval.sasrec.toy import scalar_mask, scalar_loss, data, encode
from problems.retrieval.sasrec.experiment import run


@pytest.mark.t0
def test_contract_empty_strided_precision_and_validation(impl):
    """Equal [B,T] logits/IDs, Boolean visibility, empty axes and no input mutation."""
    for b, t in ((2, 4), (0, 4), (2, 0)):
        ids = torch.ones(b, t*2, dtype=torch.long)[:, ::2]
        mask = impl.causal_item_mask(ids)
        assert mask.shape == (b, t, t) and mask.dtype == torch.bool and mask.device == ids.device
        for dtype in (torch.float32, torch.float64):
            p, n = torch.randn(b, t*2, dtype=dtype)[:, ::2], torch.randn(b, t*2, dtype=dtype)[:, ::2]
            saved = p.clone(), n.clone(), ids.clone()
            loss = impl.sampled_next_item_loss(p, n, ids)
            assert loss.shape == () and loss.dtype == dtype and loss.device == p.device
            close(p, saved[0]); close(n, saved[1]); close(ids, saved[2])
    for ids in (torch.zeros(2), torch.zeros(2, 3), torch.tensor([[-1, 2]])):
        with pytest.raises(ValueError): impl.causal_item_mask(ids)
    with pytest.raises(ValueError): impl.sampled_next_item_loss(torch.zeros(2, 3), torch.zeros(2, 4), torch.ones(2, 3, dtype=torch.long))
    with pytest.raises(ValueError): impl.sampled_next_item_loss(torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.float64), torch.ones(2, 3, dtype=torch.long))


@pytest.mark.t1
def test_hand_calculated_visibility_and_two_binary_terms(impl):
    """Padding queries/keys are hidden; every real item sees itself; valid loss sums both binary terms."""
    ids = torch.tensor([[0, 4, 2]])
    close(impl.causal_item_mask(ids), torch.tensor([[[False]*3, [False, True, False], [False, True, True]]]))
    p, n = torch.tensor([[100., 0., math.log(3)]], dtype=torch.float64), torch.tensor([[-100., 0., -math.log(3)]], dtype=torch.float64)
    wanted = math.log(2) + math.log(4/3)
    close(impl.sampled_next_item_loss(p, n, ids), torch.tensor(wanted, dtype=p.dtype), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_independent_loop_mask_oracle_padding_patterns_and_ids(impl):
    """Python query/key loops check left/right/interior padding, duplicates and all-zero rows."""
    ids = torch.tensor([[0, 0, 2, 3], [1, 2, 0, 0], [1, 0, 3, 0], [4, 4, 4, 4], [0, 0, 0, 0]])
    close(impl.causal_item_mask(ids), torch.tensor(scalar_mask(ids.tolist())))
    close(impl.causal_item_mask(ids*7), impl.causal_item_mask(ids))


@pytest.mark.t2
def test_independent_scalar_sampled_objective_in_both_precisions(impl):
    """The objective is two binary terms, not pairwise softmax, divided by nonpadding target count."""
    targets = torch.tensor([[0, 2, 3, 4], [5, 0, 6, 0]])
    for dtype in (torch.float32, torch.float64):
        p, n = torch.randn(2, 4, dtype=dtype)*3, torch.randn(2, 4, dtype=dtype)*3
        expected = scalar_loss(p.tolist(), n.tolist(), targets.tolist())
        close(impl.sampled_next_item_loss(p, n, targets), torch.tensor(expected, dtype=dtype))


@pytest.mark.t2
def test_both_logit_gradients_against_independent_finite_differences(impl):
    """Central scalar differences independently check both gradients and exact ignored zeros."""
    targets = torch.tensor([[0, 1, 2], [3, 0, 4]])
    p = torch.tensor([[.2, -.7, 1.1], [.4, 2., -.2]], dtype=torch.float64, requires_grad=True)
    n = torch.tensor([[-.6, .9, -.1], [1.4, -2., .3]], dtype=p.dtype, requires_grad=True)
    impl.sampled_next_item_loss(p, n, targets).backward()
    ps, ns = p.detach().tolist(), n.detach().tolist()
    for values, gradient in ((ps, p.grad), (ns, n.grad)):
        assert gradient is not None
        for row in range(2):
            for col in range(3):
                original = values[row][col]
                values[row][col] = original + 1e-6; above = scalar_loss(ps, ns, targets.tolist())
                values[row][col] = original - 1e-6; below = scalar_loss(ps, ns, targets.tolist())
                values[row][col] = original
                assert gradient[row, col].item() == pytest.approx((above-below)/2e-6, abs=3e-9, rel=3e-9)


@pytest.mark.t2
def test_large_logits_ignored_nonfinite_positions_and_finite_gradients(impl):
    """Stable logits avoid sigmoid/log overflow; ignored NaN/Inf do not contaminate values or gradients."""
    for dtype in (torch.float32, torch.float64):
        p = torch.tensor([[1000., -1000., math.nan]], dtype=dtype, requires_grad=True)
        n = torch.tensor([[-1000., 1000., math.inf]], dtype=dtype, requires_grad=True)
        loss = impl.sampled_next_item_loss(p, n, torch.tensor([[1, 2, 0]]))
        close(loss, torch.tensor(1000., dtype=dtype)); loss.backward()
        close(p.grad, torch.tensor([[0., -.5, 0.]], dtype=dtype)); close(n.grad, torch.tensor([[0., .5, 0.]], dtype=dtype))


@pytest.mark.t2
def test_all_padding_zero_loss_still_has_zero_gradients(impl):
    """An empty target set is a differentiable zero, not division by zero or a detached constant."""
    p, n = torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
    loss = impl.sampled_next_item_loss(p, n, torch.zeros(2, 3, dtype=torch.long))
    assert loss.item() == 0; loss.backward()
    assert p.grad is not None and n.grad is not None
    close(p.grad, torch.zeros_like(p)); close(n.grad, torch.zeros_like(n))


@pytest.mark.t2
def test_batch_permutation_padding_extension_and_objective_direction(impl):
    """Permuting examples or adding ignored positions preserves mean; better positives/negatives lower loss."""
    ids = torch.tensor([[0, 1, 2], [3, 4, 0]])
    p, n = torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 3, dtype=torch.float64)
    loss = impl.sampled_next_item_loss(p, n, ids)
    close(impl.sampled_next_item_loss(p[[1, 0]], n[[1, 0]], ids[[1, 0]]), loss)
    zero = torch.zeros(2, 2, dtype=p.dtype)
    close(impl.sampled_next_item_loss(torch.cat([p, zero], -1), torch.cat([n, zero], -1), torch.cat([ids, zero.long()], -1)), loss)
    assert impl.sampled_next_item_loss(p+1, n-1, ids).item() < loss.item()


@pytest.mark.t2
def test_provided_encoder_causality_and_future_embedding_gradients(impl):
    """Changing future IDs cannot change past states; future-only and padding embeddings receive no gradient."""
    _, _, initial = data()
    embeddings = initial.clone().requires_grad_()
    ids = torch.tensor([[0, 1, 2, 3]])
    original, weights = encode(ids, embeddings, impl.causal_item_mask)
    changed, _ = encode(torch.tensor([[0, 1, 2, 6]]), embeddings, impl.causal_item_mask)
    close(original[:, :3], changed[:, :3], atol=1e-12, rtol=1e-12)
    close(weights[:, 0], torch.zeros_like(weights[:, 0]))
    (original[0, 1]*torch.arange(1, 7, dtype=original.dtype)).sum().backward()
    close(embeddings.grad[0], torch.zeros_like(embeddings.grad[0])); close(embeddings.grad[3], torch.zeros_like(embeddings.grad[3]))
    assert embeddings.grad[1].abs().sum().item() > 1e-5


@pytest.mark.t3
def test_real_context_history_training_and_independent_objective_check(impl, artifact):
    """160 actual updates optimize synthetic next-item prediction, retaining causal and padding boundaries."""
    result = run(impl.causal_item_mask, impl.sampled_next_item_loss); artifact('sasrec_experiment', result)
    m = result['metrics']
    assert m['updates'] == 160 and m['trainable_parameters'] == 72
    assert m['final_loss'] < .05 and m['final_loss'] < m['initial_loss']*.1
    assert m['initial_training_accuracy'] == 0 and m['final_training_accuracy'] == 1
    assert m['last_item_only_upper_bound'] == pytest.approx(5/6)
    assert m['final_training_accuracy'] > m['last_item_only_upper_bound'] + .1
    assert m['maximum_scalar_error'] < 1e-12 and m['future_state_change'] < 1e-12
    assert m['empty_row_attention_mass'] == 0
    assert result['charts'][0]['x'][-1] == 160
