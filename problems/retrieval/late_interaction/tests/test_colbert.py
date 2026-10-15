import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.retrieval.late_interaction.toy import scalar_scores
from problems.retrieval.late_interaction.experiment import run


@pytest.mark.t0
def test_contract_precision_strides_empty_batches_and_validation(impl):
    """All query/document pairs preserve floating dtype/device, support strides, and never mutate input."""
    for dtype in (torch.float32, torch.float64):
        for batch, count, length in ((2, 3, 4), (0, 3, 4), (2, 0, 4), (2, 3, 0)):
            q = torch.randn(batch, length, 6, dtype=dtype)[..., ::2]
            d = torch.randn(count, 5, 6, dtype=dtype)[..., ::2]
            before_q, before_d = q.clone(), d.clone()
            score = impl.maxsim_scores(q, d)
            assert score.shape == (batch, count) and score.dtype == dtype and score.device == q.device
            close(q, before_q); close(d, before_d)
        loss = impl.pairwise_loss(torch.zeros(2, dtype=dtype), torch.ones(2, dtype=dtype))
        assert loss.shape == () and loss.dtype == dtype
    for q, d in ((torch.zeros(2, 3), torch.zeros(2, 3, 4)), (torch.zeros(1, 2, 3), torch.zeros(2, 0, 3))):
        with pytest.raises(ValueError): impl.maxsim_scores(q, d)
    with pytest.raises(ValueError): impl.maxsim_scores(torch.zeros(1, 2, 3), torch.zeros(2, 3, 3), torch.ones(1, 2))
    with pytest.raises(ValueError): impl.pairwise_loss(torch.zeros(0), torch.zeros(0))


@pytest.mark.t1
def test_hand_computed_normalized_sum_of_document_maxima(impl):
    """Two separate axis tokens score 2, one blended token sqrt(2), and an empty document minus infinity."""
    q = torch.tensor([[[2., 0.], [0., 3.]]], dtype=torch.float64)
    d = torch.tensor([[[4., 0.], [0., 7.]], [[1., 1.], [10., 0.]], [[1., 0.], [0., 1.]]], dtype=q.dtype)
    mask = torch.tensor([[True, True], [True, False], [False, False]])
    close(impl.maxsim_scores(q, d, document_mask=mask), torch.tensor([[2., math.sqrt(2), -math.inf]], dtype=q.dtype), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_independent_scalar_cosines_for_every_query_and_document(impl):
    """Python scalar norms, dot products and maxima independently check both precisions and all masks."""
    for dtype in (torch.float32, torch.float64):
        q, d = torch.randn(3, 4, 5, dtype=dtype), torch.randn(5, 6, 5, dtype=dtype)
        qm = torch.tensor([[True, False, True, True], [False]*4, [True]*4])
        dm = torch.tensor([[True, False, True, False, True, False], [True]*6, [False]*6, [False, True, False, True, False, True], [True, True, False, False, False, False]])
        expected = scalar_scores(q.tolist(), d.tolist(), qm.tolist(), dm.tolist())
        close(impl.maxsim_scores(q, d, qm, dm), torch.tensor(expected, dtype=dtype), atol=2e-6 if dtype == torch.float32 else 1e-12, rtol=2e-6 if dtype == torch.float32 else 1e-12)


@pytest.mark.t2
def test_query_and_document_gradients_against_scalar_finite_differences(impl):
    """Both normalized token gradients match independent central differences; padding gradients vanish."""
    q = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    d = torch.randn(3, 4, 3, dtype=torch.float64, requires_grad=True)
    qm = torch.tensor([[True, False, True], [True, True, False]])
    dm = torch.tensor([[True, False, True, True], [False, True, True, False], [False]*4])
    weights = [[.3, -.8], [.7, .2]]
    value = impl.maxsim_scores(q, d, qm, dm)
    (value[:, :2] * torch.tensor(weights, dtype=q.dtype)).sum().backward()
    assert q.grad is not None and d.grad is not None, 'Both encodings must retain gradients'
    qs, ds = q.detach().tolist(), d.detach().tolist()
    def objective():
        scores = scalar_scores(qs, ds, qm.tolist(), dm.tolist())
        return sum(scores[i][j]*weights[i][j] for i in range(2) for j in range(2))
    for values, gradient in ((qs, q.grad), (ds, d.grad)):
        expected = torch.zeros_like(gradient)
        for batch, tokens in enumerate(values):
            for token, vector in enumerate(tokens):
                for feature, original in enumerate(vector):
                    vector[feature] = original + 1e-5; above = objective()
                    vector[feature] = original - 1e-5; below = objective()
                    vector[feature] = original
                    expected[batch, token, feature] = (above - below) / 2e-5
        close(gradient, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.t2
def test_exact_tie_routes_gradient_to_first_valid_document_token(impl):
    """Torch max(dim) chooses the first exact winner; duplicate and padded tokens receive no gradient."""
    q = torch.tensor([[[1., 0.]]], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([[[1., 1.], [1., 1.], [100., 0.]]], dtype=q.dtype, requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    impl.maxsim_scores(q, d, document_mask=mask).sum().backward()
    assert q.grad is not None and d.grad is not None
    close(q.grad, torch.tensor([[[0., 1/math.sqrt(2)]]], dtype=q.dtype), atol=1e-12, rtol=1e-12)
    close(d.grad, torch.tensor([[[1/(2*math.sqrt(2)), -1/(2*math.sqrt(2))], [0., 0.], [0., 0.]]], dtype=q.dtype), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_permutations_positive_token_scaling_and_duplicate_query_evidence(impl):
    """Token order and positive token rescaling preserve scores; duplicating queries doubles evidence."""
    q, d = torch.randn(2, 3, 4, dtype=torch.float64), torch.randn(4, 5, 4, dtype=torch.float64)
    score = impl.maxsim_scores(q, d)
    close(impl.maxsim_scores(q[:, [2, 0, 1]], d[:, [4, 1, 0, 3, 2]]), score, atol=1e-12, rtol=1e-12)
    order = torch.tensor([3, 1, 0, 2])
    close(impl.maxsim_scores(q, d[order]), score[:, order])
    qs = torch.tensor([[[.2], [3.], [7.]], [[2.], [.4], [8.]]], dtype=q.dtype)
    ds = torch.arange(1, 21, dtype=q.dtype).reshape(4, 5, 1)
    close(impl.maxsim_scores(q*qs, d*ds), score, atol=1e-12, rtol=1e-12)
    close(impl.maxsim_scores(torch.cat([q, q], dim=1), d), 2*score, atol=1e-12, rtol=1e-12)
    close(impl.maxsim_scores(q, torch.cat([d, d], dim=1)), score, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_negative_matches_padding_empty_documents_and_zero_gradients(impl):
    """Padding cannot beat negative evidence; fully masked documents are invalid without NaN gradients."""
    q = torch.tensor([[[1., 0.], [0., 0.]], [[0., 1.], [9., 9.]]], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([[[-1., 0.], [100., 0.]], [[0., 0.], [0., 0.]]], dtype=q.dtype, requires_grad=True)
    qm = torch.tensor([[True, False], [False, False]])
    dm = torch.tensor([[True, False], [False, False]])
    score = impl.maxsim_scores(q, d, qm, dm)
    close(score, torch.tensor([[-1., -math.inf], [0., -math.inf]], dtype=q.dtype))
    score[:, :1].sum().backward()
    assert q.grad is not None and d.grad is not None
    assert bool(torch.isfinite(q.grad).all()) and bool(torch.isfinite(d.grad).all())
    close(q.grad[:, 1], torch.zeros_like(q.grad[:, 1]))
    close(d.grad[:, 1], torch.zeros_like(d.grad[:, 1])); close(d.grad[1], torch.zeros_like(d.grad[1]))
    zero = torch.zeros(1, 1, 2, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[[1., 0.]]], dtype=zero.dtype, requires_grad=True)
    value = impl.maxsim_scores(zero, valid)
    close(value, torch.zeros_like(value)); value.sum().backward()
    assert zero.grad is not None and valid.grad is not None
    close(zero.grad, torch.tensor([[[1e12, 0.]]], dtype=zero.dtype))
    close(valid.grad, torch.zeros_like(valid.grad))


@pytest.mark.t2
def test_pairwise_loss_stability_and_both_score_gradients(impl):
    """Pairwise softmax is stable at margins +/-2000 and differentiates both positive and negative scores."""
    positive = torch.tensor([1000., -1000., 0., 2.], dtype=torch.float64, requires_grad=True)
    negative = torch.tensor([-1000., 1000., 1., -3.], dtype=positive.dtype, requires_grad=True)
    margins = [n-p for p, n in zip(positive.tolist(), negative.tolist())]
    wanted = sum(max(m, 0.) + math.log1p(math.exp(-abs(m))) for m in margins) / 4
    loss = impl.pairwise_loss(positive, negative)
    close(loss, torch.tensor(wanted, dtype=positive.dtype), atol=1e-12, rtol=1e-12)
    loss.backward()
    assert positive.grad is not None and negative.grad is not None
    sigmoid = [(1/(1+math.exp(-m)) if m >= 0 else math.exp(m)/(1+math.exp(m))) / 4 for m in margins]
    close(negative.grad, torch.tensor(sigmoid, dtype=positive.dtype)); close(positive.grad, -negative.grad)


@pytest.mark.t2
def test_pairwise_common_shift_and_positive_evidence_monotonicity(impl):
    """A shared score shift cancels, and improving every positive score strictly lowers the loss."""
    for dtype in (torch.float32, torch.float64):
        positive, negative = torch.tensor([.2, -1., 3.], dtype=dtype), torch.tensor([-.7, 2., 1.], dtype=dtype)
        original = impl.pairwise_loss(positive, negative)
        close(impl.pairwise_loss(positive+7, negative+7), original)
        assert impl.pairwise_loss(positive+1, negative).item() < original.item()
        assert impl.pairwise_loss(positive, negative+1).item() > original.item()


@pytest.mark.t3
def test_synthetic_query_optimization_and_late_interaction_retrieval(impl, artifact):
    """120 real updates learn four synthetic retrieval pairs that pooled document directions cannot distinguish."""
    result = run(impl.maxsim_scores, impl.pairwise_loss)
    artifact('late_interaction_experiment', result)
    metrics = result['metrics']
    assert metrics['updates'] == 120
    assert metrics['maximum_scalar_error'] < 1e-12 and metrics['pooled_pair_gap'] < 1e-12
    assert metrics['initial_recall_at_1'] == 0 and metrics['final_recall_at_1'] == 1
    assert metrics['final_loss'] < metrics['initial_loss'] - .2
    assert metrics['minimum_final_margin'] > .4
