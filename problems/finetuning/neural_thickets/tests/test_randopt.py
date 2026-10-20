import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.finetuning.neural_thickets.experiment import run


@pytest.mark.t0
def test_candidate_shape_purity_and_detachment(impl):
    """All candidates start from the unchanged base and carry no training graph."""
    theta = torch.arange(6, dtype=torch.float64).reshape(2, 3).T.requires_grad_()
    noise = torch.ones(3, 2, 3, dtype=theta.dtype).transpose(1, 2).requires_grad_()
    scales = torch.tensor([0., .2, .5], dtype=theta.dtype, requires_grad=True)
    before = [x.detach().clone() for x in (theta, noise, scales)]
    out = impl.perturb_candidates(theta, noise, scales)
    assert out.shape == (3, 3, 2) and out.dtype == theta.dtype and not out.requires_grad
    for i, amount in enumerate([0., .2, .5]):
        close(out[i], theta.detach() + amount)
    for x, saved in zip((theta, noise, scales), before):
        close(x, saved)
    scalar = impl.perturb_candidates(torch.tensor(2.), torch.tensor([1., -1.]), torch.tensor([.5, 2.]))
    close(scalar, torch.tensor([2.5, 0.]))


@pytest.mark.t1
def test_selection_and_independent_vote_oracle(impl):
    """Algorithm 1 selects by training score then takes the unweighted mode of discrete test answers."""
    scores = torch.tensor([.9, .1, .8, .85], dtype=torch.float64)
    selected = impl.select_experts(scores, 3)
    close(selected, torch.tensor([0, 3, 2]))
    predictions = torch.tensor([[2, 0, 1], [0, 2, 2], [1, 0, 2], [2, 1, 0]])
    expected = []
    for column in zip(*[predictions.tolist()[i] for i in [0, 3, 2]]):
        counts = [column.count(c) for c in range(3)]
        expected.append(max(range(3), key=lambda c: (counts[c], -c)))
    close(impl.majority_vote(predictions, selected, 3), torch.tensor(expected))


@pytest.mark.t2
def test_stable_score_ties_and_k_boundaries(impl):
    """The declared score-tie convention preserves original indices even for all-equal scores."""
    scores = torch.tensor([2., 3., 3., -1.])
    close(impl.select_experts(scores, 4), torch.tensor([1, 2, 0, 3]))
    close(impl.select_experts(scores, 1), torch.tensor([1]))
    close(impl.select_experts(torch.ones(5), 3), torch.tensor([0, 1, 2]))


@pytest.mark.t2
def test_vote_ties_single_expert_and_empty_batch(impl):
    """Votes are modes of class IDs; ties choose the smaller ID and no samples yield an empty vector."""
    p = torch.tensor([[0, 2], [2, 2], [2, 0], [1, 0]])
    close(impl.majority_vote(p, torch.tensor([0, 1]), 3), torch.tensor([0, 2]))
    close(impl.majority_vote(p, torch.tensor([3]), 3), p[3])
    result = impl.majority_vote(p[:, :0], torch.tensor([0, 2]), 3)
    assert result.shape == (0,) and result.dtype == torch.long
    close(impl.majority_vote(torch.zeros(2, 3, dtype=torch.long), torch.tensor([1]), 1),
          torch.zeros(3, dtype=torch.long))


@pytest.mark.t2
def test_nonselected_models_and_sample_independence(impl):
    """Changing rejected experts cannot affect the ensemble; permuting test samples only permutes outputs."""
    p = torch.tensor([[2, 0, 1, 2], [1, 1, 0, 0], [2, 0, 0, 2]])
    selected = torch.tensor([0, 2])
    expected = impl.majority_vote(p, selected, 3)
    changed = p.clone(); changed[1] = 2
    close(impl.majority_vote(changed, selected, 3), expected)
    order = torch.tensor([3, 1, 0, 2])
    close(impl.majority_vote(p[:, order], selected, 3), expected[order])
    close(impl.majority_vote(p, torch.tensor([2, 0]), 3), expected)


@pytest.mark.t2
def test_candidates_are_independent_and_scale_specific(impl):
    """Equation 3 uses each candidate's own scale and the same original parameter vector."""
    theta = torch.tensor([1., -2.], dtype=torch.float64)
    noise = torch.tensor([[2., 1.], [-3., 2.], [1., -4.]], dtype=theta.dtype)
    scales = torch.tensor([.5, 2., 0.], dtype=theta.dtype)
    close(impl.perturb_candidates(theta, noise, scales),
          torch.tensor([[2., -1.5], [-5., 2.], [1., -2.]], dtype=theta.dtype))


@pytest.mark.t3
def test_random_expert_search(impl, artifact):
    """120 actual random candidates are selected on one grid and evaluated on a separate grid."""
    result = run(impl.perturb_candidates, impl.select_experts, impl.majority_vote)
    artifact('randopt_search', result)
    assert result['metrics']['selected_count'] == 9
    assert len(set(result['selected'])) == 9
    assert result['metrics']['base_accuracy'] == .5
    assert result['metrics']['ensemble_accuracy'] >= .9
