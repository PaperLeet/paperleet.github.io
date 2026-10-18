import itertools
import math
import pytest
import torch


def oracle(roles, conversations, attention):
    expected = torch.zeros_like(attention)
    for b, (row, ids, active) in enumerate(zip(roles.tolist(), conversations.tolist(), attention.tolist())):
        for j in range(len(row)):
            if j == 0 or row[j] != 2 or not active[j]:
                continue
            if active[j-1] and ids[j-1] == ids[j]:
                expected[b, j] = True
    return expected


def check(actual, expected):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == torch.bool and actual.device == expected.device
    assert torch.equal(actual, expected), {'expected': expected.tolist(), 'actual': actual.tolist()}


@pytest.mark.t0
def test_contract_empty_axes_and_singletons(impl):
    """Target masks keep both axes, including empty rows/columns; a first token has no predictor."""
    for shape in [(0, 0), (0, 4), (2, 0), (1, 1), (2, 1)]:
        roles = torch.full(shape, 2, dtype=torch.long)
        ids = torch.zeros(shape, dtype=torch.long)
        active = torch.ones(shape, dtype=torch.bool)
        check(impl.assistant_target_mask(roles, ids, active), torch.zeros_like(active))


@pytest.mark.t1
def test_hand_labelled_turns_eos_and_packed_boundaries(impl):
    """All assistant turns and their EOS are targets, except unavailable or cross-conversation predecessors."""
    from problems.foundations.multiturn_sft_mask.experiment import fixture
    roles, ids, active, _, expected = fixture()
    check(impl.assistant_target_mask(roles, ids, active), expected)
    # The assistant token at j=2 follows user context; both j=3/6 EOS are retained.
    assert expected[0, 2] and expected[0, 3] and expected[0, 6]


@pytest.mark.t2
def test_exhaustive_adjacent_pair_truth_table(impl):
    """The independent scalar rule covers all role pairs, padding pairs and equal/different IDs."""
    cases = list(itertools.product(range(4), range(4), (False, True), (False, True), (False, True)))
    roles = torch.tensor([[a, b] for a, b, _, _, _ in cases], dtype=torch.long)
    ids = torch.tensor([[7, 7 if same else 91] for _, _, _, _, same in cases], dtype=torch.long)
    active = torch.tensor([[a, b] for _, _, a, b, _ in cases], dtype=torch.bool)
    check(impl.assistant_target_mask(roles, ids, active), oracle(roles, ids, active))


@pytest.mark.t2
def test_truncation_keeps_visible_targets_without_inventing_eos(impl):
    """Right truncation preserves prior targets; left truncation only loses the first retained predictor."""
    roles = torch.tensor([[0, 1, 2, 2, 1, 2, 2, 2]], dtype=torch.long)
    ids = torch.full_like(roles, 12)
    active = torch.ones_like(roles, dtype=torch.bool)
    full = oracle(roles, ids, active)
    for end in range(9):
        check(impl.assistant_target_mask(roles[:, :end], ids[:, :end], active[:, :end]), full[:, :end])
    for start in range(9):
        expected = full[:, start:].clone()
        expected[:, :1] = False
        check(impl.assistant_target_mask(roles[:, start:], ids[:, start:], active[:, start:]), expected)


@pytest.mark.t2
def test_padding_metadata_strides_and_input_preservation(impl):
    """Padding metadata is irrelevant, gaps break predecessor links, and strided inputs stay unchanged."""
    roles = torch.tensor([[0, 9, 1, 9, 2, 9, 2, 9, 3, 9, 2, 9, 2, 9]], dtype=torch.long)[:, ::2]
    ids = torch.full((1, 14), 123456789123, dtype=torch.long)[:, ::2]
    active = torch.tensor([[True, False, True, False, True, False, False, False, True, False, True, False, True, False]])[:, ::2]
    before = [x.clone() for x in (roles, ids, active)]
    expected = oracle(roles, ids, active)
    check(impl.assistant_target_mask(roles, ids, active), expected)
    for value, saved in zip((roles, ids, active), before):
        assert torch.equal(value, saved), 'Inputs were modified'
    dirty_roles, dirty_ids = roles.clone(), ids.clone()
    dirty_roles[~active], dirty_ids[~active] = -4567, -9876
    check(impl.assistant_target_mask(dirty_roles, dirty_ids, active), expected)
    check(impl.assistant_target_mask(roles, ids, torch.zeros_like(active)), torch.zeros_like(active))


@pytest.mark.t2
def test_row_isolation_id_renaming_and_segment_independence(impl):
    """IDs label local conversations; their values and batch ordering cannot change target selection."""
    roles = torch.tensor([[2, 2, 1, 2, 2, 2], [0, 1, 2, 3, 2, 2]], dtype=torch.long)
    ids = torch.tensor([[5, 5, 5, 9, 9, 9], [5, 5, 5, 5, 5, 5]], dtype=torch.long)
    active = torch.ones_like(roles, dtype=torch.bool)
    expected = oracle(roles, ids, active)
    check(impl.assistant_target_mask(roles, ids*1009+37, active), expected)
    order = torch.tensor([1, 0, 1], dtype=torch.long)
    check(impl.assistant_target_mask(roles[order], ids[order], active[order]), expected[order])
    # Processing each packed conversation separately gives the same target labels.
    left = impl.assistant_target_mask(roles[:1, :3], ids[:1, :3], active[:1, :3])
    right = impl.assistant_target_mask(roles[:1, 3:], ids[:1, 3:], active[:1, 3:])
    check(torch.cat([left, right], dim=1), expected[:1])


@pytest.mark.t2
def test_mask_drives_only_eligible_next_token_gradients(impl):
    """Target j supervises logits j-1 once; excluded predictors and final logits have exact zero gradient."""
    from problems.foundations.multiturn_sft_mask.experiment import fixture
    roles, ids, active, tokens, expected = fixture()
    selected = impl.assistant_target_mask(roles, ids, active)
    logits = torch.zeros(2, 12, 4, dtype=torch.float64, requires_grad=True)
    nll = -logits[:, :-1].log_softmax(-1).gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    loss = torch.where(selected[:, 1:], nll, torch.zeros_like(nll)).sum() / selected[:, 1:].sum().clamp(min=1)
    loss.backward()
    analytical = torch.zeros_like(logits)
    count = int(expected.sum())
    for b, row in enumerate(expected.tolist()):
        for j, use in enumerate(row):
            if use:
                analytical[b, j-1] = .25 / count
                analytical[b, j-1, int(tokens[b, j])] -= 1. / count
    torch.testing.assert_close(logits.grad, analytical, rtol=1e-12, atol=1e-12)
    assert abs(float(loss.detach()) - math.log(4)) < 1e-12


@pytest.mark.t3
def test_submitted_mask_trains_only_assistant_targets(impl, artifact):
    """Eighty real SGD updates fit the eleven hand-labelled targets and leave excluded logits unchanged."""
    from problems.foundations.multiturn_sft_mask.experiment import run
    result = run(impl.assistant_target_mask)
    artifact('toy_experiment', result)
    metrics = result['metrics']
    assert metrics['mask_disagreements'] == 0, metrics
    assert metrics['selected_targets'] == metrics['expected_targets'] == 11, metrics
    assert abs(metrics['initial_assistant_nll'] - math.log(4)) < 1e-12, metrics
    assert metrics['final_assistant_nll'] < .08, metrics
    assert metrics['assistant_accuracy'] == 1., metrics
    assert metrics['excluded_logit_change'] == 0., metrics
