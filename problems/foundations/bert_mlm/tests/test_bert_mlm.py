import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.foundations.bert_mlm.experiment import run


def example():
    return (torch.tensor([[1, 2, 3, 0]], dtype=torch.int64),
            torch.tensor([[True, True, True, False]]),
            torch.tensor([[.2, .85, .95, .1]], dtype=torch.float64),
            torch.tensor([[3, 0, 1, 2]], dtype=torch.int64))


@pytest.mark.t0
def test_contract(impl):
    """MLM returns two int64 arrays and a scalar loss in the logit dtype."""
    corrupted, labels = impl.make_mlm_inputs(*example(), 4)
    assert corrupted.shape == labels.shape == (1, 4)
    assert corrupted.dtype == labels.dtype == torch.int64
    loss = impl.masked_lm_loss(torch.zeros(1, 4, 4, dtype=torch.float64), labels)
    assert loss.shape == () and loss.dtype == torch.float64


@pytest.mark.t1
def test_independent_scalar_example(impl):
    """Selected targets stay original through mask, random and unchanged branches."""
    corrupted, labels = impl.make_mlm_inputs(*example(), 4)
    assert corrupted.tolist() == [[4, 0, 3, 0]]
    assert labels.tolist() == [[1, 2, 3, -100]]
    logits = torch.tensor([[[0., 1., 2., 3.]] * 4], dtype=torch.float64)
    expected = math.log(sum(math.exp(i) for i in range(4))) - 2.
    assert abs(impl.masked_lm_loss(logits, labels).item() - expected) < 1e-12


@pytest.mark.t2
def test_action_boundaries(impl):
    """Half-open intervals assign .8 to random replacement and .9 to unchanged."""
    tokens = torch.tensor([[0, 1, 2, 3, 0]], dtype=torch.int64)
    selected = torch.ones_like(tokens, dtype=torch.bool)
    draws = torch.tensor([[0., .79999, .8, .89999, .9]], dtype=torch.float64)
    replacement = torch.tensor([[2, 3, 0, 1, 2]], dtype=torch.int64)
    corrupted, labels = impl.make_mlm_inputs(tokens, selected, draws, replacement, 4)
    assert corrupted.tolist() == [[4, 4, 0, 1, 0]]
    assert labels.tolist() == tokens.tolist()


@pytest.mark.t2
def test_original_labels_and_no_mutation(impl):
    """Prediction labels refer to the same positions, never next-token targets."""
    args = example();before = [x.clone() for x in args]
    corrupted, labels = impl.make_mlm_inputs(*args, 4, ignore_index=-7)
    assert labels.tolist() == [[1, 2, 3, -7]]
    for x, y in zip(args, before):
        assert torch.equal(x, y)
    corrupted[0, 0] = 9;labels[0, 1] = 9
    assert torch.equal(args[0], before[0])


@pytest.mark.t2
def test_only_selected_tokens_in_mean(impl):
    """Adding ignored positions cannot dilute the selected-token objective."""
    logits = torch.tensor([[[2., -1., 0.], [1000., -1000., 0.], [1., 2., -2.]]], dtype=torch.float64)
    labels = torch.tensor([[2, -100, 0]])
    expected = ((math.log(sum(math.exp(v) for v in [2., -1., 0.])) - 0.) + (math.log(sum(math.exp(v) for v in [1., 2., -2.])) - 1.)) / 2
    assert abs(impl.masked_lm_loss(logits, labels).item() - expected) < 1e-12
    close(impl.masked_lm_loss(logits[:, [0, 2]], labels[:, [0, 2]]), impl.masked_lm_loss(logits, labels), rtol=0, atol=1e-12)


@pytest.mark.t2
def test_analytic_logit_gradients(impl):
    """Uniform softmax gives (probability minus target)/selected_count at selected sites."""
    logits = torch.zeros(2, 3, 4, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2], [3, -100, -100]])
    loss = impl.masked_lm_loss(logits, labels)
    assert loss.requires_grad, 'Selected logits must remain differentiable'
    loss.backward();expected = torch.zeros_like(logits)
    for b, j, token in [(0, 1, 1), (0, 2, 2), (1, 0, 3)]:
        expected[b, j] = .25 / 3;expected[b, j, token] -= 1 / 3
    assert logits.grad is not None
    close(logits.grad, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.t2
def test_loss_finite_difference_and_extremes(impl):
    """Stable cross entropy stays finite at extreme logits and matches numerical derivatives."""
    x = torch.tensor([[[.2, -.7, 1.1], [.3, .4, -.2]]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[1, -100]])
    loss = impl.masked_lm_loss(x, labels);assert loss.requires_grad;loss.backward()
    expected = torch.zeros_like(x);eps = 1e-5
    for j in range(2):
        for k in range(3):
            plus, minus = x.detach().clone(), x.detach().clone()
            plus[0, j, k] += eps;minus[0, j, k] -= eps
            expected[0, j, k] = (impl.masked_lm_loss(plus, labels) - impl.masked_lm_loss(minus, labels)) / (2 * eps)
    close(x.grad, expected, rtol=1e-7, atol=1e-9)
    extreme = torch.tensor([[[1000., -1000., 0.]]], dtype=torch.float64)
    assert abs(impl.masked_lm_loss(extreme, torch.tensor([[1]])).item() - 2000) < 1e-10


@pytest.mark.t2
def test_empty_selection_connected_zero(impl):
    """No selected targets, including empty batches, produce differentiable zero."""
    for shape in [(2, 3, 4), (0, 3, 4), (2, 0, 4)]:
        x = torch.ones(*shape, dtype=torch.float64, requires_grad=True)
        labels = torch.full(shape[:2], -7, dtype=torch.int64)
        loss = impl.masked_lm_loss(x, labels, -7)
        assert loss.item() == 0 and loss.requires_grad
        loss.backward();assert x.grad is not None
        close(x.grad, torch.zeros_like(x), rtol=0, atol=0)


@pytest.mark.t2
def test_strided_inputs_and_dtype(impl):
    """The contract follows logical positions rather than contiguous storage."""
    args = example();expanded = [a.repeat(2, 2)[:, ::2] for a in args]
    a, b = impl.make_mlm_inputs(*expanded, 4)
    c, d = impl.make_mlm_inputs(*(x.contiguous() for x in expanded), 4)
    assert torch.equal(a, c) and torch.equal(b, d)
    logits = torch.arange(48, dtype=torch.float32).reshape(2, 4, 6)[:, :, ::2]
    labels = torch.tensor([[0, 1, 2, -100], [2, 1, 0, -100]])
    loss = impl.masked_lm_loss(logits, labels)
    assert loss.dtype == torch.float32
    close(loss, impl.masked_lm_loss(logits.contiguous(), labels), rtol=1e-6, atol=1e-6)


@pytest.mark.t2
def test_invalid_inputs(impl):
    """Reject ambiguous shapes, non-probability draws and out-of-vocabulary targets."""
    tokens, selected, draws, random = example()
    for bad in [torch.full_like(draws, -.1), torch.full_like(draws, 1.), torch.full_like(draws, float('nan'))]:
        with pytest.raises(ValueError): impl.make_mlm_inputs(tokens, selected, bad, random, 4)
    with pytest.raises(ValueError): impl.make_mlm_inputs(tokens, selected[:, :1], draws, random, 4)
    with pytest.raises(ValueError): impl.make_mlm_inputs(tokens.float(), selected, draws, random, 4)
    with pytest.raises(ValueError): impl.masked_lm_loss(torch.zeros(1, 4, 3), torch.tensor([[0, 1, 3, -100]]))
    with pytest.raises(ValueError): impl.masked_lm_loss(torch.zeros(1, 4, 3), torch.tensor([[0, 1, -2, -100]]))


@pytest.mark.t3
def test_bidirectional_context_learning(impl, artifact):
    """One hundred Adam updates use submitted corruption and loss on selected center targets."""
    result = run(impl.make_mlm_inputs, impl.masked_lm_loss);artifact('toy_experiment', result)
    m = result['metrics']
    assert abs(m['initial_nll'] - math.log(4)) < 1e-12, m
    assert m['final_nll'] < .1 and m['final_nll'] < .1 * m['initial_nll'], m
    assert m['context_accuracy'] >= .95, m
