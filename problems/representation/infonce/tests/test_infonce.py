import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from ..toy import scalar_loss, fixture
from ..experiment import run


@pytest.mark.t0
def test_scalar_dtype_strides_and_input_preservation(impl):
    """A mean classification loss preserves floating dtype/device and never edits its inputs."""
    for dtype in (torch.float32, torch.float64):
        logits, _ = fixture(dtype)
        logits = logits[:, ::2]
        indices = torch.tensor([0, 9, 1, 9, 0, 9, 1, 9, 1, 9, 0, 9])[::2]
        old_logits, old_indices = logits.clone(), indices.clone()
        actual = impl.info_nce_loss(logits, indices)
        assert actual.shape == () and actual.dtype == dtype and actual.device == logits.device
        close(actual, torch.tensor(scalar_loss(logits.tolist(), indices.tolist()), dtype=dtype),
              atol=2e-6, rtol=2e-6)
        close(logits, old_logits)
        close(indices, old_indices)


@pytest.mark.t0
def test_invalid_input_contract(impl):
    """Nonempty finite logits require exactly one in-range int64 positive column per row."""
    logits, indices = fixture()
    invalid = [(logits[:0], indices[:0]), (logits[:, :0], indices),
               (logits.flatten(), indices), (logits.to(torch.int64), indices),
               (logits * float('nan'), indices), (logits * float('inf'), indices),
               (logits, indices[:2]), (logits, indices[:, None]),
               (logits, indices.float()), (logits, indices.to(torch.int32)),
               (logits, indices - 4), (logits, indices + 4)]
    for values, targets in invalid:
        with pytest.raises(ValueError):
            impl.info_nce_loss(values, targets)


@pytest.mark.t1
def test_closed_forms_and_single_candidate(impl):
    """Uniform K-way scores give log(K); a single candidate has exactly zero loss and gradient."""
    for count in (1, 2, 7):
        logits = torch.full((3, count), 2.5, dtype=torch.float64)
        indices = torch.tensor([0, count - 1, 0])
        close(impl.info_nce_loss(logits, indices), torch.tensor(math.log(count), dtype=logits.dtype),
              atol=1e-12, rtol=1e-12)
    scores = torch.tensor([[math.log(2), math.log(3), math.log(5)]], dtype=torch.float64)
    close(impl.info_nce_loss(scores, torch.tensor([1])),
          torch.tensor(math.log(10 / 3), dtype=scores.dtype), atol=1e-12, rtol=1e-12)
    one = torch.tensor([[3.], [-7.]], dtype=torch.float64, requires_grad=True)
    loss = impl.info_nce_loss(one, torch.tensor([0, 0]))
    close(loss, torch.tensor(0., dtype=one.dtype), atol=0, rtol=0)
    close(torch.autograd.grad(loss, one)[0], torch.zeros_like(one), atol=0, rtol=0)


@pytest.mark.t2
def test_independent_scalar_denominator_and_reference(impl, ref):
    """CPC Equation (4) includes the positive and every negative in each row's denominator."""
    logits, indices = fixture()
    for batch in (1, 3, 6):
        for scale in (.1, 1., 7.):
            values, targets = logits[:batch] * scale, indices[:batch]
            expected = torch.tensor(scalar_loss(values.tolist(), targets.tolist()), dtype=values.dtype)
            close(impl.info_nce_loss(values, targets), expected, atol=1e-12, rtol=1e-12)
            close(impl.info_nce_loss(values, targets), ref.info_nce_loss(values, targets),
                  atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_gradient_matches_scalar_derivative_and_finite_differences(impl):
    """The derivative is (softmax(logits)-one_hot(positive))/B for every candidate."""
    values, indices = fixture()
    values = values[:3].clone().requires_grad_()
    targets = indices[:3]
    gradient = torch.autograd.grad(impl.info_nce_loss(values, targets), values)[0]
    rows, positives = values.detach().tolist(), targets.tolist()
    analytic, numeric = [], []
    for i, row in enumerate(rows):
        peak = max(row)
        weights = [math.exp(value - peak) for value in row]
        analytic.append([(weight / sum(weights) - (j == positives[i])) / len(rows)
                         for j, weight in enumerate(weights)])
        difference = []
        for j, old in enumerate(row):
            row[j] = old + 1e-6
            plus = scalar_loss(rows, positives)
            row[j] = old - 1e-6
            minus = scalar_loss(rows, positives)
            row[j] = old
            difference.append((plus - minus) / 2e-6)
        numeric.append(difference)
    close(gradient, torch.tensor(analytic, dtype=values.dtype), atol=1e-12, rtol=1e-11)
    close(gradient, torch.tensor(numeric, dtype=values.dtype), atol=2e-9, rtol=2e-7)
    close(gradient.sum(1), torch.zeros(3, dtype=values.dtype), atol=1e-12, rtol=0)

    # Interleave each candidate with an unused column, then select a strided view.
    # The parent tensor must receive the same VJP in selected columns and zero elsewhere.
    parent = torch.stack((values.detach(), torch.ones_like(values)), dim=2).reshape(3, 8)
    parent = parent.clone().requires_grad_()
    strided_loss = impl.info_nce_loss(parent[:, ::2], targets)
    assert strided_loss.requires_grad, "Strided logits must retain their parent gradient path"
    parent_gradient = torch.autograd.grad(strided_loss, parent)[0]
    expected_parent = torch.stack((torch.tensor(analytic, dtype=values.dtype),
                                   torch.zeros_like(values)), dim=2).reshape(3, 8)
    close(parent_gradient, expected_parent, atol=1e-12, rtol=1e-11)


@pytest.mark.t2
def test_row_offsets_and_row_permutation(impl):
    """Per-context score offsets cancel, and averaging makes context ordering irrelevant."""
    logits, indices = fixture()
    base = impl.info_nce_loss(logits, indices)
    offsets = torch.tensor([[10.], [-20.], [3.], [7.], [-2.], [30.]], dtype=logits.dtype)
    close(impl.info_nce_loss(logits + offsets, indices), base, atol=1e-12, rtol=1e-12)
    order = torch.tensor([3, 5, 1, 0, 4, 2])
    close(impl.info_nce_loss(logits[order], indices[order]), base, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_candidate_permutation_tracks_positive_columns(impl):
    """Relabeling candidates preserves the probability of the same positive event."""
    logits, indices = fixture()
    order = [2, 0, 3, 1]
    remapped = torch.tensor([order.index(value) for value in indices.tolist()])
    close(impl.info_nce_loss(logits[:, order], remapped), impl.info_nce_loss(logits, indices),
          atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_extreme_scores_are_stable_and_finite(impl):
    """Stable log-sum-exp must handle large score offsets without exponent overflow."""
    logits = torch.tensor([[10000., 9999., 9998.], [-10000., -10001., -10004.],
                           [10000., -10000., 0.]], dtype=torch.float64, requires_grad=True)
    indices = torch.tensor([2, 0, 1])
    actual = impl.info_nce_loss(logits, indices)
    assert bool(torch.isfinite(actual))
    close(actual, torch.tensor(scalar_loss(logits.tolist(), indices.tolist()), dtype=logits.dtype),
          atol=1e-10, rtol=1e-12)
    gradient = torch.autograd.grad(actual, logits)[0]
    assert bool(torch.isfinite(gradient).all())


@pytest.mark.t2
def test_positive_strength_and_extra_negative(impl):
    """Increasing a positive score reduces loss; adding a finite negative increases the denominator."""
    logits, indices = fixture()
    strengthened = logits.clone()
    for i, target in enumerate(indices.tolist()):
        strengthened[i, target] += 2.
    base = impl.info_nce_loss(logits, indices)
    assert float(impl.info_nce_loss(strengthened, indices)) < float(base)
    expanded = torch.cat([logits, torch.zeros((6, 1), dtype=logits.dtype)], dim=1)
    assert float(impl.info_nce_loss(expanded, indices)) > float(base)


@pytest.mark.t3
def test_actual_score_table_training_selects_the_positive(impl, artifact):
    """80 SGD updates on 24 scores must improve all six contexts using the submitted objective."""
    result = run(impl.info_nce_loss)
    artifact('infonce_training', result)
    metrics = result['metrics']
    assert metrics['updates'] == 80 and metrics['trainable_parameters'] == 24
    assert metrics['maximum_scalar_error'] < 1e-12
    assert metrics['final_accuracy'] == 1 and metrics['final_accuracy'] > metrics['initial_accuracy']
    assert metrics['final_loss'] < .1 and metrics['final_loss'] < .1 * metrics['initial_loss']
    assert metrics['final_positive_probability'] > .9 and metrics['final_minimum_margin'] > 3
