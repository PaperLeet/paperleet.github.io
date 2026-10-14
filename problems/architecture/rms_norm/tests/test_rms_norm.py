import math
import pytest
import torch
import torch.nn.functional as F
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.rms_norm.experiment import run


@pytest.mark.t0
def test_shapes_precision_strides_and_no_mutation(impl):
    """Both normalizers preserve all features, leading dimensions, strides' values and precision."""
    for dtype in (torch.float32, torch.float64):
        for x in (torch.randn(5, dtype=dtype), torch.randn(2, 3, 10, dtype=dtype)[..., ::2],
                  torch.empty(0, 3, 5, dtype=dtype)):
            weight = torch.randn(5, dtype=dtype)
            before = [value.clone() for value in (x, weight)]
            for output in (impl.rms_norm(x, weight), impl.partial_rms_norm(x, weight, 2)):
                assert output.shape == x.shape and output.dtype == dtype and output.device == x.device
            for actual, expected in zip((x, weight), before):
                close(actual, expected)


@pytest.mark.t1
def test_hand_computed_full_and_partial_affine(impl):
    """The full mean square is 169/3; a two-feature prefix has mean square 25/2."""
    x = torch.tensor([[-3., 4., 12.]], dtype=torch.float64)
    weight, eps = torch.tensor([2., -1., .5], dtype=x.dtype), .25
    numerators = [-6., -4., 6.]
    for output, mean_square in ((impl.rms_norm(x, weight, eps), 169 / 3),
                                (impl.partial_rms_norm(x, weight, 2, eps), 25 / 2)):
        expected = torch.tensor([[value / math.sqrt(mean_square + eps) for value in numerators]], dtype=x.dtype)
        close(output, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_zero_constant_single_feature_and_epsilon(impl):
    """RMS is not centered variance: constant nonzero rows stay nonzero, and zero has finite gradients."""
    for value in (0., 3., 1e-8, -1e-8):
        x = torch.full((2, 1), value, dtype=torch.float64, requires_grad=True)
        weight, eps = torch.tensor([-2.], dtype=x.dtype, requires_grad=True), 1e-4
        expected = torch.full((2, 1), -2 * value / math.sqrt(value * value + eps), dtype=x.dtype)
        for output in (impl.rms_norm(x, weight, eps), impl.partial_rms_norm(x, weight, 1, eps)):
            close(output, expected, atol=1e-13, rtol=1e-12)
            dx, dw = torch.autograd.grad(output.sum(), (x, weight))
            assert torch.isfinite(dx).all() and torch.isfinite(dw).all()
            if value == 0:
                close(dx, torch.full_like(x, -2 / math.sqrt(eps)), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_token_local_scale_sign_and_shift(impl):
    """Per-token statistics are independent; scaled epsilon gives an exact scale identity, not shift invariance."""
    x = torch.randn(2, 3, 5, dtype=torch.float64)
    weight, eps = torch.ones(5, dtype=x.dtype), 1e-4
    for normalize in (lambda a, e: impl.rms_norm(a, weight, e),
                      lambda a, e: impl.partial_rms_norm(a, weight, 2, e)):
        original = normalize(x, eps)
        close(normalize(7 * x, 49 * eps), original, atol=1e-12, rtol=1e-12)
        close(normalize(-x, eps), -original, atol=1e-12, rtol=1e-12)
        changed = x.clone(); changed[1] *= 100; changed[0, 1:] += 20
        close(normalize(changed, eps)[0, 0], original[0, 0], atol=1e-12, rtol=1e-12)
        assert (normalize(x + 8, eps) - original).abs().max().item() > .1


@pytest.mark.t2
def test_full_values_and_both_gradients_against_library(impl):
    """An independent RMSNorm operator agrees on outputs, input gradients and feature-scale gradients."""
    x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, dtype=x.dtype, requires_grad=True)
    a, b = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()
    output, expected = impl.rms_norm(x, weight, 1e-4), F.rms_norm(a, (5,), b, 1e-4)
    close(output, expected, atol=1e-12, rtol=1e-12)
    upstream = torch.randn_like(x)
    for actual, reference in zip(torch.autograd.grad((output * upstream).sum(), (x, weight)),
                                 torch.autograd.grad((expected * upstream).sum(), (a, b))):
        close(actual, reference, atol=1e-11, rtol=1e-11)


@pytest.mark.t2
def test_partial_jacobian_includes_tail_to_prefix_coupling(impl):
    """Every output affects prefix statistics' gradients; suffix inputs only have their direct path."""
    values, scales, upstream = [2., -1., 3., 4.], [1., -2., .5, 3.], [.2, 2., -1., .7]
    x = torch.tensor([values], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor(scales, dtype=x.dtype, requires_grad=True)
    eps, k = .25, 2
    output = impl.partial_rms_norm(x, weight, k, eps)
    dx, dw = torch.autograd.grad((output * torch.tensor([upstream], dtype=x.dtype)).sum(), (x, weight))
    s = math.sqrt(sum(value * value for value in values[:k]) / k + eps)
    coupled = sum(u * w * value for u, w, value in zip(upstream, scales, values))
    expected_dx = [u * w / s - (value * coupled / (k * s**3) if j < k else 0.)
                   for j, (u, w, value) in enumerate(zip(upstream, scales, values))]
    close(dx, torch.tensor([expected_dx], dtype=x.dtype), atol=1e-12, rtol=1e-12)
    close(dw, torch.tensor([u * value / s for u, value in zip(upstream, values)], dtype=x.dtype), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_prefix_selection_endpoints_and_invalid_counts(impl):
    """Only the first k features estimate scale, all features are returned, and k=D recovers full RMSNorm."""
    x = torch.tensor([[1., -2., 30., -40.]], dtype=torch.float64)
    weight = torch.tensor([2., -1., .5, 3.], dtype=x.dtype)
    before = impl.partial_rms_norm(x, weight, 2)
    changed = x.clone(); changed[..., 2:] *= 10
    after = impl.partial_rms_norm(changed, weight, 2)
    close(after[..., :2], before[..., :2], atol=1e-12, rtol=1e-12)
    close(after[..., 2:], 10 * before[..., 2:], atol=1e-12, rtol=1e-12)
    close(impl.partial_rms_norm(x, weight, 4), impl.rms_norm(x, weight), atol=1e-12, rtol=1e-12)
    for invalid in (0, -1, 5, True, False, 1.5, '2', None):
        with pytest.raises(ValueError):
            impl.partial_rms_norm(x, weight, invalid)


@pytest.mark.t2
def test_partial_gradients_match_finite_differences(impl):
    """A numerical Jacobian independently checks prefix-statistic and scale gradients."""
    x = torch.tensor([[.3, -1.2, 2.1]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([1.1, -.7, 2.], dtype=x.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: impl.partial_rms_norm(a, b, 2, .01),
                                    (x, weight), eps=1e-6, atol=1e-6, rtol=1e-5, raise_exception=False)


@pytest.mark.t3
def test_prefix_bias_and_offset_experiment(impl, artifact):
    """A low-energy prefix amplifies the suffix; full-prefix normalization and the epsilon scale identity remain exact."""
    result = run(impl.rms_norm, impl.partial_rms_norm)
    artifact('rms_prefix_experiment', result)
    metrics = result['metrics']
    assert metrics['maximum_scalar_error'] < 1e-11
    assert metrics['scale_identity_error'] < 1e-12
    assert metrics['full_prefix_error'] < 1e-12
    assert metrics['balanced_prefix_error'] < 1e-12
    assert metrics['biased_prefix_error_k1'] > 5
    assert metrics['positive_offset_mean'] > .9
