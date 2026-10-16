import math
import pytest
import torch
import torch.nn.functional as F
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.group_norm.experiment import run, scalar_expected


@pytest.mark.t0
def test_shape_dtype_strides_empty_batch_and_input_preservation(impl):
    """Preserve every axis and floating dtype; reshape must also handle strided inputs and an empty batch."""
    for dtype in (torch.float32, torch.float64):
        for shape in [(2, 6, 3), (1, 6, 2, 3), (2, 6, 2, 2, 3), (0, 6, 2, 3)]:
            x = torch.randn(*shape, dtype=dtype)
            if x.ndim >= 4:
                x = x.transpose(-1, -2)
            weight, bias = torch.randn(6, dtype=dtype), torch.randn(6, dtype=dtype)
            before = [value.clone() for value in (x, weight, bias)]
            output = impl.group_norm(x, 3, weight, bias)
            assert output.shape == x.shape and output.dtype == dtype and output.device == x.device
            assert torch.isfinite(output).all()
            for actual, saved in zip((x, weight, bias), before):
                close(actual, saved, atol=0., rtol=0.)


@pytest.mark.t1
def test_hand_computed_contiguous_groups_and_per_channel_affine(impl):
    """The two four-value groups have means 3 and 11 and population variance 5, before channel-specific affine."""
    x = torch.arange(0., 16., 2., dtype=torch.float64).reshape(1, 4, 1, 2)
    weight, bias = torch.tensor([-2., .5, 1.5, 0.], dtype=x.dtype), torch.tensor([3., 1., -2., 7.], dtype=x.dtype)
    expected = torch.tensor([[(v - (3. if c < 2 else 11.)) / math.sqrt(5.25) * w + b for v in [4. * c, 4. * c + 2.]]
                             for c, (w, b) in enumerate(zip(weight.tolist(), bias.tolist()))], dtype=x.dtype).reshape(x.shape)
    close(impl.group_norm(x, 2, weight, bias, .25), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_independent_scalar_oracle_across_ranks_and_groups(impl):
    """Scalar member collection independently defines every sample/channel/spatial output."""
    for shape in [(2, 6, 3), (2, 6, 2, 3), (1, 6, 2, 2, 2)]:
        x = torch.randn(*shape, dtype=torch.float64)
        weight, bias = torch.randn(6, dtype=x.dtype), torch.randn(6, dtype=x.dtype)
        for groups in (1, 2, 3, 6):
            close(impl.group_norm(x, groups, weight, bias, .03), scalar_expected(x, groups, weight, bias, .03), atol=2e-12, rtol=2e-12)


@pytest.mark.t2
def test_library_values_and_all_three_gradients(impl):
    """Native GroupNorm is a separate oracle for outputs and input/scale/bias gradients."""
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(2, 6, 3, 2, dtype=dtype).transpose(-1, -2).requires_grad_()
        weight, bias = torch.randn(6, dtype=dtype, requires_grad=True), torch.randn(6, dtype=dtype, requires_grad=True)
        a, w, b = [value.detach().clone().requires_grad_() for value in (x, weight, bias)]
        output, expected = impl.group_norm(x, 3, weight, bias, .01), F.group_norm(a, 3, w, b, .01)
        tolerance = 2e-5 if dtype == torch.float32 else 2e-11
        close(output, expected, atol=tolerance, rtol=tolerance)
        upstream = torch.randn_like(x)
        actual_grads = torch.autograd.grad((output * upstream).sum(), (x, weight, bias))
        expected_grads = torch.autograd.grad((expected * upstream).sum(), (a, w, b))
        for actual, reference in zip(actual_grads, expected_grads):
            close(actual, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.t2
def test_other_samples_and_other_groups_cannot_change_the_anchor(impl):
    """A sample's group statistics exclude every companion sample and every other channel group."""
    x = torch.randn(2, 6, 2, 3, dtype=torch.float64)
    weight, bias = torch.randn(6, dtype=x.dtype), torch.randn(6, dtype=x.dtype)
    original = impl.group_norm(x, 3, weight, bias)
    changed = x.clone(); changed[1] = changed[1] * 20 + 100; changed[0, 2:] += 50
    close(impl.group_norm(changed, 3, weight, bias)[0, :2], original[0, :2], atol=1e-12, rtol=1e-12)
    close(impl.group_norm(x[:1], 3, weight, bias), original[:1], atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_layer_and_instance_endpoints_include_spatial_axes(impl):
    """G=1 shares all channel/spatial statistics; G=C shares only a channel's spatial statistics."""
    x = torch.tensor([[[0., 1., 3.], [4., 9., 12.], [-5., -3., 8.], [20., 22., 30.]]], dtype=torch.float64)
    weight, bias, eps = torch.ones(4, dtype=x.dtype), torch.zeros(4, dtype=x.dtype), .02
    close(impl.group_norm(x, 1, weight, bias, eps), F.layer_norm(x, (4, 3), eps=eps), atol=1e-12, rtol=1e-12)
    instance = impl.group_norm(x, 4, weight, bias, eps)
    close(instance, scalar_expected(x, 4, weight, bias, eps), atol=1e-12, rtol=1e-12)
    close(instance.mean(dim=-1), torch.zeros(1, 4, dtype=x.dtype), atol=1e-12, rtol=1e-12)
    assert (instance - impl.group_norm(x, 1, weight, bias, eps)).abs().max().item() > .1


@pytest.mark.t2
def test_constant_tiny_and_singleton_groups_follow_the_epsilon_formula(impl):
    """Positive epsilon belongs inside sqrt; a one-value group returns bias with zero input and scale gradients."""
    weight = torch.tensor([2., -1., .5, 3.], dtype=torch.float64)
    bias = torch.tensor([3., 4., 5., 6.], dtype=weight.dtype)
    for scale in (0., 1e-8, 1e-4):
        x = (torch.arange(8, dtype=weight.dtype).reshape(1, 4, 2) - 2) * scale + .01
        close(impl.group_norm(x, 2, weight, bias, 1e-4), scalar_expected(x, 2, weight, bias, 1e-4), atol=1e-12, rtol=1e-12)
    x = torch.randn(1, 4, 1, dtype=weight.dtype, requires_grad=True)
    w, b = weight.clone().requires_grad_(), bias.clone().requires_grad_()
    output = impl.group_norm(x, 4, w, b)
    close(output, b.reshape(1, 4, 1), atol=1e-12, rtol=1e-12)
    dx, dw, db = torch.autograd.grad(output.sum(), (x, w, b))
    close(dx, torch.zeros_like(x), atol=1e-10, rtol=0.)
    close(dw, torch.zeros_like(w), atol=1e-12, rtol=0.)
    close(db, torch.ones_like(b), atol=1e-12, rtol=0.)


@pytest.mark.t2
def test_analytical_group_jacobian_and_channel_affine_gradients(impl):
    """An independent scalar Jacobian checks statistics coupling; no library normalization or autodiff is used for expected gradients."""
    x = torch.tensor([[[-2., 1., 4.], [0., 3., 8.], [-5., 2., 7.], [1., 6., 9.]]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([1., -2., .5, 3.], dtype=x.dtype, requires_grad=True)
    bias = torch.tensor([2., 0., -1., 4.], dtype=x.dtype, requires_grad=True)
    upstream = torch.tensor([[[.2, -1., 2.], [3., .7, -2.], [1., -3., .5], [-.2, 2., 4.]]], dtype=x.dtype)
    eps = .04
    output = impl.group_norm(x, 2, weight, bias, eps)
    dx, dw, db = torch.autograd.grad((output * upstream).sum(), (x, weight, bias))
    values, seeds, scales = x.detach().reshape(4, 3).tolist(), upstream.reshape(4, 3).tolist(), weight.detach().tolist()
    expected_dx, expected_dw, expected_db = [], [], []
    for first in (0, 2):
        members = [v for c in range(first, first + 2) for v in values[c]]
        mean = math.fsum(members) / 6
        inverse = 1 / math.sqrt(math.fsum((v - mean)**2 for v in members) / 6 + eps)
        z = [(v - mean) * inverse for v in members]
        q = [u * scales[c] for c in range(first, first + 2) for u in seeds[c]]
        mean_q, mean_qz = math.fsum(q) / 6, math.fsum(a * b for a, b in zip(q, z)) / 6
        expected_dx.extend([(a - mean_q - b * mean_qz) * inverse for a, b in zip(q, z)])
        for offset, c in enumerate(range(first, first + 2)):
            expected_dw.append(math.fsum(u * v for u, v in zip(seeds[c], z[offset * 3:offset * 3 + 3])))
            expected_db.append(math.fsum(seeds[c]))
    for actual, expected in [(dx, torch.tensor(expected_dx, dtype=x.dtype).reshape(x.shape)),
                             (dw, torch.tensor(expected_dw, dtype=x.dtype)), (db, torch.tensor(expected_db, dtype=x.dtype))]:
        close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.t2
def test_all_operand_gradients_match_finite_differences(impl):
    """Finite differences independently check gradients to activations, scale and bias."""
    x = torch.randn(1, 4, 2, dtype=torch.float64, requires_grad=True)
    weight, bias = torch.randn(4, dtype=x.dtype, requires_grad=True), torch.randn(4, dtype=x.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, w, b: impl.group_norm(a, 2, w, b, .05),
                                    (x, weight, bias), eps=1e-6, atol=1e-6, rtol=1e-5, raise_exception=False)


@pytest.mark.t2
def test_invalid_group_counts_are_explicit_errors(impl):
    """The group count is a positive integer divisor of C; bool is not a group count."""
    x = torch.randn(2, 6, 2, dtype=torch.float64)
    weight, bias = torch.ones(6, dtype=x.dtype), torch.zeros(6, dtype=x.dtype)
    for invalid in (0, -1, 4, 7, True, False, 2.0, '2', None):
        with pytest.raises(ValueError):
            impl.group_norm(x, invalid, weight, bias)


@pytest.mark.t3
def test_batch_context_and_group_choice_experiment(impl, artifact):
    """Call the learner's implementation in a controlled experiment; changing companions must not change the anchor's GroupNorm output."""
    result = run(impl.group_norm)
    artifact('group_norm_batch_context', result)
    assert result['metrics']['maximum_scalar_error'] < 1e-11
    assert result['metrics']['maximum_group_context_change'] < 1e-11
    assert result['metrics']['minimum_batch_context_change_with_companions'] > .1
