import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.batch_norm.experiment import run, scalar_normalized


def buffers(channels, dtype=torch.float64):
    return torch.zeros(channels, dtype=dtype), torch.ones(channels, dtype=dtype)


@pytest.mark.t0
def test_contract(impl):
    """BN preserves [N,C,*spatial] and returns separate channel-wise buffers."""
    x = torch.randn(3, 2, 4, dtype=torch.float64)
    y, mean, variance = impl.batch_norm(x, torch.ones(2, dtype=x.dtype), torch.zeros(2, dtype=x.dtype), *buffers(2))
    assert y.shape == x.shape and mean.shape == variance.shape == (2,)
    assert y.dtype == mean.dtype == variance.dtype == x.dtype and y.device == x.device


@pytest.mark.t1
def test_scalar_oracle(impl):
    """Algorithm 1 uses each channel's population moments across all samples."""
    x = torch.tensor([[-3., 2.], [1., 4.], [5., 12.]], dtype=torch.float64)
    weight, bias = torch.tensor([2., -.5], dtype=x.dtype), torch.tensor([1., 3.], dtype=x.dtype)
    y, _, _ = impl.batch_norm(x, weight, bias, *buffers(2), eps=.2)
    close(y, scalar_normalized(x, .2) * weight + bias, rtol=1e-12, atol=1e-12)


@pytest.mark.t2
def test_channel_moments_and_affine(impl):
    """Within each channel, normalized variance is v/(v+eps), not exactly one."""
    x = torch.arange(24, dtype=torch.float64).reshape(3, 2, 4)
    y, _, _ = impl.batch_norm(x, torch.ones(2, dtype=x.dtype), torch.zeros(2, dtype=x.dtype), *buffers(2), eps=.3)
    v = (x - x.mean(dim=(0, 2), keepdim=True)).square().mean(dim=(0, 2))
    close(y.mean(dim=(0, 2)), torch.zeros(2, dtype=x.dtype), rtol=0, atol=1e-12)
    close(y.square().mean(dim=(0, 2)), v / (v + .3), rtol=1e-12, atol=1e-12)


@pytest.mark.t2
def test_running_variance_uses_unbiased_estimator(impl):
    """Algorithm 2 corrects batch variance by m/(m-1), using all spatial values."""
    x = torch.tensor([[[1., 3.]], [[5., 7.]]], dtype=torch.float64)
    for momentum in (0., .25, 1.):
        _, mean, variance = impl.batch_norm(x, torch.ones(1, dtype=x.dtype), torch.zeros(1, dtype=x.dtype), *buffers(1), momentum=momentum)
        close(mean, torch.tensor([4 * momentum], dtype=x.dtype), rtol=0, atol=1e-12)
        close(variance, torch.tensor([1 - momentum + momentum * 20 / 3], dtype=x.dtype), rtol=0, atol=1e-12)


@pytest.mark.t2
def test_frozen_inference_is_companion_independent(impl):
    """Inference uses fixed population estimates, even for a one-sample batch."""
    x = torch.tensor([[1., 3.], [101., -97.]], dtype=torch.float64)
    mean, variance = torch.tensor([2., 1.], dtype=x.dtype), torch.tensor([4., 9.], dtype=x.dtype)
    args = (torch.tensor([2., -3.], dtype=x.dtype), torch.tensor([1., 2.], dtype=x.dtype), mean, variance)
    all_y, next_mean, next_var = impl.batch_norm(x, *args, training=False, eps=.1)
    alone = impl.batch_norm(x[:1], *args, training=False, eps=.1)[0]
    close(alone, all_y[:1], rtol=0, atol=0)
    close(all_y, (x - mean) / torch.sqrt(variance + .1) * args[0] + args[1], rtol=1e-12, atol=1e-12)
    close(next_mean, mean, rtol=0, atol=0)
    close(next_var, variance, rtol=0, atol=0)


@pytest.mark.t2
def test_training_jacobian(impl):
    """Differentiating the mean and variance gives the BN centering Jacobian."""
    x = torch.tensor([[1., -2.], [3., 1.], [8., 4.]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([1.3, -.7], dtype=x.dtype, requires_grad=True)
    bias = torch.tensor([.1, .2], dtype=x.dtype, requires_grad=True)
    upstream = torch.tensor([[1., 2.], [-3., .5], [.25, -2.]], dtype=x.dtype)
    output = impl.batch_norm(x, weight, bias, *buffers(2), eps=.03)[0]
    (output * upstream).sum().backward()
    centered = x.detach() - x.detach().mean(dim=0)
    inverse = torch.rsqrt(centered.square().mean(dim=0) + .03)
    normalized, q = centered * inverse, upstream * weight.detach()
    expected = inverse * (q - q.mean(dim=0) - normalized * (q * normalized).mean(dim=0))
    for value, derivative in [(x, expected), (weight, (upstream * normalized).sum(dim=0)), (bias, upstream.sum(dim=0))]:
        assert value.grad is not None, 'Differentiable batch moments and affine parameters are required'
        close(value.grad, derivative, rtol=1e-9, atol=1e-10)


@pytest.mark.t2
def test_finite_difference_gradients(impl):
    """Small double-precision perturbations independently verify all three live inputs."""
    x = torch.tensor([[1., 3.], [-2., 2.]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([.5, 1.3], dtype=x.dtype, requires_grad=True)
    bias = torch.zeros(2, dtype=x.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, w, b: impl.batch_norm(a, w, b, *buffers(2), eps=.2)[0], (x, weight, bias))


@pytest.mark.t2
def test_buffers_are_detached_and_inputs_unmodified(impl):
    """Moving estimates are state, never an extra differentiable training path."""
    x = torch.tensor([[1., 2.], [3., 7.]], dtype=torch.float64, requires_grad=True)
    mean, variance = (v.requires_grad_() for v in buffers(2))
    values = [x, mean, variance]
    saved = [v.detach().clone() for v in values]
    output, next_mean, next_var = impl.batch_norm(x, torch.ones(2, dtype=x.dtype), torch.zeros(2, dtype=x.dtype), mean, variance)
    assert not next_mean.requires_grad and not next_var.requires_grad
    for value, before in zip(values, saved):
        close(value.detach(), before, rtol=0, atol=0)
    eval_y = impl.batch_norm(x, torch.ones(2, dtype=x.dtype), torch.zeros(2, dtype=x.dtype), mean, variance, False)[0]
    eval_y.sum().backward()
    assert mean.grad is None and variance.grad is None, 'Inference buffers must remain detached'


@pytest.mark.t2
def test_strides_constant_channels_and_empty_inference(impl):
    """Spatial grouping respects channel layout and epsilon handles constant channels."""
    x = torch.arange(48, dtype=torch.float32).reshape(3, 2, 8)[:, :, ::2]
    weight, bias = torch.ones(2), torch.zeros(2)
    y = impl.batch_norm(x, weight, bias, *buffers(2, x.dtype))[0]
    for c in range(2):
        flat = x[:, c].reshape(-1)
        expected = (x[:, c] - flat.mean()) / torch.sqrt((flat - flat.mean()).square().mean() + 1e-5)
        close(y[:, c], expected, rtol=1e-5, atol=1e-6)
    constant = impl.batch_norm(torch.ones(3, 2), weight, bias, *buffers(2, torch.float32))[0]
    close(constant, torch.zeros_like(constant), rtol=0, atol=0)
    assert impl.batch_norm(x[:0], weight, bias, *buffers(2, x.dtype), training=False)[0].shape == (0, 2, 4)


@pytest.mark.t2
def test_invalid_training_and_parameters(impl):
    """The declared estimator needs m>1 and finite, valid epsilon and momentum."""
    args = (torch.ones(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64), *buffers(2))
    with pytest.raises(ValueError):
        impl.batch_norm(torch.ones(1, 2, dtype=torch.float64), *args)
    for options in ({'eps': 0}, {'eps': math.inf}, {'momentum': -.1}, {'momentum': 1.1}):
        with pytest.raises(ValueError):
            impl.batch_norm(torch.ones(2, 2, dtype=torch.float64), *args, **options)


@pytest.mark.t3
def test_affine_training_and_frozen_population(impl, artifact):
    """Eighty sampled minibatches fit live affine parameters and frozen inference stays deterministic."""
    result = run(impl.batch_norm)
    artifact('batch_norm_training', result)
    metrics = result['metrics']
    assert metrics['final_mse'] < .003, f"final affine MSE: expected < .003, got {metrics['final_mse']}"
    assert max(metrics['scale_error'], metrics['bias_error']) < .04, metrics
    assert metrics['inference_companion_change'] < 1e-12, metrics
