import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.dropout.experiment import run


@pytest.mark.t0
def test_contract(impl):
    """Explicit-mask dropout preserves floating shape, device and dtype."""
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    y = impl.inverted_dropout(x, .2, torch.ones_like(x, dtype=torch.bool))
    assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device


@pytest.mark.t1
def test_known_keep_mask(impl):
    """Paper Section 10's inverted convention scales kept units by reciprocal keep probability."""
    x = torch.tensor([[1., -2., 3.], [4., 5., -6.]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    expected = torch.tensor([[2., 0., 6.], [0., 10., 0.]], dtype=x.dtype)
    close(impl.inverted_dropout(x, .5, mask), expected, rtol=0, atol=0)


@pytest.mark.t2
def test_mask_jacobian(impl):
    """A realized keep mask makes the Jacobian diagonal with entries mask/(1-p)."""
    x = torch.tensor([[.2, -.7], [1.3, -2.]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False], [False, True]])
    upstream = torch.tensor([[2., -1.], [3., .5]], dtype=x.dtype)
    y = impl.inverted_dropout(x, .25, mask);assert y.requires_grad
    (y * upstream).sum().backward()
    expected = torch.tensor([[2 / .75, 0.], [0., .5 / .75]], dtype=x.dtype)
    close(x.grad, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.t2
def test_each_position_and_no_mutation(impl):
    """Element-wise masks must not be shared across rows or mutate either input."""
    x = torch.arange(1, 7, dtype=torch.float64).reshape(2, 3)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    a, b = x.clone(), mask.clone()
    y = impl.inverted_dropout(x, .2, mask)
    assert torch.equal(y != 0, mask) and torch.equal(x, a) and torch.equal(mask, b)
    close(y[mask], a[mask] / .8, rtol=1e-12, atol=1e-12)


@pytest.mark.t2
def test_inference_ignores_mask(impl):
    """Inference is exactly identity, without rescaling or consuming a keep mask."""
    x = torch.tensor([-2., 0., 3.], dtype=torch.float64, requires_grad=True)
    for p in (0., .3, 1.):
        y = impl.inverted_dropout(x, p, training=False)
        close(y, x, rtol=0, atol=0)
    y = impl.inverted_dropout(x, .5, torch.zeros(1, dtype=torch.bool), training=False)
    close(y, x, rtol=0, atol=0);y.sum().backward()
    close(x.grad, torch.ones_like(x), rtol=0, atol=0)


@pytest.mark.t2
def test_probability_endpoints_connected_gradients(impl):
    """No-drop is identity; drop-all is graph-connected zero rather than division by zero."""
    for p, expected in [(0., 1.), (1., 0.)]:
        x = torch.tensor([-2., 3.], dtype=torch.float64, requires_grad=True)
        y = impl.inverted_dropout(x, p)
        assert y.requires_grad;close(y, x * expected, rtol=0, atol=0)
        y.sum().backward();close(x.grad, torch.full_like(x, expected), rtol=0, atol=0)


@pytest.mark.t2
def test_bernoulli_mean_and_variance(impl):
    """Inverted Bernoulli noise has mean one and variance p/(1-p)."""
    rng = torch.Generator().manual_seed(211)
    x = torch.tensor([1., -2.], dtype=torch.float64).expand(8192, 2)
    y = impl.inverted_dropout(x, .25, torch.rand(8192, 2, generator=rng) >= .25)
    close(y.mean(dim=0), x[0], rtol=0, atol=.06)
    variance = (y - y.mean(dim=0)).square().mean(dim=0)
    close(variance, x[0].square() / 3, rtol=0, atol=.08)


@pytest.mark.t2
def test_strided_scalar_empty_and_dtype(impl):
    """Logical strides, zero-sized arrays and scalar masks follow the same element-wise rule."""
    for x in [torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2], torch.empty(0, 3, dtype=torch.float32), torch.tensor(3., dtype=torch.float32)]:
        mask = torch.ones_like(x, dtype=torch.bool)
        y = impl.inverted_dropout(x, .5, mask)
        assert y.dtype == x.dtype and y.shape == x.shape
        close(y, x * 2, rtol=0, atol=0)


@pytest.mark.t2
def test_invalid_probability_or_mask(impl):
    """Reject non-probabilities and implicit mask broadcasting for stochastic training."""
    x = torch.ones(2, 3)
    for p in [-.1, 1.1, float('inf'), float('nan')]:
        with pytest.raises(ValueError): impl.inverted_dropout(x, p)
    for mask in [None, torch.ones(1, 3, dtype=torch.bool), torch.ones(2, 3)]:
        with pytest.raises(ValueError): impl.inverted_dropout(x, .5, mask)


@pytest.mark.t3
def test_dropout_regression_and_sampling(impl, artifact):
    """One hundred Adam steps use submitted dropout; actual Bernoulli samples check its moments."""
    result = run(impl.inverted_dropout);artifact('toy_experiment', result);m = result['metrics']
    assert m['final_mse'] < .2 * m['initial_mse'], m
    assert m['mc_mean_max_error'] < .06 and m['mc_variance_max_error'] < .08, m
