import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.vae.experiment import run


@pytest.mark.t0
def test_contract(impl):
    """Samples are [B,Z], positive divergences are [B], and the objective is scalar."""
    mu, logvar, noise = (torch.randn(3, 2, dtype=torch.float64) for _ in range(3))
    x, logits = torch.ones(3, 4, dtype=mu.dtype), torch.zeros(3, 4, dtype=mu.dtype)
    sample = impl.reparameterize(mu, logvar, noise)
    kl = impl.gaussian_kl(mu, logvar)
    loss = impl.negative_elbo(logits, x, mu, logvar)
    assert sample.shape == mu.shape and kl.shape == (3,) and loss.shape == ()
    assert sample.dtype == kl.dtype == loss.dtype == mu.dtype and sample.device == mu.device


@pytest.mark.t1
def test_independent_scalar_elbo(impl):
    """Equation (10) combines exact diagonal-Gaussian KL and Bernoulli log likelihood."""
    mu = torch.tensor([[1., -2.], [.5, -.25]], dtype=torch.float64)
    logvar = torch.tensor([[math.log(4), math.log(.5)], [.2, -.3]], dtype=mu.dtype)
    logits = torch.tensor([[2., -1., .3], [-4., 1., .7]], dtype=mu.dtype)
    target = torch.tensor([[1., 0., 1.], [0., 1., 0.]], dtype=mu.dtype)
    divergences, totals = [], []
    for n in range(2):
        divergence = .5 * math.fsum(mu[n, j].item()**2 + math.exp(logvar[n, j].item()) - 1 - logvar[n, j].item() for j in range(2))
        reconstruction = math.fsum(max(value, 0.) + math.log1p(math.exp(-abs(value))) - target[n, j].item() * value for j, value in enumerate(logits[n].tolist()))
        divergences.append(divergence)
        totals.append(reconstruction + divergence)
    close(impl.gaussian_kl(mu, logvar), torch.tensor(divergences, dtype=mu.dtype), rtol=1e-12, atol=1e-12)
    close(impl.negative_elbo(logits, target, mu, logvar), torch.tensor(sum(totals) / 2, dtype=mu.dtype), rtol=1e-12, atol=1e-12)


@pytest.mark.t2
def test_reparameterization_pathwise_gradient(impl):
    """The posterior standard deviation is exp(logvar/2); noise is an explicit input."""
    mu = torch.tensor([[1., -2.]], dtype=torch.float64, requires_grad=True)
    logvar = torch.tensor([[math.log(4), math.log(9)]], dtype=mu.dtype, requires_grad=True)
    noise = torch.tensor([[.5, -1.]], dtype=mu.dtype, requires_grad=True)
    sample = impl.reparameterize(mu, logvar, noise)
    close(sample, torch.tensor([[2., -5.]], dtype=mu.dtype), rtol=0, atol=1e-12)
    sample.sum().backward()
    for value, expected in [(mu, [[1., 1.]]), (logvar, [[.5, -1.5]]), (noise, [[2., 3.]])]:
        assert value.grad is not None, 'Reparameterization must preserve every pathwise gradient'
        close(value.grad, torch.tensor(expected, dtype=mu.dtype), rtol=1e-12, atol=1e-12)


@pytest.mark.t2
def test_prior_kl_and_known_shift(impl):
    """KL to the same standard normal is zero; mean displacement costs ||mu||²/2."""
    mu = torch.tensor([[0., 0.], [3., 4.]], dtype=torch.float64)
    close(impl.gaussian_kl(mu, torch.zeros_like(mu)), torch.tensor([0., 12.5], dtype=mu.dtype), rtol=0, atol=0)
    spread = impl.gaussian_kl(torch.zeros_like(mu), torch.ones_like(mu))
    assert (spread > 0).all(), 'Non-unit covariance has strictly positive KL'


@pytest.mark.t2
def test_elbo_closed_form_gradients(impl):
    """Independent derivatives distinguish the minimization sign, event sum and KL gradients."""
    logits = torch.tensor([[.3, -1., 2.], [-.2, .7, 1.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[1., 0., 1.], [0., 1., 0.]], dtype=logits.dtype)
    mu = torch.tensor([[.2, -.4], [.8, -.1]], dtype=logits.dtype, requires_grad=True)
    logvar = torch.tensor([[.1, -.3], [.5, .2]], dtype=logits.dtype, requires_grad=True)
    impl.negative_elbo(logits, target, mu, logvar).backward()
    for value, expected in [(logits, (logits.detach().sigmoid() - target) / 2),
                            (mu, mu.detach() / 2), (logvar, .25 * (logvar.detach().exp() - 1))]:
        assert value.grad is not None, 'Both reconstruction and KL must contribute gradients'
        close(value.grad, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.t2
def test_extreme_logits_are_stable(impl):
    """Logit-space Bernoulli likelihood remains finite at magnitudes of 1000."""
    logits = torch.tensor([[1000., -1000.], [1000., -1000.]], dtype=torch.float64)
    target = torch.tensor([[1., 0.], [0., 1.]], dtype=logits.dtype)
    mu = torch.zeros(2, 2, dtype=logits.dtype)
    close(impl.negative_elbo(logits, target, mu, mu), torch.tensor(1000., dtype=logits.dtype), rtol=0, atol=1e-12)


@pytest.mark.t2
def test_event_sum_and_batch_mean(impl):
    """Duplicating the batch preserves the loss, while duplicating observations doubles NLL."""
    mu = torch.tensor([[.2, .4]], dtype=torch.float64)
    lv = torch.zeros_like(mu)
    logits, target = torch.zeros(1, 3, dtype=mu.dtype), torch.ones(1, 3, dtype=mu.dtype)
    base = impl.negative_elbo(logits, target, mu, lv)
    close(impl.negative_elbo(logits.repeat(3, 1), target.repeat(3, 1), mu.repeat(3, 1), lv.repeat(3, 1)), base, rtol=0, atol=1e-12)
    doubled = impl.negative_elbo(logits.repeat(1, 2), target.repeat(1, 2), mu, lv)
    close(doubled - base, torch.tensor(3 * math.log(2), dtype=mu.dtype), rtol=0, atol=1e-12)


@pytest.mark.t2
def test_strides_float32_and_no_mutation(impl):
    """Sampling is pure and supports noncontiguous float32 tensors and empty batches."""
    mu = torch.arange(12, dtype=torch.float32).reshape(2, 6)[:, ::2]
    before = mu.clone()
    lv, noise = torch.zeros_like(mu), torch.ones_like(mu)
    close(impl.reparameterize(mu, lv, noise), mu + 1, rtol=0, atol=0)
    close(mu, before, rtol=0, atol=0)
    assert impl.gaussian_kl(mu, lv).dtype == torch.float32
    assert impl.reparameterize(mu[:0], lv[:0], noise[:0]).shape == (0, 3)


@pytest.mark.t2
def test_invalid_shapes_are_explicit(impl):
    """Broadcasting a shared noise row or averaging an empty batch violates the contract."""
    with pytest.raises(ValueError):
        impl.reparameterize(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(1, 3))
    with pytest.raises(ValueError):
        impl.negative_elbo(torch.zeros(0, 3), torch.zeros(0, 3), torch.zeros(0, 2), torch.zeros(0, 2))
    with pytest.raises(ValueError):
        impl.gaussian_kl(torch.zeros(2, 3), torch.zeros(2, 1))


@pytest.mark.t3
def test_tiny_bernoulli_vae_training(impl, artifact):
    """All learner functions participate in 120 actual encoder/decoder updates."""
    result = run(impl.reparameterize, impl.gaussian_kl, impl.negative_elbo)
    artifact('vae_training', result)
    metrics = result['metrics']
    assert metrics['final_20_mean_negative_elbo'] < .8 * metrics['initial_negative_elbo'], metrics
    assert metrics['maximum_decomposition_error'] < 1e-10, metrics
    assert metrics['final_20_mean_kl'] >= 0, metrics
