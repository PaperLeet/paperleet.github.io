import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.gan.experiment import run


def softplus(x):
    return max(x, 0) + math.log1p(math.exp(-abs(x)))


@pytest.mark.t0
def test_contract(impl):
    """The two objectives are scalar losses with preserved floating dtype."""
    x = torch.ones(2, 3, dtype=torch.float64)
    for loss in [impl.discriminator_loss(x, -x), impl.generator_loss(x)]:
        assert loss.shape == () and loss.dtype == x.dtype


@pytest.mark.t1
def test_independent_scalar_oracle(impl):
    """Equation 1 negates discriminator log-likelihood; G uses the stated non-saturating variant."""
    real, fake = torch.tensor([2., -1.], dtype=torch.float64), torch.tensor([-.4, 1., 3.], dtype=torch.float64)
    d = sum(softplus(-v) for v in real.tolist()) / 2 + sum(softplus(v) for v in fake.tolist()) / 3
    g = sum(softplus(-v) for v in fake.tolist()) / 3
    assert abs(impl.discriminator_loss(real, fake).item() - d) < 1e-12
    assert abs(impl.generator_loss(fake).item() - g) < 1e-12


@pytest.mark.t2
def test_extreme_logits_and_non_saturation(impl):
    """A confidently rejected fake still has a non-vanishing non-saturating G gradient."""
    fake = torch.tensor([-1000.], dtype=torch.float64, requires_grad=True)
    g = impl.generator_loss(fake);assert g.requires_grad;g.backward()
    assert abs(g.item() - 1000) < 1e-10
    close(fake.grad, torch.tensor([-1.], dtype=fake.dtype), rtol=0, atol=1e-12)
    real = torch.tensor([1000.], dtype=fake.dtype)
    assert impl.discriminator_loss(real, fake.detach()).item() < 1e-10


@pytest.mark.t2
def test_discriminator_analytic_gradients(impl):
    """D's real derivative is (sigmoid-1)/N_real and fake derivative sigmoid/N_fake."""
    real = torch.tensor([-.8, .3], dtype=torch.float64, requires_grad=True)
    fake = torch.tensor([-.6, .2, .9], dtype=torch.float64, requires_grad=True)
    loss = impl.discriminator_loss(real, fake);assert loss.requires_grad;loss.backward()
    assert real.grad is not None and fake.grad is not None, 'Detach generated samples before D, not D logits inside the loss'
    expected_real = torch.tensor([(1 / (1 + math.exp(-v)) - 1) / 2 for v in real.detach().tolist()], dtype=real.dtype)
    expected_fake = torch.tensor([1 / (1 + math.exp(-v)) / 3 for v in fake.detach().tolist()], dtype=fake.dtype)
    close(real.grad, expected_real, rtol=1e-12, atol=1e-12);close(fake.grad, expected_fake, rtol=1e-12, atol=1e-12)


@pytest.mark.t2
def test_generator_analytic_gradients(impl):
    """Every fake-logit derivative is negative and equals (sigmoid-1)/N_fake."""
    x = torch.tensor([-1., .1, 2.], dtype=torch.float64, requires_grad=True)
    loss = impl.generator_loss(x);assert loss.requires_grad;loss.backward()
    expected = torch.tensor([(1 / (1 + math.exp(-v)) - 1) / 3 for v in x.detach().tolist()], dtype=x.dtype)
    close(x.grad, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.t2
def test_separate_means_and_duplication(impl):
    """Replicating either batch cannot change its expectation's weight."""
    real = torch.tensor([1., -2.], dtype=torch.float64);fake = torch.tensor([.3, -.4, 2.], dtype=real.dtype)
    baseline = impl.discriminator_loss(real, fake)
    close(impl.discriminator_loss(real.repeat(3), fake), baseline, rtol=0, atol=1e-12)
    close(impl.discriminator_loss(real, fake.repeat(4)), baseline, rtol=0, atol=1e-12)
    assert abs(impl.discriminator_loss(torch.zeros(2), torch.zeros(5)).item() - 2 * math.log(2)) < 1e-6


@pytest.mark.t2
def test_numerical_derivative(impl):
    """Central differences independently check the discriminator's fake-logit derivative."""
    real = torch.tensor([.2], dtype=torch.float64)
    fake = torch.tensor([-.7, 1.2], dtype=real.dtype, requires_grad=True)
    loss = impl.discriminator_loss(real, fake);assert loss.requires_grad;loss.backward()
    expected = torch.zeros_like(fake);eps = 1e-5
    for j in range(2):
        plus, minus = fake.detach().clone(), fake.detach().clone();plus[j] += eps;minus[j] -= eps
        expected[j] = (impl.discriminator_loss(real, plus) - impl.discriminator_loss(real, minus)) / (2 * eps)
    assert fake.grad is not None;close(fake.grad, expected, rtol=1e-7, atol=1e-9)


@pytest.mark.t2
def test_scalar_strided_and_dtype(impl):
    """Losses reduce logical elements, accept scalars and preserve float32."""
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)[:, ::2];before = x.clone()
    close(impl.generator_loss(x), impl.generator_loss(x.contiguous()), rtol=1e-6, atol=1e-6)
    assert impl.generator_loss(x).dtype == x.dtype and torch.equal(x, before)
    scalar = torch.tensor(.3, dtype=torch.float64)
    assert abs(impl.generator_loss(scalar).item() - softplus(-.3)) < 1e-12


@pytest.mark.t2
def test_empty_and_dtype_errors(impl):
    """Empty expectations and mismatched or integral logits are not silently normalized."""
    for real, fake in [(torch.empty(0), torch.ones(2)), (torch.ones(2), torch.empty(0)), (torch.ones(2), torch.ones(2, dtype=torch.float64))]:
        with pytest.raises(ValueError): impl.discriminator_loss(real, fake)
    with pytest.raises(ValueError): impl.generator_loss(torch.empty(0))
    with pytest.raises(ValueError): impl.generator_loss(torch.tensor([1, 2]))


@pytest.mark.t3
def test_alternating_gaussian_game(impl, artifact):
    """Two hundred forty real D/G update pairs learn a Gaussian shift through learner losses."""
    result = run(impl.discriminator_loss, impl.generator_loss);artifact('toy_experiment', result)
    m = result['metrics']
    assert m['initial_location_error'] == 2., m
    assert m['final_location_error'] < .5, m
    assert math.isfinite(m['final_d_loss']), m
