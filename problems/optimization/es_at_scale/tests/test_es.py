import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.optimization.es_at_scale.experiment import run


@pytest.mark.t0
def test_shapes_dtype_and_purity(impl):
    """Functional ES preserves arbitrary parameter blocks, dtype and input values."""
    theta = torch.arange(6, dtype=torch.float64).reshape(2, 3).T
    noise = torch.arange(24, dtype=theta.dtype).reshape(4, 2, 3).transpose(1, 2)
    rewards = torch.tensor([3., 1., -2., 0.], dtype=theta.dtype)
    before = [x.clone() for x in (theta, noise, rewards)]
    result = impl.es_step(theta, noise, rewards, .03)
    assert result.shape == theta.shape and result.dtype == theta.dtype
    assert impl.reward_zscores(rewards).shape == rewards.shape
    for x, saved in zip((theta, noise, rewards), before):
        close(x, saved)


@pytest.mark.t1
def test_independent_scalar_oracle(impl):
    """Algorithm 1 adds the mean of standardized-reward-weighted directions."""
    r = [-3., 1., 2., 8.]
    directions = [[2., 1.], [-1., 4.], [3., -2.], [0., 5.]]
    mean = sum(r) / len(r)
    sd = math.sqrt(sum((x - mean) ** 2 for x in r) / len(r))
    z = [(x - mean) / sd for x in r]
    expected = [base + .2 * sum(z[i] * directions[i][j] for i in range(4)) / 4
                for j, base in enumerate([1., -2.])]
    theta = torch.tensor([1., -2.], dtype=torch.float64)
    close(impl.reward_zscores(torch.tensor(r, dtype=theta.dtype)), torch.tensor(z, dtype=theta.dtype))
    close(impl.es_step(theta, torch.tensor(directions, dtype=theta.dtype),
                       torch.tensor(r, dtype=theta.dtype), .2), torch.tensor(expected, dtype=theta.dtype))


@pytest.mark.t2
def test_constant_singleton_and_epsilon(impl):
    """Zero reward variance means no update; the explicit std floor controls tiny signals."""
    theta = torch.tensor(2., dtype=torch.float64)
    for n in (1, 4):
        r = torch.ones(n, dtype=theta.dtype)
        close(impl.reward_zscores(r), torch.zeros_like(r))
        close(impl.es_step(theta, torch.arange(n, dtype=theta.dtype), r, .5), theta)
    close(impl.reward_zscores(torch.tensor([0., 2e-10], dtype=theta.dtype), eps=1e-8),
          torch.tensor([-.01, .01], dtype=theta.dtype), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_population_replication_and_reward_affine_invariance(impl):
    """Duplicating a population or positively rescaling well-resolved rewards leaves the step unchanged."""
    theta, noise, rewards = torch.randn(3), torch.randn(5, 3), torch.randn(5)
    expected = impl.es_step(theta, noise, rewards, .1)
    close(impl.es_step(theta, noise.repeat(2, 1), rewards.repeat(2), .1), expected)
    close(impl.es_step(theta, noise, rewards * 3 + 7, .1), expected)
    close(impl.es_step(theta, noise, rewards, 0.), theta)


@pytest.mark.t2
def test_reward_sign_and_permutation(impl):
    """Reward maximization reverses direction when rewards are negated and ignores population order."""
    theta, noise, rewards = torch.randn(2), torch.randn(4, 2), torch.randn(4)
    p = torch.tensor([3, 1, 0, 2])
    delta = impl.es_step(theta, noise, rewards, .1) - theta
    close(impl.es_step(theta, noise, -rewards, .1) - theta, -delta)
    close(impl.es_step(theta, noise[p], rewards[p], .1) - theta, delta)


@pytest.mark.t2
def test_no_autograd_graph(impl):
    """The black-box optimizer treats weights, noise and observed rewards as fixed records."""
    theta = torch.ones(2, requires_grad=True)
    noise = torch.randn(3, 2, requires_grad=True)
    rewards = torch.tensor([1., 2., 5.], requires_grad=True)
    assert not impl.reward_zscores(rewards).requires_grad
    assert not impl.es_step(theta, noise, rewards, .1).requires_grad


@pytest.mark.t3
def test_black_box_optimization(impl, artifact):
    """100 genuine ES steps improve a two-dimensional quadratic using only candidate rewards."""
    result = run(impl.es_step)
    artifact('es_search', result)
    assert result['metrics']['initial_loss'] == 25.
    assert result['metrics']['final_loss'] < .08
