import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.finetuning.visual_domain_prompt.experiment import run


@pytest.mark.t0
def test_canvas_shapes_and_no_mutation(impl):
    """Equation 1 adds already placed prompts without replacing or clipping original pixels."""
    x = torch.full((2, 1, 2, 3), 2., dtype=torch.float64)
    s, a = torch.ones(1, 2, 3, dtype=x.dtype), torch.full((1, 2, 3), -.5, dtype=x.dtype)
    originals = [t.clone() for t in (x, s, a)]
    y = impl.apply_prompts(x, s, a)
    assert y.shape == x.shape and y.dtype == x.dtype
    close(y, torch.full_like(x, 2.5))
    assert impl.apply_prompts(x[:0], s, a).shape == (0, 1, 2, 3)
    for t, saved in zip((x, s, a), originals):
        close(t, saved)


@pytest.mark.t1
def test_scalar_formula_oracles(impl):
    """Equations 3 and 4 sum per-domain normalized importance and all parameter penalties."""
    eta = torch.tensor([[2., 1.], [3., 4.]], dtype=torch.float64)
    delta = torch.tensor([[1., -2.], [0., 3.]], dtype=eta.dtype)
    expected = torch.tensor([2 / 1.5 + 3 / .5, 1 / 4.5 + 4 / 9.5], dtype=eta.dtype)
    close(impl.homeostatic_factor(eta, delta, xi=.5), expected)
    prompt = torch.tensor([2., -1.], dtype=eta.dtype)
    anchor = torch.tensor([1., 1.], dtype=eta.dtype)
    close(impl.homeostatic_penalty(prompt, anchor, expected, .3),
          .3 * (expected[0] + 4 * expected[1]))


@pytest.mark.t2
def test_prompt_broadcast_gradients(impl):
    """Shared canvas prompts collect gradients from every image while image gradients stay local."""
    x = torch.randn(3, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    s = torch.randn(2, 2, 2, dtype=x.dtype, requires_grad=True)
    a = torch.randn(2, 2, 2, dtype=x.dtype, requires_grad=True)
    weights = torch.arange(24, dtype=x.dtype).reshape(x.shape)
    grads = torch.autograd.grad((impl.apply_prompts(x, s, a) * weights).sum(), (x, s, a))
    close(grads[0], weights)
    close(grads[1], weights.sum(dim=0))
    close(grads[2], weights.sum(dim=0))


@pytest.mark.t2
def test_history_empty_additivity_and_sign(impl):
    """Domain history is additive, signed displacement is squared, and empty history gives zero."""
    eta = torch.rand(3, 2, 4, dtype=torch.float64).transpose(1, 2)
    delta = torch.randn(3, 2, 4, dtype=eta.dtype).transpose(1, 2)
    saved = [eta.clone(), delta.clone()]
    whole = impl.homeostatic_factor(eta, delta)
    close(whole, impl.homeostatic_factor(eta[:1], delta[:1]) + impl.homeostatic_factor(eta[1:], delta[1:]))
    close(whole, impl.homeostatic_factor(eta, -delta))
    close(impl.homeostatic_factor(eta[:0], delta[:0]), torch.zeros_like(eta[0]))
    close(impl.homeostatic_factor(eta, torch.zeros_like(delta), xi=.2), eta.sum(dim=0) / .2)
    close(eta, saved[0]); close(delta, saved[1])


@pytest.mark.t2
def test_frozen_records_and_penalty_gradient(impl):
    """Only the current DAP receives 2*alpha*Lambda*(prompt-anchor); historical records are frozen."""
    eta = torch.ones(2, 3, requires_grad=True)
    delta = torch.ones(2, 3, requires_grad=True)
    assert not impl.homeostatic_factor(eta, delta).requires_grad
    p = torch.tensor([1., 2., -1.], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([.5, -1., 2.], dtype=p.dtype, requires_grad=True)
    f = torch.tensor([2., 0., 3.], dtype=p.dtype, requires_grad=True)
    loss = impl.homeostatic_penalty(p, a, f, .4)
    assert loss.ndim == 0 and loss.dtype == p.dtype
    gp, ga, gf = torch.autograd.grad(loss, (p, a, f), allow_unused=True)
    close(gp, .8 * f.detach() * (p.detach() - a.detach()))
    assert ga is None and gf is None


@pytest.mark.t2
def test_penalty_boundaries_and_blocks(impl):
    """Zero strength/importance gives differentiable zero; independent parameter blocks add."""
    p = torch.tensor([2., -3., 4.], dtype=torch.float64, requires_grad=True)
    a, f = torch.zeros_like(p), torch.ones_like(p)
    for strength, factor in [(0., f), (2., torch.zeros_like(f))]:
        loss = impl.homeostatic_penalty(p, a, factor, strength)
        close(loss, torch.tensor(0., dtype=p.dtype))
        close(torch.autograd.grad(loss, p)[0], torch.zeros_like(p))
    close(impl.homeostatic_penalty(p, a, f, 2.),
          impl.homeostatic_penalty(p[:1], a[:1], f[:1], 2.) + impl.homeostatic_penalty(p[1:], a[1:], f[1:], 2.))


@pytest.mark.t3
def test_prompt_only_adaptation(impl, artifact):
    """Actual prompt SGD reaches the independent quadratic optima: [1,-1] and [1/9,-1]."""
    result = run(impl.apply_prompts, impl.homeostatic_factor, impl.homeostatic_penalty)
    artifact('prompt_adaptation', result)
    m = result['metrics']
    close(torch.tensor(m['unregularized_prompt']), torch.tensor([1., -1.]), atol=1e-4, rtol=1e-4)
    close(torch.tensor(m['regularized_prompt']), torch.tensor([1 / 9, -1.]), atol=1e-4, rtol=1e-4)
    assert m['unregularized_data_loss'] < 1e-8
    assert .39 < m['regularized_data_loss'] < .40
