"""NumPy-only Tensor callback workflows, including full training updates."""
import gc
import weakref

import numpy as np
import pytest
from torchwasm import UnsupportedOperation, torch


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_hooked_training_and_inline_sgd_match_independent_numpy(dtype):
    kind = getattr(torch, dtype); rng = np.random.default_rng(71)
    inputs = rng.normal(size=(8, 3)).astype(dtype); targets = rng.normal(size=(8, 1)).astype(dtype)
    weight = rng.normal(size=(1, 3)).astype(dtype); bias = np.array([.2], dtype=dtype)
    model = torch.nn.Linear(3, 1, dtype=kind)
    with torch.no_grad(): model.weight.copy_(torch.tensor(weight)); model.bias.copy_(torch.tensor(bias))
    model.weight.register_hook(lambda g: g * .5)
    model.bias.register_hook(lambda g: g.clamp(-.3, .3))
    updates = []
    def update(p):
        assert not torch.is_grad_enabled()
        p.add_(p.grad, alpha=-.02); p.grad = None; updates.append(p.shape)
    for parameter in model.parameters(): parameter.register_post_accumulate_grad_hook(update)
    x, y = torch.tensor(inputs), torch.tensor(targets)
    for _ in range(5):
        residual = inputs @ weight.T + bias - targets
        expected = np.mean(residual ** 2)
        gw = (residual.T @ inputs) * np.asarray(2 / len(inputs), dtype=dtype) * .5
        gb = np.clip(residual.mean(0) * 2, -.3, .3)
        loss = (model(x) - y).square().mean(); loss.backward()
        np.testing.assert_allclose(loss.item(), expected, rtol=3e-6)
        weight -= .02 * gw; bias -= .02 * gb
        np.testing.assert_allclose(model.weight.detach().numpy(), weight, rtol=3e-6, atol=2e-7)
        np.testing.assert_allclose(model.bias.detach().numpy(), bias, rtol=3e-6, atol=2e-7)
        assert model.weight.grad is model.bias.grad is None
    assert len(updates) == 10 and torch.is_grad_enabled()


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_intermediate_hook_and_retained_gradient_routing(dtype):
    x = torch.tensor([2., 3.], dtype=getattr(torch, dtype), requires_grad=True)
    middle = x.square(); middle.retain_grad(); seen = []
    middle.register_hook(lambda g: g * 2)
    x.register_hook(lambda g: seen.append(g.tolist()) or g * 3)
    value = middle.square().sum()
    gradient = torch.autograd.grad(value, x, retain_graph=True)[0]
    np.testing.assert_array_equal(gradient.numpy(), [192., 648.]); assert x.grad is None
    np.testing.assert_array_equal(middle.grad.numpy(), [16., 36.])
    value.backward(); np.testing.assert_array_equal(x.grad.numpy(), [192., 648.])
    np.testing.assert_array_equal(middle.grad.numpy(), [32., 72.])
    assert seen == [[64., 216.], [64., 216.]]
    with pytest.raises(RuntimeError, match='freed|second time'): value.backward()


def test_removed_hook_and_snapshot_registration_have_bounded_lifetimes():
    x = torch.tensor(2., requires_grad=True); middle = x * 2; seen = []
    handle = middle.register_hook(lambda g: seen.append(g.item()) or g * 3)
    result = middle + 1; reference = weakref.ref(middle); del middle; gc.collect()
    assert reference() is None
    result.backward(); assert x.grad.item() == 6. and seen == [1.]
    handle.remove(); handle.remove()
    events = []; handles = []
    def first(g): events.append('first'); handles[1].remove()
    handles.extend([x.register_hook(first), x.register_hook(lambda g: events.append('second'))])
    x.backward(); x.backward(); assert events == ['first', 'second', 'first']


def test_invalid_replacement_and_exception_leave_mode_and_graph_usable():
    x = torch.tensor([2., 3.], requires_grad=True); output = x.sum()
    with x.register_hook(lambda g: g.double()):
        with pytest.raises(RuntimeError, match='dtype'): output.backward()
    def fail(g): torch.set_grad_enabled(True); raise ValueError('callback failed')
    with x.register_hook(fail), torch.no_grad():
        with pytest.raises(ValueError, match='callback failed'): output.backward()
        assert not torch.is_grad_enabled()
    output.backward(); assert x.grad.tolist() == [1., 1.] and torch.is_grad_enabled()


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_reverse_over_forward_leaf_callback_keeps_analytic_gradient(dtype):
    x = torch.tensor(2., dtype=getattr(torch, dtype), requires_grad=True); seen = []
    x.register_hook(lambda g: seen.append(g.item()) or g * 2)
    primal, tangent = torch.func.jvp(lambda p: p.square(), (x,), (torch.tensor(3., dtype=getattr(torch, dtype)),))
    assert not seen and torch.autograd.grad(primal + tangent, x)[0].item() == 20.
    assert seen == [10.]


def test_requested_hook_result_keeps_its_explicitly_created_graph():
    x = torch.tensor(2., requires_grad=True); saved = []
    def replace(g):
        with torch.enable_grad(): value = x.square()
        saved.append(value); return value
    handle = x.register_hook(replace)
    gradient = torch.autograd.grad(x * 3, x)[0]
    assert gradient is saved[0] and gradient.requires_grad
    handle.remove(); assert torch.autograd.grad(gradient, x)[0].item() == 4.


def test_weak_handle_does_not_keep_leaf_alive():
    # Public weak-handle ownership is also observable without native Torch.
    x = torch.tensor(2., requires_grad=True)
    handle = x.register_hook(lambda g: g * 2); reference = weakref.ref(x)
    del x; gc.collect(); assert reference() is None
    handle.remove()


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_no_grad_copy_shares_views_and_guards_saved_state(dtype):
    x = torch.arange(12, dtype=getattr(torch, dtype)).reshape(3, 4).requires_grad_()
    view = x[:, 1:]; loss = x.square().sum(); before = x._version
    with torch.no_grad(): assert view.copy_(torch.tensor(5.)) is view
    expected = np.arange(12, dtype=dtype).reshape(3, 4); expected[:, 1:] = 5.
    np.testing.assert_array_equal(x.detach().numpy(), expected)
    assert x._version == before + 1
    with pytest.raises(RuntimeError):
        with torch.no_grad(): x[:, 1:].copy_(x[:, :-1])
    with pytest.raises(UnsupportedOperation, match='modified in-place'): loss.backward()
    boolean = torch.zeros(2, dtype=torch.bool)
    boolean.copy_(torch.tensor([0+2j, 0j], dtype=torch.complex64)); assert boolean.tolist() == [True, False]
    target = torch.tensor([5., 6.]); version = target._version
    with pytest.raises(TypeError): torch.tensor(np.array([1, object()], dtype=object))
    assert target.tolist() == [5., 6.] and target._version == version
    cast = torch.zeros(4, dtype=torch.float32)
    wide = torch.tensor([1e300, 1e-300, float('nan'), float('inf')], dtype=torch.float64)
    with np.errstate(all='raise'): cast.copy_(wide)
    assert cast._version == 1
    np.testing.assert_array_equal(cast.numpy(), [float('inf'), 0., float('nan'), float('inf')])
