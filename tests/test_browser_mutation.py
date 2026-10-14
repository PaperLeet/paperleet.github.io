"""Native parity for parameter updates and rejection of stale backward storage."""
import gc

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from paperleet.browser import numpy_autograd as wasm


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('operation', ['__iadd__', '__isub__', '__imul__', '__itruediv__'])
def test_no_grad_leaf_update_preserves_identity_aliases_and_future_gradients(dtype, operation):
    data = np.array([[.3, .7], [1.2, -.5]], dtype=dtype)
    actual, expected = wasm.Tensor(data.copy(), requires_grad=True), torch.tensor(data, requires_grad=True)
    alias, native_alias = actual.detach().T, expected.detach().T
    identity = id(actual)
    for _ in range(4):
        actual.square().sum().backward(); expected.square().sum().backward()
        with wasm.no_grad(), torch.no_grad():
            getattr(actual, operation)(actual.grad * .03 + 1.)
            getattr(expected, operation)(expected.grad * .03 + 1.)
        assert id(actual) == identity and actual.requires_grad and not actual._parents
        np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(alias.numpy(), native_alias.numpy(), rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(actual.grad.numpy(), expected.grad.numpy(), rtol=2e-6, atol=2e-6)
    # Existing .grad accumulation survives the update; the next forward uses new values.
    actual.grad = None; expected.grad = None
    actual.sin().sum().backward(); expected.sin().sum().backward()
    np.testing.assert_allclose(actual.grad.numpy(), expected.grad.numpy(), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('alias', ['identity', 'detach', 'slice', 'transpose', 'reshape', 'same_dtype'])
def test_alias_mutation_invalidates_saved_graph_before_accumulating_any_gradients(alias):
    data = np.array([[.3, -.8], [1.1, .6]])
    actual, expected = wasm.Tensor(data.copy(), requires_grad=True), torch.tensor(data, requires_grad=True)
    views = {
        'identity': lambda x:x, 'detach': lambda x:x.detach(), 'slice': lambda x:x.detach()[:, :1],
        'transpose': lambda x:x.detach().T, 'reshape': lambda x:x.detach().reshape(4),
        'same_dtype': lambda x:x.detach().to(dtype=x.dtype),
    }
    a, b = actual.square().sum(), expected.square().sum()
    with wasm.no_grad(), torch.no_grad():
        views[alias](actual)[...] = 3.
        views[alias](expected)[...] = 3.
    with pytest.raises(RuntimeError, match='modified by an inplace operation'): b.backward()
    with pytest.raises(wasm.BrowserUnsupported, match='saved for backward'): a.backward()
    assert actual.grad is None


def test_restoring_values_does_not_restore_a_stale_graph_but_clones_are_independent():
    x = wasm.Tensor([.2, .5], requires_grad=True)
    native = torch.tensor([.2, .5], dtype=torch.float64, requires_grad=True)
    a, b = x.square().sum(), native.square().sum()
    with wasm.no_grad(), torch.no_grad():
        x += 1.; x -= 1.; native += 1.; native -= 1.
    with pytest.raises(wasm.BrowserUnsupported): a.backward()
    with pytest.raises(RuntimeError): b.backward()
    clone = x.detach().clone()
    fresh = x.square().sum()
    clone[...] = 99.
    fresh.backward()
    np.testing.assert_allclose(x.grad.numpy(), 2*x.numpy())


def test_mutated_frozen_operand_output_mask_and_indices_are_guarded():
    for case in ('constant', 'output', 'where', 'index', 'gather'):
        x = wasm.Tensor([[.2, .7], [1.4, -.4]], requires_grad=True)
        if case == 'constant':
            changed = wasm.Tensor([[1., 2.], [3., 4.]])
            output = x * changed
        elif case == 'output':
            output = x.exp(); changed = output.detach()
        elif case == 'where':
            changed = wasm.Tensor([[True, False], [False, True]])
            output = wasm.where(changed, x, -x)
        else:
            changed = wasm.Tensor([[0, 1], [1, 0]], dtype=np.int64)
            output = x.gather(1, changed) if case == 'gather' else x[changed]
        changed[...] = 0
        with pytest.raises(wasm.BrowserUnsupported, match='saved for backward'): output.sum().backward()
        assert x.grad is None


@pytest.mark.parametrize('optimizer', ['Adam', 'AdamW'])
def test_optimizer_mutations_invalidate_old_graph_but_allow_fresh_graph(optimizer):
    x = wasm.Tensor([.2, -.9], requires_grad=True)
    native = torch.tensor([.2, -.9], dtype=torch.float64, requires_grad=True)
    a, b = x.square().sum(), native.square().sum()
    a.backward(retain_graph=True); b.backward(retain_graph=True)
    opt, native_opt = getattr(wasm, optimizer)([x]), getattr(torch.optim, optimizer)([native])
    opt.step(); native_opt.step()
    with pytest.raises(wasm.BrowserUnsupported): a.backward()
    with pytest.raises(RuntimeError): b.backward()
    opt.zero_grad(); native_opt.zero_grad()
    x.square().sum().backward(); native.square().sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), native.grad.numpy(), atol=1e-12)


def test_differentiable_nonleaf_and_invalid_updates_remain_explicitly_unsupported():
    x = wasm.Tensor([[1., 2.]], requires_grad=True)
    with pytest.raises(wasm.BrowserUnsupported): x += 1.
    nonleaf = x * 2
    with wasm.no_grad(), pytest.raises(wasm.BrowserUnsupported): nonleaf -= 1.
    before = x.numpy().copy()
    with wasm.no_grad(), pytest.raises(RuntimeError): x += wasm.Tensor([[1., 2.], [3., 4.]])
    np.testing.assert_array_equal(x.numpy(), before)
    with wasm.no_grad():
        with wasm.no_grad(): x -= .5
        assert not (x * 2).requires_grad
    assert (x * 2).requires_grad


def test_adamw_exposed_state_updates_also_invalidate_saved_operands():
    x = wasm.Tensor([.2, -.9], requires_grad=True)
    native = torch.tensor([.2, -.9], dtype=torch.float64, requires_grad=True)
    opt, native_opt = wasm.AdamW([x]), torch.optim.AdamW([native])
    x.square().sum().backward(); native.square().sum().backward()
    opt.step(); native_opt.step()
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        probe = wasm.Tensor([1., 2.], requires_grad=True)
        native_probe = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
        a, b = (probe*opt.state[x][key]).sum(), (native_probe*native_opt.state[native][key]).sum()
        opt.step(); native_opt.step()
        with pytest.raises(wasm.BrowserUnsupported): a.backward()
        with pytest.raises(RuntimeError): b.backward()


def test_shared_storage_registry_does_not_keep_discarded_arrays_alive():
    gc.collect(); before = len(wasm._storages)
    for _ in range(1000):
        leaf = wasm.Tensor(np.ones((2, 3)), requires_grad=True)
        output = leaf.square().sum()
        output.backward()
    del leaf, output
    gc.collect()
    assert len(wasm._storages) <= before


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('beta', [0., .25, 1., 2.])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_smooth_l1_independent_values_and_both_gradients_match_native(dtype, beta, reduction):
    data = np.array([[-3., -1., -.25, 0.], [.25, 1., 2., 4.]], dtype=dtype).T
    a, b = wasm.Tensor(data.copy(order="K"), requires_grad=True), wasm.Tensor(np.full(data.shape, .25, dtype=dtype), requires_grad=True)
    x, y = torch.tensor(data.copy(order="K"), requires_grad=True), torch.full(data.shape, .25, dtype=torch.float32 if dtype==np.float32 else torch.float64, requires_grad=True)
    actual, expected = wasm.smooth_l1_loss(a,b,beta=beta,reduction=reduction), F.smooth_l1_loss(x,y,beta=beta,reduction=reduction)
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), atol=2e-6)
    upstream = np.linspace(.2, 1.4, actual.numel(), dtype=dtype).reshape(actual.shape)
    actual.backward(wasm.Tensor(upstream)); expected.backward(torch.tensor(upstream))
    for actual, expected in ((a,x), (b,y)):
        np.testing.assert_allclose(actual.grad.numpy(), expected.grad.numpy(), atol=2e-6)


@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_smooth_l1_empty_batches_and_validation(reduction):
    a = wasm.Tensor(np.empty((0, 3)), requires_grad=True)
    b = torch.empty((0,3), dtype=torch.float64, requires_grad=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        np.testing.assert_allclose(wasm.smooth_l1_loss(a,a,reduction=reduction).numpy(), F.smooth_l1_loss(b,b,reduction=reduction).detach().numpy())
    with pytest.raises(ValueError): wasm.smooth_l1_loss(a,a,beta=-1.)
    with pytest.raises(ValueError): wasm.smooth_l1_loss(a,a,reduction='invalid')
    with pytest.raises(wasm.BrowserUnsupported): wasm.smooth_l1_loss(a,wasm.Tensor([1.]))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_mse_loss_values_and_both_gradients_match_native(dtype, reduction):
    values = np.array([[.2, -.6], [.8, 1.2]],dtype=dtype).T
    a, b = wasm.Tensor(values.copy(order="K"),requires_grad=True), wasm.Tensor(np.ones_like(values),requires_grad=True)
    x, y = torch.tensor(values,requires_grad=True), torch.tensor(np.ones_like(values),requires_grad=True)
    actual, expected = wasm.mse_loss(a,b,reduction=reduction), F.mse_loss(x,y,reduction=reduction)
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), atol=2e-6)
    actual.sum().backward(); expected.sum().backward()
    for actual, expected in ((a,x), (b,y)):
        np.testing.assert_allclose(actual.grad.numpy(), expected.grad.numpy(), atol=2e-6)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_logaddexp_extreme_values_broadcast_gradients_and_empty(dtype):
    values = np.array([[-1000., -1., 0., 1., 1000.]], dtype=dtype)
    offsets = np.array([[1000.], [0.], [-1000.]], dtype=dtype)
    a, b = wasm.Tensor(values.copy(), requires_grad=True), wasm.Tensor(offsets.copy(), requires_grad=True)
    x, y = torch.tensor(values, requires_grad=True), torch.tensor(offsets, requires_grad=True)
    actual, expected = a.logaddexp(b), torch.logaddexp(x,y)
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), atol=2e-6)
    weights = np.arange(15,dtype=dtype).reshape(3,5) / 10
    (actual * wasm.Tensor(weights)).sum().backward(); (expected * torch.tensor(weights)).sum().backward()
    for actual, expected in ((a,x), (b,y)):
        np.testing.assert_allclose(actual.grad.numpy(), expected.grad.numpy(), atol=2e-6)
    empty = wasm.Tensor(np.empty((0,5),dtype=dtype), requires_grad=True)
    empty.logaddexp(a).sum().backward()
    assert empty.grad.shape == (0,5)
