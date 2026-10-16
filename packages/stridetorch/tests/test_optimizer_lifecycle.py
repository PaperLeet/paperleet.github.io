"""Complete optimizer workflows with NumPy only, also executed in real WASM."""
import copy

import numpy as np
import pytest

from stridetorch import UnsupportedOperation, as_torch, torch


@pytest.mark.parametrize('name,options', [('SGD', {'momentum': .8}),
                                        ('Adam', {'amsgrad': True}), ('AdamW', {'amsgrad': True})])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_grouped_training_model_and_optimizer_checkpoint_resume(name, options, dtype):
    with as_torch():
        import torch as api
        from torch.optim.optimizer import Optimizer
        from torch.optim.adam import Adam
        from torch.optim.adamw import AdamW
        from torch.optim.sgd import SGD
        assert (Adam, AdamW, SGD) == (api.optim.Adam, api.optim.AdamW, api.optim.SGD)
        model = api.nn.Linear(2, 1, dtype=dtype)
        with api.no_grad():model.weight.fill_(.1);model.bias.fill_(.05)
        optimizer = getattr(api.optim, name)([{'params': [model.weight], 'lr': .03},
                                              {'params': [model.bias], 'lr': .01}], **options)
        assert isinstance(optimizer, Optimizer) and not optimizer.state
        x = api.tensor([[1., .2], [-.5, 2.]], dtype=dtype)
        target = api.tensor([[.7], [-.2]], dtype=dtype)
        def update(net, opt):
            opt.zero_grad()
            loss = (net(x) - target).square().mean();loss.backward();opt.step()
            return loss.item()
        initial = update(model, optimizer)
        for _ in range(8):update(model, optimizer)
        assert (model(x) - target).square().mean().item() < initial
        checkpoint = copy.deepcopy({'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
        restored = copy.deepcopy(model)
        restored.load_state_dict(checkpoint['model'])
        resumed = getattr(api.optim, name)([{'params': [restored.weight]}, {'params': [restored.bias]}])
        resumed.load_state_dict(checkpoint['optimizer'])
        for step in range(5):
            for opt in (optimizer, resumed):opt.param_groups[0]['lr'] = .02 / (step + 1)
            update(model, optimizer);update(restored, resumed)
        np.testing.assert_allclose(model(x).detach().numpy(), restored(x).detach().numpy(), rtol=1e-6, atol=1e-7)
        assert resumed.param_groups[0]['params'][0] is restored.weight


@pytest.mark.parametrize('name', ['SGD', 'Adam', 'AdamW'])
def test_delayed_state_add_group_closure_and_reused_gradient_buffers(name):
    p, q = [torch.tensor([1., 2.], requires_grad=True) for _ in range(2)]
    optimizer = getattr(torch.optim, name)([p], lr=.02)
    optimizer.add_param_group({'params': [q], 'lr': .01})
    calls = []
    def closure():
        calls.append(1);optimizer.zero_grad()
        loss = p.square().sum();loss.backward();return loss
    with torch.no_grad():
        result = optimizer.step(closure)
        assert not p.square().requires_grad
    assert result.item() == 5. and calls == [1] and p.square().requires_grad
    assert q not in optimizer.state
    grad = p.grad
    optimizer.zero_grad(set_to_none=False)
    assert p.grad is grad and q.grad is None
    np.testing.assert_array_equal(grad.numpy(), [0., 0.])
    q.grad = torch.ones(2)
    optimizer.step()
    if name != 'SGD':assert optimizer.state[q]['step'].item() == 1.
    before = p.detach().numpy().copy()
    optimizer.zero_grad();optimizer.step()
    np.testing.assert_array_equal(p.numpy(), before)


def test_lazy_adam_state_missing_grad_and_legacy_lr_alias():
    params = [torch.nn.Parameter(torch.ones(32, 32)) for _ in range(8)]
    optimizer = torch.optim.Adam(params)
    assert not optimizer.state and not optimizer.state_dict()['state']
    optimizer.lr = .03
    assert optimizer.param_groups[0]['lr'] == .03 and optimizer.params == params
    assert optimizer.params is optimizer.param_groups[0]['params']
    params[-1].grad = torch.ones(32, 32)
    optimizer.step()
    assert list(optimizer.state) == [params[-1]]
    assert optimizer.state_dict()['param_groups'][0]['params'] == list(range(8))
    assert set(optimizer.state_dict()['state']) == {7}


def test_bad_groups_checkpoints_options_and_state_shapes_fail_explicitly():
    p = torch.tensor([1., 2.], requires_grad=True)
    with pytest.raises(ValueError):torch.optim.SGD([])
    with pytest.raises(ValueError):torch.optim.Adam([p.square()])
    with pytest.raises(ValueError):torch.optim.Adam([{'params': [p]}, {'params': [p]}])
    optimizer = torch.optim.Adam([p])
    with pytest.raises(ValueError):optimizer.add_param_group({'params': [p]})
    with pytest.raises(ValueError):optimizer.load_state_dict({'state': {}, 'param_groups': []})
    for flag in ('fused', 'foreach', 'capturable', 'differentiable'):
        with pytest.raises(UnsupportedOperation):torch.optim.Adam([p], **{flag: True})
    with pytest.raises(UnsupportedOperation):torch.optim.Adam([p], typo=True)
    p.grad = torch.ones(2);optimizer.step()
    optimizer.state[p]['exp_avg'] = torch.zeros(3)
    with pytest.raises(RuntimeError, match='state'):optimizer.step()


def test_same_dtype_conversion_identity_preserves_graph_and_storage_guards():
    x = torch.tensor([2., 3.], requires_grad=True)
    assert x.to(dtype=x.dtype) is x and x.to('cpu') is x
    def same_dtype(value):
        assert value.to(dtype=value.dtype) is value
        return value.square()
    primal, tangent = torch.func.jvp(same_dtype, (x,), (torch.ones_like(x),))
    np.testing.assert_array_equal(primal.detach().numpy(), [4., 9.])
    np.testing.assert_array_equal(tangent.detach().numpy(), [4., 6.])
    loss = x.to(dtype=x.dtype).square().sum()
    with torch.no_grad():x.fill_(4.)
    with pytest.raises(UnsupportedOperation, match='modified in-place'):loss.backward()
