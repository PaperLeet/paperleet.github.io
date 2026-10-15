"""Model/checkpoint contracts runnable with NumPy alone, including in Pyodide."""
from collections import OrderedDict
import copy

import numpy as np
import pytest

from paperleet_torch import UnsupportedOperation, as_torch, torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_unchanged_imports_model_training_checkpoint_and_restore(dtype):
    with as_torch():
        import torch as api
        from torch import nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2, 2, dtype=dtype), nn.Linear(2, 1, dtype=dtype)])
                self.output_alias = self.layers[1]
                self.options = nn.ModuleDict({'activation': nn.SiLU()})
                self.scales = nn.ParameterList([api.ones(1, dtype=dtype)])
                self.offsets = nn.ParameterDict({'output': api.zeros(1, dtype=dtype)})
                self.register_buffer('center', api.tensor([.1, -.2], dtype=dtype))
                self.register_buffer('scratch', api.zeros(1, dtype=dtype), persistent=False)

            def forward(self, x):
                hidden = self.options['activation'](self.layers[0](x - self.center))
                return F.linear(hidden, self.layers[1].weight, self.layers[1].bias) * self.scales[0] + self.offsets['output']

        model = Model()
        with api.no_grad():
            for parameter in model.parameters(): parameter.fill_(.1)
        optimizer = api.optim.Adam(model.parameters(), lr=.02)
        x = api.tensor([[1., .5], [-.5, 2.]], dtype=dtype)
        target = api.tensor([[.8], [-.1]], dtype=dtype)
        initial_loss = (model(x) - target).square().mean().item()
        for _ in range(12):
            optimizer.zero_grad()
            (model(x) - target).square().mean().backward()
            optimizer.step()
        assert (model(x) - target).square().mean().item() < initial_loss
        expected = model.eval()(x).detach().numpy().copy()
        assert all(not child.training for child in model.modules())
        checkpoint = copy.deepcopy(model.state_dict())
        assert 'scratch' not in checkpoint
        identities = [id(parameter) for parameter in model.parameters()]
        with api.no_grad():
            for parameter in model.parameters(): parameter.fill_(0.)
            model.scratch.fill_(19.)
        restored = model.load_state_dict(checkpoint)
        assert not restored.missing_keys and not restored.unexpected_keys
        assert [id(parameter) for parameter in model.parameters()] == identities
        np.testing.assert_allclose(model(x).detach().numpy(), expected, rtol=1e-6, atol=1e-6)
        assert model.scratch.item() == 19. and model.output_alias is model.layers[1]
        model.train().zero_grad()
        (model(x) - target).square().mean().backward()
        optimizer.step()  # The pre-existing optimizer still owns restored parameters.
        assert not np.array_equal(model(x).detach().numpy(), expected)


def test_buffer_views_checkpoint_copy_and_parameter_gradient_copy_rules():
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    model.register_buffer('base', torch.arange(24, dtype=torch.float64).reshape(4, 6))
    model.register_buffer('view', model.base[1:, 1::2].T)
    model.register_buffer('reverse', model.base[::-1])
    model.register_buffer('empty', model.base[:, :0])
    shallow, independent = model.state_dict(), copy.deepcopy(model.state_dict())
    independent['base'][1, 1] = 99.
    assert independent['view'][0, 0].item() == independent['reverse'][2, 1].item() == 99.
    assert model.base[1, 1].item() == 7.
    model.base[1, 1] = 23.
    assert shallow['view'][0, 0].item() == 23.
    model(torch.ones(1, 2, dtype=torch.float64)).sum().backward()
    copied = copy.deepcopy(model)
    assert all(parameter.grad is None for parameter in copied.parameters())
    assert copied.weight is not model.weight
    copied.base.fill_(0.)
    assert model.base[1, 1].item() == 23.


def test_registered_aliases_none_and_nonpersistent_buffers():
    module = torch.nn.Module()
    module.p = torch.nn.Parameter(torch.ones(2))
    module.alias = module.p
    module.child = torch.nn.Linear(2, 1, bias=False)
    module.other = module.child
    module.register_parameter('absent', None)
    module.register_buffer('present', torch.ones(1))
    module.register_buffer('temporary', torch.ones(1), persistent=False)
    module.register_buffer('nothing', None)
    assert list(module.state_dict()) == ['p', 'alias', 'present', 'child.weight', 'other.weight']
    assert [name for name, _ in module.named_parameters()] == ['p', 'child.weight']
    assert [name for name, _ in module.named_parameters(remove_duplicate=False)] == ['p', 'alias', 'child.weight', 'other.weight']
    assert [name for name, _ in module.named_buffers()] == ['present', 'temporary']
    assert module.get_parameter('other.weight') is module.child.weight
    assert module.get_submodule('other') is module.child
    assert module.get_buffer('nothing') is None
    with pytest.raises(TypeError): module.p = torch.ones(2)
    del module.alias
    assert 'alias' not in module.state_dict()


def test_state_errors_and_unsupported_lifecycle_options_are_explicit():
    module = torch.nn.Linear(2, 1)
    parameters = torch.nn.ParameterList([torch.ones(1)])
    with pytest.raises(TypeError): parameters.extend(torch.ones(2))
    assert len(parameters) == 1
    result = module.load_state_dict({'bias': torch.ones(1), 'extra': torch.zeros(1)}, strict=False)
    assert result.missing_keys == ['weight'] and result.unexpected_keys == ['extra']
    with pytest.raises(RuntimeError, match='Missing'): module.load_state_dict({})
    with pytest.raises(RuntimeError, match='size mismatch'):
        module.load_state_dict({'weight': torch.zeros(3, 1)}, strict=False)
    with pytest.raises(RuntimeError, match='Tensor'):
        module.load_state_dict({'weight': [[1., 2.]]}, strict=False)
    with pytest.raises(UnsupportedOperation, match='Assign'):
        module.load_state_dict(module.state_dict(), assign=True)
    with pytest.raises(UnsupportedOperation): module.to('cuda')
    with pytest.raises(UnsupportedOperation): module.register_full_backward_hook(lambda *args: None)


def test_container_slicing_mutation_registration_and_named_order():
    a, b = torch.nn.Linear(1, 1), torch.nn.ReLU()
    sequence = torch.nn.Sequential(OrderedDict([('linear', a), ('relu', b)]))
    assert list(sequence[1:]._modules) == ['relu']
    del sequence[1]
    sequence.append(b).insert(0, a)
    assert list(sequence.state_dict()) == ['0.weight', '0.bias', '1.weight', '1.bias']
    modules = torch.nn.ModuleList([a, b])
    modules.insert(1, a)
    del modules[::2]
    assert len(modules) == 1 and modules[0] is a
    params = torch.nn.ParameterDict({'z': torch.ones(1), 'a': torch.zeros(1)})
    params.update(OrderedDict([('y', params['z']), ('b', torch.ones(1))]))
    assert list(params) == ['a', 'z', 'y', 'b']
    assert [name for name, _ in params.named_parameters()] == ['a', 'z', 'b']
    assert isinstance(params['b'], torch.nn.Parameter)
    with pytest.raises(UnsupportedOperation): modules.insert(-1, a)
    with pytest.raises(UnsupportedOperation): sequence[1:].insert(0, a)  # Slice retains named keys.
