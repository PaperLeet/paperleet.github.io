"""Contracts for independent usage, import isolation and the published namespace."""
import subprocess
import sys
import types

import numpy as np
import pytest

from paperleet_torch import UnsupportedOperation, as_torch, create_torch, torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_namespace_keeps_broadcast_gradients_and_storage_guards(dtype):
    x = torch.tensor([[1., 2.], [3., 4.]], dtype=dtype, requires_grad=True)
    bias = torch.tensor([2., 3.], dtype=dtype, requires_grad=True)
    (x * bias).square().sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [[8., 36.], [24., 72.]])
    np.testing.assert_allclose(bias.grad.numpy(), [40., 120.])
    value = x.square().sum()
    with torch.no_grad():
        x[0, 0] = 9.
    with pytest.raises(UnsupportedOperation, match='modified in-place'):
        value.backward()


def test_import_and_default_namespace_leave_native_torch_names_untouched():
    script = '''
import sys, types
before = {name: types.ModuleType(name) for name in ('torch', 'torch.nn', 'torch.nn.functional')}
sys.modules.update(before)
from paperleet_torch import torch, create_torch
from paperleet_torch import Tensor
assert Tensor is torch.Tensor
assert all(sys.modules[name] is value for name, value in before.items())
assert create_torch().Tensor is Tensor
assert all(sys.modules[name] is value for name, value in before.items())
x=torch.tensor([2.,3.],requires_grad=True); x.square().sum().backward()
assert x.grad.tolist()==[4.,6.]
assert all(sys.modules[name] is value for name, value in before.items())
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_explicit_import_context_restores_every_module_after_nested_exception():
    previous = {name: module for name, module in sys.modules.items()
                if name == 'torch' or name.startswith('torch.')}
    with pytest.raises(ValueError, match='intentional'):
        with as_torch() as outer:
            import torch as imported
            from torch.nn.functional import linear
            assert imported is outer
            assert linear(outer.ones(1, 2), outer.ones(1, 2)).item() == 2.
            with as_torch() as inner:
                import torch as nested
                assert nested is inner and inner is not outer
            assert sys.modules['torch'] is outer
            sys.modules['torch.extra'] = types.ModuleType('torch.extra')
            raise ValueError('intentional')
    current = {name: module for name, module in sys.modules.items()
               if name == 'torch' or name.startswith('torch.')}
    assert current.keys() == previous.keys()
    assert all(current[name] is module for name, module in previous.items())


def test_namespace_exposes_optimizers_linalg_jvp_and_capability_guards():
    layer = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(layer.parameters(), lr=.02)
    x = torch.tensor([[1., 0.], [0., 1.]])
    for _ in range(10):
        optimizer.zero_grad()
        layer(x).square().mean().backward()
        optimizer.step()
    matrix = torch.tensor([[3., 0.], [0., 2.]], dtype=torch.float64)
    np.testing.assert_allclose(torch.linalg.inv(matrix).numpy(), [[1/3, 0.], [0., .5]])
    primal, tangent = torch.func.jvp(lambda value: value.square(),
                                   (torch.tensor([2., 3.]),), (torch.ones(2),))
    np.testing.assert_array_equal(primal.numpy(), [4., 9.])
    np.testing.assert_array_equal(tangent.numpy(), [4., 6.])
    with pytest.raises(TypeError):
        torch._browser_capabilities['forged'] = None
    with pytest.raises(UnsupportedOperation):
        torch.tensor([1.], device='cuda')
    with pytest.raises(TypeError):
        torch.sum(torch.ones(2), typo=True)
