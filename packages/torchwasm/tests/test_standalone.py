"""Contracts for independent usage, import isolation and the published namespace."""
import subprocess
import sys
import types

import numpy as np
import pytest

from torchwasm import UnsupportedOperation, as_torch, create_torch, torch


def test_runtime_submodule_uses_normal_package_imports():
    from torchwasm import runtime
    assert 'torchwasm/_tensor.py' in runtime.source_files()


def test_minimal_schedules_export_is_self_contained_for_feature_detection(tmp_path):
    from torchwasm import runtime
    for name,source in runtime.source_files('schedules').items():
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(source,encoding='utf-8')
    script='''
import sys
sys.path.insert(0,sys.argv[1])
from torchwasm._legacy import install
from torchwasm import UnsupportedOperation
torch=install();x=torch.tensor([.1,.2]);sentinel=object()
for target in (torch,x):
    assert not hasattr(target,'not_a_real_symbol')
    assert getattr(target,'not_a_real_symbol',sentinel) is sentinel
    try:getattr(target,'not_a_real_symbol')
    except UnsupportedOperation as error:assert isinstance(error,AttributeError) and error.obj is target
    else:raise AssertionError('Missing symbol was accepted')
try:torch.tensor([1.],requires_grad=True)
except UnsupportedOperation as error:assert not isinstance(error,AttributeError)
else:raise AssertionError('Schedules unexpectedly accepted autograd')
assert abs(float(x.square().numpy().sum())-.05)<1e-6
assert 'torchwasm._api' not in sys.modules and 'torchwasm._tensor' not in sys.modules
'''
    subprocess.run([sys.executable,'-I','-B','-c',script,str(tmp_path)],cwd=tmp_path,check=True,capture_output=True,text=True)


@pytest.mark.parametrize('profile', ['autograd', 'schedules'])
def test_embedded_runtime_carries_redistribution_notices(profile):
    from importlib.resources import files
    from torchwasm import runtime
    sources = runtime.source_files(profile)
    for name in ('LICENSE', 'NOTICE'):
        assert sources['torchwasm/' + name] == files('torchwasm').joinpath(name).read_text(encoding='utf-8')
    assert 'Version 2.0, January 2004' in sources['torchwasm/LICENSE']
    assert 'Copyright 2026 PaperLeet contributors' in sources['torchwasm/NOTICE']


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
from torchwasm import torch, create_torch
from torchwasm import Tensor
assert Tensor is torch.Tensor
assert all(sys.modules[name] is value for name, value in before.items())
assert create_torch().Tensor is Tensor
assert all(sys.modules[name] is value for name, value in before.items())
x=torch.tensor([2.,3.],requires_grad=True); x.square().sum().backward()
assert x.grad.tolist()==[4.,6.]
assert all(sys.modules[name] is value for name, value in before.items())
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
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


def test_persistent_alias_supports_unchanged_imports_and_removes_stale_children():
    script = '''
import sys, types
sys.modules['torch'] = types.ModuleType('torch')
stale_cuda = types.ModuleType('torch.cuda')
sys.modules['torch.cuda'] = stale_cuda
from torchwasm import install_as_torch
namespace = install_as_torch()
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.linalg import inv
from torch.autograd import grad
from torch.func import jvp
from torch.testing import assert_close
assert torch is namespace and nn is torch.nn and optim is torch.optim
assert F is torch.nn.functional
assert sys.modules['torch.cuda'] is namespace.cuda and namespace.cuda is not stale_cuda
assert not namespace.cuda.is_available() and namespace.cuda.device_count() == 0
x = torch.tensor([[2., 3.]], requires_grad=True)
F.linear(x, torch.ones(1, 2)).square().sum().backward()
assert x.grad.tolist() == [[10., 10.]]
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('ending', ['', 'raise SystemExit(7)', 'raise RuntimeError("script failure")'])
def test_launcher_restores_host_state_after_success_and_errors(tmp_path, ending):
    from torchwasm._cli import main
    previous_argv, previous_path = sys.argv, sys.path[:]
    previous_modules = {name: module for name, module in sys.modules.items()
                        if name == 'torch' or name.startswith('torch.')}
    target = tmp_path / 'unchanged.py'
    target.write_text('import torch\nassert torch.Tensor is not None\n' + ending)
    if ending.startswith('raise SystemExit'):
        with pytest.raises(SystemExit) as error:
            main([str(target)])
        assert error.value.code == 7
    elif ending:
        with pytest.raises(RuntimeError, match='script failure'):
            main([str(target)])
    else:
        main([str(target)])
    assert sys.argv is previous_argv
    assert sys.path == previous_path
    current_modules = {name: module for name, module in sys.modules.items()
                       if name == 'torch' or name.startswith('torch.')}
    assert current_modules.keys() == previous_modules.keys()
    assert all(current_modules[name] is value for name, value in previous_modules.items())
