"""Installed opt-in GPU resources, import isolation and offline embedding."""
import subprocess
import sys

import numpy as np
import pytest


def test_root_import_does_not_activate_experimental_gpu_code():
    script = '''
import sys,types
native=types.ModuleType('torch');sys.modules['torch']=native
import stridetorch
assert not any(name.startswith('stridetorch.experimental') for name in sys.modules)
assert sys.modules['torch'] is native
'''
    subprocess.run([sys.executable, '-B', '-c', script], check=True, capture_output=True, text=True)


def test_all_experimental_exports_import_without_browser_bindings_or_torch_aliases():
    script = '''
import sys,types
native=types.ModuleType('torch');sys.modules['torch']=native
from stridetorch.experimental.webgpu import (MatmulOffload,ResidentMatmulOffload,
    Matrix,MatrixProgram,ProgramResult,service_sources,source_files)
assert 'js' not in sys.modules and 'pyodide' not in sys.modules
assert sys.modules['torch'] is native
assert all(callable(value) for value in (MatmulOffload,ResidentMatmulOffload,
    Matrix,MatrixProgram,ProgramResult,service_sources,source_files))
'''
    subprocess.run([sys.executable, '-B', '-c', script], check=True, capture_output=True, text=True)


def test_installed_tensor_offload_accepts_an_explicit_transport_without_pyodide():
    from stridetorch import torch
    from stridetorch.experimental.webgpu import MatmulOffload
    calls = []
    def submit(jobs):
        calls.extend(jobs)
        return [(a.T if ta else a) @ (b.T if tb else b) for a,b,ta,tb in jobs]
    a = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
    b = torch.tensor([[5.,6.],[7.,8.]], requires_grad=True)
    value = MatmulOffload(submit)(a,b)
    da,db = torch.autograd.grad(value,(a,b),torch.ones_like(value))
    np.testing.assert_array_equal(value.numpy(), [[19.,22.],[43.,50.]])
    np.testing.assert_array_equal(da.numpy(), [[11.,15.],[11.,15.]])
    np.testing.assert_array_equal(db.numpy(), [[4.,4.],[6.,6.]])
    assert len(calls) == 3


@pytest.mark.parametrize('name', ['MatrixProgram','ResidentMatmulOffload'])
def test_missing_browser_host_reports_activation_error_without_taking_ownership(monkeypatch,name):
    from stridetorch.experimental import webgpu
    monkeypatch.setitem(sys.modules,'js',None)
    with pytest.raises(RuntimeError,match='requires Pyodide and an attached MatrixRpc client'):
        getattr(webgpu,name)()


def test_host_assets_are_a_complete_adjacent_module_set_with_notices():
    import re
    from importlib.resources import files
    from stridetorch.experimental.webgpu import service_sources
    values = service_sources()
    assert set(values) == {'gpu-service.mjs','rpc-client.mjs','rpc-protocol.mjs',
                           'plan-program.mjs','LICENSE','NOTICE'}
    for name,source in values.items():
        if name.endswith('.mjs'):
            assert source == files('stridetorch.experimental.webgpu').joinpath(name).read_text(encoding='utf-8')
            for dependency in re.findall(r"from ['\"]\./([^'\"]+)['\"]", source):
                assert dependency in values
        else:
            assert source == files('stridetorch').joinpath(name).read_text(encoding='utf-8')


def test_explicit_offline_payload_preserves_default_runtime_sources():
    from stridetorch import runtime
    from stridetorch.experimental.webgpu import source_files
    before = runtime.source_files()
    optional = source_files()
    assert before == runtime.source_files()
    assert not any('/experimental/' in name for name in before)
    assert all(optional[name] == source for name,source in before.items())
    assert 'stridetorch/experimental/webgpu/__init__.py' in optional
    assert 'stridetorch/experimental/webgpu/gpu-service.mjs' in optional
    assert not any(name.startswith(('/', 'paperleet/')) or 'benchmarks/' in name for name in optional)


def test_explicit_offline_payload_runs_without_a_source_checkout(tmp_path):
    from stridetorch.experimental.webgpu import source_files
    for name,source in source_files().items():
        path = tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(source,encoding='utf-8')
    script = '''
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import stridetorch
from stridetorch.experimental.webgpu import MatmulOffload,MatrixProgram,service_sources
assert Path(stridetorch.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert len(service_sources())==6
stridetorch.install_as_torch()
import torch
x=torch.tensor([2.,3.],requires_grad=True);x.square().sum().backward()
assert x.grad.tolist()==[4.,6.]
assert 'js' not in sys.modules and 'pyodide' not in sys.modules
'''
    subprocess.run([sys.executable,'-I','-B','-c',script,str(tmp_path)],
                   cwd=tmp_path,check=True,capture_output=True,text=True)


def _program_spy(monkeypatch, used):
    from types import SimpleNamespace
    from stridetorch.experimental.webgpu import matrix_client
    class Program:
        def __init__(self,client=None):
            used.append(matrix_client.default_client() if client is None else client)
            self.values=[]
        def input(self,value):self.values.append(value);return len(self.values)-1
        def matmul(self,a,b,ta=False,tb=False):
            left,right=self.values[a],self.values[b]
            self.values.append((left.T if ta else left)@(right.T if tb else right))
            return len(self.values)-1
        def run(self,outputs,**kwargs):
            result=SimpleNamespace(arrays=[self.values[ref] for ref in outputs],
                                   matrices=[None]*len(outputs),input_matrices={})
            class Context:
                def __enter__(self):return result
                def __exit__(self,*args):pass
            # Resident transport consumes a result directly, while stateless
            # transport uses its cleanup context manager.
            return Context() if not kwargs else result
    monkeypatch.setattr(matrix_client,'MatrixProgram',Program)


def test_resident_backend_keeps_constructor_client_after_global_replacement(monkeypatch):
    from types import SimpleNamespace
    from stridetorch.experimental.webgpu import resident_bridge
    first=SimpleNamespace(requireFeatures=lambda *args:None)
    second=SimpleNamespace(requireFeatures=lambda *args:None)
    monkeypatch.setitem(sys.modules,'js',SimpleNamespace(matrix_rpc=first))
    backend=resident_bridge._RpcBackend()
    monkeypatch.setitem(sys.modules,'js',SimpleNamespace(matrix_rpc=second))
    used=[];_program_spy(monkeypatch,used)
    a=np.ones((2,2),dtype='float32');b=np.ones((2,2),dtype='float32')
    backend.program([(a,b,False,False)],{id(a):a,id(b):b},(False,),())
    assert used == [first]


def test_stateless_offload_binds_default_client_only_on_first_gpu_submission(monkeypatch):
    from types import SimpleNamespace
    from stridetorch import torch
    from stridetorch.experimental.webgpu import MatmulOffload
    monkeypatch.setitem(sys.modules,'js',None)
    call=MatmulOffload()
    first,second=object(),object();used=[];_program_spy(monkeypatch,used)
    a=torch.ones(2,2,dtype=torch.float32)
    with torch.enable_grad():
        monkeypatch.setitem(sys.modules,'js',SimpleNamespace(matrix_rpc=first))
        np.testing.assert_array_equal(call(a,a).numpy(),np.full((2,2),2.))
        monkeypatch.setitem(sys.modules,'js',SimpleNamespace(matrix_rpc=second))
        np.testing.assert_array_equal(call(a,a).numpy(),np.full((2,2),2.))
    assert used == [first,first] and call.client is first


def test_explicit_stateless_client_needs_no_global_host_binding(monkeypatch):
    from stridetorch import torch
    from stridetorch.experimental.webgpu import MatmulOffload
    monkeypatch.setitem(sys.modules,'js',None)
    client=object();used=[];_program_spy(monkeypatch,used)
    with torch.enable_grad():
        a=torch.ones(2,2,dtype=torch.float32)
        np.testing.assert_array_equal(MatmulOffload(client=client)(a,a).numpy(),np.full((2,2),2.))
    assert used == [client]


def test_explicit_resident_client_is_borrowed_without_global_lookup(monkeypatch):
    from types import SimpleNamespace
    from stridetorch.experimental.webgpu import ResidentMatmulOffload
    monkeypatch.setitem(sys.modules,'js',None)
    calls=[];client=SimpleNamespace(requireFeatures=lambda *features:calls.append(features),closed=False)
    value=ResidentMatmulOffload(client=client)
    assert value.client is client and calls==[('program-matmul-f32','program-retain-inputs')]
    value.close()
    assert not client.closed


@pytest.mark.parametrize('resident',[False,True])
def test_conflicting_client_and_injected_transport_fail_before_activation(resident):
    from stridetorch.experimental.webgpu import MatmulOffload,ResidentMatmulOffload
    with pytest.raises(ValueError,match='Choose an injected'):
        if resident:ResidentMatmulOffload(backend=object(),client=object())
        else:MatmulOffload(lambda jobs:[],client=object())
