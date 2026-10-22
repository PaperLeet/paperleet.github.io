"""Capability refusals must happen before Python ndarray conversion or RPC."""
import importlib
import sys
from types import SimpleNamespace
import pytest
from resident_bridge import ResidentMatmulOffload


def test_resident_requires_negotiating_client_before_owning_resources(monkeypatch):
    monkeypatch.setitem(sys.modules, 'js', SimpleNamespace(matrix_rpc=SimpleNamespace()))
    with pytest.raises(RuntimeError, match='capability negotiation'):
        ResidentMatmulOffload()


def test_resident_unsupported_host_fails_at_session_construction(monkeypatch):
    calls = []
    def require(*features):
        calls.append(features)
        raise RuntimeError('unsupported feature')
    monkeypatch.setitem(sys.modules, 'js', SimpleNamespace(matrix_rpc=SimpleNamespace(requireFeatures=require)))
    with pytest.raises(RuntimeError, match='unsupported feature'):
        ResidentMatmulOffload()
    assert calls == [('program-matmul-f32', 'program-retain-inputs')]


def test_resident_supported_constructor_uses_only_local_capabilities(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'js', SimpleNamespace(matrix_rpc=SimpleNamespace(requireFeatures=lambda *features:calls.append(features))))
    value = ResidentMatmulOffload()
    assert calls == [('program-matmul-f32', 'program-retain-inputs')]
    assert value.cache_info()['owned_leases'] == 0
    value.close()


def client_module(monkeypatch, client):
    def forbidden(*args, **kwargs):
        raise AssertionError('RPC or JS conversion happened before feature refusal')
    monkeypatch.setitem(sys.modules, 'js', SimpleNamespace(matrix_rpc=client, Object=SimpleNamespace(fromEntries=forbidden)))
    monkeypatch.setitem(sys.modules, 'pyodide.ffi', SimpleNamespace(run_sync=forbidden, to_js=forbidden))
    return importlib.import_module('torchwasm.experimental.webgpu.matrix_client')


class NoConvert:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Unsupported retention copied the input')


@pytest.mark.parametrize('negotiating', [False, True])
def test_program_refuses_unsupported_retention_before_array_conversion(monkeypatch, negotiating):
    calls = []
    def require(*features):
        calls.append(features)
        raise RuntimeError('unsupported retention')
    client = SimpleNamespace(requireFeatures=require) if negotiating else SimpleNamespace()
    module = client_module(monkeypatch, client)
    value = module.MatrixProgram(client)
    a, b = value.input(NoConvert()), value.input(NoConvert())
    output = value.matmul(a,b)
    with pytest.raises(RuntimeError, match='unsupported retention' if negotiating else 'capability negotiation'):
        value.run([output], retain_inputs=[a])
    assert calls == ([('program-matmul-f32', 'program-retain-inputs')] if negotiating else [])


@pytest.mark.parametrize('refs', [[True], [-1], [2], [0,0]])
def test_invalid_retained_inputs_are_rejected_before_feature_check(monkeypatch, refs):
    module = client_module(monkeypatch, SimpleNamespace())
    value = module.MatrixProgram()
    a,b = value.input(NoConvert()),value.input(NoConvert())
    output = value.matmul(a,b)
    with pytest.raises(ValueError):
        value.run([output], retain_inputs=refs)
