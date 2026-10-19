"""The legacy imports must share the canonical reentry gate and classes."""
import importlib
import pytest


@pytest.mark.parametrize('name',['torch_bridge','resident_bridge','matrix_client'])
def test_old_import_is_the_canonical_module(name):
    old = importlib.import_module(name)
    current = importlib.import_module('stridetorch.experimental.webgpu.'+name)
    assert old is current


def test_old_and_new_offloads_share_one_gate():
    from stridetorch import torch
    import torch_bridge
    from stridetorch.experimental.webgpu import MatmulOffload
    previous = torch_bridge._busy
    try:
        torch_bridge._busy = True
        with pytest.raises(RuntimeError,match='already active'):
            MatmulOffload(lambda jobs:[])(torch.ones(2,2),torch.ones(2,2))
    finally:
        torch_bridge._busy = previous
