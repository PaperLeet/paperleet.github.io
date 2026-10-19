"""Application-private and standalone runtime integration for unchanged imports."""
import sys
import numpy as np
import pytest
from paperleet._vendor import torch as private
import stridetorch as public


@pytest.mark.parametrize('distribution',[private,public],ids=['application-private','standalone'])
def test_unchanged_training_imports_clip_existing_gradient_buffers(distribution):
    previous={name:module for name,module in sys.modules.items() if name=='torch' or name.startswith('torch.')}
    with distribution.as_torch() as namespace:
        scope={}
        exec('''
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.clip_grad import clip_grad_value_
x=torch.tensor([3.,4.],dtype=torch.float64,requires_grad=True)
optimizer=torch.optim.SGD([x],lr=.1)
optimizer.zero_grad();x.square().sum().backward();buffer=x.grad
norm=clip_grad_norm_(x,1.)
clip_grad_value_([x],.7)
optimizer.step()
assert x.grad is buffer
''',scope)
        assert scope['torch'] is namespace
        np.testing.assert_allclose(scope['norm'].numpy(),10.,rtol=1e-12)
        expected=np.array([3.,4.])-.1*np.minimum(np.array([6.,8.])/(10+1e-6),.7)
        np.testing.assert_allclose(scope['x'].detach().numpy(),expected,rtol=1e-12)
    current={name:module for name,module in sys.modules.items() if name=='torch' or name.startswith('torch.')}
    assert current.keys()==previous.keys() and all(current[name] is module for name,module in previous.items())


def test_browser_profile_exports_gradient_module_from_both_distributions():
    from paperleet._vendor.torch.runtime import source_files as application
    from stridetorch.runtime import source_files as standalone
    for export in [application,standalone]:
        sources=export('autograd')
        assert 'stridetorch/_gradient_utils.py' in sources
        assert 'def clip_grad_norm_' in sources['stridetorch/_gradient_utils.py']
        assert 'def clip_grad_value_' in sources['stridetorch/_gradient_utils.py']
        assert 'stridetorch/_gradient_utils.py' not in export('schedules')
