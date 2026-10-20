"""Binary saved-value contracts from independently observed Torch 2.6 controls."""
import numpy as np
import pytest
from torchwasm import torch

# Actual native probe, not expected values computed by the compatibility runtime.
NATIVE_CONTROLS = [('multiply', [True, False], 'left', False, [[4.0, 5.0]]), ('multiply', [True, False], 'right', True, None), ('multiply', [True, False], 'output', False, [[4.0, 5.0]]), ('multiply', [False, True], 'left', True, None), ('multiply', [False, True], 'right', False, [[2.0, 3.0]]), ('multiply', [False, True], 'output', False, [[2.0, 3.0]]), ('multiply', [True, True], 'left', True, None), ('multiply', [True, True], 'right', True, None), ('multiply', [True, True], 'output', False, [[4.0, 5.0], [2.0, 3.0]]), ('divide', [True, False], 'left', False, [[0.25, 0.2]]), ('divide', [True, False], 'right', True, None), ('divide', [True, False], 'output', False, [[0.25, 0.2]]), ('divide', [False, True], 'left', True, None), ('divide', [False, True], 'right', True, None), ('divide', [False, True], 'output', False, [[-0.125, -0.12]]), ('divide', [True, True], 'left', True, None), ('divide', [True, True], 'right', True, None), ('divide', [True, True], 'output', False, [[0.25, 0.2], [-0.125, -0.12]]), ('scalar_multiply', [True, False], 'left', False, [[2.0, 2.0]]), ('scalar_multiply', [True, False], 'right', False, [[2.0, 2.0]]), ('scalar_multiply', [True, False], 'output', False, [[2.0, 2.0]]), ('scalar_multiply', [True, True], 'left', False, [[2.0, 2.0]]), ('scalar_multiply', [True, True], 'right', False, [[2.0, 2.0]]), ('scalar_multiply', [True, True], 'output', False, [[2.0, 2.0]]), ('scalar_divide', [True, False], 'left', False, [[0.5, 0.5]]), ('scalar_divide', [True, False], 'right', False, [[0.5, 0.5]]), ('scalar_divide', [True, False], 'output', False, [[0.5, 0.5]]), ('scalar_divide', [True, True], 'left', False, [[0.5, 0.5]]), ('scalar_divide', [True, True], 'right', False, [[0.5, 0.5]]), ('scalar_divide', [True, True], 'output', False, [[0.5, 0.5]]), ('reflected_divide', [True, False], 'left', False, [[-0.5, -0.2222222222222222]]), ('reflected_divide', [True, False], 'right', False, [[-0.5, -0.2222222222222222]]), ('reflected_divide', [True, False], 'output', False, [[-0.5, -0.2222222222222222]]), ('reflected_divide', [True, True], 'left', False, [[-0.5, -0.2222222222222222]]), ('reflected_divide', [True, True], 'right', False, [[-0.5, -0.2222222222222222]]), ('reflected_divide', [True, True], 'output', False, [[-0.5, -0.2222222222222222]])]

@pytest.mark.parametrize('operation,activity,mutation,error,expected',NATIVE_CONTROLS)
def test_binary_backward_guards_only_values_native_saves(operation,activity,mutation,error,expected):
    a=torch.tensor([2.,3.],dtype=torch.float64,requires_grad=activity[0])
    b=torch.tensor([4.,5.],dtype=torch.float64,requires_grad=activity[1])
    if operation=='multiply':value=a*b
    elif operation=='divide':value=a/b
    elif operation=='scalar_multiply':value=a*2.
    elif operation=='scalar_divide':value=a/2.
    else:value=2./a
    target={'left':a,'right':b,'output':value}[mutation]
    with torch.no_grad():target.detach().fill_(7.)
    inputs=[x for x in [a,b] if x.requires_grad]
    if error:
        with pytest.raises(RuntimeError):torch.autograd.grad(value.sum(),inputs,allow_unused=True)
    else:
        actual=[x.numpy() for x in torch.autograd.grad(value.sum(),inputs,allow_unused=True) if x is not None]
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

@pytest.mark.parametrize('operation',['multiply','divide'])
def test_scalar_binary_saved_graph_still_requires_retention(operation):
    x=torch.tensor([2.,3.],requires_grad=True)
    value=x*2. if operation=='multiply' else x/2.
    value.sum().backward()
    with pytest.raises(RuntimeError,match='freed|second time'):value.sum().backward()
