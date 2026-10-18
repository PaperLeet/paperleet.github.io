"""Registered operator attributes retain callable identity across model serialization."""
import copy,pickle
import numpy as np
import pytest
from stridetorch import torch

@pytest.mark.parametrize('kind',['module','transformer'])
@pytest.mark.parametrize('operator',['sin','vjp'])
@pytest.mark.parametrize('method',['copy','deepcopy','pickle2','pickle_highest'])
def test_registered_operator_attributes_survive_model_copy_pickle(kind,operator,method):
    model=torch.nn.Module() if kind=='module' else torch.nn.TransformerEncoderLayer(4,2,8,dropout=0.)
    fn=torch.sin if operator=='sin' else torch.autograd.functional.vjp
    model.operator=fn;model.same_operator=fn
    model.register_buffer('buffer',torch.tensor([1.,2.]));model.alias=model.buffer
    if method=='copy':restored=copy.copy(model)
    elif method=='deepcopy':restored=copy.deepcopy(model)
    else:restored=pickle.loads(pickle.dumps(model,protocol=2 if method=='pickle2' else pickle.HIGHEST_PROTOCOL))
    assert restored.operator is fn and restored.same_operator is fn
    assert restored.alias is restored.buffer
    if method=='copy':assert restored.buffer is model.buffer
    else:assert restored.buffer is not model.buffer
    x=torch.tensor([2.,3.],requires_grad=True)
    if operator=='sin':np.testing.assert_allclose(restored.operator(x).detach().numpy(),np.sin([2.,3.]),rtol=1e-6,atol=1e-7)
    else:
        out,g=restored.operator(lambda a:a.square().sum(),x)
        assert out.item()==13 and g.tolist()==[4.,6.] and x.grad is None

class _OrdinaryPayload:
    @property
    def __class__(self):
        raise AssertionError('State handling must not inspect a payload class property')

@pytest.mark.parametrize('kind',['module','transformer'])
def test_state_does_not_invoke_unrelated_payload_properties(kind):
    model=torch.nn.Module() if kind=='module' else torch.nn.TransformerEncoderLayer(4,2,8,dropout=0.)
    payload=_OrdinaryPayload();model.__dict__['payload']=payload
    state=model.__getstate__();assert state['payload'] is payload
    restored=torch.nn.Module() if kind=='module' else torch.nn.TransformerEncoderLayer(4,2,8,dropout=0.)
    restored.__setstate__(state);assert restored.__dict__['payload'] is payload
