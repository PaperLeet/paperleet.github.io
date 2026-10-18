"""Consumed joint callbacks must not pin saved payloads into predecessor work."""
import gc
import weakref

import numpy as np
import pytest

from stridetorch import torch, UnsupportedOperation
from stridetorch._autograd import joint_result


def weighted_pair(first,second,payload,selective):
    def backward(g,needs):
        return g*payload if needs[0] else None,g*3 if needs[1] else None
    return joint_result(first.numpy()*payload+second.numpy()*3,(first,second),backward,prune_inputs=selective)


@pytest.mark.parametrize('retained_first',[False,True])
@pytest.mark.parametrize('selective',[False,True])
@pytest.mark.parametrize('method',['backward','targeted_grad','targeted_backward'])
def test_consumed_callback_payload_is_freed_before_predecessor_hooks(retained_first,selective,method):
    x=torch.tensor([1.,2.,3.,4.],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([4.,3.,2.,1.],dtype=torch.float64,requires_grad=True)
    first=x*2;second=y*2;payload=np.full(4,2.,dtype='float64');reference=weakref.ref(payload)
    output=weighted_pair(first,second,payload,selective);del payload
    state={'retain':retained_first};observed=[]
    def inspect(g):
        observed.append(reference() is not None)
        assert (reference() is not None)==state['retain']
    first.register_hook(inspect);second.register_hook(inspect)
    seeds=[np.array([1.,-2.,3.,4.])]
    if retained_first:seeds.append(np.array([-3.,2.,-1.,5.]))
    for index,seed in enumerate(seeds):
        state['retain']=retained_first and index==0;x.grad=y.grad=None;start=len(observed)
        if method=='targeted_grad':
            gradient=torch.autograd.grad(output,x,torch.tensor(seed),retain_graph=state['retain'])[0]
            np.testing.assert_array_equal(gradient.numpy(),4*seed)
            assert x.grad is y.grad is None
        else:
            output.backward(torch.tensor(seed),retain_graph=state['retain'],**({'inputs':[x]} if method=='targeted_backward' else {}))
            np.testing.assert_array_equal(x.grad.numpy(),4*seed)
            if method=='backward':np.testing.assert_array_equal(y.grad.numpy(),6*seed)
            else:assert y.grad is None
        assert len(observed)>start
        gc.collect();assert (reference() is not None)==state['retain']


def test_callback_exception_retains_payload_for_a_valid_retry():
    x=torch.tensor([1.,2.],dtype=torch.float64,requires_grad=True);middle=x*2
    payload=np.array([2.,3.]);reference=weakref.ref(payload);mode={'fail':True}
    def build(data):
        def backward(g,needs):
            if mode['fail']:raise RuntimeError('Synthetic joint callback failure')
            return (g*data,)
        return joint_result(middle.numpy()*data,(middle,),backward)
    output=build(payload);del payload
    with pytest.raises(RuntimeError,match='Synthetic'):output.backward(torch.ones_like(output))
    assert reference() is not None and x.grad is None
    mode['fail']=False;output.backward(torch.ones_like(output));gc.collect()
    np.testing.assert_array_equal(x.grad.numpy(),[4.,6.]);assert reference() is None


@pytest.mark.parametrize('selective',[False,True])
def test_storage_guard_failure_does_not_free_a_still_owned_callback(selective):
    x=torch.tensor([1.,2.],dtype=torch.float64,requires_grad=True);y=torch.tensor([3.,4.],dtype=torch.float64,requires_grad=True)
    payload=np.array([2.,3.]);reference=weakref.ref(payload);output=weighted_pair(x,y,payload,selective);del payload
    with torch.no_grad():y.add_(1)
    with pytest.raises(UnsupportedOperation,match='modified'):output.backward(torch.ones_like(output))
    assert reference() is not None and x.grad is y.grad is None
    del output;gc.collect();assert reference() is None
