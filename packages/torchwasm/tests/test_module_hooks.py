"""NumPy-only callback workflows, also executed against exported real WASM."""
from collections import OrderedDict
import copy
import gc
import weakref

import numpy as np
import pytest

from torchwasm import UnsupportedOperation,as_torch,torch


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_hooked_training_gradient_and_checkpoint_continuation(dtype):
    kind=getattr(torch,dtype)
    x=np.array([[1.,2.],[-1.,3.]],dtype=dtype)
    targets=np.array([[.1],[.7]],dtype=dtype)
    initial=np.array([[.2,-.1]],dtype=dtype)
    def build():
        model=torch.nn.Linear(2,1,bias=False,dtype=kind)
        model.load_state_dict({'weight':torch.tensor(initial)})
        model.register_forward_pre_hook(lambda m,a:a[0]*2)
        model.register_forward_hook(lambda m,a,out:out*3+.4)
        def pack(m,state,prefix,metadata):
            metadata['format']=8
            state[prefix+'packed_weight']=state.pop(prefix+'weight')
        def unpack(m,state,prefix,metadata,strict,missing,unexpected,errors):
            if metadata.get('format')!=8:errors.append('unknown model format')
            state[prefix+'weight']=state.pop(prefix+'packed_weight')
        model.register_state_dict_post_hook(pack)
        model.register_load_state_dict_pre_hook(unpack)
        return model,torch.optim.SGD(model.parameters(),lr=.02,momentum=.3)
    model,optimizer=build();expected=initial.copy();velocity=np.zeros_like(expected)
    for _ in range(3):
        optimizer.zero_grad()
        predictions=model(torch.tensor(x))
        loss=(predictions-torch.tensor(targets)).square().mean();loss.backward()
        reference=(x*2)@expected.T*3+.4
        gradient=(2/len(x))*((reference-targets)*3).T@(x*2)
        np.testing.assert_allclose(model.weight.grad.numpy(),gradient,rtol=3e-6,atol=3e-6)
        velocity=.3*velocity+gradient;expected=expected-.02*velocity
        optimizer.step();np.testing.assert_allclose(model.weight.numpy(),expected,rtol=3e-6,atol=3e-6)
    state=copy.deepcopy({'model':model.state_dict(),'optimizer':optimizer.state_dict()})
    rebuilt,resumed=build();rebuilt.load_state_dict(state['model']);resumed.load_state_dict(state['optimizer'])
    for m,opt in [(model,optimizer),(rebuilt,resumed)]:
        opt.zero_grad();m(torch.tensor(x)).square().mean().backward();opt.step()
    np.testing.assert_array_equal(rebuilt.weight.numpy(),model.weight.numpy())


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_forward_hooks_keep_primal_tangent_and_reverse_tape(dtype):
    kind=getattr(torch,dtype);model=torch.nn.Identity()
    model.register_forward_pre_hook(lambda m,a:a[0]*2)
    model.register_forward_hook(lambda m,a,out:out.square())
    x=torch.tensor([.25,-.5],dtype=kind,requires_grad=True)
    v=torch.tensor([.3,.7],dtype=kind,requires_grad=True)
    value,tangent=torch.func.jvp(model,(x,),(v,));tangent.sum().backward()
    for actual,expected in [(value,4*x.numpy()**2),(tangent,8*x.numpy()*v.numpy()),
                            (x.grad,8*v.numpy()),(v.grad,8*x.numpy())]:
        np.testing.assert_allclose(actual.numpy(),expected,rtol=2e-6,atol=2e-6)


def test_kwargs_order_removal_context_and_direct_forward():
    class Shift(torch.nn.Module):
        def forward(self,x,*,shift=0):return x+shift
    model=Shift();events=[]
    def pre(m,args,kwargs):
        events.append('pre');return (args[0]*2,),{**kwargs,'shift':kwargs['shift']+1}
    model.register_forward_pre_hook(pre,with_kwargs=True)
    model.register_forward_pre_hook(lambda *a:events.append('first'),prepend=True)
    model.register_forward_hook(lambda m,a,k,out:out+k['shift'],with_kwargs=True)
    with model.register_forward_hook(lambda m,a,out:out*3) as handle:
        assert model(torch.tensor(2.),shift=1).item()==24
    handle.remove();assert model(torch.tensor(2.),shift=1).item()==8
    assert events==['first','pre','first','pre']
    assert model.forward(torch.tensor(2.),shift=1).item()==3


def test_always_call_runs_after_pre_error_and_does_not_replace_error():
    model=torch.nn.Identity();events=[]
    def bad_pre(*args):raise ValueError('original')
    def bad_cleanup(*args):events.append('bad');raise LookupError('secondary')
    model.register_forward_pre_hook(bad_pre)
    model.register_forward_hook(bad_cleanup,always_call=True)
    model.register_forward_hook(lambda m,a,k,out:events.append((a[0].item(),k,out)),with_kwargs=True,always_call=True)
    with pytest.warns(UserWarning,match='secondary'):
        with pytest.raises(ValueError,match='original'):model(torch.tensor(2.))
    assert events==['bad',(2.,{},None)]


def test_checkpoint_pre_post_order_and_shared_incompatible_keys():
    model=torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Identity())
    events=[]
    for name,m in model.named_modules():
        m.name=name
        m.register_state_dict_pre_hook(lambda m,p,k:events.append(('pre',m.name,p,k)))
        m.register_state_dict_post_hook(lambda m,s,p,meta:events.append(('post',m.name,p,meta['version'])))
    state=model.state_dict(keep_vars=True)
    assert events==[('pre','','',True),('pre','0','0.',True),('post','0','0.',1),
                    ('pre','1','1.',True),('post','1','1.',1),('post','','',1)]
    assert state['0.weight'] is model[0].weight
    model[0].register_load_state_dict_post_hook(lambda m,keys:keys.missing_keys.clear())
    model.register_load_state_dict_post_hook(lambda m,keys:keys.unexpected_keys.clear())
    result=model.load_state_dict({'extra':torch.ones(1)})
    assert not result.missing_keys and not result.unexpected_keys


def test_handle_alias_copy_and_lifetime():
    with as_torch():
        from torch.utils.hooks import RemovableHandle
        assert RemovableHandle is torch.utils.hooks.RemovableHandle
    primary,extra=OrderedDict(),OrderedDict()
    with RemovableHandle(primary,extra_dict=extra) as handle:
        primary[handle.id]='callback';extra[handle.id]=True
    assert not primary and not extra
    model=torch.nn.Identity();model.handle=model.register_forward_hook(lambda m,a,o:o*2,always_call=True)
    other=copy.deepcopy(model);other.handle.remove()
    assert other(torch.ones(1)).item()==1 and model(torch.ones(1)).item()==2
    reference=weakref.ref(model);handle=model.handle
    del model;gc.collect();assert reference() is None
    handle.remove();assert handle.hooks_dict_ref() is None


def test_generic_attribute_checks_and_typed_execution_boundaries():
    model=torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Dropout(),torch.nn.LayerNorm(1))
    def initialize(m):
        if getattr(m,'weight',None) is not None:
            with torch.no_grad():m.weight.fill_(2)
    model.apply(initialize)
    assert not hasattr(model[1],'weight')
    assert {'weight','bias'}<=set(dir(model[0]))
    with pytest.raises(AttributeError):model[1].unknown_attribute
    for method in ['cuda','to_empty','register_full_backward_hook']:
        with pytest.raises(UnsupportedOperation):getattr(model,method)(lambda *args:None)


def test_post_hook_wrong_returns_fail_and_callbacks_remain_removable():
    model=torch.nn.Identity()
    handle=model.register_state_dict_post_hook(lambda *a:{})
    with pytest.raises(RuntimeError,match='return None'):model.state_dict()
    handle.remove();assert not model.state_dict()
    handle=model.register_load_state_dict_post_hook(lambda *a:1)
    with pytest.raises(AssertionError,match='return None'):model.load_state_dict({})
    handle.remove();assert not model.load_state_dict({}).missing_keys
