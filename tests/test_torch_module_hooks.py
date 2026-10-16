"""Public module callbacks checked against independently installed native Torch."""
from collections import OrderedDict
import copy
import gc
import warnings
import weakref

import numpy as np
import pytest
import torch as native

from stridetorch import UnsupportedOperation, torch


def array(value):
    return value.detach().numpy().copy()


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('with_kwargs', [False, True])
@pytest.mark.parametrize('prepend', [False, True])
def test_forward_replacements_order_and_gradients(dtype, with_kwargs, prepend):
    results = []
    for api in (native, torch):
        events = []
        class Model(api.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = api.nn.Parameter(api.tensor([2., -3.], dtype=getattr(api,dtype)))
            def forward(self, x, offset=0.):
                events.append(('forward', offset))
                return x*self.weight+offset
        model = Model()
        def pre(module, args, kwargs=None):
            events.append(('pre', len(args)))
            if with_kwargs:
                return (args[0]*2,), {**kwargs, 'offset': kwargs['offset']+1}
            return (args[0]*2, args[1]+1)
        def post(module, args, *remaining):
            output = remaining[-1]
            events.append(('post', len(args)))
            return output*3
        model.register_forward_pre_hook(lambda m,a: events.append(('first_pre',len(a))))
        model.register_forward_pre_hook(pre, prepend=prepend, with_kwargs=with_kwargs)
        model.register_forward_hook(lambda m,a,o: events.append(('first_post',len(a))))
        handle = model.register_forward_hook(post, prepend=prepend, with_kwargs=with_kwargs)
        x = api.tensor([.4, -.2], dtype=getattr(api,dtype), requires_grad=True)
        output = model(x, offset=.5) if with_kwargs else model(x,.5)
        output.square().sum().backward()
        results.append((array(output),array(x.grad),array(model.weight.grad),events))
        handle.remove();handle.remove()
    for left,right in zip(results[0][:3],results[1][:3]):
        np.testing.assert_allclose(left,right,rtol=2e-6,atol=2e-6)
    assert results[0][3] == results[1][3]


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_hooked_module_jvp_and_reverse_over_forward(dtype):
    results=[]
    for api in (native,torch):
        model=api.nn.Tanh()
        model.register_forward_pre_hook(lambda m,args:args[0]*2)
        model.register_forward_hook(lambda m,args,out:out.square())
        x=api.tensor([-.4,.7],dtype=getattr(api,dtype),requires_grad=True)
        tangent=api.tensor([.3,-.2],dtype=getattr(api,dtype),requires_grad=True)
        value,jvp=api.func.jvp(model,(x,),(tangent,))
        jvp.sum().backward()
        results.append(tuple(array(v) for v in (value,jvp,x.grad,tangent.grad)))
    for expected,actual in zip(*results):np.testing.assert_allclose(actual,expected,rtol=4e-6,atol=2e-6)


@pytest.mark.parametrize('stage', ['pre','forward','post','always','success'])
@pytest.mark.parametrize('with_kwargs', [False,True])
def test_always_call_exceptions_preserve_original_and_partial_output(stage,with_kwargs):
    results=[]
    for api in (native,torch):
        events=[]
        class Model(api.nn.Module):
            def forward(self,x,*,offset=0):
                events.append(('forward',offset))
                if stage=='forward':raise ValueError('original forward')
                return x+offset
        model=Model()
        def pre(m,args):
            events.append(('pre',))
            if stage=='pre':raise ValueError('original pre')
        def post(m,args,out):
            events.append(('post',float(out.item())))
            if stage=='post':raise ValueError('original post')
            return out*2
        def always(m,args,*rest):
            out=rest[-1]
            events.append(('always',None if out is None else float(out.item()),len(rest)))
            if stage=='always' or stage=='forward':raise LookupError('cleanup failure')
            return None if out is None else out+1
        model.register_forward_pre_hook(pre)
        model.register_forward_hook(post)
        model.register_forward_hook(always,with_kwargs=with_kwargs,always_call=True)
        model.register_forward_hook(lambda m,a,o:events.append(('last',None if o is None else float(o.item()))),always_call=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:out=model(api.tensor(2.),offset=3);outcome=('value',out.item())
            except Exception as error:outcome=(type(error).__name__,str(error))
        results.append((events,outcome,len(caught),[w.category for w in caught]))
    assert results[0]==results[1]


@pytest.mark.parametrize('replacement',[(),(1,),[(),{}],42])
def test_invalid_kwargs_pre_hook_result(replacement):
    for api in (native,torch):
        model=api.nn.Identity()
        model.register_forward_pre_hook(lambda *args:replacement,with_kwargs=True)
        with pytest.raises(RuntimeError,match='pre-hook'):model(api.ones(1))


def test_nested_shared_modules_snapshot_mutation_and_direct_forward_bypass():
    results=[]
    for api in (native,torch):
        events=[]; shared=api.nn.Identity();model=api.nn.Sequential(shared,shared)
        removed=shared.register_forward_hook(lambda *args:events.append('queued'))
        added=[]
        def first(m,args,out):
            events.append('first');removed.remove()
            if not added:added.append(m.register_forward_hook(lambda *args:events.append('new')))
        shared.register_forward_hook(first,prepend=True)
        model.register_forward_pre_hook(lambda *args:events.append('root_pre'))
        model.register_forward_hook(lambda *args:events.append('root_post'))
        x=api.ones(1);assert shared.forward(x) is x
        assert model(x) is x
        results.append(events)
    assert results[0]==results[1]==['root_pre','first','queued','first','new','root_post']


def test_pre_hook_adds_post_hook_and_replaces_forward_for_next_call():
    results=[]
    for api in (native,torch):
        events=[];model=api.nn.Identity();handles=[]
        def pre(m,args):
            if not handles:
                handles.append(m.register_forward_hook(lambda m,a,o:o*2))
                m.forward=lambda x:x+3
            events.append('pre')
        model.register_forward_pre_hook(pre)
        results.append(([model(api.ones(1)).item(),model(api.ones(1)).item()],events))
    assert results[0]==results[1]==([2.,8.],['pre','pre'])


@pytest.mark.parametrize('kind',['pre','post','save_pre','save_post','load_pre','load_post'])
def test_self_removal_and_registration_during_callback_dispatch(kind):
    results=[]
    for api in (native,torch):
        events=[];model=api.nn.Identity();registered=[]
        method={'pre':'register_forward_pre_hook','post':'register_forward_hook',
                'save_pre':'register_state_dict_pre_hook','save_post':'register_state_dict_post_hook',
                'load_pre':'register_load_state_dict_pre_hook','load_post':'register_load_state_dict_post_hook'}[kind]
        def hook(*args):
            events.append('first');handle.remove()
            registered.append(getattr(model,method)(lambda *a:events.append('new')))
        handle=getattr(model,method)(hook)
        getattr(model,method)(lambda *a:events.append('second'))
        try:
            if kind.startswith('save'):model.state_dict()
            elif kind.startswith('load'):model.load_state_dict({})
            else:model(api.ones(1))
            outcome='ok'
        except RuntimeError:outcome='mutation_error'
        results.append((events,outcome))
    assert results[0]==results[1]


@pytest.mark.parametrize('keep_vars',[False,True])
@pytest.mark.parametrize('prefix',['','network.'])
def test_checkpoint_hook_order_metadata_renaming_and_restore(keep_vars,prefix):
    results=[]
    for api in (native,torch):
        events=[]
        model=api.nn.Sequential(api.nn.Linear(2,1,bias=False),api.nn.Identity())
        with api.no_grad():model[0].weight[:]=api.tensor([[2.,3.]])
        def pre(m,prefix,keep):events.append(('save_pre',m.tag,prefix,keep))
        def post(m,state,prefix,metadata):
            events.append(('save_post',m.tag,prefix,tuple(state),metadata['version']))
            metadata['format']='packed'
            if m.tag=='0':state[prefix+'packed']=state.pop(prefix+'weight')*2
        def load_pre(m,state,prefix,metadata,strict,missing,unexpected,errors):
            events.append(('load_pre',m.tag,prefix,metadata.get('format'),strict,hasattr(state,'_metadata')))
            if m.tag=='0':state[prefix+'weight']=state.pop(prefix+'packed')/2
        def load_post(m,keys):events.append(('load_post',m.tag,tuple(keys.missing_keys),tuple(keys.unexpected_keys)))
        for name,m in model.named_modules():
            m.tag=name or 'root'
            m.register_state_dict_pre_hook(pre);m.register_state_dict_post_hook(post)
            m.register_load_state_dict_pre_hook(load_pre);m.register_load_state_dict_post_hook(load_post)
        state=model.state_dict(prefix=prefix,keep_vars=keep_vars)
        assert state[prefix+'0.packed'].requires_grad==keep_vars
        # A prefix is a save-time namespace; strip it to load into the same root.
        restored=OrderedDict((k[len(prefix):],v) for k,v in state.items())
        restored._metadata=OrderedDict((k[len(prefix):] if k.startswith(prefix) else '',v)
                                       for k,v in state._metadata.items())
        with api.no_grad():model[0].weight.zero_()
        keys=model.load_state_dict(restored)
        assert list(restored)==['0.packed']
        results.append((events,array(model[0].weight),keys.missing_keys,keys.unexpected_keys))
    assert results[0][0]==results[1][0]
    np.testing.assert_array_equal(results[0][1],results[1][1]);assert results[0][2:]==results[1][2:]


@pytest.mark.parametrize('strict',[False,True])
@pytest.mark.parametrize('clear',[False,True])
def test_load_post_hook_changes_shared_incompatible_lists(strict,clear):
    results=[]
    for api in (native,torch):
        events=[];model=api.nn.Sequential(api.nn.Linear(1,1))
        def post(m,keys):
            events.append((m.tag,tuple(keys.missing_keys),tuple(keys.unexpected_keys)))
            if clear:keys.missing_keys.clear();keys.unexpected_keys.clear()
            else:keys.missing_keys.append('synthetic')
        model.tag='root';model[0].tag='child'
        model.register_load_state_dict_post_hook(post);model[0].register_load_state_dict_post_hook(post)
        try:
            result=model.load_state_dict({'stray':api.ones(1)},strict=strict)
            outcome=(result.missing_keys,result.unexpected_keys)
        except RuntimeError:outcome='error'
        results.append((events,outcome))
    assert results[0]==results[1]


@pytest.mark.parametrize('kind',['save_post','load_post','load_error'])
def test_invalid_checkpoint_hook_returns_and_error_messages(kind):
    for api in (native,torch):
        model=api.nn.Identity()
        if kind=='save_post':
            model.register_state_dict_post_hook(lambda *args:{})
            with pytest.raises(RuntimeError):model.state_dict()
        elif kind=='load_post':
            model.register_load_state_dict_post_hook(lambda *args:1)
            with pytest.raises(AssertionError):model.load_state_dict({})
        else:
            model.register_load_state_dict_pre_hook(lambda *args:args[-1].append('checkpoint rejected'))
            with pytest.raises(RuntimeError,match='checkpoint rejected'):model.load_state_dict({},strict=False)


def test_handles_context_copy_and_weak_lifetime():
    for api in (native,torch):
        model=api.nn.Identity()
        with model.register_forward_hook(lambda m,args,out:out*2) as handle:
            assert model(api.ones(1)).item()==2
        handle.remove();assert model(api.ones(1)).item()==1
        model.handle=model.register_forward_hook(lambda m,args,out:out*3,with_kwargs=False,always_call=True)
        cloned=copy.deepcopy(model)
        cloned.handle.remove()
        assert cloned(api.ones(1)).item()==1
        assert model(api.ones(1)).item()==3
        reference=weakref.ref(model);handle=model.handle
        del model;gc.collect();assert reference() is None
        handle.remove()


def test_python_attribute_introspection_and_generic_initialization():
    for api in (native,torch):
        model=api.nn.Sequential(api.nn.Linear(2,2),api.nn.Dropout(),api.nn.LayerNorm(2))
        seen=[]
        def initialize(module):
            if hasattr(module,'weight') and module.weight is not None:
                with api.no_grad():module.weight.fill_(2)
                seen.append(type(module).__name__)
        model.apply(initialize)
        assert seen==['Linear','LayerNorm']
        assert not hasattr(model[1],'weight')
        assert getattr(model[1],'optional_setting',17)==17
        for module in model.modules():
            with pytest.raises(AttributeError):module.unknown_public_attribute
        root=api.nn.Module();root.register_parameter('sample',api.nn.Parameter(api.ones(1)))
        root.register_buffer('sample_buffer',None);root.add_module('0',model)
        assert {'sample','sample_buffer'}<=set(dir(root)) and '0' not in dir(root)


def test_dynamic_registration_collision_and_computed_submodule():
    for api in (native,torch):
        class Model(api.nn.Module):
            @property
            def virtual(self):return self.__dict__['_hidden']
            @property
            def reserved(self):return 12
            @property
            def unavailable(self):raise AttributeError('not set')
        model=Model();model.__dict__['_hidden']=api.nn.Identity()
        assert model.get_submodule('virtual') is model.virtual
        with pytest.raises(KeyError):model.register_buffer('reserved',api.ones(1))
        model.register_buffer('unavailable',api.ones(1));assert model.unavailable.item()==1


def test_unsupported_known_module_methods_remain_typed():
    model=torch.nn.Identity()
    for name in ['cuda','to_empty','compile','register_full_backward_hook']:
        assert hasattr(model,name)
        with pytest.raises(UnsupportedOperation):getattr(model,name)()


@pytest.mark.parametrize('kind',['tuple','dict','number'])
def test_hooks_can_replace_structured_outputs(kind):
    results=[]
    for api in (native,torch):
        class Model(api.nn.Module):
            def forward(self,x):return {'features':x.square(),'tag':'model'}
        model=Model()
        def post(m,args,out):
            if kind=='tuple':return (out['features']*2,out['tag'])
            if kind=='dict':return {'values':out['features']*2}
            return 17
        model.register_forward_hook(post)
        x=api.tensor([2.,-3.],requires_grad=True);out=model(x)
        if kind=='number':results.append(out);continue
        value=out[0] if kind=='tuple' else out['values']
        value.sum().backward();results.append((array(value).tolist(),array(x.grad).tolist()))
    assert results[0]==results[1]


@pytest.mark.parametrize('method',['register_forward_hook','register_state_dict_post_hook','register_load_state_dict_pre_hook'])
def test_hook_registration_requires_module_initialization(method):
    for api in (native,torch):
        model=api.nn.Module.__new__(api.nn.Module)
        with pytest.raises(AttributeError):getattr(model,method)(lambda *a:None)


def test_base_exception_skips_always_call():
    for api in (native,torch):
        events=[];model=api.nn.Identity()
        def pre(*args):raise SystemExit(7)
        model.register_forward_pre_hook(pre)
        model.register_forward_hook(lambda *args:events.append('always'),always_call=True)
        with pytest.raises(SystemExit):model(api.ones(1))
        assert not events


def test_registration_and_removal_on_models_with_frozen_attributes():
    for api in (native,torch):
        class Frozen(api.nn.Identity):
            def __init__(self):
                super().__init__();object.__setattr__(self,'locked',True)
            def __setattr__(self,name,value):
                if self.__dict__.get('locked'):raise RuntimeError('attributes are locked')
                super().__setattr__(name,value)
        model=Frozen();handle=model.register_forward_hook(lambda m,a,o:o*2)
        assert model(api.ones(1)).item()==2
        handle.remove();assert model(api.ones(1)).item()==1
        assert model(api.ones(1)).item()==1
