"""Python feature probes coexist with typed unsupported-backend diagnostics."""
import copy
import gc
import importlib
import pickle
import weakref

import pytest
from torchwasm import UnsupportedOperation, as_torch, create_torch, torch


NAMESPACES = tuple(torch._compat_modules)
MISSING = 'not_a_real_torch_symbol_37'


@pytest.mark.parametrize('path', NAMESPACES)
def test_namespace_missing_symbols_support_default_and_typed_lookup(path):
    module = torch._compat_modules[path]
    sentinel = object()
    for name in (MISSING, '', '不存在的接口'):
        assert not hasattr(module, name)
        assert getattr(module, name, sentinel) is sentinel
        with pytest.raises(AttributeError) as observed:getattr(module, name)
        error = observed.value
        assert isinstance(error, UnsupportedOperation) and isinstance(error, NotImplementedError)
        assert error.name == name and error.obj is module
        assert 'outside the CPU/WASM' in str(error)
    for name in ('_private_missing', '__not_a_protocol__'):
        assert not hasattr(module, name)
        with pytest.raises(AttributeError) as observed:getattr(module, name)
        assert not isinstance(observed.value, UnsupportedOperation)
    sentinel = object()
    try:
        setattr(module, MISSING, sentinel)
        assert hasattr(module, MISSING) and getattr(module, MISSING) is sentinel
    finally:delattr(module, MISSING)
    assert not hasattr(module, MISSING)


@pytest.mark.parametrize('kind', ['Tensor','Parameter','Optimizer','Module'])
def test_object_optional_attribute_protocol_and_existing_values(kind):
    x = torch.tensor([1.], requires_grad=True)
    target = {'Tensor':x, 'Parameter':torch.nn.Parameter(x),
              'Optimizer':torch.optim.SGD([x], lr=.1), 'Module':torch.nn.Linear(1,1)}[kind]
    sentinel = object()
    assert not hasattr(target, MISSING) and getattr(target, MISSING, sentinel) is sentinel
    with pytest.raises(AttributeError) as observed:getattr(target, MISSING)
    assert observed.value.name == MISSING and observed.value.obj is target
    if kind != 'Module':assert isinstance(observed.value, UnsupportedOperation)
    assert getattr(target, 'training' if kind=='Module' else 'param_groups' if kind=='Optimizer' else 'dtype', sentinel) is not sentinel
    for name in ('_private_missing','__missing_protocol__'):
        assert not hasattr(target, name) and getattr(target, name, sentinel) is sentinel


@pytest.mark.parametrize('path', ['torch','torch.nn','torch.nn.functional','torch.linalg','torch.optim'])
def test_standard_import_failure_does_not_install_a_fake_symbol(path):
    with as_torch() as namespace:
        module = importlib.import_module(path)
        assert module is namespace._compat_modules[path]
        with pytest.raises(ImportError):exec(f'from {path} import {MISSING}', {})
        assert MISSING not in vars(module)
        assert not hasattr(module, MISSING)
        with pytest.raises(UnsupportedOperation):getattr(module, MISSING)


@pytest.mark.parametrize('kind', ['namespace','tensor','optimizer'])
def test_missing_symbol_exception_roundtrips_without_serializing_its_owner(kind):
    x = torch.tensor([1.], requires_grad=True)
    owner = torch.nn if kind=='namespace' else x if kind=='tensor' else torch.optim.SGD([x], lr=.1)
    try:getattr(owner, MISSING)
    except AttributeError as error:
        assert error.obj is owner
        add_note = getattr(error, 'add_note', None)
        if add_note:add_note('optional diagnostic')
        for operation in (copy.copy, copy.deepcopy, lambda value:pickle.loads(pickle.dumps(value))):
            restored = operation(error)
            reference = operation(AttributeError(str(error), name=MISSING, obj=owner))
            assert type(restored) is type(error) and restored.args == error.args
            # Python 3.10 drops name as well as obj; newer interpreters retain
            # name. Follow the actual standard exception's serialization rule.
            assert restored.name == reference.name and restored.obj is reference.obj is None
            assert isinstance(restored, AttributeError) and isinstance(restored, UnsupportedOperation)
            if add_note:assert restored.__notes__ == ['optional diagnostic']
    else:raise AssertionError('Missing symbol did not fail')


def test_valid_call_unsupported_option_is_not_hidden_by_attribute_defaulting():
    class Workload:
        @property
        def result(self):return torch.nn.functional.relu(torch.tensor([-1.]), inplace=True)
    for operation in (hasattr, lambda obj,key:getattr(obj,key,None)):
        with pytest.raises(UnsupportedOperation) as observed:operation(Workload(), 'result')
        assert not isinstance(observed.value, AttributeError)
    x=torch.tensor([2.],requires_grad=True)
    with pytest.raises(UnsupportedOperation) as observed:(x//2).sum().backward()
    assert not isinstance(observed.value, AttributeError)


def test_builtin_name_validation_and_real_descriptor_errors_are_preserved():
    for target in (torch, torch.nn, torch.tensor([1.])):
        for name in (None, 1):
            with pytest.raises(TypeError):hasattr(target, name)
            with pytest.raises(TypeError):getattr(target, name, None)
    class Broken:
        @property
        def value(self):raise ValueError('actual workload error')
    with pytest.raises(ValueError,match='actual workload'):hasattr(Broken(),'value')
    with pytest.raises(ValueError,match='actual workload'):getattr(Broken(),'value',None)


def test_optional_tensor_probes_do_not_retain_tensors_or_alias_namespaces():
    def probe():
        x=torch.tensor([1.]);reference=weakref.ref(x)
        for _ in range(50):
            assert not hasattr(x,MISSING)
            assert getattr(x,MISSING,None) is None
        return reference
    reference=probe();gc.collect();assert reference() is None
    def namespace_probe():
        namespace=create_torch(name='feature_probe');reference=weakref.ref(namespace)
        assert not hasattr(namespace,MISSING) and not hasattr(namespace.nn,MISSING)
        return reference
    reference=namespace_probe();gc.collect();assert reference() is None


@pytest.mark.parametrize('kind',['namespace','tensor'])
def test_schedules_profile_keeps_lookup_and_unsupported_mode_protocols(kind):
    from torchwasm import _legacy
    namespace=_legacy.install();x=namespace.tensor([.1,.2])
    target=namespace if kind=='namespace' else x
    sentinel=object()
    assert not hasattr(target,MISSING) and getattr(target,MISSING,sentinel) is sentinel
    with pytest.raises(AttributeError) as observed:getattr(target,MISSING)
    assert isinstance(observed.value,UnsupportedOperation)
    assert observed.value.name==MISSING and observed.value.obj is target
    with pytest.raises(UnsupportedOperation) as observed:namespace.tensor([1.],requires_grad=True)
    assert not isinstance(observed.value,AttributeError)
    np_result=x.square().numpy()
    assert np_result.shape==(2,) and abs(float(np_result.sum())-.05)<1e-6
