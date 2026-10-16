"""Native Python-facing feature discovery on modules, tensors and optimizers."""
import importlib
import pytest
import torch as native
from stridetorch import torch as lite


PATHS = tuple(lite._compat_modules)
MISSING = 'not_a_real_torch_symbol_37'


@pytest.mark.parametrize('path', PATHS)
def test_every_advertised_namespace_matches_native_optional_lookup(path):
    expected=importlib.import_module(path);actual=lite._compat_modules[path]
    sentinel=object()
    for name in (MISSING,'_not_a_protocol_37','不存在的接口',''):
        assert hasattr(actual,name)==hasattr(expected,name)==False
        assert getattr(actual,name,sentinel) is getattr(expected,name,sentinel) is sentinel
        for target in (actual,expected):
            with pytest.raises(AttributeError):getattr(target,name)


@pytest.mark.parametrize('kind',['Tensor','Parameter','Optimizer','Module'])
def test_native_object_feature_detection_and_dynamic_attributes(kind):
    for lib in (lite,native):
        x=lib.tensor([1.],requires_grad=True)
        target={'Tensor':x,'Parameter':lib.nn.Parameter(x),'Optimizer':lib.optim.SGD([x],lr=.1),'Module':lib.nn.Linear(1,1)}[kind]
        sentinel=object()
        assert not hasattr(target,MISSING) and getattr(target,MISSING,sentinel) is sentinel
        with pytest.raises(AttributeError):getattr(target,MISSING)
        setattr(target,MISSING,sentinel);assert hasattr(target,MISSING) and getattr(target,MISSING) is sentinel
        delattr(target,MISSING);assert not hasattr(target,MISSING)
