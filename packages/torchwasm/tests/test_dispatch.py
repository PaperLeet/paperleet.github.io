"""Repeated public calls retain validation, dynamic modes and object lifetimes."""
import gc
import inspect
import itertools
import types
import weakref

import numpy as np
import pytest

from torchwasm import UnsupportedOperation, torch
from torchwasm._registry import OperatorRegistry


def registered(function, **policy):
    module = types.ModuleType('example')
    OperatorRegistry({'example': module}).add('example.run', function, **policy)
    return module.run


def test_repeated_mixed_signatures_preserve_python_argument_validation():
    def fixed(a, b=2, *, flag=3): return a, b, flag
    def positional(a, /, b=2, *, flag): return a, b, flag
    def variadic(a, /, *rest, flag=3, **extras): return a, rest, flag, extras
    def keywords(*, flag, **extras): return flag, extras
    names = ('a', 'b', 'flag', 'other')
    for function in (fixed, positional, variadic, keywords):
        call = registered(function, reverse='control')
        signature = inspect.signature(function)
        # Revisit valid patterns with different values after interleaved errors,
        # keyword orders, duplicate assignments and more than 32 valid patterns.
        for value in (1, 7):
            for count in range(5):
                args = tuple(range(value, value + count))
                for size in range(5):
                    for keys in itertools.permutations(names, size):
                        kwargs = {key: value + index for index, key in enumerate(keys)}
                        try:
                            signature.bind(*args, **kwargs)
                        except TypeError as error:
                            with pytest.raises(TypeError) as observed: call(*args, **kwargs)
                            assert str(observed.value) == str(error)
                        else:
                            # Python 3.13's binder can accept a missing positional-
                            # only parameter when **extras is present. The actual
                            # Python function still rejects it, as must dispatch.
                            try: expected = function(*args, **kwargs)
                            except TypeError as error:
                                with pytest.raises(TypeError) as observed: call(*args, **kwargs)
                                assert str(observed.value) == str(error)
                            else: assert call(*args, **kwargs) == expected
        assert inspect.signature(call) == signature


def test_value_errors_and_unsupported_options_remain_dynamic_after_warm_calls():
    def function(value, *, scale=1):
        if scale == 'broken': raise TypeError('workload bug')
        return value * scale
    call = registered(function, reverse='control', unsupported_keywords=('out',))
    for value in range(40): assert call(value, scale=2, out=None) == 2 * value
    with pytest.raises(TypeError, match='workload bug'): call(1, scale='broken', out=None)
    with pytest.raises(UnsupportedOperation, match='out'): call(1, scale=2, out=object())
    with pytest.raises(TypeError): call(1, scale=2, out=object(), typo=True)
    assert call(3, scale=4, out=None) == 12


def test_warmed_unvalidated_operator_rechecks_nested_tensors_and_ad_modes():
    call = registered(lambda payload: payload['rows'][0].square())
    for _ in range(4): np.testing.assert_array_equal(call({'rows': [torch.tensor([2.])]}).numpy(), [4.])
    tracked = torch.tensor([3.], requires_grad=True)
    with pytest.raises(UnsupportedOperation, match='reverse-mode'): call({'rows': [tracked]})
    with torch.no_grad():
        result = call({'rows': [tracked]})
        assert not result.requires_grad
        np.testing.assert_array_equal(result.numpy(), [9.])
    with pytest.raises(UnsupportedOperation, match='reverse-mode'): call({'rows': [tracked]})
    with pytest.raises(UnsupportedOperation, match='forward-mode'):
        torch.func.jvp(lambda x: call({'rows': [x]}), (torch.tensor([2.]),), (torch.ones(1),))
    np.testing.assert_array_equal(call({'rows': [tracked.detach()]}).numpy(), [9.])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_public_reduction_rechecks_values_dtypes_gradients_and_output_options(dtype):
    for _ in range(5): assert torch.sum(torch.ones(2), dim=0, out=None).item() == 2
    x = torch.tensor([2., 3.], dtype=dtype, requires_grad=True)
    output = torch.sum(x, dim=0, dtype=torch.float64, out=None)
    assert output.dtype == torch.float64
    output.backward()
    np.testing.assert_array_equal(x.grad.numpy(), [1., 1.])
    with torch.no_grad(): assert not torch.sum(x, dim=0, out=None).requires_grad
    assert torch.sum(x, dim=0, out=None).requires_grad
    with pytest.raises((IndexError, RuntimeError)): torch.sum(x, dim=100, out=None)
    with pytest.raises(UnsupportedOperation, match='out'): torch.sum(x, dim=0, out=torch.zeros(1))
    with pytest.raises(TypeError): torch.sum(x, input=x)
    with pytest.raises(TypeError): torch.sum(x, dim=0, typo=True)
    primal, tangent = torch.func.jvp(lambda value: torch.sum(value, dim=0, out=None),
                                    (x.detach(),), (torch.tensor([4., 5.], dtype=dtype),))
    assert primal.item() == 5 and tangent.item() == 9
    assert torch.sum(torch.tensor([2, 3]), dim=0, out=None).dtype == torch.int64


def test_dispatch_does_not_retain_argument_values_or_returned_tensors():
    call = registered(lambda value, **options: None, reverse='control')
    class Payload: pass
    refs = []
    for number in range(96):
        value = Payload(); other = torch.tensor([float(number)], requires_grad=True)
        refs.extend((weakref.ref(value), weakref.ref(other)))
        assert call(value, **{f'option_{number}': other}) is None
    del value, other
    gc.collect()
    assert all(reference() is None for reference in refs)
    refs = []
    for number in range(40):
        tensor = torch.tensor([float(number)])
        output = torch.exp(input=tensor)
        refs.extend((weakref.ref(tensor), weakref.ref(output)))
    del tensor, output
    gc.collect()
    assert all(reference() is None for reference in refs)


def test_keyword_name_subclasses_do_not_retain_user_payloads():
    class Keyword(str): pass
    call = registered(lambda **options: None, reverse='control')
    refs = []
    for number in range(40):
        key = Keyword(f'option_{number}')
        key.payload = torch.tensor([float(number)], requires_grad=True)
        refs.extend((weakref.ref(key), weakref.ref(key.payload)))
        assert call(**{key: None}) is None
    del key
    gc.collect()
    assert all(reference() is None for reference in refs)
