"""Gradient mode behavior against native contexts, recursive and generator calls."""
import inspect

import numpy as np
import pytest
import torch as native

from paperleet._vendor.torch import torch as lite


@pytest.mark.parametrize('api', [lite, native], ids=['lite', 'native'])
def test_nested_modes_factories_direct_setter_and_exception_restoration(api):
    before = api.is_grad_enabled()
    try:
        api.set_grad_enabled(True)
        x = api.tensor([2., 3.], requires_grad=True)
        with api.no_grad() as entered:
            assert entered is None and not api.is_grad_enabled()
            assert not x.square().requires_grad
            assert api.tensor([1.], requires_grad=True).requires_grad
            assert api.nn.Parameter(api.ones(1)).requires_grad
            with api.enable_grad() as entered:
                assert entered is None and api.is_grad_enabled()
                loss = x.square().sum()
            assert not api.is_grad_enabled()
            with pytest.raises(ValueError):
                with api.set_grad_enabled(True):
                    assert api.is_grad_enabled();raise ValueError('restore')
            assert not api.is_grad_enabled()
        assert api.is_grad_enabled()
        loss.backward();np.testing.assert_array_equal(x.grad.numpy(),[4.,6.])
        mode = api.set_grad_enabled(False)
        assert not api.is_grad_enabled()
        mode.__exit__(None,None,None)
        assert api.is_grad_enabled()
        for invalid in (None, 0, 1, np.bool_(True), 'yes'):
            with pytest.raises(TypeError):api.set_grad_enabled(invalid)
            assert api.is_grad_enabled()
    finally:api.set_grad_enabled(before)


@pytest.mark.parametrize('mode', ['no_grad', 'enable_grad', 'disabled', 'enabled'])
def test_recursive_decorators_and_error_calls_restore_outer_mode(mode):
    outcomes = []
    for api in (lite, native):
        before = api.is_grad_enabled()
        try:
            api.set_grad_enabled(True)
            decorator = (getattr(api,mode)() if mode in ('no_grad','enable_grad')
                         else api.set_grad_enabled(mode=='enabled'))
            @decorator
            def recursive(depth):
                flags.append(api.is_grad_enabled())
                if depth:return recursive(depth-1)
                return x.square()
            assert api.is_grad_enabled()  # Decorating undoes set_grad_enabled's constructor side effect.
            x=api.tensor([2.],requires_grad=True);flags=[]
            with api.no_grad():
                result=recursive(3)
                assert not api.is_grad_enabled()
            expected = mode in ('enable_grad','enabled')
            assert result.requires_grad==expected and flags==[expected]*4
            outcomes.append((flags,result.requires_grad))
            @decorator
            def failing():raise ValueError('failure')
            with pytest.raises(ValueError,match='failure'):failing()
            assert api.is_grad_enabled()
            @api.no_grad
            def plain(x):return x.square()
            @api.enable_grad
            def enabled(x):return x.square()
            assert not plain(x).requires_grad
            with api.no_grad():assert enabled(x).requires_grad
        finally:api.set_grad_enabled(before)
    assert outcomes[0]==outcomes[1]


@pytest.mark.parametrize('mode', ['no_grad','enable_grad','disabled','enabled'])
def test_generator_send_throw_close_return_and_caller_mode_match_native(mode):
    outcomes=[]
    for api in (lite,native):
        before=api.is_grad_enabled()
        try:
            api.set_grad_enabled(True)
            decorator=(getattr(api,mode)() if mode in ('no_grad','enable_grad')
                       else api.set_grad_enabled(mode=='enabled'))
            flags=[]
            @decorator
            def stream(x):
                try:
                    flags.append(('first',api.is_grad_enabled()))
                    received=yield x.square()
                    flags.append(('send',received,api.is_grad_enabled()))
                    try:yield x*3
                    except ValueError:
                        flags.append(('throw',api.is_grad_enabled()));yield x*4
                    return 17
                finally:flags.append(('close',api.is_grad_enabled()))
            assert inspect.isgeneratorfunction(stream) and api.is_grad_enabled()
            x=api.tensor([2.],requires_grad=True);it=stream(x)
            values=[next(it).requires_grad];assert api.is_grad_enabled()
            values.append(it.send(9).requires_grad);assert api.is_grad_enabled()
            values.append(it.throw(ValueError('handled')).requires_grad);assert api.is_grad_enabled()
            with pytest.raises(StopIteration) as done:next(it)
            assert done.value.value==17 and api.is_grad_enabled()
            second=stream(x);next(second);second.close();assert api.is_grad_enabled()
            outcomes.append((flags,values))
        finally:api.set_grad_enabled(before)
    assert outcomes[0]==outcomes[1]


@pytest.mark.parametrize('api',[lite,native],ids=['lite','native'])
def test_no_grad_keeps_forward_ad_while_enable_grad_restores_reverse_edges(api):
    before=api.is_grad_enabled()
    try:
        x=api.tensor([2.,3.],requires_grad=True)
        with api.no_grad():
            primal,tangent=api.func.jvp(lambda t:t.square(),(x,),(api.ones_like(x),))
            assert not primal.requires_grad and not tangent.requires_grad
            with api.enable_grad():
                output=x.square().sum()
            assert not api.is_grad_enabled()
        output.backward();np.testing.assert_array_equal(x.grad.numpy(),[4.,6.])
        np.testing.assert_array_equal(tangent.numpy(),[4.,6.])
    finally:api.set_grad_enabled(before)
