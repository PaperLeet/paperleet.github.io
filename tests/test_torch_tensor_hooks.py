"""Tensor reverse callbacks compared with independent native Torch execution."""
import copy
import gc
import weakref

import numpy as np
import pytest
import torch as native

from torchwasm import UnsupportedOperation, torch


def close(actual, expected):
    np.testing.assert_allclose(actual.detach().numpy(), expected.detach().numpy(), rtol=3e-6, atol=2e-6)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('kind', ['leaf', 'nonleaf'])
@pytest.mark.parametrize('method', ['backward', 'grad', 'inputs'])
def test_gradient_replacement_retention_and_post_accumulation(dtype, kind, method):
    results = []
    for api in (torch, native):
        x = api.tensor([2., 3.], dtype=getattr(api, dtype), requires_grad=True)
        middle = x.square(); target = x if kind == 'leaf' else middle
        events = []; target.retain_grad()
        def first(g):
            events.append(('first', g.tolist(), api.is_grad_enabled(), g.requires_grad))
            return g * 2
        target.register_hook(first)
        target.register_hook(lambda g: events.append(('second', g.tolist(), api.is_grad_enabled(), g.requires_grad)))
        if kind == 'leaf':
            target.register_post_accumulate_grad_hook(lambda p: events.append(('post', p.grad.tolist(), api.is_grad_enabled(), p is target)))
        output = middle.square().sum()
        value = api.autograd.grad(output, target)[0] if method == 'grad' else None
        if method == 'backward': output.backward()
        elif method == 'inputs': output.backward(inputs=[target])
        results.append((events, target.grad, value, x.grad))
    assert results[0][0] == results[1][0]
    for actual, expected in zip(results[0][1:], results[1][1:]):
        if actual is None or expected is None: assert actual is expected
        else: close(actual, expected)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_shared_roots_call_hook_once_on_combined_gradient(dtype):
    results = []
    for api in (torch, native):
        x = api.tensor([.2, .5], dtype=getattr(api, dtype), requires_grad=True)
        middle = x.exp(); events = []; middle.retain_grad()
        middle.register_hook(lambda g: events.append(g.tolist()))
        a, b = middle.square(), middle * 3
        gradients = api.autograd.grad((a, b, a), (x, middle, x), (api.ones_like(a), api.ones_like(b), api.ones_like(a)), retain_graph=True)
        api.autograd.backward((a, b)) if a.numel() == 1 else api.autograd.backward((a, b), (api.ones_like(a), api.ones_like(b)))
        assert len(events) == 2
        results.append((events, gradients, x.grad, middle.grad))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=2e-6)
    for actual, expected in zip(results[0][1], results[1][1]): close(actual, expected)
    close(results[0][2], results[1][2]); close(results[0][3], results[1][3])


@pytest.mark.parametrize('post', [False, True])
@pytest.mark.parametrize('change', ['add', 'remove'])
def test_registration_mutation_uses_phase_snapshot(post, change):
    for api in (torch, native):
        x = api.tensor(2., requires_grad=True); events = []; handles = []
        register = x.register_post_accumulate_grad_hook if post else x.register_hook
        def first(g):
            events.append('first')
            if change == 'remove': handles[1].remove()
            else: handles.append(register(lambda g: events.append('added')))
        handles.extend([register(first), register(lambda g: events.append('second'))])
        x.backward(); x.backward()
        assert events == (['first', 'second', 'first'] if change == 'remove' else ['first', 'second', 'first', 'second', 'added'])


def test_regular_phase_can_install_first_post_accumulate_hook():
    for api in (torch, native):
        x = api.tensor(2., requires_grad=True); events = []
        def first(g): x.register_post_accumulate_grad_hook(lambda p: events.append(p.grad.item()))
        x.register_hook(first); x.square().backward()
        assert events == [4.]


@pytest.mark.parametrize('post', [False, True])
@pytest.mark.parametrize('kind', ['number', 'dtype', 'shape', 'tensor', 'none', 'noncallable'])
def test_invalid_callback_results_and_recovery(post, kind):
    for api in (torch, native):
        x = api.ones(2, requires_grad=True)
        replacement = {'number': 3, 'dtype': api.ones(2, dtype=api.float64), 'shape': api.ones(3), 'tensor': api.ones(2), 'none': None, 'noncallable': None}[kind]
        hook = 3 if kind == 'noncallable' else lambda g: replacement
        handle = (x.register_post_accumulate_grad_hook if post else x.register_hook)(hook)
        expected = TypeError if kind in ['number', 'noncallable'] else RuntimeError
        fail = kind in ['number', 'dtype', 'shape', 'noncallable'] or post and kind == 'tensor'
        output = x.sum()
        if fail:
            with pytest.raises(expected): output.backward()
        else: output.backward()
        assert api.is_grad_enabled()
        handle.remove(); x.grad = None; output.backward(); assert x.grad.tolist() == [1., 1.]


@pytest.mark.parametrize('outer', [False, True])
@pytest.mark.parametrize('post', [False, True])
def test_callback_grad_mode_changes_are_scoped_to_node(outer, post):
    for api in (torch, native):
        x = api.tensor(2., requires_grad=True); seen = []
        register = x.register_post_accumulate_grad_hook if post else x.register_hook
        def first(g): seen.append(api.is_grad_enabled()); api.set_grad_enabled(True)
        register(first); register(lambda g: seen.append(api.is_grad_enabled()))
        with api.set_grad_enabled(outer):
            x.backward(); assert api.is_grad_enabled() == outer
        assert seen == [False, True] and api.is_grad_enabled()


def test_mode_change_flows_from_regular_to_post_but_not_to_parent_node():
    for api in (torch, native):
        x = api.tensor(2., requires_grad=True); middle = x.square(); seen = []
        def first(g): seen.append(('middle', api.is_grad_enabled())); api.set_grad_enabled(True)
        middle.register_hook(first)
        def leaf(g): seen.append(('leaf', api.is_grad_enabled())); api.set_grad_enabled(True)
        x.register_hook(leaf)
        x.register_post_accumulate_grad_hook(lambda p: seen.append(('post', api.is_grad_enabled())))
        middle.backward(); assert seen == [('middle', False), ('leaf', False), ('post', True)]
        assert api.is_grad_enabled()


@pytest.mark.parametrize('post', [False, True])
def test_callback_exception_restores_mode_and_handle_removal(post):
    for api in (torch, native):
        x = api.tensor(2., requires_grad=True)
        def broken(g): api.set_grad_enabled(True); raise ValueError('broken callback')
        with (x.register_post_accumulate_grad_hook if post else x.register_hook)(broken):
            with api.no_grad(), pytest.raises(ValueError, match='broken callback'): x.backward()
            assert api.is_grad_enabled()
        x.grad = None; x.backward(); assert x.grad.item() == 1.


def test_retained_hook_survives_intermediate_wrapper_collection_and_graph_release():
    for api in (torch, native):
        x = api.tensor(2., requires_grad=True); middle = x * 2; events = []
        handle = middle.register_hook(lambda g: events.append(g.item()) or g * 3)
        output = middle + 1; reference = weakref.ref(middle); del middle; gc.collect()
        assert reference() is None
        output.backward(); assert events == [1.] and x.grad.item() == 6.
        handle.remove(); x.grad = None
        with pytest.raises(RuntimeError, match='freed|second time'): output.backward()
        assert events == [1.]


@pytest.mark.parametrize('post', [False, True])
def test_handles_are_weak_and_deepcopy_drops_tensor_callbacks(post):
    for api in (torch, native):
        x = api.tensor(2., requires_grad=True); events = []
        handle = (x.register_post_accumulate_grad_hook if post else x.register_hook)(lambda g: events.append('called'))
        y = copy.deepcopy(x); y.backward(); assert events == []
        ref = weakref.ref(x); del x; gc.collect(); assert ref() is None
        handle.remove(); handle.remove()


def test_requested_inputs_and_unused_materialized_inputs_do_not_fire_unneeded_hooks():
    for api in (torch, native):
        x = api.tensor(2., requires_grad=True); middle = x.square(); unused = api.tensor(3., requires_grad=True); events = []
        x.register_hook(lambda g: events.append('leaf')); middle.register_hook(lambda g: events.append('middle'))
        unused.register_hook(lambda g: events.append('unused'))
        middle.register_hook(lambda g: g * 2)
        a, b = api.autograd.grad(middle, (middle, unused), materialize_grads=True)
        assert events == ['middle'] and a.item() == 2. and b.item() == 0.
        assert x.grad is unused.grad is None


def test_post_hook_inline_sgd_update_and_gradient_clear():
    for api in (torch, native):
        x = api.tensor([2., 3.], requires_grad=True); old = x.detach().clone(); events = []
        def update(p):
            assert not api.is_grad_enabled()
            p.add_(p.grad, alpha=-.1); events.append(p.grad.tolist()); p.grad = None
        x.register_post_accumulate_grad_hook(update)
        x.square().sum().backward()
        np.testing.assert_allclose(x.detach().numpy(), old.numpy() * .8, rtol=2e-6)
        assert events == [[4., 6.]] and x.grad is None


def test_regular_hook_can_disable_leaf_accumulation():
    for api in (torch, native):
        x = api.tensor(2., requires_grad=True)
        def disable(g): x.requires_grad_(False)
        x.register_hook(disable); x.square().backward(); assert x.grad is None
        x.requires_grad_(True)
        with pytest.raises(RuntimeError): api.ones(2).register_hook(lambda g: g)
        with pytest.raises(RuntimeError): (x * 2).register_post_accumulate_grad_hook(lambda p: None)


def test_register_inside_forward_ad_is_explicitly_unsupported():
    x = torch.tensor(2., requires_grad=True)
    with pytest.raises(UnsupportedOperation, match='forward AD'):
        torch.func.jvp(lambda v: v.register_hook(lambda g: g), (x,), (torch.ones_like(x),))


@pytest.mark.parametrize('method', ['backward', 'grad'])
@pytest.mark.parametrize('live', [False, True])
def test_returned_hook_tensor_identity_and_graph_at_requested_input(method, live):
    for api in (torch, native):
        x = api.tensor(2., requires_grad=True); stored = []
        def replace(g):
            with api.enable_grad(): value = x.square() if live else api.tensor(7.)
            stored.append(value); return value
        handle = x.register_hook(replace)
        output = x * 3
        if method == 'backward':
            output.backward(); value = x.grad
            assert value.item() == (4. if live else 7.) and not value.requires_grad
        else:
            value = api.autograd.grad(output, x)[0]
            assert value is stored[0] and value.requires_grad == live
            handle.remove()
            if live: assert api.autograd.grad(value, x)[0].item() == 4.


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_reverse_over_forward_applies_original_leaf_hook_once(dtype):
    for api in (torch, native):
        x = api.tensor(2., dtype=getattr(api, dtype), requires_grad=True); events = []
        def double(g): events.append(g.item()); return g * 2
        x.register_hook(double)
        primal, tangent = api.func.jvp(lambda p: p.square(), (x,), (api.tensor(3., dtype=getattr(api, dtype)),))
        assert events == []
        assert api.autograd.grad(primal + tangent, x)[0].item() == 20.
        assert events == [10.]
