"""Optimizer lifecycle and numerical contracts against independent native Torch."""
import copy

import numpy as np
import pytest
import torch as native

from paperleet._vendor.torch import UnsupportedOperation, torch as lite


OPTIONS = [('SGD', {'momentum': .8, 'dampening': .1, 'weight_decay': .03}),
           ('SGD', {'momentum': .8, 'nesterov': True, 'maximize': True}),
           ('Adam', {'betas': (.7, .92), 'weight_decay': .03}),
           ('Adam', {'amsgrad': True, 'maximize': True}),
           ('AdamW', {'betas': (.7, .92), 'weight_decay': .03}),
           ('AdamW', {'amsgrad': True, 'maximize': True})]


def tensor(api, values, dtype='float64', requires_grad=False):
    return api.tensor(np.asarray(values), dtype=getattr(api, dtype), requires_grad=requires_grad)


def compare_state(actual, expected, tolerance):
    assert set(actual['state']) == set(expected['state'])
    assert len(actual['param_groups']) == len(expected['param_groups'])
    for a, b in zip(actual['param_groups'], expected['param_groups']):
        for key, value in a.items():
            if key in b:
                assert value == b[key], (key, value, b[key])
    for key in actual['state']:
        assert set(actual['state'][key]) == set(expected['state'][key])
        for name, a in actual['state'][key].items():
            b = expected['state'][key][name]
            assert a.dtype.name == str(b.dtype).split('.')[-1]
            np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('name,options', OPTIONS)
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_group_updates_lazy_state_dynamic_lr_addition_and_native_checkpoint(name, options, dtype):
    runs = []
    for api in (lite, native):
        params = [tensor(api, [.4, -.9], dtype, True), tensor(api, [[.3, -.2]], dtype, True)]
        groups = [{'params': [params[0]], 'lr': .03, 'tag': 'first'},
                  {'params': [params[1]], 'lr': .01, 'weight_decay': .02, 'tag': 'second'}]
        optimizer = getattr(api.optim, name)(groups, **options)
        assert isinstance(optimizer, api.optim.Optimizer)
        assert optimizer.state_dict()['state'] == {}
        for step in range(8):
            if step == 3:
                params.append(tensor(api, [.7], dtype, True))
                optimizer.add_param_group({'params': [params[-1]], 'lr': .02, 'tag': 'late'})
            optimizer.param_groups[0]['lr'] = .03 / (1 + step)
            for index, parameter in enumerate(params):
                parameter.grad = (None if (index == 1 and step < 4) or (index == 2 and step % 2 == 0)
                                  else tensor(api, np.full(tuple(parameter.shape), .2 + step * .05 + index), dtype))
            optimizer.step()
            if step < 4: assert params[1] not in optimizer.state
        runs.append((params, optimizer))
    tolerance = 4e-6 if dtype == 'float32' else 1e-12
    for a, b in zip(runs[0][0], runs[1][0]):
        np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy(), rtol=tolerance, atol=tolerance)
    compare_state(runs[0][1].state_dict(), runs[1][1].state_dict(), tolerance)


@pytest.mark.parametrize('name,options', OPTIONS)
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_train_checkpoint_resume_with_new_model_and_optimizer_matches_uninterrupted(name, options, dtype):
    outcomes = []
    for api in (lite, native):
        model = api.nn.Sequential(api.nn.Linear(2, 3, dtype=getattr(api, dtype)), api.nn.SiLU(),
                                  api.nn.Linear(3, 1, dtype=getattr(api, dtype)))
        with api.no_grad():
            for index, p in enumerate(model.parameters()):p.fill_(.1 + .05 * index)
        params = list(model.parameters())
        optimizer = getattr(api.optim, name)([{'params': params[:2], 'lr': .02}, {'params': params[2:], 'lr': .01}], **options)
        x = tensor(api, [[.2, -.7], [1., .3]], dtype)
        target = tensor(api, [[.1], [.6]], dtype)
        def update(net, opt):
            opt.zero_grad()
            (net(x) - target).square().mean().backward()
            opt.step()
        for _ in range(4):update(model, optimizer)
        checkpoint = copy.deepcopy({'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
        resumed = copy.deepcopy(model)
        cloned_params = list(resumed.parameters())
        restored = getattr(api.optim, name)([{'params': cloned_params[:2]}, {'params': cloned_params[2:]}], lr=.7)
        parameter_ids = [id(p) for group in restored.param_groups for p in group['params']]
        resumed.load_state_dict(checkpoint['model'])
        assert restored.load_state_dict(checkpoint['optimizer']) is None
        assert [id(p) for group in restored.param_groups for p in group['params']] == parameter_ids
        for step in range(5):
            for opt in (optimizer, restored):opt.param_groups[0]['lr'] = .013 / (step + 1)
            update(model, optimizer); update(resumed, restored)
        for a, b in zip(model.parameters(), resumed.parameters()):
            np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy(), rtol=1e-6, atol=1e-7)
        outcomes.append((resumed, restored))
    tolerance = 6e-6 if dtype == 'float32' else 1e-12
    for a, b in zip(outcomes[0][0].parameters(), outcomes[1][0].parameters()):
        np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy(), rtol=tolerance, atol=tolerance)
    compare_state(outcomes[0][1].state_dict(), outcomes[1][1].state_dict(), tolerance)


@pytest.mark.parametrize('name', ['SGD', 'Adam', 'AdamW'])
def test_closure_enabled_inside_no_grad_returns_loss_and_restores_mode_on_errors(name):
    outcomes = []
    for api in (lite, native):
        p = tensor(api, [1., 2.], requires_grad=True)
        optimizer = getattr(api.optim, name)([p], lr=.03)
        calls = []
        def closure():
            calls.append('called');optimizer.zero_grad()
            loss = p.square().sum();loss.backward();return loss
        with api.no_grad():
            loss = optimizer.step(closure)
            assert not p.square().requires_grad
            with pytest.raises(ValueError, match='failure'):
                optimizer.step(lambda: (_ for _ in ()).throw(ValueError('failure')))
            assert not p.square().requires_grad
        assert p.square().requires_grad and calls == ['called']
        outcomes.append((loss.item(), p.detach().numpy()))
    assert outcomes[0][0] == outcomes[1][0]
    np.testing.assert_allclose(outcomes[0][1], outcomes[1][1], atol=1e-12)


@pytest.mark.parametrize('name', ['SGD', 'Adam', 'AdamW'])
def test_zero_grad_preserves_existing_storage_missing_grad_and_frozen_leaf_gradient(name):
    for api in (lite, native):
        p, unused = [tensor(api, [1., 2.], requires_grad=True) for _ in range(2)]
        optimizer = getattr(api.optim, name)([p, unused])
        p.grad = tensor(api, [3., 4.], requires_grad=True)
        grad, pointer = p.grad, p.grad.data_ptr()
        optimizer.zero_grad(set_to_none=False)
        assert p.grad is grad and p.grad.data_ptr() == pointer and not p.grad.requires_grad
        np.testing.assert_array_equal(p.grad.numpy(), [0., 0.])
        assert unused.grad is None
        optimizer.zero_grad();assert p.grad is None


@pytest.mark.parametrize('name', ['SGD', 'Adam', 'AdamW'])
def test_state_checkpoint_views_same_dtype_aliases_and_cross_dtype_cast_match_native(name):
    for api in (lite, native):
        p = tensor(api, [1., 2.], requires_grad=True)
        options = {'momentum': .8} if name == 'SGD' else {}
        optimizer = getattr(api.optim, name)([p], **options)
        p.grad = tensor(api, [.3, .7]);optimizer.step()
        state = optimizer.state_dict()
        clone = getattr(api.optim, name)([p], **options)
        clone.load_state_dict(state)
        for key, value in state['state'][0].items():assert value is clone.state[p][key]
        q = tensor(api, [1., 2.], 'float32', True)
        casted = getattr(api.optim, name)([q], **options)
        casted.load_state_dict(state)
        for key, value in state['state'][0].items():
            target = casted.state[q][key]
            if key == 'step':assert target is value
            else:assert target.dtype == api.float32 and target.data_ptr() != value.data_ptr()
        q.grad = tensor(api, [.2, .5], 'float32');casted.step()


@pytest.mark.parametrize('name', ['SGD', 'Adam', 'AdamW'])
def test_group_validation_duplicate_semantics_and_failed_load_preserve_groups(name):
    outcomes = []
    for api in (lite, native):
        p, q = [tensor(api, [1., 2.], requires_grad=True) for _ in range(2)]
        cls = getattr(api.optim, name)
        with pytest.raises(ValueError):cls([])
        with pytest.raises(TypeError):cls(p)
        assert cls({p}).param_groups[0]['params'] == [p]
        with pytest.raises(ValueError):cls([p.square()])
        with pytest.raises(TypeError):cls([3])
        with pytest.raises(ValueError):cls([{'params': [p]}, {'params': [p]}])
        with pytest.raises(TypeError):cls([{'params': {p}}])
        with pytest.raises(ValueError):cls([{'params': [p], 'lr': .1}], lr=-.1)
        if name == 'SGD':assert np.isnan(cls([p], lr=float('nan')).param_groups[0]['lr'])
        else:
            with pytest.raises((ValueError, TypeError)):cls([p], lr=float('nan'))
        optimizer = cls([p])
        before = optimizer.param_groups
        for state in ({'state': {}, 'param_groups': []}, {'state': {}, 'param_groups': [{'params': [0, 1]}]}):
            with pytest.raises(ValueError):optimizer.load_state_dict(state)
            assert optimizer.param_groups is before
        options = {'momentum': .8} if name == 'SGD' else {}
        with pytest.warns(UserWarning):duplicate = cls([q, q], **options)
        for _ in range(3):q.grad = tensor(api, [.3, .7]);duplicate.step()
        outcomes.append((q.detach().numpy(), duplicate.state_dict()))
    np.testing.assert_allclose(outcomes[0][0], outcomes[1][0], atol=1e-12)
    compare_state(outcomes[0][1], outcomes[1][1], 1e-12)


def test_named_parameters_global_state_metadata_and_missing_saved_names():
    for api in (lite, native):
        model = api.nn.Linear(2, 1, dtype=api.float64)
        optimizer = api.optim.Adam(model.named_parameters())
        assert optimizer.param_groups[0]['param_names'] == ['weight', 'bias']
        state = optimizer.state_dict();state['state']['global'] = {'epoch': 17, 'step': 18}
        state['param_groups'][0].pop('param_names')
        restored = api.optim.Adam(model.named_parameters())
        restored.load_state_dict(state)
        assert restored.param_groups[0]['param_names'] == ['weight', 'bias']
        assert restored.state['global'] == {'epoch': 17, 'step': 18}
        with pytest.raises(ValueError):restored.add_param_group({'params': [tensor(api, [1.], requires_grad=True)]})


@pytest.mark.parametrize('name', ['SGD', 'Adam', 'AdamW'])
def test_exposed_state_mutation_guards_old_graphs(name):
    for api in (lite, native):
        p = tensor(api, [.2, -.9], requires_grad=True)
        options = {'momentum': .7} if name == 'SGD' else {'amsgrad': True}
        optimizer = getattr(api.optim, name)([p], **options)
        p.grad = tensor(api, [.3, .5]);optimizer.step()
        state = list(optimizer.state[p].values())
        old = sum((tensor(api, np.ones(tuple(value.shape)), requires_grad=True) * value).sum() for value in state)
        optimizer.step()
        with pytest.raises((RuntimeError, UnsupportedOperation)):old.backward()


def test_bounded_execution_options_unknown_keywords_state_shapes_and_jvp_fail_explicitly():
    p = tensor(lite, [1., 2.], requires_grad=True)
    for name in ('SGD', 'Adam', 'AdamW'):
        cls = getattr(lite.optim, name)
        for options in ({'foreach': True}, {'fused': True}, {'differentiable': True}, {'typo': True}):
            with pytest.raises(UnsupportedOperation):cls([p], **options)
        if name != 'SGD':
            with pytest.raises(UnsupportedOperation):cls([p], capturable=True)
        opt = cls([p])
        p.grad = tensor(lite, [1.])
        with pytest.raises(RuntimeError, match='shape/dtype'):opt.step()
        p.grad = tensor(lite, [.3, .5])
        with pytest.raises(UnsupportedOperation, match='JVP'):
            lite.func.jvp(lambda x: (opt.step(), x.square())[1], (p,), (lite.ones_like(p),))
    opt = lite.optim.Adam([p]);opt.step()
    opt.state[p]['exp_avg'] = lite.zeros(1, dtype=lite.float64)
    with pytest.raises(RuntimeError, match='state'):opt.step()


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_same_dtype_to_is_identity_and_changed_dtype_keeps_gradients(dtype):
    for api in (lite, native):
        p = tensor(api, [2., 3.], dtype, True)
        assert p.to(dtype=getattr(api, dtype)) is p and p.to('cpu') is p
        target = api.float64 if dtype == 'float32' else api.float32
        converted = p.to(dtype=target)
        assert converted is not p
        converted.square().sum().backward()
        np.testing.assert_array_equal(p.grad.numpy(), [4., 6.])
        def same_dtype(value):
            assert value.to(dtype=value.dtype) is value
            return value.square()
        primal, tangent = api.func.jvp(same_dtype, (p,), (api.ones_like(p),))
        np.testing.assert_array_equal(primal.detach().numpy(), [4., 9.])
        np.testing.assert_array_equal(tangent.detach().numpy(), [4., 6.])


@pytest.mark.parametrize('name', ['SGD', 'Adam', 'AdamW'])
@pytest.mark.parametrize('shape_kind', ['scalar', 'empty', 'transposed', 'gradient_alias'])
def test_scalar_empty_strided_parameters_and_gradient_storage_aliases(name, shape_kind):
    results = []
    for api in (lite, native):
        if shape_kind == 'scalar':p = tensor(api, .7, 'float32', True)
        elif shape_kind == 'empty':p = api.empty(0, 3, dtype=api.float32, requires_grad=True)
        elif shape_kind == 'transposed':p = api.arange(12, dtype=api.float32).reshape(3, 4).T.requires_grad_()
        else:p = tensor(api, [1., 2.], 'float32', True)
        options = {'momentum': .8} if name == 'SGD' else {'amsgrad': True}
        optimizer = getattr(api.optim, name)([p], lr=.02, weight_decay=.1, **options)
        for step in range(4):
            p.grad = p.detach() if shape_kind == 'gradient_alias' else api.ones_like(p) * (.3 + step * .1)
            optimizer.step()
        results.append((p.detach().numpy(), optimizer.state_dict()))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=4e-6, atol=4e-6)
    compare_state(results[0][1], results[1][1], 4e-6)


def test_checkpoint_remaps_by_position_not_parameter_name_and_preserves_loaded_names():
    results = []
    for api in (lite, native):
        p, q = [tensor(api, [1., 2.], requires_grad=True) for _ in range(2)]
        optimizer = api.optim.Adam([('first', p), ('second', q)])
        p.grad, q.grad = tensor(api, [.3, .5]), tensor(api, [.8, .9]);optimizer.step()
        state = copy.deepcopy(optimizer.state_dict())
        a, b = [tensor(api, [1., 2.], requires_grad=True) for _ in range(2)]
        restored = api.optim.Adam([('second', a), ('first', b)])
        restored.load_state_dict(state)
        assert restored.param_groups[0]['param_names'] == ['first', 'second']
        assert restored.state[a]['exp_avg'] is state['state'][0]['exp_avg']
        assert restored.state[b]['exp_avg'] is state['state'][1]['exp_avg']
        a.grad, b.grad = tensor(api, [.2, .4]), tensor(api, [.6, .3]);restored.step()
        results.append(restored.state_dict())
    compare_state(results[0], results[1], 1e-12)


@pytest.mark.parametrize('name', ['SGD', 'Adam', 'AdamW'])
def test_tensor_learning_rate_live_updates_and_legacy_numeric_step_checkpoint(name):
    results = []
    for api in (lite, native):
        p = tensor(api, [1., 2.], requires_grad=True)
        rate = api.tensor(.03, dtype=api.float64)
        optimizer = getattr(api.optim, name)([p], lr=rate)
        for step in range(4):
            rate.fill_(.03 / (step + 1))
            p.grad = tensor(api, [.3, .5]);optimizer.step()
        state = copy.deepcopy(optimizer.state_dict())
        if name != 'SGD':state['state'][0]['step'] = state['state'][0]['step'].item()
        restored = getattr(api.optim, name)([p]);restored.load_state_dict(state)
        assert restored.param_groups[0]['lr'].item() == rate.item()
        p.grad = tensor(api, [.2, .4]);restored.step()
        # Group tensor metadata is compared as a scalar by the fixture below.
        packed = restored.state_dict();packed['param_groups'][0]['lr'] = packed['param_groups'][0]['lr'].item()
        results.append((p.detach().numpy(), packed))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-12)
    compare_state(results[0][1], results[1][1], 1e-12)
