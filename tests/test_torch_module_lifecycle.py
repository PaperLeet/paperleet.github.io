"""Native-reference model lifecycle contracts beyond PaperLeet's lessons."""
from collections import OrderedDict
import copy

import numpy as np
import pytest
import torch as native

from paperleet._vendor.torch import UnsupportedOperation, torch as lite


def tensor(api, values, dtype='float64'):
    return api.tensor(np.asarray(values), dtype=getattr(api, dtype))


def build_model(api, dtype='float64'):
    class Model(api.nn.Module):
        _version = 3

        def __init__(self):
            super().__init__()
            self.root = api.nn.Parameter(tensor(api, [.3, -.5], dtype))
            self.alias = self.root
            self.left = api.nn.Linear(2, 2, dtype=getattr(api, dtype))
            self.right = self.left
            self.layers = api.nn.ModuleList([self.left, api.nn.Linear(2, 1, dtype=getattr(api, dtype))])
            self.choices = api.nn.ModuleDict(OrderedDict([('second', self.layers[1]), ('first', self.left)]))
            self.knobs = api.nn.ParameterList([self.root, tensor(api, [.25], dtype)])
            self.named = api.nn.ParameterDict({'z': self.root, 'a': tensor(api, [.75], dtype)})
            self.register_parameter('absent_parameter', None)
            self.register_buffer('running', tensor(api, [1., 2.], dtype))
            self.register_buffer('running_alias', self.running)
            self.register_buffer('scratch', tensor(api, [-3.], dtype), persistent=False)
            self.register_buffer('absent_buffer', None)
            self.add_module('absent_child', None)
            self.plain_tensor = tensor(api, [7.], dtype)
            with api.no_grad():
                for index, parameter in enumerate(self.parameters()):
                    parameter[...] = tensor(api, np.arange(parameter.numel()).reshape(tuple(parameter.shape)) / 10 + index, dtype)

        def forward(self, x):
            return self.layers[1](self.left(x) + self.root)

    return Model()


def alias_pattern(items):
    first = {}
    result = []
    for name, value in items:
        result.append((name, first.setdefault(id(value), name)))
    return result


@pytest.mark.parametrize('kind', ['named_modules', 'named_parameters', 'named_buffers'])
@pytest.mark.parametrize('duplicates', [False, True])
def test_named_traversal_order_and_shared_identity_match_native(kind, duplicates):
    actual, expected = build_model(lite), build_model(native)
    options = dict(prefix='model', remove_duplicate=duplicates)
    assert alias_pattern(getattr(actual, kind)(**options)) == alias_pattern(getattr(expected, kind)(**options))
    if kind != 'named_modules':
        options['recurse'] = False
        assert alias_pattern(getattr(actual, kind)(**options)) == alias_pattern(getattr(expected, kind)(**options))
    assert [name for name, _ in actual.named_children()] == [name for name, _ in expected.named_children()]
    assert len(list(actual.parameters())) == len(list(expected.parameters()))
    assert len(list(actual.buffers())) == len(list(expected.buffers()))


def test_named_modules_supplied_memo_and_paths_match_native():
    for remove_duplicate in (True, False):
        actual, expected = build_model(lite), build_model(native)
        assert [name for name, _ in actual.named_modules(memo={actual.left}, remove_duplicate=remove_duplicate)] == [
            name for name, _ in expected.named_modules(memo={expected.left}, remove_duplicate=remove_duplicate)]
    for api in (lite, native):
        model = build_model(api)
        assert model.get_submodule('') is model
        assert model.get_submodule('layers.0') is model.left
        assert model.get_parameter('right.weight') is model.left.weight
        assert model.get_buffer('absent_buffer') is None
        for method, path in [('get_submodule', 'left.weight'), ('get_submodule', 'missing'),
                             ('get_parameter', 'running'), ('get_parameter', 'absent_parameter'),
                             ('get_buffer', 'root')]:
            with pytest.raises(AttributeError): getattr(model, method)(path)


def test_train_eval_apply_and_zero_grad_match_native_without_breaking_autograd():
    for api in (lite, native):
        model = build_model(api)
        ids = [id(value) for value in model.parameters()]
        assert model.eval() is model and all(not child.training for child in model.modules())
        assert model.train() is model and all(child.training for child in model.modules())
        with pytest.raises(ValueError): model.train(1)
        assert [id(value) for value in model.parameters()] == ids
        model(tensor(api, [[1., 2.], [-.5, .2]])).square().mean().backward()
        before = [parameter.grad for parameter in model.parameters()]
        model.zero_grad(set_to_none=False)
        for parameter, old in zip(model.parameters(), before):
            assert parameter.grad is old
            if old is not None: np.testing.assert_array_equal(old.numpy(), np.zeros(tuple(old.shape)))
        model.zero_grad()
        assert all(parameter.grad is None for parameter in model.parameters())
        visits = []
        assert model.apply(lambda child: visits.append(child._get_name())) is model
        if api is lite: actual_visits = visits
        else: assert visits == actual_visits


@pytest.mark.parametrize('keep_vars', [False, True])
def test_state_dictionary_keys_metadata_storage_and_prefix_match_native(keep_vars):
    actual, expected = build_model(lite), build_model(native)
    states = [model.state_dict(keep_vars=keep_vars) for model in (actual, expected)]
    assert list(states[0]) == list(states[1])
    assert states[0]._metadata == states[1]._metadata
    assert states[0]['root'].requires_grad == states[1]['root'].requires_grad
    for name in states[0]:
        np.testing.assert_array_equal(states[0][name].detach().numpy(), states[1][name].detach().numpy())
    for model, state, api in zip((actual, expected), states, (lite, native)):
        assert (state['root'] is model.root) == keep_vars
        assert state['root'].data_ptr() == model.root.data_ptr() == state['alias'].data_ptr()
        with api.no_grad(): model.root.fill_(19.)
        np.testing.assert_array_equal(state['root'].detach().numpy(), [19., 19.])
        destination = OrderedDict(sentinel='kept')
        assert model.state_dict(destination=destination, prefix='net.') is destination
        assert list(destination)[0] == 'sentinel' and 'net.left.weight' in destination


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_state_roundtrip_keeps_parameter_identity_optimizer_references_and_gradients(dtype):
    models = [build_model(api, dtype) for api in (lite, native)]
    for model, api in zip(models, (lite, native)):
        saved = copy.deepcopy(model.state_dict())
        identities = [id(p) for p in model.parameters()]
        optimizer = api.optim.SGD(model.parameters(), lr=.03)
        with api.no_grad():
            for p in model.parameters(): p.fill_(0.)
            model.running.fill_(0.)
            model.scratch.fill_(99.)
        result = model.load_state_dict(saved)
        assert result.missing_keys == result.unexpected_keys == []
        assert repr(result) == '<All keys matched successfully>'
        assert [id(p) for p in model.parameters()] == identities
        np.testing.assert_array_equal(model.scratch.numpy(), [99.])
        model(tensor(api, [[.2, -.7], [1., .3]], dtype)).square().mean().backward()
        optimizer.step()
    for (_, a), (_, b) in zip(models[0].named_parameters(), models[1].named_parameters()):
        tolerance = 4e-6 if dtype == 'float32' else 1e-12
        np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('strict', [False, True])
def test_missing_unexpected_key_order_and_partial_native_loading(strict):
    results = []
    for api in (lite, native):
        model = build_model(api)
        state = OrderedDict([('left.unknown', tensor(api, [1.])), ('unknown', tensor(api, [2.])),
                             ('root', tensor(api, [8., 9.])), ('right.unknown', tensor(api, [3.]))])
        if strict:
            with pytest.raises(RuntimeError): model.load_state_dict(state)
        else:
            result = model.load_state_dict(state, strict=False)
            results.append((result.missing_keys, result.unexpected_keys))
        np.testing.assert_array_equal(model.root.detach().numpy(), [8., 9.])
    if not strict: assert results[0] == results[1]


def test_bad_shape_and_non_tensor_values_raise_even_when_not_strict():
    for api in (lite, native):
        for value in (tensor(api, [1., 2., 3.]), [1., 2.]):
            model = build_model(api)
            with pytest.raises(RuntimeError): model.load_state_dict({'root': value}, strict=False)
        with pytest.raises(TypeError): model.load_state_dict([])


def test_scalar_buffer_legacy_shape_dtype_cast_and_saved_graph_version_guards():
    for api in (lite, native):
        model = api.nn.Linear(2, 1, dtype=api.float64)
        model.register_buffer('count', api.tensor(0, dtype=api.int64))
        before = model.weight
        state = {'weight': tensor(api, [[1., 2.]], 'float32'), 'bias': tensor(api, [3.], 'float32'),
                 'count': api.tensor([8], dtype=api.int32)}
        model.load_state_dict(state)
        assert model.weight is before and model.weight.dtype == api.float64
        assert model.count.item() == 8 and model.count.dtype == api.int64
        # A trainable input makes the old parameter values necessary for VJP.
        x = tensor(api, [[2., 3.]]).requires_grad_()
        old = model(x).sum()
        model.load_state_dict(state)
        with pytest.raises((RuntimeError, UnsupportedOperation)):
            old.backward()


def test_deepcopied_state_preserves_buffer_views_and_module_parameter_grad_reset():
    outputs = []
    for api in (lite, native):
        model = api.nn.Linear(2, 2, dtype=api.float64)
        model.register_buffer('base', api.arange(12, dtype=api.float64).reshape(3, 4))
        model.register_buffer('window', model.base[:, 1::2])
        state = copy.deepcopy(model.state_dict())
        state['base'][0, 1] = 99.
        outputs.append(state['window'][0, 0].item())
        model(tensor(api, [[1., 2.]])).sum().backward()
        copied = copy.deepcopy(model)
        assert all(parameter.grad is None for parameter in copied.parameters())
        assert copied.weight.data_ptr() != model.weight.data_ptr()
    assert outputs[0] == outputs[1] == 99.


def test_custom_state_extra_data_and_load_metadata_callbacks():
    outputs = []
    for api in (lite, native):
        class Custom(api.nn.Module):
            _version = 5
            def __init__(self):
                super().__init__(); self.epoch = 3; self.weight = api.nn.Parameter(tensor(api, [2.]))
            def get_extra_state(self): return {'epoch': self.epoch}
            def set_extra_state(self, state): self.epoch = state['epoch']
            def _load_from_state_dict(self, state, prefix, metadata, *args):
                self.seen_version = metadata.get('version')
                super()._load_from_state_dict(state, prefix, metadata, *args)
        model = api.nn.Sequential(OrderedDict([('custom', Custom())]))
        state = copy.deepcopy(model.state_dict())
        model.custom.epoch = 90
        model.load_state_dict(state)
        outputs.append((model.custom.epoch, model.custom.seen_version, list(state)))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('kind', ['parameter', 'buffer', 'module'])
def test_registration_names_types_replacement_and_deletion(kind):
    outcomes = []
    for api in (lite, native):
        model = api.nn.Module()
        value = {'parameter': lambda: api.nn.Parameter(tensor(api, [1.])),
                 'buffer': lambda: tensor(api, [1.]), 'module': lambda: api.nn.Linear(1, 1)}[kind]()
        register = {'parameter': model.register_parameter, 'buffer': model.register_buffer, 'module': model.add_module}[kind]
        errors = []
        for name in ['', 'a.b', 3, 'training']:
            try: register(name, value)
            except Exception as error: errors.append(type(error).__name__)
        register('item', value)
        model.item = None
        assert model.item is None
        with pytest.raises(TypeError): model.item = 3
        del model.item
        register('item', value)
        outcomes.append((errors, list(model.state_dict())))
    assert outcomes[0] == outcomes[1]


def container_snapshot(container):
    return ([(name, module._get_name() if module is not None else None)
             for name, module in container._modules.items()], list(container.state_dict()))


def test_sequential_and_module_list_slices_and_mutations_match_native():
    outcomes = []
    for api in (lite, native):
        a, b, c = api.nn.Linear(2, 2), api.nn.ReLU(), api.nn.Linear(2, 1, bias=False)
        sequence = api.nn.Sequential(OrderedDict([('first', a), ('activation', b), ('last', c)]))
        snapshots = [container_snapshot(sequence), container_snapshot(sequence[1:])]
        assert sequence[-1] is c
        sequence[-1] = a
        del sequence[1:2]
        snapshots.append(container_snapshot(sequence))
        assert sequence.append(b) is sequence
        assert sequence.insert(1, c) is sequence
        snapshots.append(container_snapshot(sequence))
        snapshots.append(container_snapshot(sequence + api.nn.Sequential(b)))
        sequence += api.nn.Sequential(c)
        snapshots.append(container_snapshot(sequence))
        modules = api.nn.ModuleList([a, b, c, None])
        assert modules[-2] is c and modules[-1] is None
        snapshots.append(container_snapshot(modules[::-1]))
        modules[0] = c
        modules.insert(1, a)
        del modules[::2]
        assert modules.extend([c]) is modules
        modules += [a]
        snapshots.append(container_snapshot(modules))
        assert modules.insert(len(modules), b) is None
        snapshots.append(container_snapshot(modules))
        outcomes.append(snapshots)
    assert outcomes[0] == outcomes[1]


def test_mapping_and_parameter_containers_match_native_order_aliases_and_updates():
    outcomes = []
    for api in (lite, native):
        parameter = api.nn.Parameter(tensor(api, [1.]))
        values = api.nn.ParameterList([parameter, tensor(api, [2.]), None])
        assert values[0] is parameter and isinstance(values[1], api.nn.Parameter)
        assert values[1].requires_grad and values[-1] is None
        assert values.append(tensor(api, [3.])) is values
        values[1] = parameter
        assert values.extend([tensor(api, [4.])]) is values
        with pytest.raises(TypeError): values.extend(tensor(api, [5., 6.]))
        assert len(values) == 5
        sliced = values[1::2]
        named = api.nn.ParameterDict({'z': parameter, 'a': tensor(api, [8.])})
        assert list(named) == ['a', 'z']
        named.update(OrderedDict([('new_z', parameter), ('new_a', tensor(api, [9.]))]))
        named.update([('last', tensor(api, [10.]))])
        assert named.setdefault('new_z', None) is parameter
        assert named.get('missing', 17) == 17
        assert named.pop('new_z') is parameter
        named['last'] = None
        modules = api.nn.ModuleDict(OrderedDict([('z', api.nn.ReLU()), ('a', api.nn.SiLU())]))
        modules.update({'b': api.nn.Linear(1, 1), 'c': None})
        assert modules.pop('a')._get_name() == 'SiLU'
        del modules['c']
        outcomes.append((alias_pattern(values.named_parameters(remove_duplicate=False)),
                         alias_pattern(sliced.named_parameters(remove_duplicate=False)),
                         list(named), alias_pattern(named.named_parameters(remove_duplicate=False)),
                         container_snapshot(modules)))
        named.clear(); modules.clear()
        assert not len(named) and not list(named.parameters()) and not len(modules)
    assert outcomes[0] == outcomes[1]


def test_lifecycle_respects_child_overrides_and_recursion_options():
    outcomes = []
    for api in (lite, native):
        visits = []
        class Child(api.nn.Linear):
            def train(self, mode=True):
                visits.append(('train', mode))
                return super().train(mode)
            def apply(self, fn):
                visits.append(('apply',))
                return super().apply(fn)
            def state_dict(self, *args, **kwargs):
                visits.append(('state', kwargs['prefix']))
                return super().state_dict(*args, **kwargs)
        model = api.nn.Sequential(Child(1, 1), api.nn.Sequential(Child(1, 1)))
        model.eval(); model.apply(lambda child: None); model.state_dict()
        assert model.requires_grad_(False) is model
        assert not any(p.requires_grad for p in model.parameters())
        assert model.requires_grad_() is model
        assert all(p.requires_grad for p in model.parameters())
        assert list(model.parameters(recurse=False)) == []
        outcomes.append(visits)
    assert outcomes[0] == outcomes[1]


def test_deepcopy_tensor_views_gradients_and_distinct_shared_parameters_match_native():
    for api in (lite, native):
        base = api.arange(24, dtype=api.float64).reshape(4, 6)
        window = base[1:, 1::2].T
        empty = base[:, :0]
        x = tensor(api, [2., 3.]).requires_grad_()
        x.square().sum().backward()
        data = copy.deepcopy({'base': base, 'window': window, 'empty': empty, 'x': x})
        assert data['base'].data_ptr() != base.data_ptr()
        data['base'][1, 1] = 120.
        assert data['window'][0, 0].item() == 120. and window[0, 0].item() == 7.
        assert tuple(data['empty'].shape) == (4, 0)
        np.testing.assert_array_equal(data['x'].grad.numpy(), [4., 6.])
        assert data['x'].grad is not x.grad
        a, b = api.nn.Parameter(base), api.nn.Parameter(base)
        copied = copy.deepcopy([a, b, a])
        assert copied[0] is copied[2] and copied[0] is not copied[1]
        copied[0].data.fill_(99.)
        assert copied[1][0, 0].item() == 0.
