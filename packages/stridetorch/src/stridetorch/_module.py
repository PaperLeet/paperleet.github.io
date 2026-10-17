"""Model registration, traversal and in-memory state independent of the judge."""
from collections import OrderedDict, namedtuple
from collections.abc import Mapping
from functools import partial
from reprlib import recursive_repr

import numpy as np

from . import _defaults, _tensor
from ._module_conversion import ModuleConversion
from ._module_hooks import ModuleHooks


_UNSUPPORTED_MODULE_METHODS = frozenset({
    'register_backward_hook', 'register_full_backward_hook', 'register_full_backward_pre_hook',
    'cuda', 'mps', 'xpu', 'ipu', 'mtia', 'half', 'bfloat16', 'to_empty', 'share_memory', 'compile',
    '_register_state_dict_hook', '_register_load_state_dict_pre_hook',
})


def _unsupported_method(name, *args, **kwargs):
    return _errors.unsupported(name)


class Parameter(_tensor.Tensor):
    def __init__(self, data=None, requires_grad=True):
        super().__init__(np.empty(0, dtype=_defaults._float_dtype) if data is None else _ops.wrap(data)._data,
                         requires_grad=requires_grad)

    def __deepcopy__(self, memo):
        # Parameters clone independently of other storage views and start with
        # no accumulated gradient. Repeated references still share one clone.
        value = type(self)(np.array(self._data, copy=True, order='K'), self.requires_grad)
        memo[id(self)] = value
        return value


class _IncompatibleKeys(namedtuple('IncompatibleKeys', 'missing_keys unexpected_keys')):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super().__repr__()


class Module(ModuleHooks, ModuleConversion):
    """Registered models and buffers with native-style shared-object semantics.

    Local forward/checkpoint hooks are supported; backward hooks and assign-based
    checkpoint loading remain unsupported interfaces.
    State dictionaries are in-memory Tensor mappings with optional model extra state.
    """
    _version = 1

    def __init__(self):
        object.__setattr__(self, 'training', True)
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_buffers', {})
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_non_persistent_buffers_set', set())
        object.__setattr__(self, '_has_forward_hooks', False)

    def __getattr__(self, name):
        for registry in ('_parameters', '_buffers', '_modules'):
            values = self.__dict__.get(registry, {})
            if name in values:
                return values[name]
        if name in _UNSUPPORTED_MODULE_METHODS:
            return partial(_unsupported_method, f'nn.Module.{name}')
        raise AttributeError(f'{type(self).__name__!r} object has no attribute {name!r}')

    def __dir__(self):
        names = set(super().__dir__())
        for registry in ('_parameters', '_buffers', '_modules'):
            names.update(self.__dict__.get(registry, {}))
        return sorted(name for name in names if not name[0].isdigit())

    def __setattr__(self, name, value):
        parameters = self.__dict__.get('_parameters')
        modules = self.__dict__.get('_modules')
        buffers = self.__dict__.get('_buffers')
        if isinstance(value, Parameter):
            if parameters is None:
                raise AttributeError('Cannot assign parameters before Module.__init__()')
            self.__dict__.pop(name, None)
            buffers.pop(name, None)
            modules.pop(name, None)
            self._non_persistent_buffers_set.discard(name)
            self.register_parameter(name, value)
        elif parameters is not None and name in parameters:
            if value is not None:
                raise TypeError(f'Parameter {name!r} requires Parameter or None')
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            if modules is None:
                raise AttributeError('Cannot assign modules before Module.__init__()')
            self.__dict__.pop(name, None)
            buffers.pop(name, None)
            self._non_persistent_buffers_set.discard(name)
            modules[name] = value
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError(f'Child module {name!r} requires Module or None')
            modules[name] = None
        elif buffers is not None and name in buffers:
            self.register_buffer(name, value, name not in self._non_persistent_buffers_set)
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        for registry in ('_parameters', '_buffers', '_modules'):
            values = self.__dict__.get(registry, {})
            if name in values:
                del values[name]
                self._non_persistent_buffers_set.discard(name)
                return
        object.__delattr__(self, name)

    def _check_registration_name(self, name, registry):
        if registry not in self.__dict__:
            raise AttributeError('Registration requires Module.__init__()')
        if not isinstance(name, str):
            raise TypeError('A registration name must be a string')
        if not name or '.' in name:
            raise KeyError('A registration name must be nonempty and contain no dots')
        if hasattr(self, name) and name not in self.__dict__[registry]:
            raise KeyError(f'Attribute {name!r} already exists')

    def register_parameter(self, name, param):
        self._check_registration_name(name, '_parameters')
        if param is not None and not isinstance(param, Parameter):
            raise TypeError('A parameter must be Parameter or None')
        if param is not None and param._parents:
            raise ValueError('A registered parameter must be a leaf')
        self._parameters[name] = param

    def register_buffer(self, name, tensor, persistent=True):
        self._check_registration_name(name, '_buffers')
        if tensor is not None and not isinstance(tensor, _tensor.Tensor):
            raise TypeError('A buffer must be Tensor or None')
        self._buffers[name] = tensor
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)

    def add_module(self, name, module):
        if module is not None and not isinstance(module, Module):
            raise TypeError('A child must be Module or None')
        self._check_registration_name(name, '_modules')
        self._modules[name] = module

    register_module = add_module

    def __call__(self, *args, **kwargs):
        if self._has_forward_hooks:
            return self._call_with_hooks(args, kwargs)
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'{type(self).__name__} does not define forward()')

    def named_children(self):
        seen = set()
        for name, module in self._modules.items():
            if module is not None and id(module) not in seen:
                seen.add(id(module))
                yield name, module

    def children(self):
        return (module for _, module in self.named_children())

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        memo = set() if memo is None else memo
        pending = [(prefix, self, frozenset())]
        while pending:
            path, module, ancestors = pending.pop()
            if module in memo:
                continue
            if id(module) in ancestors:
                raise _errors.BrowserUnsupported('Cyclic module graphs are unsupported')
            if remove_duplicate:
                memo.add(module)
            yield path, module
            ancestors = ancestors | {id(module)}
            for name, child in reversed(tuple(module._modules.items())):
                if child is not None:
                    pending.append((path + ('.' if path else '') + name, child, ancestors))

    def modules(self):
        return (module for _, module in self.named_modules())

    def _named_members(self, registry, prefix, recurse, remove_duplicate):
        seen = set()
        modules = self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, self)]
        for path, module in modules:
            for name, value in module.__dict__[registry].items():
                if value is None or remove_duplicate and id(value) in seen:
                    continue
                seen.add(id(value))
                yield path + ('.' if path else '') + name, value

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self._named_members('_parameters', prefix, recurse, remove_duplicate)

    def parameters(self, recurse=True):
        return (value for _, value in self.named_parameters(recurse=recurse))

    def named_buffers(self, prefix='', recurse=True, remove_duplicate=True):
        return self._named_members('_buffers', prefix, recurse, remove_duplicate)

    def buffers(self, recurse=True):
        return (value for _, value in self.named_buffers(recurse=recurse))

    def get_submodule(self, target):
        module = self
        for name in target.split('.') if target else ():
            child = getattr(module, name, None)
            if not isinstance(child, Module):
                raise AttributeError(f'{name!r} is not a child module')
            module = child
        return module

    def get_parameter(self, target):
        path, _, name = target.rpartition('.')
        value = self.get_submodule(path)._parameters.get(name)
        if not isinstance(value, Parameter):
            raise AttributeError(f'{target!r} is not a parameter')
        return value

    def get_buffer(self, target):
        path, _, name = target.rpartition('.')
        module = self.get_submodule(path)
        if name not in module._buffers:
            raise AttributeError(f'{target!r} is not a buffer')
        return module._buffers[name]

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError('Training mode must be bool')
        if self.__dict__.get('_training_in_progress'):
            raise _errors.BrowserUnsupported('Cyclic module training is unsupported')
        self.training = mode
        self.__dict__['_training_in_progress'] = True
        try:
            for child in self.children():
                child.train(mode)
        finally:
            self.__dict__.pop('_training_in_progress', None)
        return self

    def eval(self):
        return self.train(False)

    def requires_grad_(self, requires_grad=True):
        for parameter in self.parameters():
            parameter.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters():
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    if parameter.grad._parents or parameter.grad.requires_grad:
                        raise _errors.BrowserUnsupported('Clearing differentiable gradient buffers is unsupported')
                    parameter.grad.zero_()

    def apply(self, fn):
        if self.__dict__.get('_apply_in_progress'):
            raise _errors.BrowserUnsupported('Cyclic module application is unsupported')
        self.__dict__['_apply_in_progress'] = True
        try:
            for child in self.children():
                child.apply(fn)
            fn(self)
        finally:
            self.__dict__.pop('_apply_in_progress', None)
        return self

    def get_extra_state(self):
        raise RuntimeError('get_extra_state() requires a model implementation')

    def set_extra_state(self, state):
        raise RuntimeError('set_extra_state() requires a model implementation')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for registry in ('_parameters', '_buffers'):
            for name, value in self.__dict__[registry].items():
                if value is None or registry == '_buffers' and name in self._non_persistent_buffers_set:
                    continue
                destination[prefix + name] = value if keep_vars else value.detach()
        if type(self).get_extra_state is not Module.get_extra_state:
            destination[prefix + '_extra_state'] = self.get_extra_state()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if self.__dict__.get('_saving_state'):
            raise _errors.BrowserUnsupported('Cyclic module state is unsupported')
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        local_metadata = {'version': self._version}
        metadata = getattr(destination, '_metadata', None)
        if metadata is not None:
            metadata[prefix[:-1]] = local_metadata
        self.__dict__['_saving_state'] = True
        try:
            pre_hooks = self.__dict__.get('_state_dict_pre_hooks')
            if pre_hooks:
                for hook in pre_hooks.values():
                    hook(self, prefix, keep_vars)
            self._save_to_state_dict(destination, prefix, keep_vars)
            for name, module in self._modules.items():
                if module is not None:
                    module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
            post_hooks = self.__dict__.get('_state_dict_hooks')
            if post_hooks:
                for hook in post_hooks.values():
                    if hook(self, destination, prefix, local_metadata) is not None:
                        raise RuntimeError('state_dict post-hook must return None')
        finally:
            self.__dict__.pop('_saving_state', None)
        return destination

    def _load_from_state_dict(self, state, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        pre_hooks = self.__dict__.get('_load_state_dict_pre_hooks')
        if pre_hooks:
            for hook in pre_hooks.values():
                hook(self, state, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        local = {name: value for name, value in self._parameters.items() if value is not None}
        local.update((name, value) for name, value in self._buffers.items()
                     if value is not None and name not in self._non_persistent_buffers_set)
        for name, target in local.items():
            key = prefix + name
            if key not in state:
                missing_keys.append(key)
                continue
            source = state[key]
            if not isinstance(source, _tensor.Tensor):
                error_msgs.append(f'Expected a Tensor for {key!r}')
                continue
            data = source._data[0] if not target.ndim and source.ndim == 1 and source.numel() else source._data
            if np.shape(data) != target.shape:
                error_msgs.append(f'size mismatch for {key}: checkpoint {np.shape(data)}, model {target.shape}')
                continue
            if target._parents:
                raise _errors.BrowserUnsupported('Loading non-leaf buffer storage is unsupported')
            try:
                with _autograd.no_grad():
                    target[...] = data
            except (ValueError, TypeError, RuntimeError) as error:
                error_msgs.append(f'While copying {key}: {error}')
        extra = prefix + '_extra_state'
        if type(self).set_extra_state is not Module.set_extra_state:
            if extra in state:
                self.set_extra_state(state[extra])
            else:
                missing_keys.append(extra)
        elif extra in state:
            unexpected_keys.append(extra)
        for key in state:
            if key.startswith(prefix) and key != extra:
                parts = key[len(prefix):].split('.', 1)
                if parts[0] not in (self._modules if len(parts) > 1 else local):
                    unexpected_keys.append(key)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if assign:
            raise _errors.BrowserUnsupported('Assign-based state loading is unsupported')
        if not isinstance(state_dict, Mapping):
            raise TypeError('state_dict must be a mapping')
        state = OrderedDict(state_dict)
        if any(not isinstance(key, str) for key in state):
            raise TypeError('State dictionary keys must be strings')
        metadata = getattr(state_dict, '_metadata', None)
        if metadata is not None:
            state._metadata = metadata
        missing, unexpected, errors = [], [], []
        active = set()

        def load(module, values, prefix):
            if id(module) in active:
                raise _errors.BrowserUnsupported('Cyclic module state is unsupported')
            active.add(id(module))
            try:
                module._load_from_state_dict(values, prefix, {} if metadata is None else metadata.get(prefix[:-1], {}),
                                             True, missing, unexpected, errors)
                for name, child in module._modules.items():
                    if child is not None:
                        child_prefix = prefix + name + '.'
                        load(child, {key: value for key, value in values.items()
                                     if key.startswith(child_prefix)}, child_prefix)
                post_hooks = module.__dict__.get('_load_state_dict_post_hooks')
                if post_hooks:
                    incompatible = _IncompatibleKeys(missing, unexpected)
                    for hook in post_hooks.values():
                        if hook(module, incompatible) is not None:
                            raise AssertionError('load_state_dict post-hook must return None')
            finally:
                active.remove(id(module))

        load(self, state, '')
        if strict:
            if missing:
                errors.insert(0, 'Missing key(s): ' + ', '.join(missing))
            if unexpected:
                errors.insert(0, 'Unexpected key(s): ' + ', '.join(unexpected))
        if errors:
            raise RuntimeError('Error loading state_dict for ' + type(self).__name__ + ':\n' + '\n'.join(errors))
        return _IncompatibleKeys(missing, unexpected)

    def _get_name(self):
        return type(self).__name__

    def extra_repr(self):
        return ''

    @recursive_repr()
    def __repr__(self):
        extra = self.extra_repr()
        lines = extra.split('\n') if extra else []
        for name, child in self._modules.items():
            lines.append('(' + name + '): ' + repr(child).replace('\n', '\n  '))
        if not lines:
            return self._get_name() + '()'
        if len(lines) == 1 and not self._modules:
            return self._get_name() + '(' + lines[0] + ')'
        return self._get_name() + '(\n  ' + '\n  '.join(lines) + '\n)'


from . import _autograd, _errors, _ops
