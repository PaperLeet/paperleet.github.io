"""Lazy Tensor gradient callbacks, attached to reverse history when needed."""
from collections import OrderedDict
import weakref

from . import _errors, _state
from ._module_hooks import RemovableHandle


class NodeCallbacks(weakref.ref):
    __slots__ = ('hooks',)

    def __init__(self, tensor):
        self.hooks = None


def callbacks_for(tensor):
    node = tensor._grad_node
    if node.callbacks is None:
        node.callbacks = NodeCallbacks(tensor)
    return node.callbacks


def register(tensor, hook, *, post=False):
    if _state._jvp_active:
        raise _errors.BrowserUnsupported('Registering Tensor gradient hooks inside forward AD is unsupported')
    if not tensor.requires_grad:
        raise RuntimeError('Cannot register a hook on a tensor that does not require gradient')
    if post and not tensor.is_leaf:
        raise RuntimeError('Post accumulate grad hooks cannot be registered on non-leaf tensors')
    name = '_post_accumulate_grad_hooks' if post else '_backward_hooks'
    registry = getattr(tensor, name)
    if registry is None:
        registry = OrderedDict()
        setattr(tensor, name, registry)
        if not post and not tensor.is_leaf:
            callbacks_for(tensor).hooks = registry
    handle = RemovableHandle(registry)
    registry[handle.id] = hook
    return handle


def _validate(value, reference):
    from ._tensor import Tensor
    if not isinstance(value, Tensor):
        raise TypeError('A Tensor gradient hook must return a Tensor or None')
    if value.dtype != reference.dtype:
        raise RuntimeError('A Tensor gradient hook changed the gradient dtype')
    if value.shape != reference.shape:
        raise RuntimeError('A Tensor gradient hook changed the gradient shape')


def apply(gradient, hooks, owner=None):
    from ._tensor import Tensor
    previous = _state._grad_enabled
    _state._grad_enabled = False
    try:
        value = None
        if hooks:
            value = Tensor(gradient)
            # Each phase snapshots registrations. Changes affect its next run.
            for hook in tuple(hooks.values()):
                replacement = hook(value)
                if replacement is not None:
                    _validate(replacement, value)
                    value = replacement
            gradient = value._data
        if owner is not None and owner.requires_grad:
            owner._accumulate_gradient(gradient)
            post_hooks = owner._post_accumulate_grad_hooks
            if post_hooks:
                for hook in tuple(post_hooks.values()):
                    result = hook(owner)
                    if result is not None:
                        _validate(result, owner)
                        raise RuntimeError('Tensor post accumulate grad hooks must return None')
        return gradient, value
    finally:
        _state._grad_enabled = previous
