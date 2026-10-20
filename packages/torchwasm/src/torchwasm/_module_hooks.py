"""Local module callbacks and weak, removable registration handles."""
from collections import OrderedDict
import warnings
import weakref


_EMPTY = {}


class RemovableHandle:
    next_id = 0

    def __init__(self, hooks_dict, *, extra_dict=None):
        extras = [extra_dict] if isinstance(extra_dict, dict) else extra_dict if isinstance(extra_dict, list) else []
        self._registries = tuple(weakref.ref(value) for value in [hooks_dict, *extras])
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    @property
    def hooks_dict_ref(self):
        return self._registries[0]

    @property
    def extra_dict_ref(self):
        return self._registries[1:]

    def remove(self):
        for reference in self._registries:
            registry = reference()
            if registry is not None:
                registry.pop(self.id, None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __getstate__(self):
        return self.id, tuple(reference() for reference in self._registries)

    def __setstate__(self, state):
        self.id, registries = state
        self._registries = tuple(weakref.ref(value if value is not None else OrderedDict())
                                 for value in registries)
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)


class ModuleHooks:
    def _hook_registry(self, name):
        # Most models have no callbacks. Allocate only on registration, keeping
        # construction and checkpoint state free of unused ordered dictionaries.
        if '_parameters' not in self.__dict__:
            raise AttributeError('Hook registration requires Module.__init__()')
        registry = self.__dict__.get(name)
        if registry is None:
            registry = self.__dict__[name] = OrderedDict()
        return registry

    def _register_hook(self, name, hook, prepend=False, flags=()):
        registry = self._hook_registry(name)
        extras = [self._hook_registry(flag) for flag in flags]
        handle = RemovableHandle(registry, extra_dict=extras)
        registry[handle.id] = hook
        for extra in extras:
            extra[handle.id] = True
        if prepend:
            registry.move_to_end(handle.id, last=False)
        if name in ('_forward_pre_hooks', '_forward_hooks'):
            object.__setattr__(self, '_has_forward_hooks', True)
        return handle

    def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
        flags = ('_forward_pre_hooks_with_kwargs',) if with_kwargs else ()
        return self._register_hook('_forward_pre_hooks', hook, prepend, flags)

    def register_forward_hook(self, hook, *, prepend=False, with_kwargs=False, always_call=False):
        flags = tuple(name for name, active in (
            ('_forward_hooks_with_kwargs', with_kwargs),
            ('_forward_hooks_always_called', always_call),
        ) if active)
        return self._register_hook('_forward_hooks', hook, prepend, flags)

    def register_state_dict_pre_hook(self, hook):
        return self._register_hook('_state_dict_pre_hooks', hook)

    def register_state_dict_post_hook(self, hook):
        return self._register_hook('_state_dict_hooks', hook)

    def register_load_state_dict_pre_hook(self, hook):
        return self._register_hook('_load_state_dict_pre_hooks', hook)

    def register_load_state_dict_post_hook(self, hook):
        return self._register_hook('_load_state_dict_post_hooks', hook)

    def _call_with_hooks(self, args, kwargs):
        attributes = self.__dict__
        if not (attributes.get('_forward_pre_hooks') or attributes.get('_forward_hooks')):
            # Removing a handle needs no owner callback or strong model ref.
            # Pay one empty check on the next call, then regain the fast path.
            object.__setattr__(self, '_has_forward_hooks', False)
            return self.forward(*args, **kwargs)
        forward = self.forward
        result = None
        called = set()
        try:
            for key, hook in tuple(attributes.get('_forward_pre_hooks', _EMPTY).items()):
                if key in attributes.get('_forward_pre_hooks_with_kwargs', ()):
                    replacement = hook(self, args, kwargs)
                    if replacement is not None:
                        if not isinstance(replacement, tuple) or len(replacement) != 2:
                            raise RuntimeError('forward pre-hook must return None or (new_args, new_kwargs)')
                        args, kwargs = replacement
                else:
                    replacement = hook(self, args)
                    if replacement is not None:
                        args = replacement if isinstance(replacement, tuple) else (replacement,)
            result = forward(*args, **kwargs)
            for key, hook in tuple(attributes.get('_forward_hooks', _EMPTY).items()):
                if key in attributes.get('_forward_hooks_always_called', ()):
                    called.add(key)
                replacement = (hook(self, args, kwargs, result)
                               if key in attributes.get('_forward_hooks_with_kwargs', ())
                               else hook(self, args, result))
                if replacement is not None:
                    result = replacement
            return result
        except Exception:
            # Cleanup observes the current registrations. A failing always-call
            # callback must not hide the original forward/pre/post exception.
            for key, hook in attributes.get('_forward_hooks', _EMPTY).items():
                if key in attributes.get('_forward_hooks_always_called', ()) and key not in called:
                    try:
                        replacement = (hook(self, args, kwargs, result)
                                       if key in attributes.get('_forward_hooks_with_kwargs', ())
                                       else hook(self, args, result))
                        if replacement is not None:
                            result = replacement
                    except Exception as error:
                        warnings.warn(f'forward hook with always_call=True failed during exception cleanup: {error}',
                                      UserWarning, stacklevel=2)
            raise
