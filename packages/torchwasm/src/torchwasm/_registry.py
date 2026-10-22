"""NumPy-backed registry implementation; extracted from PaperLeet."""
import functools
import inspect
import types
import weakref
from collections import namedtuple


_EMPTY_IMPLEMENTATION_KEYWORDS = frozenset()

OperatorCapability = namedtuple('OperatorCapability', 'reverse forward unsupported_keywords signature')
_operator_names = weakref.WeakKeyDictionary()


def _restore_operator(name):
    # Resolve inside this distribution, including PaperLeet's private copy.
    # Never import torch: it may refer to a native installation in the host.
    from . import torch
    value = torch
    for part in name.split('.')[1:]:
        value = getattr(value, part)
    return value


class _OperatorReference:
    """A pickle-only reference; live models keep their original callable."""
    def __init__(self, function, name):
        self.function, self.name = function, name

    def __reduce__(self):
        return _restore_operator, (self.name,)

    def __deepcopy__(self, memo):
        return self.function


def pickle_operator(function):
    name = _operator_names.get(function) if isinstance(function, types.FunctionType) else None
    return _OperatorReference(function, name) if name is not None else function


def contains_tensor(value, predicate, seen=None):
    if isinstance(value, _tensor.Tensor): return predicate(value)
    if isinstance(value, (tuple, list, dict)):
        if seen is None:seen=set()
        if id(value) in seen:return False
        seen.add(id(value))
        if isinstance(value,dict):return any(contains_tensor(v,predicate,seen) for pair in value.items() for v in pair)
        return any(contains_tensor(v,predicate,seen) for v in value)
    return False


class OperatorRegistry:
    """Export and guard each public operator from the same capability declaration.

    Shape/dtype-specific restrictions stay with the numerical implementation.
    Missing AD declarations fail closed; unknown argument typos remain TypeError.
    """
    def __init__(self, modules):
        self.modules, self.capabilities = modules, {}

    def add(self, name, function, *, reverse='unsupported', forward=False, unsupported_keywords=(), implementation_keywords=()):
        if name in self.capabilities: raise ValueError(f'Duplicate browser operator: {name}')
        if reverse not in ('analytic', 'nondifferentiable', 'factory', 'control', 'unsupported'):
            raise ValueError(f'Invalid reverse-mode capability: {reverse}')
        prefix, attribute = name.rsplit('.', 1)
        signature = inspect.signature(function)
        unsupported_keywords = tuple(unsupported_keywords)
        implementation_keywords = frozenset(implementation_keywords) if implementation_keywords else _EMPTY_IMPLEMENTATION_KEYWORDS
        if not implementation_keywords.issubset(unsupported_keywords):
            raise ValueError('Implementation keywords must declare their unsupported boundary')
        if not implementation_keywords.issubset(signature.parameters):
            raise ValueError('Implementation keywords must have explicit function parameters')
        capability = OperatorCapability(reverse, forward, tuple(unsupported_keywords), str(signature))
        self.capabilities[name] = capability
        option_names = tuple(keyword for keyword in unsupported_keywords if keyword not in implementation_keywords)
        validated_patterns = None

        @functools.wraps(function)
        def call(*args, **kwargs):
            nonlocal validated_patterns
            # Some operators validate an option's type before reporting their
            # unsupported mode. Keep those options in the bound implementation
            # call while retaining the public capability restriction.
            options = {keyword: kwargs.pop(keyword) for keyword in option_names if keyword in kwargs} if option_names else None
            # Binding validates argument structure only. Keep values and AD state
            # out of this bounded cache, and run their guards on every call.
            pattern = (len(args), tuple(kwargs))
            if validated_patterns is None or pattern not in validated_patterns:
                signature.bind(*args, **kwargs)
                if validated_patterns is None: validated_patterns = set()
                # A str subclass may carry arbitrary payloads through attributes.
                # Retain only ordinary keyword names, never such user objects.
                if len(validated_patterns) < 32 and all(type(keyword) is str for keyword in kwargs):
                    validated_patterns.add(pattern)
            if options:
                for keyword, value in options.items():
                    if value is not None: raise _errors.BrowserUnsupported(f'{name} argument {keyword} requires the server')
            if _state._jvp_active and not capability.forward and contains_tensor((args, kwargs), lambda v: isinstance(v, _autograd.ForwardTensor)):
                raise _errors.BrowserUnsupported(f'{name} has no validated forward-mode rule; use the server')
            if capability.reverse == 'unsupported' and _state._grad_enabled and contains_tensor((args, kwargs), lambda v: v.requires_grad):
                raise _errors.BrowserUnsupported(f'{name} has no validated reverse-mode rule; use the server')
            return function(*args, **kwargs)

        _operator_names[call] = name
        setattr(self.modules[prefix], attribute, call)

    def group(self, prefix, functions, **capabilities):
        for name, function in functions.items(): self.add(f'{prefix}.{name}', function, **capabilities)

    def method(self, name, *, reverse='analytic', forward=False, unsupported_keywords=()):
        def call(input, *args, **kwargs):
            return getattr(_ops.wrap(input), name)(*args, **kwargs)
        call.__name__ = name
        signature = inspect.signature(getattr(_tensor.Tensor, name))
        parameters = list(signature.parameters.values())
        call.__signature__ = signature.replace(parameters=[parameters[0].replace(name='input'), *parameters[1:]])
        self.add(f'torch.{name}', call, reverse=reverse, forward=forward, unsupported_keywords=unsupported_keywords)


# Late module references break Tensor/autograd/operator cycles without rebinding functions.
from . import _autograd, _errors, _ops, _state, _tensor
