"""NumPy-backed registry implementation; extracted from PaperLeet."""
import functools
import inspect
from collections import namedtuple


OperatorCapability = namedtuple('OperatorCapability', 'reverse forward unsupported_keywords signature')


def contains_tensor(value, predicate):
    if isinstance(value, _tensor.Tensor): return predicate(value)
    if isinstance(value, (tuple, list)): return any(contains_tensor(v, predicate) for v in value)
    if isinstance(value, dict): return any(contains_tensor(v, predicate) for v in value.values())
    return False


class OperatorRegistry:
    """Export and guard each public operator from the same capability declaration.

    Shape/dtype-specific restrictions stay with the numerical implementation.
    Missing AD declarations fail closed; unknown argument typos remain TypeError.
    """
    def __init__(self, modules):
        self.modules, self.capabilities = modules, {}

    def add(self, name, function, *, reverse='unsupported', forward=False, unsupported_keywords=()):
        if name in self.capabilities: raise ValueError(f'Duplicate browser operator: {name}')
        if reverse not in ('analytic', 'nondifferentiable', 'factory', 'control', 'unsupported'):
            raise ValueError(f'Invalid reverse-mode capability: {reverse}')
        prefix, attribute = name.rsplit('.', 1)
        signature = inspect.signature(function)
        capability = OperatorCapability(reverse, forward, tuple(unsupported_keywords), str(signature))
        self.capabilities[name] = capability
        validated_patterns = None

        @functools.wraps(function)
        def call(*args, **kwargs):
            nonlocal validated_patterns
            options = {keyword: kwargs.pop(keyword) for keyword in capability.unsupported_keywords if keyword in kwargs}
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
            for keyword, value in options.items():
                if value is not None: raise _errors.BrowserUnsupported(f'{name} argument {keyword} requires the server')
            if _state._jvp_active and not capability.forward and contains_tensor((args, kwargs), lambda v: isinstance(v, _autograd.ForwardTensor)):
                raise _errors.BrowserUnsupported(f'{name} has no validated forward-mode rule; use the server')
            if capability.reverse == 'unsupported' and _state._grad_enabled and contains_tensor((args, kwargs), lambda v: v.requires_grad):
                raise _errors.BrowserUnsupported(f'{name} has no validated reverse-mode rule; use the server')
            return function(*args, **kwargs)

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
