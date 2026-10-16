"""Registered model/parameter containers; numerical kernels live elsewhere."""
from collections import OrderedDict
from collections.abc import Iterable, Mapping
import operator

from ._module import Module, Parameter
from ._tensor import Tensor
from ._errors import BrowserUnsupported


def _index(index, length):
    index = operator.index(index)
    if not -length <= index < length:
        raise IndexError('Container index out of range')
    return index % length


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        items = modules[0].items() if len(modules) == 1 and isinstance(modules[0], OrderedDict) else enumerate(modules)
        for name, module in items:
            self.add_module(str(name), module)

    def __len__(self): return len(self._modules)
    def __iter__(self): return iter(self._modules.values())
    def __getitem__(self, index):
        items = list(self._modules.items())
        return type(self)(OrderedDict(items[index])) if isinstance(index, slice) else items[_index(index, len(self))][1]

    def __setitem__(self, index, module):
        self.add_module(tuple(self._modules)[_index(index, len(self))], module)

    def __delitem__(self, index):
        names = list(self._modules)
        for name in names[index] if isinstance(index, slice) else [names[_index(index, len(self))]]:
            delattr(self, name)
        self._modules = dict((str(i), value) for i, value in enumerate(self._modules.values()))

    def append(self, module):
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        for module in list(modules) if modules is self else modules: self.append(module)
        return self

    def insert(self, index, module):
        index = operator.index(index)
        if not -len(self) <= index <= len(self): raise IndexError('Container index out of range')
        if not isinstance(module, Module): raise TypeError('Sequential insertion requires a Module')
        if index < 0: index += len(self)
        if tuple(self._modules) != tuple(str(i) for i in range(len(self))):
            raise BrowserUnsupported('Insertion into a named Sequential is unsupported; use append or rebuild it')
        values = list(self); values.insert(index, module)
        self._modules = dict((str(i), value) for i, value in enumerate(values))
        return self

    def __add__(self, other):
        if not isinstance(other, Sequential): raise ValueError('Sequential addition requires Sequential')
        return type(self)(*self, *other)

    def __iadd__(self, other):
        if not isinstance(other, Sequential): raise ValueError('Sequential addition requires Sequential')
        return self.extend(list(other))

    def forward(self, input):
        for module in self: input = module(input)
        return input


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None: self.extend(modules)

    def __len__(self): return len(self._modules)
    def __iter__(self): return iter(self._modules.values())
    def __getitem__(self, index):
        return type(self)(list(self)[index]) if isinstance(index, slice) else self._modules[str(_index(index, len(self)))]

    def __setitem__(self, index, module):
        self.add_module(str(_index(index, len(self))), module)

    def __delitem__(self, index):
        values = list(self)
        del values[index if isinstance(index, slice) else _index(index, len(self))]
        self._modules = dict((str(i), value) for i, value in enumerate(values))

    def append(self, module):
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        # Snapshot self-extension: adding a dictionary key during iteration is
        # otherwise an interpreter-dependent partial mutation.
        if not isinstance(modules, Iterable):
            raise TypeError('ModuleList.extend requires an iterable')
        for module in list(modules) if modules is self else modules: self.append(module)
        return self

    def insert(self, index, module):
        if module is not None and not isinstance(module, Module): raise TypeError('Expected a Module')
        index = operator.index(index)
        if not 0 <= index <= len(self):
            raise BrowserUnsupported('ModuleList insertion requires an index from zero to its length')
        values = list(self); values.insert(index, module)
        self._modules = dict((str(i), value) for i, value in enumerate(values))

    def __iadd__(self, modules): return self.extend(modules)
    def __add__(self, modules): return type(self)(list(self) + list(modules))


class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None: self.update(modules)

    def __len__(self): return len(self._modules)
    def __iter__(self): return iter(self._modules)
    def __contains__(self, key): return key in self._modules
    def __getitem__(self, key): return self._modules[key]
    def __setitem__(self, key, module): self.add_module(key, module)
    def __delitem__(self, key): del self._modules[key]
    def keys(self): return self._modules.keys()
    def values(self): return self._modules.values()
    def items(self): return self._modules.items()
    def clear(self): self._modules.clear()
    def pop(self, key): return self._modules.pop(key)
    def update(self, modules):
        items = modules.items() if isinstance(modules, (Mapping, ModuleDict)) else modules
        for key, module in items: self[key] = module


class ParameterList(Module):
    def __init__(self, values=None):
        super().__init__()
        self._size = 0
        if values is not None: self.extend(values)

    def __len__(self): return self._size
    def __iter__(self): return (getattr(self, str(i)) for i in range(len(self)))
    def __getitem__(self, index):
        return type(self)(list(self)[index]) if isinstance(index, slice) else getattr(self, str(_index(index, len(self))))

    def __setitem__(self, index, value):
        name = str(_index(index, len(self)))
        setattr(self, name, Parameter(value) if isinstance(value, Tensor) and not isinstance(value, Parameter) else value)

    def append(self, value):
        self._size += 1
        self[-1] = value
        return self

    def extend(self, values):
        if not isinstance(values, Iterable) or isinstance(values, Tensor):
            raise TypeError('ParameterList.extend requires an iterable of values, not a Tensor')
        for value in list(values) if values is self else values: self.append(value)
        return self

    def __iadd__(self, values): return self.extend(values)


class ParameterDict(Module):
    def __init__(self, parameters=None):
        super().__init__()
        self._keys = {}
        if parameters is not None: self.update(parameters)

    def __len__(self): return len(self._keys)
    def __iter__(self): return iter(self._keys)
    def __contains__(self, key): return key in self._keys
    def __getitem__(self, key):
        if not isinstance(key, str): raise TypeError('ParameterDict keys must be strings')
        if key not in self._keys: raise AttributeError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        if not isinstance(key, str): raise TypeError('ParameterDict keys must be strings')
        setattr(self, key, Parameter(value) if isinstance(value, Tensor) and not isinstance(value, Parameter) else value)
        self._keys[key] = None

    def __delitem__(self, key):
        if key not in self._keys: raise AttributeError(key)
        delattr(self, key); del self._keys[key]

    def keys(self): return self._keys.keys()
    def values(self): return (self[key] for key in self)
    def items(self): return ((key, self[key]) for key in self)
    def clear(self):
        for key in list(self): del self[key]

    def pop(self, key):
        value = self[key]; del self[key]; return value

    def get(self, key, default=None): return self[key] if key in self else default
    def setdefault(self, key, default=None):
        if key not in self: self[key] = default
        return self[key]

    def update(self, parameters):
        if isinstance(parameters, (OrderedDict, ParameterDict)):
            items = parameters.items()
        elif isinstance(parameters, Mapping):
            items = sorted(parameters.items())
        else:
            items = parameters
        for key, value in items: self[key] = value
