"""NumPy-backed errors implementation; extracted from PaperLeet."""


class BrowserUnsupported(NotImplementedError):
    pass


class BrowserUnsupportedAttributeError(AttributeError, BrowserUnsupported):
    """A missing symbol with Python lookup and backend fallback semantics."""


def missing_attribute(obj, name, qualified=None):
    if name.startswith('_'):
        raise AttributeError(name, name=name, obj=obj)
    raise BrowserUnsupportedAttributeError(
        f'{qualified or name} is outside the CPU/WASM autodiff backend',
        name=name, obj=obj)


def unsupported(name):
    if name.startswith('_'):
        raise AttributeError(name)
    raise BrowserUnsupported(f'{name} is outside the CPU/WASM autodiff backend')
