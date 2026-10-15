"""NumPy-backed errors implementation; extracted from PaperLeet."""


class BrowserUnsupported(NotImplementedError):
    pass


def unsupported(name):
    if name.startswith('_'):
        raise AttributeError(name)
    raise BrowserUnsupported(f'{name} is outside the CPU/WASM autodiff backend')
