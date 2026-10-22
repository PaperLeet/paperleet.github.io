"""Lazy browser bindings keep the optional namespace importable in CPython."""


def _unavailable():
    return RuntimeError('WebGPU transport requires Pyodide and an attached MatrixRpc client; '
                        'await client.ready and assign self.matrix_rpc in the host Worker')


def default_client():
    try:
        from js import matrix_rpc
    except (ImportError, AttributeError):
        raise _unavailable() from None
    return matrix_rpc


def run_sync(value):
    try:
        from pyodide.ffi import run_sync as wait
    except ImportError:
        raise _unavailable() from None
    return wait(value)


def to_js(value, **kwargs):
    try:
        from pyodide.ffi import to_js as convert
    except ImportError:
        raise _unavailable() from None
    return convert(value, **kwargs)


def dict_to_js(value):
    try:
        from js import Object
    except (ImportError, AttributeError):
        raise _unavailable() from None
    return to_js(value, dict_converter=Object.fromEntries)
