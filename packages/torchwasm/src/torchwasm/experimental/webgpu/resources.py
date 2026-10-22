"""Export installed, self-contained resources for an explicitly hosted service."""
from importlib.resources import files

_JAVASCRIPT = ('gpu-service.mjs', 'plan-program.mjs', 'rpc-client.mjs', 'rpc-protocol.mjs')


def service_sources():
    """Return the four adjacent JS modules and their redistribution notices.

    Serve these files in one host directory. The host owns service creation,
    MessagePort delivery, shutdown and its application isolation boundary.
    """
    package = files(__package__)
    root = files('torchwasm')
    return {name: (root if name in ('LICENSE', 'NOTICE') else package)
            .joinpath(name).read_text(encoding='utf-8')
            for name in (*_JAVASCRIPT, 'LICENSE', 'NOTICE')}


def source_files():
    """Return the complete optional package payload for an offline Pyodide FS.

    This explicit payload includes the normal package's root modules and the
    experimental integration. It does not change runtime.source_files() or
    automatically enable GPU dispatch. JS files must also be served by the host.
    """
    root = files('torchwasm')
    result = {'torchwasm/' + item.name: item.read_text(encoding='utf-8')
              for item in root.iterdir()
              if item.name.endswith('.py') or item.name in ('LICENSE', 'NOTICE')}
    result['torchwasm/experimental/__init__.py'] = root.joinpath(
        'experimental', '__init__.py').read_text(encoding='utf-8')
    for item in files(__package__).iterdir():
        if item.name.endswith(('.py', '.mjs')):
            result['torchwasm/experimental/webgpu/' + item.name] = item.read_text(encoding='utf-8')
    return result
