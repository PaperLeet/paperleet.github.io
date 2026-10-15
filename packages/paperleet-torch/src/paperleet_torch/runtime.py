"""Source files needed to embed the library in an offline Pyodide runtime."""
from pathlib import Path


def source_files(profile='autograd'):
    """Return Python sources and redistribution notices from installed resources.

    No PaperLeet modules or source checkout are required. The small schedules
    profile retains its original unsupported-AD behavior.
    """
    package = Path(__file__).resolve().parent
    if profile == 'schedules':
        names = ['__init__.py', '_integration.py', '_legacy.py']
    elif profile == 'autograd':
        names = [p.name for p in sorted(package.glob('*.py'))
                 if p.name not in ('_legacy.py', 'runtime.py', '_cli.py', '__main__.py')]
    else:
        raise ValueError(f'Unknown runtime profile: {profile}')
    names += ['LICENSE', 'NOTICE']
    return {f'paperleet_torch/{name}': (package / name).read_text(encoding='utf-8')
            for name in names}
