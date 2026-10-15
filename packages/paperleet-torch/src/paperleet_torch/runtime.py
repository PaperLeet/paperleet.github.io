"""Source files needed to embed the library in an offline Pyodide runtime."""
from pathlib import Path


def source_files(profile='autograd'):
    """Return self-contained Python sources, using installed package resources.

    No PaperLeet modules or source checkout are required. The small schedules
    profile retains its original unsupported-AD behavior.
    """
    package = Path(__file__).resolve().parent
    if profile == 'schedules':
        names = ['__init__.py', '_integration.py', '_legacy.py']
    elif profile == 'autograd':
        names = [p.name for p in sorted(package.glob('*.py'))
                 if p.name not in ('_legacy.py', 'runtime.py')]
    else:
        raise ValueError(f'Unknown runtime profile: {profile}')
    return {f'paperleet_torch/{name}': (package / name).read_text(encoding='utf-8')
            for name in names}
