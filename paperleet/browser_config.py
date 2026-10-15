"""Validated per-problem browser configuration; no problem-ID routing tables."""
from pathlib import PurePosixPath
import re

BACKENDS = {
    'numpy-wasm': 'schedules_numpy.py',
    'numpy-autograd-wasm': 'numpy_autograd.py',
}
DEFAULT_BUDGETS = {'calibration': 90_000, 'run': 30_000, 'total': 120_000}
OPTIONAL_PACKAGES = frozenset({'pyyaml'})


def browser_config(entry):
    config = entry.get('browser')
    if config is None: return None
    label = entry['id'] + ': browser'
    if not isinstance(config, dict) or set(config) - {'backend', 'resources', 'packages', 'budgets'}:
        raise ValueError(label + ' must contain only backend, resources, packages and budgets')
    backend = config.get('backend')
    if not isinstance(backend, str) or backend not in BACKENDS:
        raise ValueError(label + ' has an unvalidated backend')
    resources = config.get('resources', [])
    if not isinstance(resources, list) or any(not isinstance(name, str) for name in resources):
        raise ValueError(label + ' resources must be repository-relative paths')
    if len(set(resources)) != len(resources): raise ValueError(label + ' has duplicate resources')
    for name in resources:
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or path.as_posix() != name
                or not re.fullmatch(r'(?:[a-zA-Z0-9_]+/)+(?:[a-zA-Z0-9_]+\.py|problem\.yaml)', name)
                or not name.startswith(('paperleet/common/', 'problems/'))):
            raise ValueError(label + ' has an unsafe or unsupported resource: ' + name)
    packages = config.get('packages', [])
    if (not isinstance(packages, list) or any(not isinstance(name, str) or name not in OPTIONAL_PACKAGES for name in packages)
            or len(set(packages)) != len(packages)):
        raise ValueError(label + ' packages must be distinct validated optional packages')
    if any(name.endswith('/problem.yaml') for name in resources) and 'pyyaml' not in packages:
        raise ValueError(label + ' YAML resources require the pyyaml package')
    overrides = config.get('budgets', {})
    if not isinstance(overrides, dict) or set(overrides) - DEFAULT_BUDGETS.keys():
        raise ValueError(label + ' has unknown budget fields')
    budgets = {**DEFAULT_BUDGETS, **overrides}
    if (any(type(value) is not int or not 0 < value <= 300_000 for value in budgets.values())
            or max(budgets['calibration'], budgets['run']) > budgets['total']):
        raise ValueError(label + ' budgets must be positive milliseconds within the five-minute hard cap')
    return {'backend': backend, 'resources': list(resources), 'packages': list(packages), 'budgets': budgets}
