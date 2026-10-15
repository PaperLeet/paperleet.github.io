"""Configuration expansion, full multi-file suites and typed compatibility failures."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from paperleet.browser_config import DEFAULT_BUDGETS, browser_config
from paperleet.browser_judge import export_browser_bundle, require_browser_catalog


@pytest.mark.parametrize('change', [
    {'backend': 'unvalidated'}, {'backend': []}, {'typo': True},
    {'resources': ['../secret.py']}, {'resources': ['/absolute.py']},
    {'resources': ['paperleet/common/../secret.py']}, {'resources': ['paperleet/common//test.py']},
    {'resources': ['settings.yaml']}, {'resources': ['paperleet/accounts.py']},
    {'resources': ['problems/demo/example/problem.yaml']},
    {'resources': ['paperleet/common/asserts.py'] * 2},
    {'packages': ['torch']}, {'packages': ['pyyaml', 'pyyaml']}, {'packages': [True]},
    {'budgets': {'run': True}}, {'budgets': {'total': 300001}},
    {'budgets': {'run': 0}}, {'budgets': {'run': 120001}}, {'budgets': {'typo': 1000}},
])
def test_invalid_execution_config_fails_before_publication(change):
    entry = {'id': 'demo/example', 'browser': {'backend': 'numpy-autograd-wasm', **change}}
    with pytest.raises(ValueError): require_browser_catalog([entry])


def test_defaults_and_explicit_resources_packages_and_budgets():
    assert browser_config({'id': 'demo/example'}) is None
    assert browser_config({'id': 'demo/example', 'browser': {'backend': 'numpy-autograd-wasm'}}) == {
        'backend': 'numpy-autograd-wasm', 'resources': [], 'packages': [], 'budgets': DEFAULT_BUDGETS,
    }
    config = {'backend': 'numpy-autograd-wasm', 'resources': ['problems/demo/example/problem.yaml'],
              'packages': ['pyyaml'], 'budgets': {'run': 20000}}
    assert browser_config({'id': 'demo/example', 'browser': config})['budgets']['run'] == 20000
    assert 'total' not in config['budgets'], 'Normalizing must not mutate catalog input'


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.setenv('PAPERLEET_ROOT', str(tmp_path))
    directory = tmp_path / 'problems/demo/example'
    (directory / 'tests').mkdir(parents=True)
    (directory / 'mutants').mkdir()
    (directory / 'solution.py').write_text('def value(): return 3\n')
    (directory / 'mutants/wrong.py').write_text('def value(): return 4\n')
    for name in ('test_a.py', 'test_b.py'):
        (directory / 'tests' / name).write_text('import pytest\n@pytest.mark.t0\ndef test_value(impl): assert impl.value() == 3\n')
    entry = {'id': 'demo/example', 'number': 999, 'directory': directory,
             'tests': {'T0': 2, 'T1': 0, 'T2': 0, 'T3': 0},
             'browser': {'backend': 'numpy-autograd-wasm', 'budgets': {'run': 20000}}}
    output = tmp_path / 'exports'
    metadata = export_browser_bundle(entry, output)
    manifest = json.loads((output / metadata['bundle_url']).read_text())
    runtime = json.loads((output / manifest['runtime_bundle_url']).read_text())
    bundle = {**manifest, 'adapter': runtime['adapter'], 'harness': runtime['harness'],
              'files': {**runtime['files'], **manifest['files']}}
    path = tmp_path / 'bundle.json'
    path.write_text(json.dumps(bundle))
    return entry, manifest, metadata, path, tmp_path / 'judge'


def execute_bundle(exported, script):
    *_, path, directory = exported
    prelude = '''
import json, sys
from paperleet.browser.harness import prepare, calibrate, run_suite
bundle = json.load(open(sys.argv[1]))
prepare(bundle, bundle['adapter'], sys.argv[2])
'''
    result = subprocess.run([sys.executable, '-c', prelude + script, str(path), str(directory)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_new_module_exports_and_runs_every_test_file_without_registry_edits(exported):
    _, manifest, metadata, *_ = exported
    assert manifest['version'] == 3 and len(manifest['test_files']) == 2
    assert manifest['budgets'] == metadata['budgets'] == {**DEFAULT_BUDGETS, 'run': 20000}
    execute_bundle(exported, '''
c = calibrate(bundle, sys.argv[2]); assert c['compatible'], c
r = run_suite(bundle, bundle['reference'], sys.argv[2]); assert r['passed'], r
assert len(r['tests']) == 2 and len({row['nodeid'] for row in r['tests']}) == 2
assert c['reference_passed'] == 2 and c['mutants_caught'] == {'wrong':['test_value','test_value']}
# Older immutable manifests with one test_file still run unchanged.
bundle.pop('test_files'); bundle['tests']['T0'] = 1
assert run_suite(bundle, bundle['reference'], sys.argv[2])['passed']
''')


def test_real_harness_distinguishes_unsupported_options_from_student_failures(exported):
    execute_bundle(exported, '''
import torch
capabilities = torch._browser_capabilities
assert capabilities['torch.sum'].reverse == 'analytic'
assert capabilities['torch.sum'].forward and 'out' in capabilities['torch.sum'].unsupported_keywords
assert 'dim' in capabilities['torch.sum'].signature
assert not capabilities['torch.nan_to_num'].forward
try: capabilities['forged'] = None
except TypeError: pass
else: raise AssertionError('Capabilities must be immutable')
suffixes = [
    ('\\nimport torch\\ntorch.sum(torch.ones(2,2),dim=0,out=torch.empty(2))', True),
    ('\\nimport torch\\ntorch.sum(torch.ones(2),typo=True)', False),
    ('\\nraise TypeError("student error")', False),
    ('\\nassert False, "BrowserUnsupported ImportError are just assertion text"', False),
    ('\\nimport missing_native_extension', True),
]
for suffix, unsupported in suffixes:
    report = run_suite(bundle, bundle['reference'] + suffix, sys.argv[2])
    assert not report.get('error') and not report['passed'], report
    assert all(row['unsupported_api'] is unsupported for row in report['tests']), report
bundle['mutants'] = {'unsupported': bundle['reference'] + suffixes[0][0]}
assert calibrate(bundle, sys.argv[2])['reason'] == 'mutant_api_unsupported'
''')


def test_resource_symlinks_cannot_escape_repository(exported, tmp_path):
    entry, *_ = exported
    target = tmp_path.parent / (tmp_path.name + '-outside.py')
    target.write_text('outside = True\n')
    try:
        link = entry['directory'] / 'escape.py'
        link.symlink_to(target)
        entry['browser']['resources'] = ['problems/demo/example/escape.py']
        with pytest.raises(ValueError, match='escapes the repository'): export_browser_bundle(entry, tmp_path / 'unsafe')
    finally:
        target.unlink()
