"""Built wheels work alone and can coexist without sharing installed files."""
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def runtime_wheels(tmp_path_factory):
    temporary = tmp_path_factory.mktemp('runtime-wheels')
    source = temporary / 'source'
    source.mkdir()
    shutil.copy2(ROOT / 'pyproject.toml', source / 'pyproject.toml')
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info', 'build', '.pytest_cache')
    shutil.copytree(ROOT / 'paperleet', source / 'paperleet', ignore=ignore)
    def library_ignore(directory, names):
        skipped = set(ignore(directory, names))
        # Raw benchmark archives are not wheel inputs. Keep the benchmark
        # programs and every distributable source without copying the archives.
        if Path(directory) == ROOT / 'packages/stridetorch/benchmarks':
            skipped.add('results')
        return skipped
    shutil.copytree(ROOT / 'packages/stridetorch', source / 'packages/stridetorch', ignore=library_ignore)
    output = temporary / 'wheels'
    output.mkdir()
    result = {}
    for project, distribution in [('packages/stridetorch', 'stridetorch'), ('.', 'paperleet')]:
        built = subprocess.run([sys.executable, '-c',
            'import sys; from setuptools import build_meta; build_meta.build_wheel(sys.argv[1])', str(output)],
            cwd=source / project, capture_output=True, text=True, timeout=60)
        assert built.returncode == 0, built.stdout + built.stderr
        result[distribution] = next(output.glob(distribution + '-*.whl'))
    with zipfile.ZipFile(result['paperleet']) as main, zipfile.ZipFile(result['stridetorch']) as library:
        assert not set(main.namelist()).intersection(library.namelist()), 'Distributions must not own the same installed paths'
        assert not any(name.startswith('stridetorch/') for name in main.namelist())
        application_sources = {str(path.relative_to(ROOT)) for path in (ROOT / 'paperleet').rglob('*.py')}
        assert application_sources <= set(main.namelist()), 'The main wheel must retain all existing application modules'
        runtime = ROOT / 'packages/stridetorch/src/stridetorch'
        for path in runtime.iterdir():
            if path.suffix == '.py' or path.name in ('LICENSE', 'NOTICE'):
                expected = path.read_bytes()
                assert library.read('stridetorch/' + path.name) == expected
                assert main.read('paperleet/_vendor/torch/' + path.name) == expected
    return result


def isolated_runtime(target, distribution):
    script = '''
import importlib, importlib.abc, pathlib, sys
sys.path.insert(0, sys.argv[1])
standalone = sys.argv[2] == 'stridetorch'
class NoApplicationOrNativeTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        forbidden = name=='torch' or name.startswith('torch.')
        if name=='paperleet' or name.startswith('paperleet.'):
            forbidden = forbidden or standalone or not (name=='paperleet' or name=='paperleet._vendor' or name.startswith('paperleet._vendor.'))
        if forbidden: raise ImportError('Runtime must not import native torch or application features: '+name)
sys.meta_path.insert(0,NoApplicationOrNativeTorch())
name='stridetorch' if standalone else 'paperleet._vendor.torch'
library=importlib.import_module(name)
assert pathlib.Path(library.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
torch=library.torch
x=torch.tensor([2.,3.],requires_grad=True);x.square().sum().backward()
assert x.grad.tolist()==[4.,6.]
sources=importlib.import_module(name+'.runtime').source_files()
assert 'stridetorch/_tensor.py' in sources and 'stridetorch/_autograd.py' in sources
for profile in ('autograd', 'schedules'):
    resources=importlib.import_module(name+'.runtime').source_files(profile)
    for filename in ('LICENSE', 'NOTICE'):
        installed=pathlib.Path(library.__file__).parent / filename
        assert resources['stridetorch/'+filename] == installed.read_text(encoding='utf-8')
    assert 'Version 2.0, January 2004' in resources['stridetorch/LICENSE']
assert 'torch' not in sys.modules
assert ('paperleet' not in sys.modules) if standalone else ('stridetorch' not in sys.modules)
'''
    result = subprocess.run([sys.executable, '-I', '-c', script, str(target), distribution],
                            cwd=target.parent, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('distribution', ['stridetorch', 'paperleet'])
def test_built_wheels_work_without_checkout_paths(tmp_path, runtime_wheels, distribution):
    target = tmp_path / 'installed'
    with zipfile.ZipFile(runtime_wheels[distribution]) as archive:
        archive.extractall(target)
        if distribution == 'stridetorch':
            assert not any(name.startswith(('paperleet/', 'torch/')) for name in archive.namelist())
            metadata_name = next(name for name in archive.namelist()
                                 if name.startswith('stridetorch-') and name.endswith('.dist-info/METADATA'))
            metadata = archive.read(metadata_name).decode()
            requirements = [line for line in metadata.splitlines() if line.startswith('Requires-Dist:')]
            assert len(requirements) == 1 and requirements[0].startswith('Requires-Dist: numpy')
            assert 'License-Expression: Apache-2.0\n' in metadata
            for filename in ('LICENSE', 'NOTICE'):
                assert 'License-File: ' + filename + '\n' in metadata
                canonical = (ROOT / 'packages/stridetorch' / filename).read_bytes()
                assert archive.read('stridetorch/' + filename) == canonical
                notice = next(name for name in archive.namelist()
                              if name.endswith('.dist-info/licenses/' + filename))
                assert archive.read(notice) == canonical
    isolated_runtime(target, distribution)


def test_application_wheel_exports_without_the_standalone_distribution(tmp_path, runtime_wheels):
    target, catalog = tmp_path / 'installed', tmp_path / 'catalog'
    with zipfile.ZipFile(runtime_wheels['paperleet']) as archive:
        archive.extractall(target)
    shutil.copytree(ROOT / 'problems/architecture/attention', catalog / 'problems/architecture/attention')
    script = '''
import importlib.abc, json, os, pathlib, sys
sys.path.insert(0, sys.argv[1])
os.environ['PAPERLEET_ROOT'] = sys.argv[2]
class OnlyTheApplicationDistribution(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'torch' or name.startswith('torch.') or name == 'stridetorch' or name.startswith('stridetorch.'):
            raise ImportError('Unexpected standalone/native dependency: ' + name)
sys.meta_path.insert(0, OnlyTheApplicationDistribution())
from paperleet.catalog import problem
from paperleet import browser_judge
assert pathlib.Path(browser_judge.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
output = pathlib.Path(sys.argv[2]) / 'exports'
metadata = browser_judge.export_browser_bundle(problem('architecture/attention'), output)
manifest = json.loads((output / metadata['bundle_url']).read_text())
runtime = json.loads((output / manifest['runtime_bundle_url']).read_text())
assert 'stridetorch/_autograd.py' in runtime['files']
assert 'stridetorch/_nn.py' in runtime['files']
assert 'Version 2.0, January 2004' in runtime['files']['stridetorch/LICENSE']
assert 'Copyright 2026 PaperLeet contributors' in runtime['files']['stridetorch/NOTICE']
assert 'stridetorch' not in sys.modules and 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-I', '-c', script, str(target), str(catalog)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('removed,remaining', [('stridetorch', 'paperleet'), ('paperleet', 'stridetorch')])
def test_uninstalling_one_distribution_cannot_remove_the_other_runtime(tmp_path, runtime_wheels, removed, remaining):
    target = tmp_path / 'installed'
    for wheel in runtime_wheels.values():
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(target)
    with zipfile.ZipFile(runtime_wheels[removed]) as archive:
        for name in archive.namelist():
            path = target / name
            if path.is_file():
                path.unlink()
    isolated_runtime(target, remaining)


@pytest.mark.parametrize('entry', ['module', 'console'])
@pytest.mark.parametrize('exit_code', [0, 7])
def test_wheel_launches_unchanged_scripts_outside_checkout(tmp_path, runtime_wheels, entry, exit_code):
    target = tmp_path / 'installed'
    with zipfile.ZipFile(runtime_wheels['stridetorch']) as archive:
        archive.extractall(target)
    scripts = tmp_path / 'user scripts'
    scripts.mkdir()
    (scripts / 'sibling.py').write_text('VALUE = 5\n')
    script = scripts / 'train.py'
    script.write_text('''
import sys
import torch
from torch import nn
import torch.nn.functional as F
from sibling import VALUE
assert __name__ == '__main__'
assert sys.argv[1:] == ['--epochs', '2', 'argument with spaces']
assert torch.nn is nn and torch.nn.functional is F
x = torch.tensor([2., 3.], requires_grad=True)
x.square().sum().backward()
assert x.grad.tolist() == [4., 6.]
print('unchanged script passed', VALUE)
raise SystemExit(''' + str(exit_code) + ''')
''')
    bootstrap = '''
import importlib.abc, importlib.metadata, runpy, sys
target, entry, script = sys.argv[1:]
sys.path.insert(0, target)
class NoNativeTorchOrPaperLeet(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'torch' or name.startswith('torch.') or name == 'paperleet' or name.startswith('paperleet.'):
            raise ImportError('Unexpected dependency: ' + name)
sys.meta_path.insert(0, NoNativeTorchOrPaperLeet())
sys.argv = ['stridetorch', script, '--epochs', '2', 'argument with spaces']
if entry == 'module':
    runpy.run_module('stridetorch', run_name='__main__')
else:
    entries = importlib.metadata.distribution('stridetorch').entry_points
    next(e for e in entries if e.group == 'console_scripts' and e.name == 'stridetorch').load()()
'''
    result = subprocess.run([sys.executable, '-I', '-c', bootstrap, str(target), entry, str(script)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert 'unchanged script passed 5' in result.stdout
