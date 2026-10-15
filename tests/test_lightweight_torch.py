"""Verify real built distributions work independently of the source checkout."""
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('project,distribution', [
    ('packages/paperleet-torch', 'paperleet_torch'), ('.', 'paperleet'),
])
def test_built_wheels_include_the_independent_runtime_without_checkout_paths(tmp_path, project, distribution):
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    result = subprocess.run([sys.executable, '-c',
        'import sys; from setuptools import build_meta; build_meta.build_wheel(sys.argv[1])', str(wheels)],
        cwd=ROOT / project, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next(wheels.glob(distribution + '-*.whl'))
    target = tmp_path / 'installed'
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(target)
        if distribution == 'paperleet_torch':
            assert not any(name.startswith(('paperleet/', 'torch/')) for name in archive.namelist())
            metadata = archive.read('paperleet_torch-0.1.0.dist-info/METADATA').decode()
            requirements = [line for line in metadata.splitlines() if line.startswith('Requires-Dist:')]
            assert len(requirements) == 1 and requirements[0].startswith('Requires-Dist: numpy')
    script = '''
import importlib.abc, pathlib, sys
sys.path.insert(0, sys.argv[1])
class NoApplicationOrNativeTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='torch' or name.startswith('torch.') or name=='paperleet' or name.startswith('paperleet.'):
            raise ImportError('Runtime must not import native torch or the application: '+name)
sys.meta_path.insert(0,NoApplicationOrNativeTorch())
import paperleet_torch
assert pathlib.Path(paperleet_torch.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
from paperleet_torch import torch
x=torch.tensor([2.,3.],requires_grad=True);x.square().sum().backward()
assert x.grad.tolist()==[4.,6.]
from paperleet_torch.runtime import source_files
sources=source_files()
assert 'paperleet_torch/_tensor.py' in sources and 'paperleet_torch/_autograd.py' in sources
assert 'torch' not in sys.modules and 'paperleet' not in sys.modules
'''
    isolated = subprocess.run([sys.executable, '-I', '-c', script, str(target)],
                             cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert isolated.returncode == 0, isolated.stdout + isolated.stderr
