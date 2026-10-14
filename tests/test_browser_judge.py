"""Keep browser compatibility checks tied to the maintained problem suites."""
import json
from pathlib import Path
import subprocess
import sys

from paperleet.browser_judge import PILOTS, export_browser_bundle
from paperleet.catalog import problem, root


def test_pilot_bundles_use_unchanged_tests_and_are_content_addressed(tmp_path):
    for identifier in PILOTS:
        entry = problem(identifier)
        metadata = export_browser_bundle(entry, tmp_path)
        path = tmp_path / metadata['bundle_url']
        bundle = json.loads(path.read_text())
        assert bundle['problem'] == identifier
        assert bundle['tests'] == entry['tests']
        for test in (entry['directory'] / 'tests').glob('test_*.py'):
            name = test.relative_to(root()).as_posix()
            assert bundle['files'][name] == test.read_text()
        assert metadata == export_browser_bundle(entry, tmp_path)
        assert metadata['backend'] == ('numpy-wasm' if identifier == 'diffusion/schedules' else 'numpy-autograd-wasm')
    assert export_browser_bundle({'id': 'other/example'}, tmp_path) is None


def test_schedules_reference_mutants_and_failure_semantics(tmp_path):
    entry = problem('diffusion/schedules')
    metadata = export_browser_bundle(entry, tmp_path)
    # Installing the bounded adapter must never replace native torch in the
    # server/pytest process. A browser worker has this same process isolation.
    script = r'''
import json, sys
from paperleet.browser.harness import prepare, calibrate, run_suite
bundle = json.load(open(sys.argv[1]))
prepare(bundle, bundle['adapter'], sys.argv[2])
calibration = calibrate(bundle, sys.argv[2])
assert calibration['compatible'], calibration
assert calibration['reference_passed'] == 8
assert len(calibration['mutants_caught']) == 5
source = bundle['reference']
assert run_suite(bundle, source, sys.argv[2])['passed']
wrong = source.replace('return ab.sqrt() * x0 + (1 - ab).sqrt() * noise', 'return x0')
assert not run_suite(bundle, wrong, sys.argv[2])['passed']
syntax = run_suite(bundle, 'def broken(', sys.argv[2])
assert not syntax['passed'] and any('SyntaxError' in row['message'] for row in syntax['tests'])
missing = run_suite(bundle, source + '\nimport torch\ntorch.linalg\n', sys.argv[2])
assert not missing['passed'] and any('BrowserUnsupported' in row['message'] for row in missing['tests'])
import torch
assert torch.tensor([1]).dtype == torch.long
assert torch.arange(3).dtype == torch.long
assert torch.full((3,), 1).dtype == torch.long
assert torch.full((3,), 1.0).dtype == torch.float32
assert str(torch.tensor([1.0], dtype=torch.float64).dtype) == 'float64'
assert float(torch.tensor([1 + 2**-40], dtype=torch.float64)[0]) > 1
try:
    torch.tensor([1.], requires_grad=True)
except NotImplementedError:
    pass
else:
    raise AssertionError('Autograd must not be simulated')
print(json.dumps(calibration))
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path / metadata['bundle_url']), str(tmp_path / 'judge')],
                            cwd=root(), capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(json.loads(result.stdout)['mutants_caught']) == 5


def test_attention_and_lora_unchanged_suites_and_mutants(tmp_path):
    for identifier, count, mutants in [('architecture/attention', 7, 3), ('finetuning/lora', 6, 4)]:
        folder = tmp_path / identifier.replace('/', '-')
        metadata = export_browser_bundle(problem(identifier), folder)
        script = r'''
import json,sys
from paperleet.browser.harness import prepare, calibrate, run_suite
b=json.load(open(sys.argv[1]));prepare(b,b['adapter'],sys.argv[2])
c=calibrate(b,sys.argv[2]);assert c['compatible'],c
assert c['reference_passed']==int(sys.argv[3]);assert len(c['mutants_caught'])==int(sys.argv[4])
r=run_suite(b,b['reference'],sys.argv[2]);assert r['passed'],r
assert any(t['artifacts'] for t in r['tests'])
syntax=run_suite(b,'def broken(',sys.argv[2]);assert not syntax['passed'] and not syntax.get('error')
unsupported=run_suite(b,b['reference']+'\nimport torch\ntorch.cuda.synchronize()\n',sys.argv[2])
assert not unsupported['passed'] and any('BrowserUnsupported' in t['message'] for t in unsupported['tests'])
b['mutants']={'unsupported':b['reference']+'\nimport torch\ntorch.cuda.synchronize()\n'}
rejected=calibrate(b,sys.argv[2]);assert not rejected['compatible'] and rejected['reason']=='mutant_api_unsupported',rejected
print(json.dumps({'calibration':c,'report':r}))
'''
        out = subprocess.run([sys.executable, '-c', script, str(folder / metadata['bundle_url']), str(folder / 'judge'), str(count), str(mutants)],
                             cwd=root(), capture_output=True, text=True, timeout=45)
        assert out.returncode == 0, out.stdout + out.stderr
        payload = json.loads(out.stdout)
        assert payload['report']['runtime'] == 'browser-numpy-wasm'
