"""Keep browser compatibility checks tied to the maintained problem suites."""
import hashlib
import json
import subprocess
import sys

import pytest

from paperleet.browser_judge import PILOTS, SHARED_FILES, export_browser_bundle
from paperleet.catalog import problem, root


def hydrate_bundle(output, metadata):
    """Materialize the actual exported manifest/runtime pair for isolated CPU checks."""
    manifest = json.loads((output / metadata['bundle_url']).read_text())
    runtime = json.loads((output / manifest['runtime_bundle_url']).read_text())
    assert runtime['backend'] == metadata['backend']
    return {**manifest, 'adapter': runtime['adapter'], 'harness': runtime['harness'],
            'files': {**runtime['files'], **manifest['files']}}


def hydrated_path(output, metadata):
    path = output / 'hydrated.json'
    path.write_text(json.dumps(hydrate_bundle(output, metadata)))
    return path


def test_pilot_bundles_use_unchanged_tests_and_are_content_addressed(tmp_path):
    old_bytes = 0
    for identifier in PILOTS:
        entry = problem(identifier)
        metadata = export_browser_bundle(entry, tmp_path)
        path = tmp_path / metadata['bundle_url']
        manifest = json.loads(path.read_text())
        assert manifest['version'] == 2
        assert not {'adapter', 'harness'}.intersection(manifest)
        assert not set(SHARED_FILES).intersection(manifest['files'])
        for asset in (path, tmp_path / manifest['runtime_bundle_url']):
            assert hashlib.sha256(asset.read_bytes()).hexdigest()[:20] in asset.name
        bundle = hydrate_bundle(tmp_path, metadata)
        assert bundle['adapter'] == (root() / 'paperleet/browser' / ('schedules_numpy.py' if identifier == 'diffusion/schedules' else 'numpy_autograd.py')).read_text()
        assert bundle['harness'] == (root() / 'paperleet/browser/harness.py').read_text()
        assert bundle['files']['paperleet/common/asserts.py'] == (root()/'paperleet/common/asserts.py').read_text()
        assert bundle['problem'] == identifier
        assert bundle['tests'] == entry['tests']
        if identifier in {'diffusion/ddpm_sampling','diffusion/ddim','diffusion/flow_matching'}:
            for name in ('toydata.py', 'oracles.py', 'metrics.py', 'models.py'):
                assert bundle['files'][f'paperleet/common/{name}'] == (root() / 'paperleet/common' / name).read_text()
            yaml_path = entry['directory'] / 'problem.yaml'
            assert bundle['files'][yaml_path.relative_to(root()).as_posix()] == yaml_path.read_text()
        for helper in ('experiment.py', 'toy.py'):
            path = entry['directory'] / helper
            if path.exists():
                assert bundle['files'][path.relative_to(root()).as_posix()] == path.read_text()
        assert bundle['files']['paperleet/common/comparisons.py'] == (root()/'paperleet/common/comparisons.py').read_text()
        for test in (entry['directory'] / 'tests').glob('test_*.py'):
            name = test.relative_to(root()).as_posix()
            assert bundle['files'][name] == test.read_text()
        if identifier in {'rl/gae', 'rl/ppo_clip', 'rl/grpo', 'rl/dr_grpo'}:
            assert bundle['files']['paperleet/common/rl_envs.py'] == (root() / 'paperleet/common/rl_envs.py').read_text()
        if identifier == 'rl/dr_grpo':
            for name in ('solution.py', 'experiment.py'):
                relative = f'problems/rl/grpo/{name}'
                assert bundle['files'][relative] == (root() / relative).read_text()
        assert metadata == export_browser_bundle(entry, tmp_path)
        assert metadata['backend'] == ('numpy-wasm' if identifier == 'diffusion/schedules' else 'numpy-autograd-wasm')
        legacy = {key: value for key, value in bundle.items() if key != 'runtime_bundle_url'}
        legacy['version'] = 1
        old_bytes += len(json.dumps(legacy, ensure_ascii=False, separators=(',', ':')).encode())
    runtimes = list((tmp_path / 'browser-judge').glob('runtime-*.json'))
    assert len(runtimes) == 2, 'All autodiff modules must share one runtime'
    for runtime in runtimes:
        assert set(json.loads(runtime.read_text())['files']) == set(SHARED_FILES)
    new_bytes = sum(asset.stat().st_size for asset in (tmp_path / 'browser-judge').glob('*.json'))
    assert new_bytes < old_bytes * 0.35, 'Public runtime deduplication must save at least 65% across the catalog'
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
updates=[]
activities=[]
calibration = calibrate(bundle, sys.argv[2], progress=lambda *row: updates.append(row), activity=lambda *row: activities.append(row))
assert updates == [('reference', i, 8) for i in range(9)] + [('mutants', i, 5) for i in range(6)], updates
assert [(row[0],row[1],row[2],row[3],row[7]) for row in activities] == [('reference',i,8,i,'') for i in range(8)] + [('mutants',m,5,i,name) for m,name in enumerate(bundle['mutants']) for i in range(8)], activities
assert all(row[4].startswith('test_') and row[5] in {'T0','T1','T2','T3'} and row[6] for row in activities)
assert calibration['compatible'], calibration
assert calibration['reference_passed'] == 8
assert len(calibration['mutants_caught']) == 5
source = bundle['reference']
assert run_suite(bundle, source, sys.argv[2])['passed']
wrong = source.replace('return ab.sqrt() * x0 + (1 - ab).sqrt() * noise', 'return x0')
updates.clear()
activities.clear()
wrong_report=run_suite(bundle,wrong,sys.argv[2],progress=lambda *row:updates.append(row),activity=lambda *row:activities.append(row))
assert not wrong_report['passed']
assert updates == [('tests', i, 8) for i in range(9)], updates
assert [(row[0],row[1],row[2],row[3]) for row in activities]==[('tests',i,8,i) for i in range(8)],activities
assert any(row.get('comparison',{}).get('kind')=='value' for row in wrong_report['tests'])
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
    result = subprocess.run([sys.executable, '-c', script, str(hydrated_path(tmp_path, metadata)), str(tmp_path / 'judge')],
                            cwd=root(), capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(json.loads(result.stdout)['mutants_caught']) == 5


@pytest.mark.parametrize('identifier,count,mutants', [
    ('architecture/attention',7,3),('finetuning/lora',6,4),
    ('diffusion/param_conversion',7,4),('architecture/layer_norm',7,4),
    ('architecture/positional_encoding',7,4),('rl/dpo',9,5),
    ('architecture/multihead_attention',7,4),('architecture/positionwise_ffn',7,4),('optimization/adamw',7,4),
    ('rl/gae',11,4),('rl/ppo_clip',11,5),('rl/grpo',9,5),('rl/dr_grpo',6,3),
    ('rl/dapo',7,4),('rl/gspo',7,4),('representation/clip',7,4),('architecture/situ_glu',9,4),
    ('finetuning/qlora',6,4),('architecture/block_attention_residual',7,4),
    ('architecture/differential_attention',7,4),('representation/vicreg',9,4),
    ('representation/rotation_trick',7,4),('world_models/lewm_prediction',8,4),
    ('world_models/lewm_planning',8,4),('world_models/causal_jepa',8,4),
    ('representation/repa',7,4),('representation/ijepa',9,4),('finetuning/distillm2',9,4),
    ('finetuning/opsd_alignment',7,4),('finetuning/opsd_clipping',7,4),('rl/sdpo_feedback',7,4),('rl/sdpo_topk',7,4),
    ('optimization/muon_split',7,4),('finetuning/svdquant',9,4),('architecture/flash_attention',7,4),('architecture/kimi_kda',7,4),
    ('rl/glm5_offpolicy',7,4),('rl/opd',6,4),('diffusion/consistency',7,4),('diffusion/progressive_distillation',6,4),
    ('diffusion/ddpm_sampling',10,5),('diffusion/ddim',9,5),
    ('diffusion/flow_matching',8,4),
    ('diffusion/daps',9,4),('representation/sigreg',9,4),('representation/visreg',9,4),
    ('representation/moge_alignment',9,4),('representation/var_tokenizer',8,4),('architecture/var_next_scale',8,4),
    ('diffusion/block_diffusion',7,4),('rl/pkpo',9,4),('diffusion/diffusion_nft',9,4),('diffusion/fd_population',9,4),
    ('diffusion/jit',6,3),('diffusion/drifting',7,4),('diffusion/dmd',7,4),('diffusion/dmd2',6,4),
    ('diffusion/fd_loss',9,4),
    ('diffusion/meanflow',6,3),
    ('architecture/rms_norm',9,4),
    ('architecture/gqa',10,5),
    ('architecture/rope',10,6),
    ('retrieval/semantic_ids',8,4),('retrieval/generative_retrieval',8,5),('retrieval/late_interaction',10,6),
    ('retrieval/sasrec',10,7),
    ('retrieval/bpr',11,8),
    ('retrieval/lightgcn',10,8),
])
def test_browser_modules_unchanged_suites_and_mutants(tmp_path,identifier,count,mutants):
    folder = tmp_path / identifier.replace('/', '-')
    metadata = export_browser_bundle(problem(identifier), folder)
    script = r'''
import json,sys
from paperleet.browser.harness import prepare, calibrate, run_suite
b=json.load(open(sys.argv[1]));prepare(b,b['adapter'],sys.argv[2])
updates=[]
activities=[]
c=calibrate(b,sys.argv[2],progress=lambda *row:updates.append(row),activity=lambda *row:activities.append(row));assert c['compatible'], {**{k:v for k,v in c.items() if k!='reference'}, 'error':c.get('reference',{}).get('error'), 'failures': [(t['id'],t['message']) for t in c.get('reference',{}).get('tests',[]) if not t['passed']]}
assert c['reference_passed']==int(sys.argv[3]);assert len(c['mutants_caught'])==int(sys.argv[4])
count,mutants=int(sys.argv[3]),int(sys.argv[4])
assert updates==[('reference',i,count) for i in range(count+1)]+[('mutants',i,mutants) for i in range(mutants+1)],updates
assert [(row[0],row[1],row[2],row[3],row[7]) for row in activities]==[('reference',i,count,i,'') for i in range(count)]+[('mutants',m,mutants,i,name) for m,name in enumerate(b['mutants']) for i in range(count)],activities
updates.clear()
activities.clear()
r=run_suite(b,b['reference'],sys.argv[2],progress=lambda *row:updates.append(row),activity=lambda *row:activities.append(row));assert r['passed'],r
assert updates==[('tests',i,count) for i in range(count+1)],updates
assert [(row[0],row[1],row[2],row[3]) for row in activities]==[('tests',i,count,i) for i in range(count)],activities
assert [(row[4],row[5],row[6]) for row in activities]==[(row['nodeid'].split('::')[-1],row['layer'],row['why']) for row in r['tests']],activities
assert not b['tests'].get('T3') or any(t['artifacts'] for t in r['tests'])
if b['problem']=='architecture/attention':
    invalid=b['reference'].replace('return weights @ v, weights','return (weights @ v) * float("nan"), weights')
    invalid_report=run_suite(b,invalid,sys.argv[2]);assert not invalid_report['passed']
    assert any(t.get('comparison',{}).get('actual')=='NaN' for t in invalid_report['tests'])
    assert any(t['artifacts'] for t in invalid_report['tests'])
    assert 'nan' in json.dumps(invalid_report,allow_nan=False)
syntax=run_suite(b,'def broken(',sys.argv[2]);assert not syntax['passed'] and not syntax.get('error')
unsupported=run_suite(b,b['reference']+'\nimport torch\ntorch.cuda.synchronize()\n',sys.argv[2])
assert not unsupported['passed'] and any('BrowserUnsupported' in t['message'] for t in unsupported['tests'])
b['mutants']={'unsupported':b['reference']+'\nimport torch\ntorch.cuda.synchronize()\n'}
rejected=calibrate(b,sys.argv[2]);assert not rejected['compatible'] and rejected['reason']=='mutant_api_unsupported',rejected
print(json.dumps({'calibration':c,'report':r}))
'''
    out = subprocess.run([sys.executable, '-c', script, str(hydrated_path(folder, metadata)), str(folder / 'judge'), str(count), str(mutants)],
                         cwd=root(), capture_output=True, text=True, timeout=300 if identifier == 'diffusion/flow_matching' else 120 if identifier in {'diffusion/ddpm_sampling','diffusion/ddim'} else 45)
    assert out.returncode == 0, out.stdout + out.stderr
    payload = json.loads(out.stdout)
    assert payload['report']['runtime'] == 'browser-numpy-wasm'
