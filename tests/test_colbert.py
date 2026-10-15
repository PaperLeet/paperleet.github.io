"""The ColBERT teaching contract accepts another differentiable implementation."""
import json
from pathlib import Path
import subprocess
import sys

from paperleet.browser_judge import export_browser_bundle
from paperleet.catalog import problem, root
from paperleet.judge.runner import run_problem


ALTERNATIVE = '''import torch

def maxsim_scores(query, documents, query_mask=None, document_mask=None):
    qm, dm = _checked_masks(query, documents, query_mask, document_mask)
    batch, count = query.shape[0], documents.shape[0]
    if not batch or not count:
        return query.sum(dim=(1,2))[:,None].expand(batch,count)
    q = query / query.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    d = documents / documents.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    result = []
    for b in range(batch):
        row = []
        for n in range(count):
            value = q[b].sum()*0 + d[n].sum()*0
            if not bool(dm[n].any()):
                value = value + float('-inf')
            else:
                kept = d[n][dm[n]]
                for i in range(query.shape[1]):
                    if bool(qm[b,i]):
                        value = value + (kept @ q[b,i]).max(dim=0).values
            row.append(value)
        result.append(torch.stack(row))
    return torch.stack(result)

def pairwise_loss(positive,negative):
    _check_pairs(positive,negative)
    logits = torch.stack([positive,negative],dim=1)
    return (torch.logsumexp(logits,dim=1)-positive).mean()
'''


def test_independent_loop_learner_passes_native_and_browser_suite(tmp_path):
    entry = problem('retrieval/late_interaction')
    starter = (entry['directory']/'starter.py').read_text()
    source = ALTERNATIVE + '\n' + starter[starter.index('# Provided validation.'):]
    implementation = tmp_path/'learner.py'; implementation.write_text(source)
    native = run_problem(entry['id'], implementation)
    assert native['passed'], native
    output = tmp_path/'bundles'
    metadata = export_browser_bundle(entry, output)
    script = '''import json,sys
from pathlib import Path
from paperleet.browser.harness import prepare,run_suite
output=Path(sys.argv[1]);metadata=json.loads(sys.argv[2])
bundle=json.loads((output/metadata['bundle_url']).read_text())
runtime=json.loads((output/bundle['runtime_bundle_url']).read_text())
bundle={**bundle,'adapter':runtime['adapter'],'harness':runtime['harness'],'files':{**runtime['files'],**bundle['files']}}
prepare(bundle,bundle['adapter'],sys.argv[3])
report=run_suite(bundle,Path(sys.argv[4]).read_text(),sys.argv[3])
assert report['passed'],report
print(json.dumps({'tests':len(report['tests']),'passed':report['passed']}))
'''
    result = subprocess.run([sys.executable, '-c', script, str(output), json.dumps(metadata), str(tmp_path/'judge'), str(implementation)],
                            cwd=root(), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {'tests':10, 'passed':True}


def test_browser_export_keeps_actual_training_oracle_tests_and_all_mutants(tmp_path):
    entry = problem('retrieval/late_interaction')
    metadata = export_browser_bundle(entry, tmp_path)
    bundle = json.loads((tmp_path/metadata['bundle_url']).read_text())
    directory = entry['directory']
    for name in ('toy.py', 'experiment.py', 'tests/test_colbert.py'):
        assert bundle['files']['problems/retrieval/late_interaction/'+name] == (directory/name).read_text()
    assert bundle['reference'] == (directory/'solution.py').read_text()
    assert len(bundle['mutants']) == 6
    assert sum(bundle['tests'].values()) == 10
    assert bundle['mutants'] == {path.stem:path.read_text() for path in sorted((directory/'mutants').glob('*.py'))}
