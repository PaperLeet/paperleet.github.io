// Real pinned Python/NumPy WASM, separate runtimes per bundle. No mocked interpreter.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {loadPyodide} from 'pyodide';
import {loadJudgePackages} from '../public/browser-judge/packages.mjs';
import {unsupportedReport} from '../public/browser-judge/diagnostics.mjs';

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const smokeCases = ['diffusion/schedules', 'architecture/attention', 'finetuning/lora', 'optimization/adamw', 'optimization/muon_split', 'representation/moco', 'representation/simclr',
  'foundations/next_token_loss', 'foundations/sft_loss_mask', 'foundations/top_k_sampling',
  'foundations/top_p_sampling', 'foundations/gradient_accumulation', 'foundations/gradient_clipping'];
const index = JSON.parse(fs.readFileSync(path.join(web, 'public/data/index.json')));
const catalog = index.problems;
const ids = catalog.map(entry => entry.id);
assert(ids.length && new Set(ids).size === ids.length, 'The exported catalog must have distinct modules');
for (const entry of catalog) assert(entry.browser_judge?.bundle_url, `${entry.id}: missing browser bundle`);
const selection = process.argv[2];
try {
  if (!selection || selection === '--all' || selection === '--smoke') {
    const cases = selection === '--smoke' ? smokeCases : ids;
    const results = [];
    const started = performance.now();
    for (const id of cases) {
      const entry = catalog.find(entry => entry.id === id);
      assert(entry, `Unknown smoke module: ${id}`);
      try {
        const output = execFileSync(process.execPath, [fileURLToPath(import.meta.url), id, '--json'], {
          cwd:web, encoding:'utf8', timeout:Math.max(180_000, entry.browser_judge.budgets.total * 3),
          maxBuffer:1024*1024,
        });
        const result = JSON.parse(output);
        assert.equal(result.id, id);
        results.push({...result, passed:true});
        if (result.browser_budget_exceeded.length) {
          const notice = `${id}: correctness passed; ${result.browser_budget_exceeded.join(', ')} exceeded the browser budget on this runner. Browser timeouts still require native fallback.`;
          console.warn(process.env.GITHUB_ACTIONS ? `::warning::${notice}` : `WARN ${notice}`);
        }
        console.log(`PASS real Pyodide ${result.pyodide} / NumPy ${result.numpy}: ${id}, ${result.tests} tests, ${result.mutants} mutants (${(result.elapsed_ms / 1000).toFixed(2)}s)`);
      } catch(error) {
        const detail = String(error?.stderr || error?.message || error).slice(-12000);
        results.push({id,passed:false,error:detail});
        console.error(`FAIL real WASM ${id}: ${detail}`);
      }
    }
    const report = {schema_version:1, scope:selection === '--smoke' ? 'smoke' : 'catalog',
      catalog_modules:ids.length, modules:results.length, passed:results.every(result => result.passed),
      elapsed_ms:Math.round(performance.now() - started),
      browser_budget_exceeded:results.filter(result => result.passed && result.browser_budget_exceeded.length).map(result => result.id), results};
    const reportPath = path.join(web, '../reports/browser-wasm.json');
    fs.mkdirSync(path.dirname(reportPath), {recursive:true});
    fs.writeFileSync(reportPath, JSON.stringify(report,null,2)+'\n');
    console.log(`Real WASM ${report.scope}: ${results.filter(result => result.passed).length}/${results.length} modules passed; ${reportPath}`);
    if (!report.passed) process.exitCode = 1;
  } else {
    const started = performance.now();
    const id = selection;
    const entry = catalog.find(p => p.id === id);
    assert(entry, `Unknown catalog module: ${id}`);
    const read = relative => {
      const bytes = fs.readFileSync(path.join(web, 'public/data', relative));
      assert(relative.includes(createHash('sha256').update(bytes).digest('hex').slice(0,20)), `Asset hash mismatch: ${relative}`);
      return JSON.parse(bytes);
    };
    const manifest = read(entry.browser_judge.bundle_url);
    const runtime = read(manifest.runtime_bundle_url);
    assert.equal(manifest.problem,id);
    assert.deepEqual(manifest.tests,entry.tests);
    assert.equal(runtime.backend,entry.browser_judge.backend);
    const bundle = {...manifest, adapter:runtime.adapter, harness:runtime.harness, files:{...runtime.files,...manifest.files}};
    const py = await loadPyodide({indexURL:path.join(web,'node_modules/pyodide/'),
      packageCacheDir:path.join(web,'public/browser-runtime/pyodide-0.29.4/'),stdout(){},stderr(){}});
    await loadJudgePackages(py, manifest.packages);
    assert.equal(py.runPython("__import__('numpy').__version__"), '2.2.5');
    py.globals.set('bundle_json',JSON.stringify(bundle));
    py.runPython("import json\nbundle=json.loads(bundle_json)\nexec(compile(bundle['harness'],'browser_harness.py','exec'))\nprepare(bundle,bundle['adapter'],'/judge')");
    const prepared = performance.now();
    const calibration = JSON.parse(py.runPython("json.dumps(calibrate(bundle,'/judge'))"));
    assert.equal(calibration.compatible,true,JSON.stringify(calibration));
    assert.equal(calibration.reference_passed,Object.values(entry.tests).reduce((a,b)=>a+b,0));
    assert.equal(Object.keys(calibration.mutants_caught).length,Object.keys(bundle.mutants).length);
    const calibrated = performance.now();
    const report = JSON.parse(py.runPython("json.dumps(run_suite(bundle,bundle['reference'],'/judge'))"));
    const completed = performance.now();
    assert(report.passed && !unsupportedReport(report));
    assert.equal(report.tests.length, Object.values(entry.tests).reduce((a,b) => a+b,0));
    assert.deepEqual(Object.fromEntries(Object.keys(entry.tests).map(layer => [layer,
      report.tests.filter(row => row.layer === layer).length])), entry.tests);
    assert.equal(report.source_sha256,createHash('sha256').update(bundle.reference).digest('hex'));
    if (id === 'foundations/top_p_sampling') {
      const wrong = Object.entries(bundle.mutants).find(([name]) => name.includes('m2_extra_boundary'));
      assert(wrong);
      py.globals.set('wrong_boundary_source', wrong[1]);
      const rejected = JSON.parse(py.runPython("json.dumps(run_suite(bundle,wrong_boundary_source,'/judge'))"));
      assert(!rejected.passed && !unsupportedReport(rejected));
      const failure = rejected.tests.find(row => row.id === 'test_contract');
      assert.equal(failure.comparison.kind, 'value');
      assert.deepEqual(failure.comparison.index, [0, 0, 2]);
      assert.equal(failure.comparison.expected, 0.);
      assert(Math.abs(failure.comparison.actual - 1/3) < 1e-6);
      assert.equal(failure.comparison.rtol, 1.3e-6);
      assert.equal(failure.comparison.atol, 1e-5);
      assert(failure.message.includes('AssertionError'), 'Keep the full original framework diagnostic');
    }
    // Both adapters must keep torch.tensor ownership separate from as_tensor sharing.
    py.runPython(`
import numpy as np, torch
for dtype in (np.float32,np.float64):
    a=np.arange(12,dtype=dtype).reshape(3,4).T[:,::2]; expected=a.copy()
    x=torch.tensor(a); a[:]=99.
    np.testing.assert_array_equal(x.numpy(),expected); assert x.dtype==np.dtype(dtype)
    shared=torch.as_tensor(a); shared[0,0]=5.; assert a[0,0]==5.
`);
    if (id === 'architecture/attention') {
      const extra = 'problems/architecture/attention/tests/test_browser_multifile_regression.py';
      py.FS.writeFile('/judge/'+extra,'import pytest\n@pytest.mark.t0\ndef test_additional_file(): assert 2 + 2 == 4\n');
      py.globals.set('extra_test_file',extra);
      const multi=JSON.parse(py.runPython(`
multi={**bundle,'test_files':[*bundle['test_files'],extra_test_file],
       'tests':{**bundle['tests'],'T0':bundle['tests']['T0']+1}}
json.dumps(run_suite(multi,bundle['reference'],'/judge'))
`));
      assert(multi.passed && multi.tests.length===report.tests.length+1,'WASM must collect every test file');
    }
    if (runtime.backend === 'numpy-autograd-wasm') {
      if (id === 'architecture/attention') {
        // Run the independent NumPy-only model/checkpoint suite against the
        // exported library, including ordinary torch aliases in a real VM.
        for (const suite of ['module_lifecycle', 'optimizer_lifecycle', 'conversion_modes', 'scheduler_lifecycle', 'nn_workflows', 'module_hooks', 'graph_lifecycle', 'tensor_hooks', 'numpy_inputs', 'construction', 'promotion', 'linear_wasm', 'arithmetic', 'ieee', 'batchnorm', 'groupnorm', 'classification', 'convolution', 'layout']) {
          const testPath = `/judge/test_lightweight_${suite}.py`;
          py.FS.writeFile(testPath, fs.readFileSync(path.join(web,
            `../packages/paperleet-torch/tests/test_${suite}.py`), 'utf8'));
          const exitCode = py.runPython(`__import__('pytest').main([
            '${testPath}', '-q', '--noconftest', '-p', 'no:cacheprovider', '--import-mode=importlib'])`);
          assert.equal(Number(exitCode), 0, `Standalone ${suite} contracts must pass on WASM`);
        }
      }
      if (id === 'architecture/attention') py.runPython(`
import numpy as np, torch
from paperleet_torch import as_torch, install_as_torch
previous_torch = torch
attention_operand=torch.ones(1,3,2,dtype=torch.float32)
assert torch.nn.functional.scaled_dot_product_attention(attention_operand,attention_operand,attention_operand).dtype==torch.float32
with as_torch():
    namespace = install_as_torch()
    import torch as aliased
    from torch import nn
    import torch.nn.functional as F
    assert aliased is namespace and nn is namespace.nn and F is nn.functional
    value = aliased.tensor([[2.,3.]], requires_grad=True)
    F.linear(value, aliased.ones(1,2)).square().sum().backward()
    np.testing.assert_array_equal(value.grad.numpy(), [[10.,10.]])
import torch
assert torch is previous_torch
# Optimized basic slicing must keep advanced-index duplicate accumulation.
selected=torch.arange(12,dtype=torch.float64).reshape(3,4).requires_grad_()
selected[:,1::2].sum().backward()
np.testing.assert_array_equal(selected.grad.numpy(),[[0.,1.,0.,1.]]*3)
selected.grad=None
selected[torch.tensor([1,1,0])].sum().backward()
np.testing.assert_array_equal(selected.grad.numpy(),[[1.]*4,[2.]*4,[0.]*4])
# A delayed first gradient uses step one, regardless of other parameters.
early=torch.tensor([1.],requires_grad=True);late=torch.tensor([1.],requires_grad=True)
optimizer=torch.optim.Adam([early,late],lr=.04)
for iteration in range(3):
    early.grad=torch.tensor([.3]);optimizer.step()
    if iteration==0: first=early.numpy().copy()
early.grad=None;late.grad=torch.tensor([.3]);optimizer.step()
np.testing.assert_allclose(late.numpy(),first,atol=1e-7,rtol=1e-6)
optimizer.zero_grad(set_to_none=False);assert early.grad is None
# Finite-difference the independent shared SDPA pullback on actual WASM.
rng=np.random.default_rng(817)
sdpa_inputs=tuple(torch.tensor(rng.normal(size=shape),dtype=torch.float64,requires_grad=True)
                  for shape in ((1,2,3),(1,3,3),(1,3,2),(2,3)))
assert torch.autograd.gradcheck(lambda q,k,v,m:torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=m),sdpa_inputs)
x=torch.tensor([2.,3.],requires_grad=True)
(torch.nan_to_num(x)*x).sum().backward()
np.testing.assert_array_equal(x.grad.numpy(),[4.,6.])
for dtype in (np.float32,np.float64):
    a=np.array([2.,3.],dtype=dtype)
    x=torch.tensor(a,requires_grad=True); loss=x.square().sum(); a[0]=100.; loss.backward()
    np.testing.assert_array_equal(x.numpy(),[2.,3.]);np.testing.assert_array_equal(x.grad.numpy(),[4.,6.])
    shared=torch.as_tensor(a); shared[0]=5.; assert a[0]==5.
    t=torch.triu(torch.tensor([[1.,2.],[float('inf'),3.]],dtype=dtype))
    np.testing.assert_array_equal(t.numpy(),[[1.,2.],[0.,3.]])
    t=torch.tril(torch.tensor([[1.,float('inf')],[2.,3.]],dtype=dtype))
    np.testing.assert_array_equal(t.numpy(),[[1.,0.],[2.,3.]])
`);
      for (const [suffix, expected] of [
        ['\nimport torch\ntorch.sum(torch.ones(2,2),dim=0,out=torch.empty(2))',true],
        ['\nraise TypeError("student bug")',false],
        ['\nassert False, "BrowserUnsupported ImportError are assertion text"',false],
      ]) {
        py.globals.set('suffix',suffix);
        const failed=JSON.parse(py.runPython("json.dumps(run_suite(bundle,bundle['reference']+suffix,'/judge'))"));
        assert(!failed.passed && !failed.error);
        assert.equal(unsupportedReport(failed),expected);
      }
    }
    const result = {id, pyodide:py.version, numpy:'2.2.5', backend:runtime.backend,
      tests:report.tests.length, mutants:Object.keys(calibration.mutants_caught).length,
      packages:manifest.packages, bundle_url:entry.browser_judge.bundle_url,
      runtime_bundle_url:manifest.runtime_bundle_url, source_sha256:report.source_sha256,
      preparation_ms:Math.round(prepared-started), calibration_ms:Math.round(calibrated-prepared),
      learner_ms:Math.round(completed-calibrated), elapsed_ms:Math.round(performance.now()-started),
      budgets:entry.browser_judge.budgets,
      browser_budget_exceeded:[
        ...(calibrated-prepared > entry.browser_judge.budgets.calibration ? ['calibration'] : []),
        ...(completed-calibrated > entry.browser_judge.budgets.run ? ['run'] : []),
        ...(completed-started > entry.browser_judge.budgets.total ? ['total'] : []),
      ]};
    if (process.argv[3] === '--json') console.log(JSON.stringify(result));
    else console.log(`PASS real Pyodide ${py.version} / NumPy 2.2.5: ${id}, ${result.tests} tests, ${result.mutants} mutants, full reference/mutants/learner suite`);
  }
} catch(error) {
  console.error(String(error?.stderr || error?.message || error).slice(-12000));
  process.exitCode=1;
}
