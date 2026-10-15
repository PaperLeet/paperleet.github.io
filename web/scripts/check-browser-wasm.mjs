// Real pinned Python/NumPy WASM, separate runtimes per bundle. No mocked interpreter.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {loadPyodide} from 'pyodide';
import {loadJudgePackages} from '../public/browser-judge/packages.mjs';
import {unsupportedReport} from '../public/browser-judge/diagnostics.mjs';

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const cases = ['diffusion/schedules', 'architecture/attention', 'finetuning/lora', 'optimization/adamw', 'optimization/muon_split', 'representation/moco', 'representation/simclr',
  'foundations/next_token_loss', 'foundations/sft_loss_mask', 'foundations/top_k_sampling',
  'foundations/top_p_sampling', 'foundations/gradient_accumulation', 'foundations/gradient_clipping'];
try {
  if (!process.argv[2]) {
    for (const id of cases) {
      const output = execFileSync(process.execPath, [fileURLToPath(import.meta.url), id], {
        cwd:web, encoding:'utf8', timeout:120_000, maxBuffer:1024*1024,
      });
      process.stdout.write(output);
    }
  } else {
    const id = process.argv[2];
    assert(cases.includes(id));
    const index = JSON.parse(fs.readFileSync(path.join(web, 'public/data/index.json')));
    const entry = index.problems.find(p => p.id === id);
    const read = relative => JSON.parse(fs.readFileSync(path.join(web, 'public/data', relative)));
    const manifest = read(entry.browser_judge.bundle_url);
    const runtime = read(manifest.runtime_bundle_url);
    const bundle = {...manifest, adapter:runtime.adapter, harness:runtime.harness, files:{...runtime.files,...manifest.files}};
    const py = await loadPyodide({indexURL:path.join(web,'node_modules/pyodide/'),
      packageCacheDir:path.join(web,'public/browser-runtime/pyodide-0.29.4/'),stdout(){},stderr(){}});
    await loadJudgePackages(py, ['pyyaml']);
    assert.equal(py.runPython("__import__('numpy').__version__"), '2.2.5');
    py.globals.set('bundle_json',JSON.stringify(bundle));
    py.runPython("import json\nbundle=json.loads(bundle_json)\nexec(compile(bundle['harness'],'browser_harness.py','exec'))\nprepare(bundle,bundle['adapter'],'/judge')");
    const calibration = JSON.parse(py.runPython("json.dumps(calibrate(bundle,'/judge'))"));
    assert.equal(calibration.compatible,true,JSON.stringify(calibration));
    assert.equal(calibration.reference_passed,Object.values(entry.tests).reduce((a,b)=>a+b,0));
    assert.equal(Object.keys(calibration.mutants_caught).length,Object.keys(bundle.mutants).length);
    const report = JSON.parse(py.runPython("json.dumps(run_suite(bundle,bundle['reference'],'/judge'))"));
    assert(report.passed && !unsupportedReport(report));
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
        for (const suite of ['module_lifecycle', 'optimizer_lifecycle', 'conversion_modes', 'scheduler_lifecycle', 'nn_workflows', 'module_hooks', 'graph_lifecycle', 'tensor_hooks', 'numpy_inputs', 'construction', 'promotion']) {
          const testPath = `/judge/test_lightweight_${suite}.py`;
          py.FS.writeFile(testPath, fs.readFileSync(path.join(web,
            `../packages/paperleet-torch/tests/test_${suite}.py`), 'utf8'));
          const exitCode = py.runPython(`__import__('pytest').main([
            '${testPath}', '-q', '--noconftest', '-p', 'no:cacheprovider', '--import-mode=importlib'])`);
          assert.equal(Number(exitCode), 0, `Standalone ${suite} contracts must pass on WASM`);
        }
      }
      py.runPython(`
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
    console.log(`PASS real Pyodide ${py.version} / NumPy 2.2.5: ${id}, full reference/mutants/learner suite${runtime.backend==='numpy-autograd-wasm'?', numerical regressions and typed fallback':''}`);
  }
} catch(error) {
  console.error(String(error?.stderr || error?.message || error).slice(-12000));
  process.exitCode=1;
}
