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
const cases = ['diffusion/schedules', 'architecture/attention', 'finetuning/lora', 'optimization/adamw', 'optimization/muon_split'];
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
      py.runPython(`
import numpy as np, torch
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
