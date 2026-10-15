import assert from 'node:assert/strict';
import fs from 'node:fs';
import { corePackages, loadJudgePackages } from '../public/browser-judge/packages.mjs';

const lock = JSON.parse(fs.readFileSync(new URL('../node_modules/pyodide/pyodide-lock.json', import.meta.url)));
await assert.rejects(loadJudgePackages({}, ['torch']), /Unvalidated/);
await assert.rejects(loadJudgePackages({}, ['pyyaml','pyyaml']), /Unvalidated/);
for(const optional of [[], ['pyyaml']]) {
  const imports=[];
  const py={loadedPackages:{},async loadPackage(names){for(const name of names)this.loadedPackages[name]='default channel';},runPython(source){imports.push(source);}};
  await loadJudgePackages(py,optional);
  assert.deepEqual(imports,[`import numpy, pytest${optional.length?', yaml':''}`]);
}
const closure = new Set();
function include(name) {
  if (closure.has(name)) return;
  closure.add(name);
  lock.packages[name].depends.forEach(include);
}
include('numpy'); include('pytest');
assert.deepEqual([...corePackages].sort(), [...closure].sort(), 'Runtime closure must match the pinned distribution');

for (const mode of ['complete', 'partial', 'throw', 'permanent', 'bad-import']) {
  let attempts = 0, imports = 0;
  const py = {
    loadedPackages: {},
    async loadPackage(required, options) {
      attempts++;
      if (mode === 'throw' && attempts === 1) throw Error('temporary download failure');
      for (const name of required) {
        if (name === 'pluggy' && (mode === 'permanent' || (mode === 'partial' && attempts === 1))) continue;
        this.loadedPackages[name] = 'default channel';
      }
      if (!this.loadedPackages.pluggy) options.errorCallback('pluggy download failed');
    },
    runPython(source) {
      imports++;
      assert.equal(source, 'import numpy, pytest, yaml');
      if (mode === 'bad-import') throw Error('broken wheel');
    },
  };
  if (mode === 'permanent' || mode === 'bad-import') {
    await assert.rejects(loadJudgePackages(py, true), /Browser judge dependencies could not be loaded/);
    assert.equal(attempts, 2);
    if (mode === 'permanent') assert.equal(imports, 0, 'Incomplete packages must never enter the harness');
  } else {
    await loadJudgePackages(py, true);
    assert.equal(attempts, mode === 'complete' ? 1 : 2);
    assert.equal(imports, 1);
  }
}
console.log('PASS browser runtime packages: pinned dependency closure, silent partial loads, bounded recovery and import validation');
