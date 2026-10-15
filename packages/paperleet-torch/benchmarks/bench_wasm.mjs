// Real pinned Pyodide/NumPy benchmark, with a fresh process per source version.
// Dev tooling uses the repository's installed web/node_modules/pyodide runtime.
import fs from 'node:fs';
import path from 'node:path';
import {parseArgs} from 'node:util';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';

const here = path.dirname(fileURLToPath(import.meta.url));
const {values} = parseArgs({options: {
  source: {type:'string', default:path.resolve(here, '../src/paperleet_torch')},
  runtime: {type:'string', default:path.resolve(here, '../../../web/node_modules/pyodide')},
  cache: {type:'string', default:path.resolve(here, '../../../web/public/browser-runtime/pyodide-0.29.4')},
  output: {type:'string'}, repeat: {type:'string', default:'11'}, iterations: {type:'string', default:'30'},
  benchmark: {type:'string', default:'runtime'},
  dtype: {type:'string', default:'float32'}, case: {type:'string', multiple:true},
}});
const repeat = Number(values.repeat), iterations = Number(values.iterations);
if (!Number.isInteger(repeat) || !Number.isInteger(iterations) || Math.min(repeat,iterations) < 1) {
  throw new Error('repeat and iterations must be positive integers');
}
if (!['runtime','schedulers','nn','module_hooks','tensor_hooks'].includes(values.benchmark)) throw new Error('Unknown benchmark');
if (values.benchmark === 'schedulers' && (values.case || values.dtype !== 'float32')) {
  throw new Error('Scheduler suite uses fixed float32 fixtures and all cases');
}
const {loadPyodide} = await import(pathToFileURL(path.join(path.resolve(values.runtime), 'pyodide.mjs')).href);
const py = await loadPyodide({indexURL:path.resolve(values.runtime), packageCacheDir:path.resolve(values.cache),
  stdout(){}, stderr(){}});
await py.loadPackage('numpy');
py.FS.mkdirTree('/runtime/paperleet_torch');
const sourceHashes = {};
for (const name of fs.readdirSync(values.source).filter(name => name.endsWith('.py'))) {
  const raw = fs.readFileSync(path.join(values.source,name));
  sourceHashes[name] = createHash('sha256').update(raw).digest('hex');
  py.FS.writeFile('/runtime/paperleet_torch/'+name, raw.toString('utf8'));
}
py.runPython('import sys\nsys.path.insert(0,"/runtime")');
py.FS.writeFile('/runtime/bench_runtime.py',fs.readFileSync(path.join(here,`bench_${values.benchmark}.py`),'utf8'));
const options = values.benchmark === 'schedulers' ? {repeat,iterations} : {repeat,iterations,dtype:values.dtype,cases:values.case};
py.globals.set('benchmark_options_json',JSON.stringify(options));
const report = JSON.parse(py.runPython('import json, bench_runtime\njson.dumps(bench_runtime.run_benchmarks(**json.loads(benchmark_options_json)))'));
report.pyodide = py.version;
report.source = path.resolve(values.source);
report.source_hashes = sourceHashes;
const output = JSON.stringify(report,null,2)+'\n';
if (values.output) fs.writeFileSync(values.output,output);
process.stdout.write(output);
