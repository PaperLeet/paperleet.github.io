import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createServer } from 'vite';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
const tick = () => new Promise(resolve => setImmediate(resolve));
const signal = () => new AbortController().signal;
const clone = value => structuredClone(value);
const base = 'https://paperleet.example/preview/';
try {
  const { createBrowserBundleLoader } = await server.ssrLoadModule('/src/browserBundles.ts');
  const index = JSON.parse(fs.readFileSync('public/data/index.json', 'utf8'));
  const pilots = index.problems.filter(p => p.browser_judge);
  const attention = pilots.find(p => p.id === 'architecture/attention');
  const lora = pilots.find(p => p.id === 'finetuning/lora');
  const manifest = JSON.parse(fs.readFileSync(`public/data/${attention.browser_judge.bundle_url}`, 'utf8'));
  const runtime = JSON.parse(fs.readFileSync(`public/data/${manifest.runtime_bundle_url}`, 'utf8'));
  const requests = [];
  const files = async (url, options) => {
    requests.push({ url: String(url), options });
    assert.equal(options.credentials, 'omit');
    assert.equal(options.cache, 'force-cache');
    assert(options.signal instanceof AbortSignal);
    assert(!options.body && !options.headers, 'Public asset fetches must contain no learner code or credentials');
    assert(new URL(url).pathname.startsWith('/preview/data/browser-judge/'));
    return new Response(fs.readFileSync(`public${new URL(url).pathname.slice('/preview'.length)}`));
  };
  const load = createBrowserBundleLoader(files);
  const bundles = await Promise.all(pilots.map(p => load(p, base, signal())));
  assert.equal(requests.length, pilots.length + 2, 'One shared download per backend, even with concurrent requests');
  for (const [i, bundle] of bundles.entries()) {
    assert.equal(bundle.problem, pilots[i].id);
    assert.equal(bundle.harness, fs.readFileSync('../paperleet/browser/harness.py', 'utf8'));
    const adapter = pilots[i].id === 'diffusion/schedules' ? 'schedules_numpy.py' : 'numpy_autograd.py';
    assert.equal(bundle.adapter, fs.readFileSync(`../paperleet/browser/${adapter}`, 'utf8'));
    assert.equal(bundle.files['paperleet/common/asserts.py'], fs.readFileSync('../paperleet/common/asserts.py', 'utf8'));
    assert.equal(bundle.files['paperleet/common/comparisons.py'], fs.readFileSync('../paperleet/common/comparisons.py', 'utf8'));
    for (const [name, source] of Object.entries(bundle.files)) {
      if (/\/tests\/test_[^/]+\.py$/.test(name)) assert.equal(source, fs.readFileSync(`../${name}`, 'utf8'));
      if (name.endsWith('/problem.yaml')) assert.equal(source, fs.readFileSync(`../${name}`, 'utf8'));
    }
  }
  const first = await load(attention, base, signal());
  first.files['paperleet/common/asserts.py'] = 'changed local copy';
  first.adapter = 'changed local adapter';
  const next = await load(lora, base, signal());
  assert.equal(next.files['paperleet/common/asserts.py'], runtime.files['paperleet/common/asserts.py']);
  assert.equal(next.adapter, runtime.adapter);
  assert.equal(requests.filter(r => r.url.includes('/runtime-')).length, 2);

  const legacy = { version: 1, problem: attention.id, adapter: 'legacy', harness: 'legacy', files: {} };
  const legacyPaths = [];
  const legacyLoad = createBrowserBundleLoader(async url => { legacyPaths.push(String(url)); return Response.json(legacy); });
  assert.deepEqual(await legacyLoad(attention, base, signal()), legacy);
  assert.equal(legacyPaths.length, 1);

  async function invalidProblem(value, target = attention) {
    let reads = 0;
    const reject = createBrowserBundleLoader(async () => { reads++; return Response.json(value); });
    await assert.rejects(reject(target, base, signal()));
    assert(reads <= 1, 'Bad manifests cannot fetch arbitrary secondary resources');
  }
  for (const value of [null, [], { ...manifest, version: 4 }, { ...manifest, version:'3' }, { ...manifest, problem: lora.id },
    { ...manifest, files: { ...manifest.files, 'paperleet/common/asserts.py': 'override' } },
    { ...manifest, files: { ...manifest.files, 'paperleet_torch/_tensor.py': 'override' } },
    { ...manifest, files: { '../escape.py': '' } }, { ...manifest, files: [] },
    { ...manifest, reference: 42 }, { ...manifest, mutants: { bad: 42 } }, { ...manifest, test_file: 'missing.py' },
    { ...manifest, files:{...manifest.files,'../problem.yaml':'unsafe'} },
    { ...manifest, files:{...manifest.files,'settings.yaml':'unapproved resource'} }])
    await invalidProblem(value);
  for(const change of [
    {test_files:[]}, {test_files:[manifest.test_file,manifest.test_file]}, {test_files:['missing.py']},
    {test_files:undefined}, {test_files:['../escape.py']}, {packages:['torch']}, {packages:['pyyaml','pyyaml']},
    {packages:undefined}, {budgets:undefined}, {budgets:{calibration:1,run:1,total:300001}},
    {files:{...manifest.files,'problems/architecture/attention/tests/test_extra.py':''}},
  ]) await invalidProblem({...manifest,...change});
  // v2 clients had only test_file and inferred optional packages. Keep those assets usable.
  const version2={...manifest,version:2};delete version2.test_files;delete version2.packages;delete version2.budgets;
  const oldLoad=createBrowserBundleLoader(async url=>Response.json(String(url).includes('/runtime-')?runtime:version2));
  assert.equal((await oldLoad(attention,base,signal())).version,2);
  for (const path of ['https://other.example/runtime.json', '/api/private', '../browser-judge/runtime-' + 'a'.repeat(20) + '.json',
    'browser-judge/runtime-' + 'a'.repeat(20) + '.json?token=private', 'browser-judge/runtime-%2e%2e.json']) {
    await invalidProblem({ ...manifest, runtime_bundle_url: path });
    await invalidProblem(manifest, { ...attention, browser_judge: { ...attention.browser_judge, bundle_url: path } });
  }
  for (const value of [{ ...runtime, version: 2 }, { ...runtime, backend: 'unknown' }, { ...runtime, files: {} },
    { ...runtime, adapter: null }, { ...runtime, files: { ...runtime.files, 'extra.py': 'override' } },
    { ...runtime, files: { ...runtime.files, 'paperleet_torch/../escape.py': 'override' } },
    { ...runtime, files: { ...runtime.files, 'paperleet/common/asserts.py': 12 } }]) {
    const reject = createBrowserBundleLoader(async url => Response.json(String(url).includes('/runtime-') ? value : manifest));
    await assert.rejects(reject(attention, base, signal()));
  }
  const mismatch = createBrowserBundleLoader(async url => Response.json(String(url).includes('/runtime-') ? runtime : manifest));
  await assert.rejects(mismatch({ ...attention, browser_judge: { ...attention.browser_judge, backend: 'numpy-wasm' } }, base, signal()), /backend mismatch/);

  let tries = 0;
  const retry = createBrowserBundleLoader(async url => {
    if (!String(url).includes('/runtime-')) return Response.json(manifest);
    if (++tries === 1) return new Response('', { status: 503 });
    return Response.json(runtime);
  });
  await assert.rejects(retry(attention, base, signal()));
  assert.equal((await retry(attention, base, signal())).adapter, runtime.adapter);
  assert.equal(tries, 2, 'Failed shared downloads must not poison later retries');

  let release, sharedSignal, sharedReads = 0;
  const paused = createBrowserBundleLoader(async (url, options) => {
    if (!String(url).includes('/runtime-')) return Response.json(manifest);
    sharedReads++; sharedSignal = options.signal;
    return new Promise(resolve => { release = () => resolve(Response.json(runtime)); });
  });
  const stopped = new AbortController();
  const waiting = paused(attention, base, stopped.signal);
  const rejected = assert.rejects(waiting, { name: 'AbortError' });
  await tick();
  const continuing = paused(attention, base, signal());
  await tick();
  stopped.abort();
  await rejected;
  assert.equal(sharedSignal.aborted, false, 'One caller cannot cancel another caller’s public runtime download');
  release();
  assert.equal((await continuing).adapter, runtime.adapter);
  assert.equal(sharedReads, 1);
  const alreadyStopped = new AbortController(); alreadyStopped.abort();
  await assert.rejects(paused(attention, base, alreadyStopped.signal), { name: 'AbortError' });

  let releaseManifest, afterAbortReads = 0;
  const delayedManifest = createBrowserBundleLoader(async () => {
    afterAbortReads++;
    return new Promise(resolve => { releaseManifest = () => resolve(Response.json(manifest)); });
  });
  const beforeRuntime = new AbortController();
  const cancelledManifest = delayedManifest(attention, base, beforeRuntime.signal);
  beforeRuntime.abort(); releaseManifest();
  await assert.rejects(cancelledManifest, { name: 'AbortError' });
  assert.equal(afterAbortReads, 1, 'Cancellation during the manifest must prevent a secondary runtime download');

  const paths = [];
  const bounded = createBrowserBundleLoader(async url => {
    paths.push(String(url));
    const suffix = new URL(url).pathname.match(/\/([0-9]+)-[a-f0-9]{20}\.json$/)?.[1];
    const m = { ...manifest, runtime_bundle_url: `browser-judge/runtime-${String(suffix).padStart(20, '0')}.json` };
    return Response.json(String(url).includes('/runtime-') ? runtime : m);
  });
  const target = number => ({ ...attention, browser_judge: { ...attention.browser_judge, bundle_url: `browser-judge/${number}-${'a'.repeat(20)}.json` } });
  for (let number = 1; number <= 5; number++) await bounded(target(number), base, signal());
  await bounded(target(1), base, signal());
  assert.equal(paths.filter(path => path.includes('/runtime-')).length, 6, 'The immutable public cache must remain bounded');
  await bounded(target(1), 'https://paperleet.example/another-preview/', signal());
  assert.equal(paths.filter(path => path.includes('/runtime-')).length, 7, 'Different deployment bases must have isolated cache keys');
  console.log(`PASS browser bundles: all ${pilots.length} source-faithful exports, two shared downloads, legacy compatibility, bounded cache, identity/path/backend validation, retry and independent cancellation`);
} finally {
  await server.close();
}
