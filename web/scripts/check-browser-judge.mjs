import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createServer } from 'vite';
import { calibrationDiagnostic } from '../public/browser-judge/diagnostics.mjs';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
const originals = Object.fromEntries(['Worker', 'window', 'document', 'location', 'fetch'].map((key) => [key, globalThis[key]]));
try {
  const diagnostic = calibrationDiagnostic({reason:'reference_failed',reference:{tests:[
    {id:'wrong_number',passed:false,message:'expected 1; got 2'},
    {id:'passed',passed:true,message:'must not appear'},
  ],artifacts:{private_or_large_payload:'must not appear'}}});
  assert(diagnostic.includes('wrong_number') && diagnostic.includes('expected 1; got 2'));
  assert(!diagnostic.includes('must not appear'));
  assert(calibrationDiagnostic({reason:'mutant_not_caught',mutant:'wrong'}).includes('mutant: wrong'));
  assert(calibrationDiagnostic({reason:'failed',reference:{error:'x'.repeat(10000),tests:Array.from({length:10},()=>({passed:false,message:'y'.repeat(10000)}))}}).length <= 4000);
  const { runBrowserJudge } = await server.ssrLoadModule('/src/browserJudge.ts');
  const { parseReport } = await server.ssrLoadModule('/src/reports.ts');
  const listeners = new Set();
  const frames = [];
  const requests = [];
  globalThis.location = new URL('https://paperleet.example/');
  globalThis.Worker = class {};
  globalThis.window = {
    addEventListener(name, fn) { assert.equal(name, 'message'); listeners.add(fn); },
    removeEventListener(name, fn) { listeners.delete(fn); },
  };
  const frame = () => ({
    attrs: {}, hidden: false, removed: false, messages: [],
    setAttribute(name, value) { this.attrs[name] = value; },
    remove() { this.removed = true; },
    contentWindow: { postMessage() {} },
  });
  globalThis.document = {
    createElement(tag) { assert.equal(tag, 'iframe'); return frame(); },
    body: { append(value) { frames.push(value); } },
  };
  const problem = { id: 'diffusion/schedules', browser_judge: { backend: 'numpy-wasm', bundle_url: 'browser-judge/6-0123456789abcdef0123.json' } };
  globalThis.fetch = async (url, options) => {
    requests.push(String(url));
    assert.equal(options.credentials, 'omit');
    return Response.json({ version: 1, problem: problem.id });
  };
  const stages = [];
  const pending = runBrowserJudge(problem, 'frozen learner source', new AbortController().signal, (stage) => stages.push(stage));
  await new Promise((done) => setImmediate(done));
  const sandbox = frames.at(-1);
  assert.equal(sandbox.attrs.sandbox, 'allow-scripts', 'Learner code must never get same-origin privileges');
  assert.match(sandbox.srcdoc, /connect-src https:\/\/paperleet.example\/browser-runtime\//);
  assert(!sandbox.srcdoc.includes('/api/'));
  const nonce = sandbox.srcdoc.match(/const nonce = "([^"]+)"/)[1];
  const emit = (payload, source = sandbox.contentWindow, token = nonce) => {
    for (const fn of listeners) fn({ source, data: { nonce: token, payload } });
  };
  emit({ type: 'unsupported', reason: 'injected' }, {});
  emit({ type: 'unsupported', reason: 'injected' }, sandbox.contentWindow, 'wrong-token');
  assert(!sandbox.removed);
  emit({ type: 'progress', stage: 'running' });
  emit({ type: 'progress', stage: 'loading' });
  assert.deepEqual(stages, ['running'], 'Stale progress cannot extend learner execution');
  const report = { problem: problem.id, runtime: 'browser-numpy-wasm', tests: [{ id: 'failed', layer: 'T0', passed: false }], summary: { T0: [0, 1] } };
  emit({ type: 'result', report, calibration: { reference_passed: 8, mutants_caught: { wrong: ['failed'] } } });
  assert.equal((await pending).type, 'result', 'A learner assertion failure is feedback, not a compatibility fallback');
  assert.equal(listeners.size, 0);
  assert(sandbox.removed);
  assert.equal(parseReport(report).runtime, 'browser-numpy-wasm');
  assert.deepEqual(requests, ['https://paperleet.example/data/browser-judge/6-0123456789abcdef0123.json']);

  const controller = new AbortController();
  const stopped = runBrowserJudge(problem, 'while True: pass', controller.signal, () => {});
  await new Promise((done) => setImmediate(done));
  controller.abort();
  await assert.rejects(stopped, { name: 'AbortError' });
  assert(frames.at(-1).removed);
  assert.equal(listeners.size, 0);

  const nativeSetTimeout = globalThis.setTimeout;
  const nativeClearTimeout = globalThis.clearTimeout;
  const deadlines = new Map();
  try {
    globalThis.setTimeout = (fn, ms) => { const key = {}; deadlines.set(key, { fn, ms }); return key; };
    globalThis.clearTimeout = (key) => deadlines.delete(key);
    const timed = runBrowserJudge(problem, 'while True: pass', new AbortController().signal, () => {});
    await new Promise((done) => setImmediate(done));
    const timedFrame = frames.at(-1);
    const timedNonce = timedFrame.srcdoc.match(/const nonce = "([^"]+)"/)[1];
    for (const fn of listeners) fn({ source: timedFrame.contentWindow, data: { nonce: timedNonce, payload: { type: 'progress', stage: 'running' } } });
    [...deadlines.values()].find(({ ms }) => ms === 30_000).fn();
    assert.equal((await timed).reason, 'timeout');
    assert(timedFrame.removed);
    assert.equal(deadlines.size, 0);
    assert.equal(listeners.size, 0);
  } finally {
    globalThis.setTimeout = nativeSetTimeout;
    globalThis.clearTimeout = nativeClearTimeout;
  }

  for (const id of ['architecture/attention', 'finetuning/lora', 'diffusion/param_conversion', 'architecture/layer_norm', 'architecture/positional_encoding', 'rl/dpo', 'architecture/multihead_attention', 'architecture/positionwise_ffn', 'optimization/adamw', 'rl/gae', 'rl/ppo_clip', 'rl/grpo', 'rl/dr_grpo', 'rl/dapo', 'rl/gspo', 'representation/clip', 'architecture/situ_glu', 'finetuning/qlora', 'architecture/block_attention_residual', 'architecture/differential_attention', 'representation/vicreg', 'representation/rotation_trick', 'world_models/lewm_prediction', 'world_models/lewm_planning', 'world_models/causal_jepa']) {
    const wasm = runBrowserJudge({ ...problem, id, browser_judge: { ...problem.browser_judge, backend: 'numpy-autograd-wasm' } }, 'private source', new AbortController().signal, () => {});
    // The mock response intentionally has the wrong identity, so it must never
    // be executed; All autodiff modules use the same isolated CPU/WASM path.
    assert.equal((await wasm).reason, 'runtime_failed');
  }
  const index = JSON.parse(fs.readFileSync('public/data/index.json', 'utf8'));
  assert.equal(index.problems.filter((entry) => entry.browser_judge).length, 26);

  const frameCount = frames.length;
  for (const fail of [async () => { throw new TypeError('offline'); }, async () => new Response('', { status: 503 }), async () => new Response('invalid JSON')]) {
    globalThis.fetch = fail;
    assert.equal((await runBrowserJudge(problem, 'private source', new AbortController().signal, () => {})).reason, 'runtime_failed');
    assert.equal(frames.length, frameCount, 'Failed public downloads must not create a learner sandbox');
  }
  const attention = index.problems.find(p => p.id === 'architecture/attention');
  let releaseRuntime;
  globalThis.fetch = async url => {
    if (String(url).includes('/runtime-')) return new Promise(resolve => { releaseRuntime = resolve; });
    return new Response(fs.readFileSync(`public${new URL(url).pathname}`));
  };
  const beforeWorker = new AbortController();
  const cancelledDownload = runBrowserJudge(attention, 'never execute', beforeWorker.signal, () => {});
  const cancelled = assert.rejects(cancelledDownload, { name: 'AbortError' });
  await new Promise(done => setImmediate(done));
  beforeWorker.abort();
  await cancelled;
  assert.equal(frames.length, frameCount, 'Stop while downloading shared assets must never start a learner worker');
  releaseRuntime(new Response('', { status: 503 }));
  await new Promise(done => setImmediate(done));
  console.log('PASS browser judge: opaque isolation, restricted network, nonce/source validation, report parsing, assertion failures, abort cleanup and all twenty-six CPU/WASM routes');
} finally {
  Object.assign(globalThis, originals);
  await server.close();
}
