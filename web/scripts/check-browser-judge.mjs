import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createServer } from 'vite';
import { calibrationDiagnostic, runtimeDiagnostic } from '../public/browser-judge/diagnostics.mjs';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
const originals = Object.fromEntries(['Worker', 'window', 'document', 'location', 'fetch'].map((key) => [key, globalThis[key]]));
try {
  const runtimeError = runtimeDiagnostic('Traceback header\n' + 'framework frame\n'.repeat(500) + 'ImportError: precise root cause');
  assert(runtimeError.startsWith('Traceback header') && runtimeError.endsWith('ImportError: precise root cause') && runtimeError.length <= 4000);
  assert.equal(runtimeDiagnostic(new Error('short')), 'Error: short');
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
  const problem = { id: 'diffusion/schedules', tests: {T0:1,T1:1,T2:6,T3:0}, browser_judge: { backend: 'numpy-wasm', bundle_url: 'browser-judge/6-0123456789abcdef0123.json' } };
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
  assert.deepEqual(stages, ['loading', 'running'], 'Stale progress cannot extend learner execution');
  const report = { problem: problem.id, runtime: 'browser-numpy-wasm', tests: [{ id: 'failed', layer: 'T0', passed: false }], summary: { T0: [0, 1] } };
  emit({ type: 'result', report, calibration: { reference_passed: 8, mutants_caught: { wrong: ['failed'] } } });
  assert.equal((await pending).type, 'result', 'A learner assertion failure is feedback, not a compatibility fallback');
  assert.equal(listeners.size, 0);
  assert(sandbox.removed);
  assert.equal(parseReport(report).runtime, 'browser-numpy-wasm');
  assert.deepEqual(requests, ['https://paperleet.example/data/browser-judge/6-0123456789abcdef0123.json']);

  for(const valid of [true,false]){
    const source='bad = (',syntaxStages=[];
    const syntaxRun=runBrowserJudge(problem,source,new AbortController().signal,stage=>syntaxStages.push(stage));
    await new Promise(done=>setImmediate(done));
    const currentFrame=frames.at(-1),token=currentFrame.srcdoc.match(/const nonce = "([^"]+)"/)[1];
    const diagnostic={kind:'SyntaxError',message:'unclosed parenthesis',line:1,column:valid?7:-1};
    for(const fn of listeners)fn({source:currentFrame.contentWindow,data:{nonce:token,payload:{type:'syntax_error',diagnostic,source:'forged source',passed:true}}});
    const outcome=await syntaxRun;
    assert.equal(outcome.type,valid?'syntax_error':'unsupported');
    if(valid){assert.equal(outcome.failure.source,source);assert.deepEqual(outcome.failure.diagnostic,diagnostic);assert(!outcome.report);}
    assert.deepEqual(syntaxStages,['loading']);
    assert(currentFrame.removed&&listeners.size===0);
  }

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
    const runningDeadline = [...deadlines.entries()].find(([, value]) => value.ms === 30_000)[0];
    for (let completed=0; completed<=8; completed++) {
      for (const fn of listeners) fn({ source: timedFrame.contentWindow, data: { nonce: timedNonce, payload: { type: 'progress', stage: 'running', step: 'tests', completed, total:8 } } });
      assert(deadlines.has(runningDeadline), 'Detailed progress must never renew the learner deadline');
      assert(!timedFrame.removed, 'Finishing test counters is not a result message');
    }
    [...deadlines.values()].find(({ ms }) => ms === 30_000).fn();
    assert.equal((await timed).reason, 'timeout');
    assert(timedFrame.removed);
    assert.equal(deadlines.size, 0);
    assert.equal(listeners.size, 0);

    const { browserBudget } = await server.ssrLoadModule('/src/browserBudgets.ts');
    assert.deepEqual(browserBudget({id:'unknown/custom'}),{calibration:90_000,run:30_000,total:120_000});
    const flowProblem={...problem,id:'diffusion/flow_matching'};
    globalThis.fetch=async()=>Response.json({version:1,problem:flowProblem.id});
    const long=runBrowserJudge(flowProblem,'full original training',new AbortController().signal,()=>{});
    await new Promise(done=>setImmediate(done));
    const longFrame=frames.at(-1),longNonce=longFrame.srcdoc.match(/const nonce = "([^"]+)"/)[1];
    const sendLong=payload=>{for(const fn of listeners)fn({source:longFrame.contentWindow,data:{nonce:longNonce,payload}});};
    const hard=[...deadlines.entries()].find(([,v])=>v.ms===300_000);
    assert(hard,'Heavy training still has an absolute five-minute cap');
    sendLong({type:'progress',stage:'calibrating',step:'reference',completed:0,total:8});
    const calibration=[...deadlines.entries()].find(([,v])=>v.ms===240_000)[0];
    for(let completed=1;completed<=8;completed++){
      sendLong({type:'progress',stage:'calibrating',step:'reference',completed,total:8});
      assert(deadlines.has(calibration)&&deadlines.has(hard[0]),'Calibration progress cannot extend either cap');
    }
    sendLong({type:'progress',stage:'running'});
    assert([...deadlines.values()].some(v=>v.ms===60_000));
    hard[1].fn();
    assert.equal((await long).reason,'timeout');
    assert(longFrame.removed&&deadlines.size===0&&listeners.size===0);
    globalThis.fetch=async()=>Response.json({version:1,problem:problem.id});
  } finally {
    globalThis.setTimeout = nativeSetTimeout;
    globalThis.clearTimeout = nativeClearTimeout;
  }

  for (const id of ['architecture/attention', 'finetuning/lora', 'diffusion/param_conversion', 'architecture/layer_norm', 'architecture/positional_encoding', 'rl/dpo', 'architecture/multihead_attention', 'architecture/positionwise_ffn', 'optimization/adamw', 'rl/gae', 'rl/ppo_clip', 'rl/grpo', 'rl/dr_grpo', 'rl/dapo', 'rl/gspo', 'representation/clip', 'architecture/situ_glu', 'finetuning/qlora', 'architecture/block_attention_residual', 'architecture/differential_attention', 'representation/vicreg', 'representation/rotation_trick', 'world_models/lewm_prediction', 'world_models/lewm_planning', 'world_models/causal_jepa', 'representation/repa', 'representation/ijepa', 'finetuning/distillm2', 'finetuning/opsd_alignment', 'finetuning/opsd_clipping', 'rl/sdpo_feedback', 'rl/sdpo_topk', 'optimization/muon_split', 'finetuning/svdquant', 'architecture/flash_attention', 'architecture/kimi_kda', 'rl/glm5_offpolicy', 'rl/opd', 'diffusion/consistency', 'diffusion/progressive_distillation', 'diffusion/ddpm_sampling', 'diffusion/ddim', 'diffusion/flow_matching']) {
    const wasm = runBrowserJudge({ ...problem, id, browser_judge: { ...problem.browser_judge, backend: 'numpy-autograd-wasm' } }, 'private source', new AbortController().signal, () => {});
    // The mock response intentionally has the wrong identity, so it must never
    // be executed; All autodiff modules use the same isolated CPU/WASM path.
    assert.equal((await wasm).reason, 'runtime_failed');
  }
  const index = JSON.parse(fs.readFileSync('public/data/index.json', 'utf8'));
  assert.equal(index.problems.filter((entry) => entry.browser_judge).length, 44);

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
  console.log('PASS browser judge: opaque isolation, restricted network, nonce/source validation, report parsing, assertion failures, abort cleanup and all forty-four CPU/WASM routes');
} finally {
  Object.assign(globalThis, originals);
  await server.close();
}
