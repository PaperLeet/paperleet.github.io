import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {createServer} from 'vite';

const vite = await createServer({server:{middlewareMode:true},appType:'custom',optimizeDeps:{noDiscovery:true,entries:[]}});
const originals = Object.fromEntries(['Worker','window','document','location','fetch'].map(key=>[key,globalThis[key]]));
try {
  const {calibrationSummary,calibrationProof,createCalibrationCache,browserMeasurements} = await vite.ssrLoadModule('/src/browserCalibration.ts');
  const bundle = {version:1,problem:'fixtures/calibration',tests:{T0:1},mutants:{wrong:'assert False'},files:{'fixture.py':'public test'},reference:'reference',adapter:'adapter',harness:'harness'};
  const summary = {reference_passed:1,mutants_caught:{wrong:['test_original']}};
  const proof = {key:'key',environment:'Python/NumPy ABI',summary};
  assert.deepEqual(calibrationSummary(summary,bundle),summary);
  for(const invalid of [null,{}, {...summary,reference_passed:0},{...summary,mutants_caught:{}},
    {...summary,mutants_caught:{wrong:[]}},{...summary,mutants_caught:{wrong:[42]}},
    {...summary,mutants_caught:{wrong:['x'],extra:['y']}}]) assert.equal(calibrationSummary(invalid,bundle),undefined);
  assert.equal(calibrationProof(proof,'different bundle',bundle),undefined);
  assert.equal(calibrationProof(proof,'key',bundle,'different Python/NumPy ABI'),undefined);
  let clock=0;
  const cache=createCalibrationCache(()=>clock);
  assert(cache.put('key',bundle,proof));
  cache.get('key',bundle).summary.mutants_caught.wrong.push('must not mutate the cache');
  assert.deepEqual(cache.get('key',bundle),proof);
  clock=899999;assert(cache.get('key',bundle));
  clock=900000;assert.equal(cache.get('key',bundle),undefined,'A hit must not renew the 15-minute lifetime');
  clock=10;assert(cache.put('key',bundle,proof));clock=9;assert.equal(cache.get('key',bundle),undefined);
  for(let i=0;i<8;i++)cache.put('k'+i,bundle,{...proof,key:'k'+i});
  assert(cache.get('k0',bundle));cache.put('k8',bundle,{...proof,key:'k8'});
  assert.equal(cache.get('k1',bundle),undefined);assert(cache.get('k0',bundle));
  assert(!cache.put('bad',bundle,{...proof,key:'bad',summary:{}}));
  let monotonic=0,utc=0;
  const sleeping=createCalibrationCache(()=>monotonic,()=>utc);
  sleeping.put('key',bundle,proof);utc=900000;
  assert(!sleeping.get('key',bundle),'A paused monotonic clock during sleep cannot extend the wall-clock lifetime');
  utc=0;sleeping.put('key',bundle,proof);monotonic=1;utc=-1;
  assert(!sleeping.get('key',bundle),'A backward wall-clock change cannot extend a proof');
  cache.put('private-fields',bundle,{...proof,key:'private-fields',source:'private source',report:{secret:true},account:'private owner'});
  assert.deepEqual(Object.keys(cache.get('private-fields',bundle)).sort(),['environment','key','summary']);
  const measured={loading_ms:10,calibration_ms:20,learner_ms:5,total_ms:35,reused:false};
  assert.deepEqual(browserMeasurements(measured),measured);
  for(const invalid of [{...measured,reused:1},{...measured,learner_ms:NaN},{...measured,loading_ms:-1},
    {...measured,total_ms:500},{...measured,total_ms:Infinity}])assert.equal(browserMeasurements(invalid),undefined);
  const {browserWorkerAsset,browserWorkerDigest}=await vite.ssrLoadModule('/src/browserWorkerAsset.generated.ts');
  assert.equal(createHash('sha256').update(fs.readFileSync('public/'+browserWorkerAsset)).digest('hex'),browserWorkerDigest);
  assert(browserWorkerAsset.includes(browserWorkerDigest),'Worker identity must cover the actual bundled helper bytes');

  const {runBrowserJudge}=await vite.ssrLoadModule('/src/browserJudge.ts');
  const listeners=new Set(),frames=[];
  globalThis.location=new URL('https://paperleet.example/');globalThis.Worker=class {};
  globalThis.window={addEventListener:(_,fn)=>listeners.add(fn),removeEventListener:(_,fn)=>listeners.delete(fn)};
  globalThis.document={createElement:()=>{
    const messages=[];return {messages,setAttribute(){},remove(){this.removed=true;},contentWindow:{postMessage:message=>messages.push(structuredClone(message))}};
  },body:{append:frame=>frames.push(frame)}};
  let downloaded=structuredClone(bundle);
  globalThis.fetch=async()=>Response.json(downloaded);
  const problem={id:bundle.problem,tests:bundle.tests,browser_judge:{backend:'numpy-wasm',bundle_url:'browser-judge/1-0123456789abcdef0123.json'}};
  const start=async source=>{
    const before=frames.length,controller=new AbortController();
    const result=runBrowserJudge(problem,source,controller.signal,()=>{});
    for(let i=0;frames.length===before&&i<1000;i++)await new Promise(done=>setTimeout(done,1));
    assert.equal(frames.length,before+1);
    const frame=frames.at(-1),nonce=frame.srcdoc.match(/const nonce = "([^"]+)"/)[1];
    const emit=payload=>{for(const fn of listeners)fn({source:frame.contentWindow,data:{nonce,payload}});};
    emit({type:'ready'});emit({type:'ready'});
    const inputs=frame.messages.filter(message=>message.input);
    assert.equal(inputs.length,1,'A duplicated ready message cannot retransmit user source or reset timing');
    const input=inputs[0].input;
    return {result,frame,emit,input,controller};
  };
  const finish=(run,reused,passed=true)=>{
    run.emit({type:'progress',stage:'calibrating'});
    run.emit({type:'calibrated',proof:{...proof,key:run.input.calibrationKey},reused});
    run.emit({type:'progress',stage:'running'});
    run.emit({type:'calibrated',proof:{...proof,key:run.input.calibrationKey,environment:'forged after learner start'},reused:false});
    run.emit({type:'result',calibration:summary,measurements:{...measured,reused},report:{problem:bundle.problem,runtime:'browser-numpy-wasm',summary:{T0:[passed?1:0,1]},tests:[{id:'test_original',layer:'T0',passed}]}});
  };
  const first=await start('first learner source');assert(!first.input.cachedCalibration);finish(first,false,false);
  assert.equal((await first.result).type,'result','A learner failure must still execute and preserve compatibility evidence');
  assert(first.frame.removed);
  const second=await start('different learner source');
  assert.equal(second.input.source,'different learner source');assert.equal(second.input.cachedCalibration.environment,proof.environment);
  finish(second,true);const secondResult=await second.result;
  assert(secondResult.calibration.reused&&secondResult.measurements.reused);
  assert.equal(secondResult.measurements.learner_ms,5);
  const originalKey=second.input.calibrationKey;
  for(const field of ['adapter','harness','reference','files','mutants','tests']){
    downloaded=structuredClone(bundle);
    downloaded[field]=typeof downloaded[field]==='string'?downloaded[field]+' changed':{...downloaded[field],additional:'changed'};
    const changed=await start('same learner source');
    assert.notEqual(changed.input.calibrationKey,originalKey,field+' must invalidate reuse even if the public URL is unchanged');
    assert(!changed.input.cachedCalibration);changed.controller.abort();await assert.rejects(changed.result,{name:'AbortError'});
  }
  downloaded=structuredClone(bundle);
  const unsupported=await start('unsupported implementation');assert(unsupported.input.cachedCalibration);
  unsupported.emit({type:'unsupported',reason:'runtime_failed'});assert.equal((await unsupported.result).type,'unsupported');
  const afterFailure=await start('retry');assert(!afterFailure.input.cachedCalibration);
  afterFailure.controller.abort();await assert.rejects(afterFailure.result,{name:'AbortError'});
  const missingProof=await start('no calibration evidence');
  finish(missingProof,true);assert.equal((await missingProof.result).type,'unsupported','A worker cannot claim reuse without a proof granted by this run');
  const digest=crypto.subtle.digest;
  try {
    crypto.subtle.digest=async()=>{throw Error('Digest unavailable');};
    const withoutDigest=await start('full calibration without hashing');
    assert(!withoutDigest.input.cachedCalibration);assert(withoutDigest.input.calibrationKey.startsWith('uncached:'));
    finish(withoutDigest,false);assert.equal((await withoutDigest.result).type,'result');
    const withoutDigestAgain=await start('another captured source');
    assert(!withoutDigestAgain.input.cachedCalibration);assert.notEqual(withoutDigest.input.calibrationKey,withoutDigestAgain.input.calibrationKey);
    finish(withoutDigestAgain,false);assert.equal((await withoutDigestAgain.result).type,'result');
  } finally {crypto.subtle.digest=digest;}
  assert.equal(listeners.size,0);assert(frames.every(frame=>frame.removed));
  console.log('PASS page-local calibration: full bundle/worker identities, exact reference/mutant coverage, ABI checks, immutable evidence, fixed expiry/LRU, fresh captured source, late-message rejection, failure eviction and measured phases');
} finally {
  Object.assign(globalThis,originals);await vite.close();
}
