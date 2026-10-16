import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {createServer} from 'vite';
import React from 'react';
import {localizedRenderer} from './render-locale.mjs';

const vite = await createServer({server:{middlewareMode:true},appType:'custom',optimizeDeps:{noDiscovery:true,entries:[]}});
const originals = Object.fromEntries(['Worker','window','document','location','fetch'].map(key=>[key,globalThis[key]]));
try {
  const {calibrationSummary,calibrationProof,createCalibrationCache,browserMeasurements} = await vite.ssrLoadModule('/src/browserCalibration.ts');
  const {browserRunDetails}=await vite.ssrLoadModule('/src/browserRunDetails.ts');
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
  const {parseReport,parseSaved}=await vite.ssrLoadModule('/src/reports.ts');
  const {parseBrowserLearning}=await vite.ssrLoadModule('/src/browserLearningStore.ts');
  const {browserLearningView,learningKey}=await vite.ssrLoadModule('/src/browserLearning.ts');
  const {default:Details}=await vite.ssrLoadModule('/src/JudgeRuntimeDetails.tsx');
  const recorded={version:1,calibration:{reference_passed:1,mutants_caught:1,reused:false},measurements:measured};
  const report={problem:bundle.problem,runtime:'browser-numpy-wasm',source_sha256:createHash('sha256').update('historical source').digest('hex'),summary:{T0:[1,1]},tests:[{id:'test_original',layer:'T0',passed:true}],browser_run:recorded};
  const restored=parseReport(JSON.parse(JSON.stringify(report)));assert.deepEqual(restored.browser_run,recorded);
  assert.deepEqual(parseSaved({[report.problem]:{code:'newer edited draft',report:restored}})[report.problem].report.browser_run,recorded,'Guest draft recovery keeps timing with its original report, even when the draft is newer');
  const learned=parseBrowserLearning(JSON.parse(JSON.stringify({owner:'user:alice',id:report.problem,suite:'bundle-v1',version:'v1',report:restored,acceptedReport:restored})),'user:alice');
  assert.deepEqual(learned.report.browser_run,recorded);assert.equal(parseBrowserLearning(learned,'user:bob'),undefined,'Run history stays with its existing learning owner');
  const changedSuite={id:report.problem,tests:{T0:1},browser_judge:{bundle_url:'bundle-v2'}};
  assert.equal(browserLearningView('user:alice',[changedSuite],{},{[learningKey(report.problem,'bundle-v1')]:learned}).records[report.problem],undefined,'An old suite cannot attach its run details to a new suite report');
  for(const browser_run of [null,{},[],{...recorded,version:2},...[-1,0,.5,100001,'1',true].map(reference_passed=>({...recorded,calibration:{...recorded.calibration,reference_passed}})),...[-1,.5,1001,'1',true].map(mutants_caught=>({...recorded,calibration:{...recorded.calibration,mutants_caught}})),{...recorded,calibration:{...recorded.calibration,reused:1}}]){
    const parsed=parseReport({...report,browser_run});assert.equal(parsed.browser_run,undefined);assert.equal(parsed.tests[0].passed,true,'Bad optional run history cannot discard a usable result');
  }
  for(const invalid of [{...measured,total_ms:500},{...measured,learner_ms:Infinity},{...measured,reused:true}])assert.deepEqual(browserRunDetails({...recorded,measurements:invalid}),{version:1,calibration:recorded.calibration});
  const coldLoad={loading_ms:100000,calibration_ms:150000,learner_ms:60000,total_ms:310000,reused:false};assert.equal(browserMeasurements(coldLoad),undefined,'Worker measurements retain their existing bound');assert.deepEqual(browserRunDetails({...recorded,measurements:coldLoad}).measurements,coldLoad,'Persist the completed end-to-end measurement including loading before the worker deadline');assert.equal(browserRunDetails({...recorded,measurements:{...coldLoad,loading_ms:3600001,total_ms:3810001}}).measurements,undefined);
  assert.equal(parseReport({...report,runtime:'native-cpu'}).browser_run,undefined);assert.equal(parseReport({...report,source_sha256:undefined}).browser_run,undefined);
  assert.deepEqual(browserRunDetails({...recorded,source:'private source',key:'proof key',environment:'ABI',calibration:{...recorded.calibration,mutants:{secret:'details'}}}),recorded,'History serializes only bounded counts and timings');
  assert.equal(calibrationProof({...recorded,key:'key',environment:proof.environment},'key',bundle),undefined,'A persisted historical summary is not a reusable calibration proof');
  for(const locale of ['zh','en']){
    const render=await localizedRenderer(vite,locale),html=render(React.createElement(Details,{server:false,browserResult:true,calibration:restored.browser_run.calibration,measurement:restored.browser_run.measurements}));
    assert(html.includes(locale==='en'?'Timing for this result':'该结果耗时'));assert(html.replace(/<!--.*?-->/g,'').includes('0.02s'));assert(html.includes(locale==='en'?'mutants caught':'个错误实现被抓住'));
    const reused=render(React.createElement(Details,{server:false,browserResult:true,calibration:{...recorded.calibration,reused:true}}));assert(reused.includes(locale==='en'?'This result reused calibration':'该结果复用了兼容性校验'));
    const legacy=render(React.createElement(Details,{server:false,browserResult:true}));assert(!legacy.includes(locale==='en'?'Timing for this result':'该结果耗时'),'An old report without timing cannot invent measurements');
  }
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
    run.emit({type:'result',calibration:summary,measurements:{...measured,reused},report:{problem:bundle.problem,runtime:'browser-numpy-wasm',source_sha256:createHash('sha256').update(run.input.source).digest('hex'),browser_run:{...recorded,calibration:{reference_passed:999,mutants_caught:999,reused:true}},summary:{T0:[passed?1:0,1]},tests:[{id:'test_original',layer:'T0',passed}]}});
  };
  const first=await start('first learner source');assert(!first.input.cachedCalibration);finish(first,false,false);
  const firstResult=await first.result;assert.equal(firstResult.type,'result','A learner failure must still execute and preserve compatibility evidence');
  assert.deepEqual(firstResult.report.browser_run.calibration,recorded.calibration,'Persist the coordinator-checked calibration, not fields returned inside the report');assert.equal(firstResult.report.browser_run.measurements.learner_ms,5);
  assert(first.frame.removed);
  const second=await start('different learner source');
  assert.equal(second.input.source,'different learner source');assert.equal(second.input.cachedCalibration.environment,proof.environment);
  finish(second,true);const secondResult=await second.result;
  assert(secondResult.calibration.reused&&secondResult.measurements.reused);
  assert.equal(parseReport(JSON.parse(JSON.stringify(secondResult.report))).browser_run.calibration.reused,true);
  assert.notEqual(firstResult.report.source_sha256,secondResult.report.source_sha256,'Each run keeps its own captured source identity');
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
  console.log('PASS browser run history and page-local calibration: source/owner/suite-bound recovery, bounded advisory records, bilingual restored details, coordinator-owned counts, full bundle/worker identities, exact reference/mutant coverage, ABI checks, immutable evidence, fixed expiry/LRU, fresh captured source, late-message rejection, failure eviction and measured phases');
} finally {
  Object.assign(globalThis,originals);await vite.close();
}
