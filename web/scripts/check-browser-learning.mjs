import assert from 'node:assert/strict';
import fs from 'node:fs';
import React from 'react';
import {renderToString} from 'react-dom/server';
import 'fake-indexeddb/auto';

export async function checkBrowserLearning(server) {
  const {mergeBrowserLearning,browserLearningView,learningKey}=await server.ssrLoadModule('/src/browserLearning.ts');
  const {loadBrowserLearning,saveBrowserLearning,dismissBrowserLearning,parseBrowserLearning}=await server.ssrLoadModule('/src/browserLearningStore.ts');
  const {status,isPassed}=await server.ssrLoadModule('/src/types.ts');
  const {parseSaved}=await server.ssrLoadModule('/src/reports.ts');
  const {achievementProgress}=await server.ssrLoadModule('/src/achievementProgress.ts');
  const {nextPractice}=await server.ssrLoadModule('/src/practice-recommendation.ts');
  const {BrowserLearningNotice}=await server.ssrLoadModule('/src/BrowserLearningNotice.tsx');
  const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
  const {problems,papers}=JSON.parse(fs.readFileSync('public/data/catalog.json','utf8'));
  const attention=problems.find(p=>p.id==='architecture/attention'),owner='anonymous';
  const key=learningKey(attention.id,attention.browser_judge.bundle_url);
  const report={problem:attention.id,runtime:'browser-numpy-wasm',source_sha256:'a'.repeat(64),
    tests:Object.entries(attention.tests).flatMap(([layer,count])=>Array.from({length:count},(_,i)=>({id:`${layer}-${i}`,layer,passed:true}))),
    summary:Object.fromEntries(Object.entries(attention.tests).map(([layer,count])=>[layer,[count,count]]))};
  const failed={...report,tests:report.tests.map((row,i)=>({...row,passed:i!==0})),summary:{...report.summary,T0:[0,1]}};
  assert(!mergeBrowserLearning(undefined,owner,attention,{...report,tests:report.tests.slice(0,1),summary:{T0:[1,1]}},'subset').acceptedReport,'Partial suites never unlock learning completion');
  await new Promise((resolve,reject)=>{
    const request=indexedDB.open('paperleet.learning.v1',1);
    request.onupgradeneeded=()=>request.result.createObjectStore('results',{keyPath:['owner','id']});
    request.onerror=()=>reject(request.error);
    request.onsuccess=()=>{
      const db=request.result,tx=db.transaction('results','readwrite');
      tx.objectStore('results').put(mergeBrowserLearning(undefined,owner,attention,report,'legacy-pass'));
      tx.oncomplete=()=>{db.close();resolve();};tx.onabort=()=>{db.close();reject(tx.error);};
    };
  });
  assert(isPassed(attention,(await loadBrowserLearning(owner))[key].acceptedReport),'Schema upgrade retains older browser learning results');
  await saveBrowserLearning(owner,attention,report,'passed');
  await saveBrowserLearning(owner,attention,failed,'failed');
  let records=await loadBrowserLearning(owner),row=records[key];
  assert(!isPassed(attention,row.report)&&isPassed(attention,row.acceptedReport),'A later failure retains a complete earlier browser pass');
  const nativeFailed={...failed,runtime:'native-torch'};
  const saved={[attention.id]:{code:'# newer private code',updated:'draft-token',report:nativeFailed}};
  const before=structuredClone(saved);
  let projection=browserLearningView(owner,problems,saved,records);
  assert.equal(status(attention,saved[attention.id]),'started');
  assert.equal(status(attention,projection.saved[attention.id]),'done');
  assert.equal(projection.localPasses,1);
  assert.equal(projection.saved[attention.id].code,saved[attention.id].code);
  assert.equal(projection.saved[attention.id].updated,'draft-token');
  assert.deepEqual(saved,before,'Learning projections must never change account/source records');
  const transformer=papers.find(p=>p.id==='transformer');
  assert.equal(nextPractice(attention,transformer.module_ids,problems,projection.saved)?.id,transformer.module_ids[1]);
  const serverAccepted={[attention.id]:{...saved[attention.id],acceptedReport:{...report,runtime:'native-torch'}}};
  assert.equal(browserLearningView(owner,problems,serverAccepted,records).localPasses,0,'Native acceptance takes precedence over local completion');
  assert.equal(browserLearningView('user:bob',problems,saved,records).localPasses,0,'A projection cannot use another owner’s cache');
  const changed={...attention,browser_judge:{...attention.browser_judge,bundle_url:'browser-judge/changed-suite.json'}};
  const historical=browserLearningView(owner,[changed],saved,records);
  assert.equal(historical.localPasses,1,'Runtime bundle changes must not erase completed learning');
  assert.equal(status(changed,historical.saved[attention.id]),'done');
  assert.equal(historical.records[attention.id],undefined,'An older bundle report must not appear as a current test result');
  assert.equal(historical.saved[attention.id].report,nativeFailed,'Preserve the account report without substituting older browser evidence');
  assert.equal(historical.saved[attention.id].code,saved[attention.id].code);
  assert.equal(historical.saved[attention.id].updated,'draft-token');
  const expanded={...changed,tests:{...changed.tests,T0:changed.tests.T0+1}};
  const expandedHistory=browserLearningView(owner,[expanded],saved,records);
  assert.equal(status(expanded,expandedHistory.saved[attention.id]),'done','A past completion remains a learning milestone after tests are expanded');
  assert(!isPassed(expanded,report),'Historical mastery must not relax current-suite test coverage');
  assert.equal(achievementProgress([expanded],papers,expandedHistory.saved).solved,1,'Local learning milestones agree with the library');
  assert.equal(status(expanded,parseSaved(JSON.parse(JSON.stringify(expandedHistory.saved)))[attention.id]),'started','Projection-only history must not become imported or account evidence');
  assert.equal(status(expanded,parseSaved({[attention.id]:{...saved[attention.id],browserLearningPassed:true}})[attention.id]),'started','JSON flags cannot manufacture historical mastery');
  const currentFailure=mergeBrowserLearning(undefined,owner,changed,failed,'new-suite-failure');
  const failedAgain=browserLearningView(owner,[changed],saved,{...records,[learningKey(changed.id,changed.browser_judge.bundle_url)]:currentFailure});
  assert.equal(status(changed,failedAgain.saved[attention.id]),'done','A failure under the current runtime cannot erase a prior completed module');
  assert.equal(failedAgain.records[attention.id].report,currentFailure.report,'The current suite still owns the displayed browser report');
  assert.equal(failedAgain.localPasses,1,'Multiple suite records count a module only once');
  for(const invalid of [
    {...row,owner:'user:other'},
    {...row,id:'another/module'},
    {...row,acceptedReport:undefined,report},
    {...row,acceptedReport:{...report,runtime:'native-torch'}},
    {...row,acceptedReport:{...report,summary:{T0:[1,1]}}},
    {...row,acceptedReport:{...report,error:'run failed'}},
    {...row,acceptedReport:{...report,mutation:{}}},
    {...row,acceptedReport:failed},
  ])assert.equal(browserLearningView(owner,[expanded],saved,{[key]:invalid}).localPasses,0,'Historical learning needs a valid accepted browser report for this owner and problem');
  assert.equal(browserLearningView(owner,[{...attention,browser_judge:undefined}],saved,records).localPasses,0);
  assert(!parseBrowserLearning({...row,owner:'user:bob'},owner));
  assert(!parseBrowserLearning('{broken',owner));
  assert(!parseBrowserLearning({...row,report:{bad:true},acceptedReport:{...report,runtime:'server'}},owner).acceptedReport);
  assert(!mergeBrowserLearning(row,owner,changed,failed,'new-suite').acceptedReport);
  for(const bad of [{...report,problem:'rl/ppo_clip'},{...report,runtime:'server'}])await assert.rejects(saveBrowserLearning(owner,attention,bad,'invalid'));
  await assert.rejects(loadBrowserLearning('unresolved'));
  await assert.rejects(saveBrowserLearning('user:',attention,report,'invalid-owner'));
  await saveBrowserLearning('user:alice',attention,report,'alice-only');
  assert.equal(Object.keys(await loadBrowserLearning('user:bob')).length,0);
  assert.equal((await loadBrowserLearning(owner))[key].version,'failed');
  await Promise.all(Array.from({length:12},(_,i)=>{const p={...attention,id:`learning/${i}`};return saveBrowserLearning(owner,p,{...report,problem:p.id},`module-${i}`);}));
  assert.equal(Object.keys(await loadBrowserLearning(owner)).length,13,'Concurrent module results must not replace the whole learning record map');
  const stale=(await loadBrowserLearning(owner))[key];
  await saveBrowserLearning(owner,attention,report,'newer-browser');
  await dismissBrowserLearning(owner,attention.id,attention.browser_judge.bundle_url,stale.version,'late-native');
  assert.equal((await loadBrowserLearning(owner))[key].version,'newer-browser','A late native dismissal cannot hide an unseen newer browser run');
  await dismissBrowserLearning(owner,attention.id,attention.browser_judge.bundle_url,'newer-browser','native-finished');
  row=(await loadBrowserLearning(owner))[key];
  assert(!row.report&&isPassed(attention,row.acceptedReport),'Native results supersede the cached latest browser report without erasing learning history');
  const put=IDBObjectStore.prototype.put;
  try{
    IDBObjectStore.prototype.put=function(...args){if(this.name==='results')throw new DOMException('Storage full','QuotaExceededError');return put.apply(this,args);};
    await assert.rejects(saveBrowserLearning(owner,attention,failed,'must-roll-back'),{name:'QuotaExceededError'});
  }finally{IDBObjectStore.prototype.put=put;}
  assert.equal((await loadBrowserLearning(owner))[key].version,'native-finished');
  await saveBrowserLearning(owner,changed,report,'new-suite-pass');
  await saveBrowserLearning(owner,attention,failed,'older-tab-late');
  records=await loadBrowserLearning(owner);
  assert.equal(browserLearningView(owner,[changed],saved,records).localPasses,1,'An older curriculum tab cannot erase a pass under the current suite');
  assert.equal(records[learningKey(attention.id,changed.browser_judge.bundle_url)].version,'new-suite-pass');
  const originalStorage=globalThis.localStorage;
  try{
    for(const english of [false,true])for(const count of [1,2]){
      globalThis.localStorage={getItem:()=>english?'en':'zh'};
      const html=renderToString(React.createElement(LocaleProvider,null,React.createElement(BrowserLearningNotice,{count,failed:true,saving:false,onRetry(){}})));
      assert(html.includes(english?'account results and statistics still require server verification':'账号成绩与统计仍需服务端确认'));
      assert(html.includes(english?'Retry saving progress':'重试保存学习记录')&&html.includes('role="alert"'));
      if(english)assert(html.includes(`${count} ${count === 1 ? 'module' : 'modules'} passed`));
    }
  }finally{if(originalStorage===undefined)delete globalThis.localStorage;else globalThis.localStorage=originalStorage;}
  console.log('PASS browser learning: durable full-suite history across runtime/test updates, strict current reports, projection-only milestones, native authority, owner isolation, source safety, concurrent saves, stale dismissal and quota rollback');
}
