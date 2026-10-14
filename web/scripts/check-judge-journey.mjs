import assert from 'node:assert/strict';
import fs from 'node:fs';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { createServer } from 'vite';

const server=await createServer({server:{middlewareMode:true},appType:'custom'});
try {
  const { problems,papers }=JSON.parse(fs.readFileSync('public/data/catalog.json','utf8'));
  const attention=problems.find(p=>p.id==='architecture/attention');
  const { paperJudgeCapability,PaperJudgeBadge,ModuleJudgeBadge }=await server.ssrLoadModule('/src/JudgeCapability.tsx');
  const { paperModules,PaperDetail }=await server.ssrLoadModule('/src/PaperViews.tsx');
  const mixed=papers.find(p=>p.id==='ddpm');
  // Exercise mixed-runtime rendering even when current multi-module papers
  // have uniform coverage; exported problem data remain unchanged.
  const fixtureProblems=problems.map(p=>p.id==='diffusion/ddpm_sampling'?{...p,browser_judge:undefined}:p);
  const modules=paperModules(mixed,fixtureProblems);
  assert.deepEqual(paperJudgeCapability(modules),{browser:1,server:1,total:2});
  assert(renderToString(React.createElement(PaperJudgeBadge,{modules})).includes('部分浏览器 1/2'));
  const detail=renderToString(React.createElement(PaperDetail,{paper:mixed,problems:fixtureProblems,saved:{},onGuide(){}}));
  assert(detail.includes('浏览器判题 · 无需 GPU') && detail.includes('服务端判题'));
  assert(renderToString(React.createElement(ModuleJudgeBadge,{problem:modules.find(p=>!p.browser_judge)})).includes('服务端判题'));
  assert.equal(renderToString(React.createElement(PaperJudgeBadge,{modules:[]})),'');

  const { verificationReason,matchesBrowserSubmission }=await server.ssrLoadModule('/src/browserVerification.ts');
  const { fingerprintSource }=await server.ssrLoadModule('/src/reportSource.ts');
  const report={problem:attention.id,runtime:'browser-numpy-wasm',summary:Object.fromEntries(Object.entries(attention.tests).map(([l,n])=>[l,[n,n]])),tests:Object.entries(attention.tests).flatMap(([layer,n])=>Array.from({length:n},(_,i)=>({id:`${layer}-${i}`,layer,passed:true}))),source_sha256:(await fingerprintSource(attention.starter)).sha256};
  assert(await matchesBrowserSubmission(attention,report,attention.starter));
  assert(!await matchesBrowserSubmission(attention,report,attention.starter+'\n'));
  assert(!await matchesBrowserSubmission(attention,{...report,problem:'rl/dpo'},attention.starter));
  assert(!await matchesBrowserSubmission(attention,{...report,runtime:'server'},attention.starter));
  assert(!await matchesBrowserSubmission(attention,{...report,tests:[]},attention.starter));
  assert(!await matchesBrowserSubmission(attention,{...report,source_sha256:undefined},attention.starter));
  const ready={passed:true,sourceState:'current',signedIn:true,online:true,ready:true,conflict:false,syncError:false,busy:false};
  assert.equal(verificationReason(ready),'ready');
  for (const [change,reason] of [[{busy:true},'busy'],[{passed:false},'failed'],[{signedIn:false},'signin'],[{ready:false},'workspace'],[{conflict:true},'conflict'],[{syncError:true},'sync'],[{sourceState:'changed'},'changed'],[{sourceState:'checking'},'version'],[{sourceState:'unknown'},'version'],[{online:false},'offline']]) {
    assert.equal(verificationReason({...ready,...change}),reason);
  }
  const { BrowserResultActions }=await server.ssrLoadModule('/src/BrowserResultActions.tsx');
  const props={...ready,problem:attention,report,onSignIn(){},onVerify(){},onRefresh(){},onLab(){}};
  const guest=renderToString(React.createElement(BrowserResultActions,{...props,signedIn:false}));
  assert(guest.includes('保留本题代码并登录') && !guest.includes('服务端确认成绩'));
  const changed=renderToString(React.createElement(BrowserResultActions,{...props,sourceState:'changed'}));
  assert(changed.includes('代码已改变') && changed.includes('disabled=""'));
  const offline=renderToString(React.createElement(BrowserResultActions,{...props,online:false}));
  assert(offline.includes('重试连接') && offline.includes('本机测试结果仍可查看'));

  const { HANDOFF_KEY,HANDOFF_TTL,parseHandoff,saveHandoff,claimHandoff,consumeHandoff }=await server.ssrLoadModule('/src/draftHandoff.ts');
  const values=new Map();
  const storage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
  const held={version:1,id:'handoff_0123456789',problem:attention.id,code:'# <script>私有草稿</script>\n',created:Date.now()};
  saveHandoff(storage,{...held,report:{passed:true},access_code:'never-copy'});
  assert.deepEqual(parseHandoff(storage.getItem(HANDOFF_KEY)),held,'Only source and bounded transfer metadata may be held');
  assert.equal(claimHandoff(storage,'unresolved'),undefined);
  assert.equal(claimHandoff(storage,'anonymous').code,held.code);
  assert.equal(claimHandoff(storage,'user:alice').owner,'user:alice');
  assert.equal(claimHandoff(storage,'user:bob'),undefined,'An account switch cannot inherit another account’s import prompt');
  assert.equal(storage.getItem(HANDOFF_KEY),null);
  saveHandoff(storage,held);claimHandoff(storage,'user:alice');
  assert.equal(claimHandoff(storage,'anonymous'),undefined,'Logout clears the claimed transfer');
  saveHandoff(storage,held);
  assert(!consumeHandoff(storage,'older_handoff_001') && storage.getItem(HANDOFF_KEY));
  assert(consumeHandoff(storage,held.id) && !storage.getItem(HANDOFF_KEY));
  assert(parseHandoff(JSON.stringify(held),held.created+HANDOFF_TTL-1));
  assert(!parseHandoff(JSON.stringify(held),held.created+HANDOFF_TTL));
  for (const change of [{created:Date.now()+1000},{problem:'../private'},{problem:'https://outside.test/'},{code:'你'.repeat(50_000)},{owner:'anonymous'},{owner:'user:<script>'},{version:2}]) {
    assert(!parseHandoff(JSON.stringify({...held,...change})));
  }
  assert(!parseHandoff('{broken'));
  assert.throws(()=>saveHandoff({...storage,setItem(){throw Error('blocked')}},held));

  const { draftSnapshotMatches }=await server.ssrLoadModule('/src/PracticeWorkspace.tsx');
  const expected={owner:'user:alice',code:'existing',updated:'v1'};
  assert(draftSnapshotMatches({code:'existing',updated:'v1'},expected));
  assert(!draftSnapshotMatches({code:'new edit',updated:'v1'},expected));
  assert(!draftSnapshotMatches({code:'existing',updated:'v2'},expected));
  assert(draftSnapshotMatches(undefined,{owner:'user:alice'}));
  assert(!draftSnapshotMatches({code:''},{owner:'user:alice'}));
  const { DraftHandoffNotice }=await server.ssrLoadModule('/src/DraftHandoffNotice.tsx');
  const notice=renderToString(React.createElement(DraftHandoffNotice,{pending:{...held,owner:'user:alice'},problem:attention,current:{code:'existing'},username:'alice',open:true,ready:true,conflict:false,onImport(){},onDiscard(){}}));
  assert(notice.includes('备份并替换草稿') && notice.includes('@alice'));
  assert(notice.includes('&lt;script&gt;') && !notice.includes('<script>'));
  const waiting=renderToString(React.createElement(DraftHandoffNotice,{pending:held,problem:attention,current:{},username:'alice',open:true,ready:false,conflict:false,onImport(){},onDiscard(){}}));
  assert(waiting.includes('正在加载账号草稿') && !waiting.includes('私有草稿'));
  console.log('PASS judge journey: honest mixed/server badges, exact-source verification, guest sign-in actions, tab-scoped source-only handoff, TTL/account isolation, overwrite review and stale-draft guards');
} finally { await server.close(); }
