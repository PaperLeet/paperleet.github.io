import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const project=fileURLToPath(new URL('../',import.meta.url)).replace(/\/$/,'');
const require=createRequire(`${project}/package.json`), {JSDOM}=require('jsdom');
const React=require('react'), {act}=React;
const dom=new JSDOM('<!doctype html><div id="a"></div><div id="b"></div>',{url:'https://workspace-sync.invalid/',pretendToBeVisual:true});
for(const [key,value] of Object.entries({window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,
  CustomEvent:dom.window.CustomEvent,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,
  BroadcastChannel:undefined,IS_REACT_ACT_ENVIRONMENT:true}))Object.defineProperty(globalThis,key,{configurable:true,writable:true,value});
const {createRoot}=require('react-dom/client'), {createServer}=await import('vite');
globalThis.__workspaceAccountContext=React.createContext(null);
let releaseCache;globalThis.__workspaceCacheGate=new Promise(resolve=>{releaseCache=resolve;});
const server=await createServer({root:project,configFile:false,server:{middlewareMode:true},appType:'custom',optimizeDeps:{noDiscovery:true,entries:[]},plugins:[{
  name:'workspace-account-fixture',enforce:'pre',
  resolveId(source,importer){
    if(importer?.endsWith('/PracticeWorkspace.tsx')){
      if(source==='./social')return '\0workspace-social';
      if(source==='./hostedDraftCache')return '\0workspace-cache';
    }
  },
  load(id){
    if(id==='\0workspace-cache')return `await globalThis.__workspaceCacheGate; export * from ${JSON.stringify(`${project}/src/hostedDraftCache.ts`)};`;
    if(id==='\0workspace-social')return `import {useContext} from 'react';
    export function useAccount(){return useContext(globalThis.__workspaceAccountContext);}
    export async function socialApi(path,body){return globalThis.__workspaceApi(path,body);}
    export class SocialError extends Error {constructor(message,status){super(message);this.status=status;}}
    export const errorMessage=e=>e.message;`;},
}]});
let roots={},views={},userIds={a:'alice',b:'alice'},requests=0,writes=0,version=1;
const id='rl/dpo',problem={id,test_count:1};
const passed={problem:id,tests:[{id:'case',layer:'T0',passed:true}],summary:{T0:[1,1]},selected:'all'};
const failed={...passed,tests:[{...passed.tests[0],passed:false}],summary:{T0:[0,1]}};
let workspace={[id]:{code:'base',updated:'v1'}};
let delayedRead,pauseRead=false,readError=false;
function copy(value){return JSON.parse(JSON.stringify(value));}
globalThis.__workspaceApi=async(path,body)=>{
  if(path==='practice/workspace'){
    requests++;
    if(readError)throw Error('Workspace temporarily unavailable');
    const snapshot={user_id:'alice',workspace:copy(workspace)};
    if(pauseRead){pauseRead=false;return new Promise(resolve=>{delayedRead=()=>resolve(snapshot);});}
    return snapshot;
  }
  assert.equal(path,'practice/draft');writes++;
  if(body.expected_updated!==(workspace[body.problem]?.updated||'')){
    const {SocialError}=await server.ssrLoadModule('\0workspace-social');throw new SocialError('Draft changed',409);
  }
  workspace[body.problem]={...workspace[body.problem],code:body.code,updated:`v${++version}`};
  return {draft:copy(workspace[body.problem])};
};
try{
  const {usePracticeWorkspace}=await server.ssrLoadModule('/src/PracticeWorkspace.tsx');
  const {publishPracticeChange,subscribePracticeChanges}=await server.ssrLoadModule('/src/practiceChanges.ts');
  const hints=[],unsubscribe=subscribePracticeChanges(change=>hints.push(change));
  function Probe({name}){views[name]=usePracticeWorkspace();return React.createElement('pre',null,views[name].saved[id]?.code||'');}
  async function render(name){await act(async()=>{
    const root=roots[name]??=createRoot(document.querySelector(`#${name}`));
    root.render(React.createElement(globalThis.__workspaceAccountContext.Provider,{value:{health:{mode:'hosted'},user:{id:userIds[name]},loading:false,refresh:async()=>{}}},React.createElement(Probe,{name})));
  });}
  async function settle(predicate,label){for(let i=0;i<200&&!predicate();i++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),label);}
  async function delay(ms){await act(async()=>new Promise(resolve=>setTimeout(resolve,ms)));}
  async function signal(owner='user:alice'){await act(async()=>publishPracticeChange({owner,kind:'submission',problem:id,source:'remote-fixture'}));}
  await render('a');
  assert.equal(views.a.ready,false,'hosted editing stays disabled until the lazy runtime is available');
  assert.equal(views.a.update(id,{code:'before runtime initialization'}),false);
  assert.equal(requests,0,'no hosted workspace request races runtime initialization');
  userIds.a='bob';await render('a');
  await act(async()=>releaseCache());
  await settle(()=>!views.a.loading,'delayed runtime resolves after an account switch');
  assert.equal(views.a.owner,'user:bob');
  assert.deepEqual(views.a.saved,{},'delayed initialization cannot expose the prior account');
  assert.equal(requests,1,'obsolete owner does not initiate a late API read');
  userIds.a='alice';requests=0;await render('a');await render('b');
  await settle(()=>views.a.ready&&views.b.ready,'both independent hook instances hydrate');
  assert.equal(requests,2);
  console.log('PASS lazy hosted runtime gates editing and rejects obsolete owner initialization');
  await act(async()=>views.a.update(id,{code:'window A edit'}));
  await act(async()=>views.a.flush());
  await settle(()=>views.b.saved[id]?.code==='window A edit','successful save reaches peer without reload');
  assert.equal(views.a.saved[id].code,'window A edit');
  assert.equal(requests,3,'sender ignores own invalidation; one peer fetch');
  assert(hints.some(change=>change.kind==='draft'));
  for(const hint of hints)assert.deepEqual(Object.keys(hint).filter(key=>!['owner','kind','problem','revision','source','nonce'].includes(key)),[],'hints contain no private code/results');
  console.log('PASS saved draft refreshes the peer; sender self-ignore and metadata-only hints');

  await act(async()=>{views.a.update(id,{code:'A conflicting edit'});views.b.update(id,{code:'B conflicting edit'});});
  const recoveryKeys=Object.keys(localStorage).filter(key=>key.includes(':window:'));
  assert.equal(recoveryKeys.length,2,'each document owns a separate recovery journal');
  assert(recoveryKeys.some(key=>localStorage.getItem(key).includes('A conflicting edit')));
  assert(recoveryKeys.some(key=>localStorage.getItem(key).includes('B conflicting edit')));
  await act(async()=>views.a.flush());
  await settle(()=>Boolean(views.b.conflicts[id]),'peer unsynced edit produces an explicit conflict');
  assert.equal(views.b.saved[id].code,'B conflicting edit','refresh preserves unsaved editor source');
  assert.equal(views.b.conflicts[id].code,'A conflicting edit');
  assert(Object.keys(localStorage).some(key=>key.includes(':window:')&&localStorage.getItem(key).includes('B conflicting edit')),'peer flush cannot erase unsynced recovery source');
  await act(async()=>views.b.resolveConflict(id,false));
  console.log('PASS concurrent edits preserve both sources and require conflict resolution');

  workspace[id]={...workspace[id],report:passed,acceptedReport:passed};
  await act(async()=>views.a.applyReport(problem,passed,'user:alice'));
  await settle(()=>views.b.saved[id]?.acceptedReport?.tests[0].passed,'peer progress/report refresh');
  const beforeBurst=requests;
  await act(async()=>{for(let i=0;i<10;i++)publishPracticeChange({owner:'user:alice',kind:'submission',problem:id,source:'burst'});});
  await delay(200);assert.equal(requests,beforeBurst+2,'burst invalidations coalesce once per window');
  const beforeForeign=requests;await signal('user:bob');await delay(200);assert.equal(requests,beforeForeign,'foreign account invalidation does not fetch');
  console.log('PASS accepted progress sync, burst coalescing and owner isolation');

  pauseRead=true;await signal();await settle(()=>Boolean(delayedRead),'slow background refresh started');
  workspace[id]={...workspace[id],report:failed,acceptedReport:passed};
  await act(async()=>views.a.applyReport(problem,failed,'user:alice'));
  await act(async()=>delayedRead());delayedRead=undefined;
  assert.equal(views.a.saved[id].report.tests[0].passed,false,'late snapshot cannot replace a newer local server report');
  await delay(250);assert.equal(views.a.saved[id].acceptedReport.tests[0].passed,true);
  console.log('PASS late snapshots cannot roll back newer submission results');

  const beforeFocus=requests;
  await delay(2050);workspace[id]={...workspace[id],code:'changed on other device',updated:`v${++version}`};
  await act(async()=>{window.dispatchEvent(new dom.window.Event('focus'));document.dispatchEvent(new dom.window.Event('visibilitychange'));});
  await settle(()=>views.a.saved[id]?.code==='changed on other device'&&views.b.saved[id]?.code==='changed on other device','focus refresh catches missed notifications');
  assert.equal(requests,beforeFocus+2,'focus and visibility event pair coalesces');
  console.log('PASS focus/visibility refresh recovers missed notifications without polling');

  readError=true;await signal();
  await settle(()=>views.a.error&&views.b.error,'background workspace read fails visibly');
  const beforeRetry=requests;
  workspace[id]={...workspace[id],code:'remote edit during outage',updated:`v${++version}`};
  readError=false;
  await act(async()=>views.a.retry());
  await settle(()=>views.a.saved[id]?.code==='remote edit during outage','Retry reloads a hydrated workspace after a failed background read');
  assert.equal(requests,beforeRetry+1,'manual read retry makes one authoritative request');
  assert.equal(views.a.error,'');
  console.log('PASS manual Retry recovers failed background workspace reads');

  pauseRead=true;await signal();await settle(()=>Boolean(delayedRead),'account switch request pending');
  const staleA=views.a;userIds.a='bob';await render('a');
  await act(async()=>delayedRead());delayedRead=undefined;
  assert.deepEqual(views.a.saved,{},'account switch never reveals prior user snapshot');
  assert.equal(staleA.update(id,{code:'stale alice edit'}),false,'retained callback cannot edit another account');
  const beforeStale=writes;await act(async()=>staleA.flush());assert.equal(writes,beforeStale);
  await act(async()=>roots.a.unmount());delete roots.a;
  const staleB=views.b;await act(async()=>roots.b.unmount());delete roots.b;
  assert.equal(staleB.update(id,{code:'after unmount'}),false);
  const afterUnmount=requests;await signal();await delay(200);assert.equal(requests,afterUnmount,'unmounted hooks unsubscribe');
  unsubscribe();console.log('PASS owner changes/unmount reject stale responses and callbacks');
  const {createHostedDraftCache,practiceCacheKey}=await server.ssrLoadModule('/src/hostedDraftCache.ts');
  localStorage.clear();sessionStorage.clear();
  const owner='user:alice',legacy=practiceCacheKey(owner,location.origin);
  const recovered={ [id]:{code:'legacy unsynced source',updated:'2026-10-15T01:00:00Z',base:'old-server',revision:1} };
  localStorage.setItem(legacy,JSON.stringify(recovered));
  const firstCache=createHostedDraftCache('document-1');
  assert.deepEqual(firstCache.read(owner),recovered,'legacy pending source remains recoverable');
  firstCache.write(owner,recovered);
  assert(localStorage.getItem(legacy),'migration retains legacy copy until exact server acknowledgment');
  const reloaded=createHostedDraftCache('document-2');
  assert.deepEqual(reloaded.read(owner),recovered,'reload recovers the tab session');
  reloaded.write(owner,recovered);
  assert.equal(Object.keys(localStorage).filter(key=>key.includes(':window:')).length,2,'reload/duplicate uses a fresh writable document key');
  sessionStorage.clear();
  const reopened=createHostedDraftCache('document-3');
  assert.deepEqual(reopened.read(owner),recovered,'fresh session can recover closed-window drafts');
  assert.deepEqual(reopened.read('user:bob'),{},'recovery journals are account scoped');
  reopened.acknowledge(owner,{[id]:{code:'different source'}});
  assert(localStorage.getItem(legacy),'unrelated server source cannot erase unsynced draft');
  reopened.acknowledge(owner,{[id]:{code:recovered[id].code}});
  assert.equal(Object.keys(localStorage).filter(key=>key.includes(':window:')).length,0,'exact server acknowledgement removes matching immutable copies only');
  assert(localStorage.getItem(legacy),'legacy mutable map is never deleted underneath an older window');
  sessionStorage.clear();
  assert.deepEqual(reopened.read(owner),{},'acknowledged legacy source is not replayed');
  localStorage.setItem(legacy,JSON.stringify({[id]:{...recovered[id],code:'newer legacy write',revision:2}}));
  assert.equal(reopened.read(owner)[id].code,'newer legacy write','legacy acknowledgement cannot hide a later old-client edit');

  // Interleave a different browser process after acknowledge reads an old key
  // and before it removes that key. The newer immutable key must survive.
  localStorage.clear();sessionStorage.clear();
  const writer=createHostedDraftCache('adversarial-writer'), cleaner=createHostedDraftCache('adversarial-reader');
  writer.write(owner,recovered);
  cleaner.read(owner);
  const originalGet=dom.window.Storage.prototype.getItem;
  let interleaved=false;
  dom.window.Storage.prototype.getItem=function(key){
    const result=originalGet.call(this,key);
    if(this===localStorage&&String(key).includes(':window:adversarial-writer:')&&!interleaved){
      interleaved=true;writer.write(owner,{[id]:{...recovered[id],code:'new unsynced concurrent edit',updated:'2026-10-15T02:00:00Z',revision:2}});
    }
    return result;
  };
  try{cleaner.acknowledge(owner,{[id]:{code:recovered[id].code}});}finally{dom.window.Storage.prototype.getItem=originalGet;}
  assert(interleaved,'the adversarial write occurred between the read and deletion');
  assert(Object.keys(localStorage).some(key=>key.includes(':window:')&&localStorage.getItem(key).includes('new unsynced concurrent edit')),'acknowledging an old immutable key cannot delete the concurrent unsynced write');
  sessionStorage.clear();assert.equal(cleaner.read(owner)[id].code,'new unsynced concurrent edit');
  console.log('PASS adversarial peer acknowledgement cannot erase a concurrent immutable draft revision');
  console.log('PASS isolated recovery journals survive reload/closed tabs and migrate legacy source safely');

  localStorage.clear();sessionStorage.clear();
  const otherId='rl/gae',oldWriter=createHostedDraftCache('old-writer');
  oldWriter.write(owner,{...recovered,[otherId]:{...recovered[id],code:'another problem'}});
  sessionStorage.clear();const recovering=createHostedDraftCache('recovering');recovering.read(owner);
  recovering.write(owner,{[id]:{...recovered[id],code:'new unsynced edit'}});
  const latePeer=createHostedDraftCache('late-peer');
  latePeer.write(owner,{[id]:{...recovered[id],code:'unseen peer edit'}});
  recovering.retireRecovery(owner,id);
  const durable=Object.values(localStorage).join('\n');
  assert(durable.includes('new unsynced edit'),'retiring recovery preserves this document\'s newer unsynced source');
  assert(durable.includes('unseen peer edit'),'retiring recovery preserves an unseen peer revision');
  sessionStorage.clear();assert.equal(recovering.read(owner)[otherId].code,'another problem','retirement is scoped to one problem');
  console.log('PASS recovery retirement preserves newer unsynced edits, unseen peer revisions and other problems');

  localStorage.clear();sessionStorage.clear();
  const sessionDraft={...recovered[id],code:'this tab session source'};
  const sessionWriter=createHostedDraftCache('session-writer');sessionWriter.write(owner,{[id]:sessionDraft});
  localStorage.setItem(legacy,JSON.stringify(recovered));
  const sessionReader=createHostedDraftCache('session-reader');
  assert.equal(sessionReader.read(owner)[id].code,sessionDraft.code,'the tab session takes precedence over another window\'s legacy source');
  sessionReader.retireRecovery(owner,id);
  sessionStorage.clear();
  assert.equal(createHostedDraftCache('fresh-reader').read(owner)[id]?.code,recovered[id].code,
    'retiring session recovery must not acknowledge a legacy source the tab never recovered');
  console.log('PASS session recovery retirement preserves ignored legacy drafts from other windows');

  for(const keepLocal of [true,false]){
    localStorage.clear();sessionStorage.clear();
    localStorage.setItem(legacy,JSON.stringify(recovered));
    const closedWindow=createHostedDraftCache('closed-window');
    closedWindow.write(owner,recovered);
    sessionStorage.clear();userIds.a='alice';await render('a');
    await settle(()=>views.a.ready&&views.a.conflicts[id],'closed-window source conflicts with the current server draft');
    assert.equal(views.a.saved[id].code,recovered[id].code);
    await act(async()=>views.a.update(id,{code:`edited recovered source ${keepLocal}`}));
    await act(async()=>views.a.resolveConflict(id,keepLocal));
    if(keepLocal)await act(async()=>views.a.flush());
    assert.equal(views.a.saved[id].code,workspace[id].code,'the selected source matches the server');
    await act(async()=>roots.a.unmount());delete roots.a;
    sessionStorage.clear();await render('a');
    await settle(()=>views.a.ready,'a fresh browser session hydrates after resolving recovery');
    assert.equal(views.a.saved[id].code,workspace[id].code,'superseded recovery must not reappear in a fresh session');
    assert.deepEqual(views.a.conflicts,{});
    await act(async()=>roots.a.unmount());delete roots.a;
  }
  console.log('PASS saved and explicitly discarded recovered drafts retire older journal and legacy copies');
}finally{
  for(const root of Object.values(roots))if(root)await act(async()=>root.unmount());
  await server.close();dom.window.close();
}
