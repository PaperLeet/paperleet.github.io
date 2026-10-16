import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const project=fileURLToPath(new URL('../',import.meta.url)).replace(/\/$/,'');
const require=createRequire(`${project}/package.json`), {JSDOM}=require('jsdom');
const React=require('react'), {act}=React;
const dom=new JSDOM('<!doctype html><div id="a"></div><div id="b"></div>',{url:'https://offline-recovery.invalid/'});
for(const [key,value] of Object.entries({window:dom.window,document:dom.window.document,navigator:dom.window.navigator,
  HTMLElement:dom.window.HTMLElement,CustomEvent:dom.window.CustomEvent,localStorage:dom.window.localStorage,
  sessionStorage:dom.window.sessionStorage,location:dom.window.location,BroadcastChannel:undefined,IS_REACT_ACT_ENVIRONMENT:true}))
  Object.defineProperty(globalThis,key,{configurable:true,writable:true,value});
const {createRoot}=require('react-dom/client'), {createServer}=await import('vite');
globalThis.__offlineAccount=React.createContext(null);
const server=await createServer({root:project,configFile:false,server:{middlewareMode:true,ws:false},appType:'custom',
  optimizeDeps:{noDiscovery:true,entries:[]},plugins:[{
    name:'offline-account-fixture',enforce:'pre',
    resolveId(source,importer){if(importer?.endsWith('/PracticeWorkspace.tsx')&&source==='./social')return '\0offline-social';},
    load(id){if(id==='\0offline-social')return `import {useContext} from 'react';
      export function useAccount(){return useContext(globalThis.__offlineAccount);}
      export async function socialApi(path,body){return globalThis.__offlineApi(path,body);}
      export class SocialError extends Error {constructor(message,status){super(message);this.status=status;}}
      export const errorMessage=e=>e.message;`;},
  }]});
const id='rl/gae', code='# offline source 🧪\r\ndef gae():\r\n\treturn 42\r\n';
const remote={alice:{[id]:{code:'alice original',updated:'alice-1'}},bob:{}};
const roots={}, views={}, successfulWrites=[];
let account='alice', offline=false, revision=1;
const copy=value=>JSON.parse(JSON.stringify(value));
globalThis.__offlineApi=async(path,body)=>{
  if(offline)throw Error('Fixture network offline');
  if(path==='practice/workspace')return {user_id:account,workspace:copy(remote[account])};
  assert.equal(path,'practice/draft');
  const {SocialError}=await server.ssrLoadModule('\0offline-social');
  if(body.expected_user_id!==account||body.expected_updated!==(remote[account][body.problem]?.updated||''))
    throw new SocialError('Draft or account changed',409);
  const draft={code:body.code,updated:`${account}-${++revision}`};
  remote[account][body.problem]=draft;
  successfulWrites.push({owner:account,code:body.code});
  return {draft:copy(draft)};
};
try{
  const {usePracticeWorkspace}=await server.ssrLoadModule('/src/PracticeWorkspace.tsx');
  function Probe({name}){views[name]=usePracticeWorkspace();return React.createElement('pre',null,views[name].saved[id]?.code||'');}
  async function render(name){await act(async()=>{
    const root=roots[name]??=createRoot(document.querySelector(`#${name}`));
    root.render(React.createElement(globalThis.__offlineAccount.Provider,{value:{health:{mode:'hosted'},user:{id:account},loading:false,refresh:async()=>{}}},React.createElement(Probe,{name})));
  });}
  async function settle(predicate,label){for(let i=0;i<250&&!predicate();i++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),label);}
  async function reconnect(){offline=false;await act(async()=>window.dispatchEvent(new dom.window.Event('online')));}
  await render('a');await render('b');
  await settle(()=>views.a.ready&&views.b.ready,'both workspace instances hydrate');
  offline=true;
  await act(async()=>views.a.update(id,{code}));
  await act(async()=>assert.rejects(views.a.flush(),/network offline/));
  assert.equal(views.a.saved[id].code,code);
  assert.equal(remote.alice[id].code,'alice original');
  assert(Object.values(localStorage).some(value=>value.includes('offline source')),'unsaved source is durably journaled');
  const staleAlice=views.a;
  account='bob';await render('a');await render('b');
  await settle(()=>views.a.error&&views.b.error,'offline account hydration reports failure');
  assert.deepEqual(views.a.saved,{});assert.deepEqual(views.b.saved,{});
  assert.equal(staleAlice.update(id,{code:'stale account source'}),false);
  await act(async()=>staleAlice.flush());
  assert.equal(successfulWrites.length,0,'stale callbacks cannot save under the new account');
  await reconnect();
  await settle(()=>views.a.ready&&views.b.ready&&!views.a.error&&!views.b.error,'online event rehydrates both new-account workspaces');
  assert.deepEqual(remote.bob,{});assert.deepEqual(views.a.saved,{});
  await act(async()=>views.a.update(id,{code:'bob private source'}));
  await act(async()=>views.a.flush());
  await settle(()=>views.b.saved[id]?.code==='bob private source','the new account still synchronizes between instances');
  account='alice';await render('a');await render('b');
  await settle(()=>views.a.ready&&views.b.ready,'returning account hydrates');
  assert.equal(views.a.saved[id].code,code,'returning account recovers exact unsaved Unicode/CRLF source');
  await settle(()=>remote.alice[id].code===code&&!views.a.pending&&!views.b.pending,'recovered draft saves after reconnection');
  assert.equal(remote.bob[id].code,'bob private source','recovery cannot replace another account source');
  assert(successfulWrites.every(row=>row.owner==='alice'?row.code===code:row.code==='bob private source'));
  console.log('PASS failed offline save → account switch → reconnect → account-scoped exact-source recovery and peer synchronization');

  offline=true;const conflictCode=code+'# second offline edit\n';
  await act(async()=>views.a.update(id,{code:conflictCode}));
  await act(async()=>assert.rejects(views.a.flush(),/network offline/));
  remote.alice[id]={code:'changed remotely during outage',updated:`alice-${++revision}`};
  const before=successfulWrites.length;
  await reconnect();
  await settle(()=>views.a.conflicts[id]&&views.b.saved[id]?.code==='changed remotely during outage','reconnection exposes concurrent server source');
  assert.equal(views.a.saved[id].code,conflictCode,'reconnection preserves the unsaved editor source');
  assert.equal(views.a.conflicts[id].code,'changed remotely during outage');
  await act(async()=>new Promise(resolve=>setTimeout(resolve,850)));
  assert.equal(successfulWrites.length,before,'conflicting source cannot be automatically overwritten');
  await act(async()=>views.a.resolveConflict(id,true));
  await act(async()=>views.a.flush());
  await settle(()=>views.b.saved[id]?.code===conflictCode,'explicit resolution reaches the peer');
  assert.equal(remote.alice[id].code,conflictCode);
  assert.equal(remote.bob[id].code,'bob private source');
  console.log('PASS online recovery detects concurrent remote edits, preserves both versions and waits for explicit resolution');
}finally{
  for(const root of Object.values(roots))await act(async()=>root.unmount());
  await server.close();dom.window.close();
  delete globalThis.__offlineAccount;delete globalThis.__offlineApi;
}
