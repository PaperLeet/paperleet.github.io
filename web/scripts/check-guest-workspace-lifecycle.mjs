import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
const project=fileURLToPath(new URL('../',import.meta.url)).replace(/\/$/,'');
const require=createRequire(`${project}/package.json`);
const {JSDOM}=require('jsdom');
require('fake-indexeddb/auto');
const React=require('react');
const {act}=React;
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://guest-hook-test.invalid/'});
const bindings={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,
  localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,
  BroadcastChannel:undefined,IS_REACT_ACT_ENVIRONMENT:true};
for(const [key,value] of Object.entries(bindings))Object.defineProperty(globalThis,key,{configurable:true,writable:true,value});
let networkCalls=0;
globalThis.fetch=async()=>{networkCalls++;throw Error('Hook persistence must not contact the network');};
const {createRoot}=require('react-dom/client');
const {createServer}=await import(pathToFileURL(require.resolve('vite')).href);
const server=await createServer({root:project,configFile:false,server:{middlewareMode:true},appType:'custom',optimizeDeps:{noDiscovery:true,entries:[]}});
let root,view;
const id='architecture/hook-lifecycle-fixture';
const initial='# lifecycle fixture 🧪\r\ndef solve():\r\n\treturn 42\r\n';
const quotaCode='# preserve failed draft\ndef solve():\n    return 43\n';
let originalPut=IDBObjectStore.prototype.put;
const checkpoints=[];
try{
  const {useGuestWorkspace}=await server.ssrLoadModule('/src/useGuestWorkspace.ts');
  function Probe({owner}){view=useGuestWorkspace(owner);return React.createElement('pre',null,view.saved[id]?.code??'');}
  async function render(owner){await act(async()=>{if(!root)root=createRoot(document.querySelector('#root'));root.render(React.createElement(Probe,{owner}));});}
  async function settle(predicate,label){for(let i=0;i<200&&!predicate();i++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),`${label}: ${JSON.stringify({ready:view?.ready,failed:view?.failed,saving:view?.saving,pending:view?.pending,owner:view?.owner,error:view?.error})}`);}
  async function unmount(){if(root)await act(async()=>root.unmount());root=undefined;}
  await render('anonymous');
  await settle(()=>view.ready&&!view.loading,'initial guest hydration');
  assert.equal(view.saved[id],undefined);
  await act(async()=>assert.equal(view.update(id,{code:initial}),true));
  await settle(()=>!view.saving&&!view.pending&&!view.failed,'first durable guest save');
  const staleBeforeUnmount=view;
  await unmount();
  assert.equal(staleBeforeUnmount.update(id,{code:'# stale unmounted mutation'}),false,'Unmounted hook callback cannot mutate guest source');
  await render('anonymous');
  await settle(()=>view.ready&&!view.loading,'guest remount hydration');
  assert.equal(view.saved[id].code,initial,'Unmount/remount restores exact source from IndexedDB');
  checkpoints.push('anonymous edit → durable save → unmount/remount restores exact source');

  const beforeAccount=view;
  await render(undefined);
  assert.deepEqual(view.saved,{},'A signed-in/unresolved owner never exposes guest source');
  assert.equal(view.update(id,{code:'# inactive'}),false,'Inactive owner cannot edit the guest cache');
  assert.equal(beforeAccount.update(id,{code:'# stale owner'}),false,'A retained guest callback cannot mutate after owner changes');
  await render('anonymous');
  await settle(()=>view.ready&&!view.loading,'returning guest hydration');
  assert.equal(view.saved[id].code,initial,'Returning to guest preserves the separate guest draft');
  checkpoints.push('undefined owner hides guest and rejects stale callback; anonymous return restores guest');

  IDBObjectStore.prototype.put=function(...args){if(this.name==='records')throw new DOMException('Fixture quota exhausted','QuotaExceededError');return originalPut.apply(this,args);};
  await act(async()=>assert.equal(view.update(id,{code:quotaCode}),true));
  await settle(()=>view.failed&&!view.saving,'quota failure visible in hook');
  assert.equal(view.saved[id].code,quotaCode,'A storage failure retains the current in-memory code');
  assert(view.error,'Storage failure exposes a user-facing error');
  await unmount();
  await render('anonymous');
  await settle(()=>view.ready&&view.failed,'failed draft is retried and marked failed after remount');
  assert.equal(view.saved[id].code,quotaCode,'Session safety restores an uncommitted source after remount');
  IDBObjectStore.prototype.put=originalPut;
  await act(async()=>view.retry());
  await settle(()=>view.ready&&!view.failed&&!view.saving&&!view.pending,'quota recovery retry');
  await unmount();
  await render('anonymous');
  await settle(()=>view.ready&&!view.loading,'retry persistence remount');
  assert.equal(view.saved[id].code,quotaCode,'A successful retry becomes durable');
  assert.equal(networkCalls,0,'No account/practice network requests are allowed');
  checkpoints.push('quota failure keeps memory + error; remount restores safety draft; retry persists');
  for(const message of checkpoints)console.log(`PASS ${message}`);
  console.log('PASS actual React effects and cleanup; zero server requests');
}finally{
  IDBObjectStore.prototype.put=originalPut;
  if(root)await act(async()=>root.unmount());
  await server.close();dom.window.close();
}
