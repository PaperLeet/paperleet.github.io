import assert from 'node:assert/strict';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';

const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: 'https://paperleet.test/#/account', pretendToBeVisual: true });
const context = { account: { user: { id: 'owner_a' }, loading: false }, locale: 'en', request: async () => { throw Error('Unexpected request'); } };
const globals = { window: dom.window, document: dom.window.document, HTMLElement: dom.window.HTMLElement, Element: dom.window.Element, navigator: dom.window.navigator, location: dom.window.location, IS_REACT_ACT_ENVIRONMENT: true, __aiSettingsTest: context };
const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
let storageWrites = 0; const originalSet = dom.window.Storage.prototype.setItem;
dom.window.Storage.prototype.setItem = function(...args) { storageWrites++; return originalSet.apply(this,args); };
const mocks = {
  './social': 'export const useAccount=()=>globalThis.__aiSettingsTest.account;export const socialApi=(...args)=>globalThis.__aiSettingsTest.request(...args);',
  './i18n': 'export const useI18n=()=>({locale:globalThis.__aiSettingsTest.locale,t:(zh,en)=>globalThis.__aiSettingsTest.locale==="zh"?zh:en});',
};
const cacheDir = await mkdtemp(`${tmpdir()}/paperleet-ai-settings-`);
const server = await createServer({ configFile: false, cacheDir, esbuild: { jsx: 'automatic' }, server: { middlewareMode: true, watch: null, hmr: false, ws: false }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] },
  plugins: [{ name: 'personal-ai-test', enforce: 'pre', resolveId(source) { if (mocks[source]) return '\0ai:' + source; }, load(id) { return id.startsWith('\0ai:') ? mocks[id.slice(4)] : undefined; } }],
});
const deferred = () => { let resolve, reject; const promise = new Promise((yes,no) => { resolve=yes; reject=no; }); return {promise,resolve,reject}; };
const problem = (status,code,message=code) => Object.assign(Error(message),{status,code});
let root;
try {
  const { createRoot } = await import('react-dom/client');
  const { default: AISettings } = await server.ssrLoadModule('/src/AISettings.tsx');
  root = createRoot(document.querySelector('#root'));
  let revision = 0, active = null, available = true, saved = {}, changed = 0, calls = [];
  const documentValue = (owner=context.account.user.id) => ({ user_id:owner, available, active_provider:active, revision, providers:['gemini','openai','anthropic'].map(id => ({id,label:id,configured:!!saved[id],model:saved[id]||null,updated_at:saved[id]?'2030-01-01T00:00:00Z':null})) });
  const defaultRequest = async (path,body) => {
    calls.push({path,body:body&&structuredClone(body)});
    if (!body) return documentValue();
    assert.equal(body.expected_user_id, context.account.user.id);
    if (path.endsWith('/models')) return {user_id:context.account.user.id,models:[{id:'test-model',name:'Test Model'}]};
    if (path.endsWith('/test')) return {user_id:context.account.user.id,ok:true,provider:body.provider,model:body.model};
    assert.equal(body.expected_revision,revision); revision++;
    if (path.endsWith('/remove')) { delete saved[body.provider]; if(active===body.provider) active=null; }
    else { saved[body.provider]=body.model; active=body.provider; }
    return documentValue();
  };
  context.request = defaultRequest;
  const render = async (props={}) => act(async () => { root.render(React.createElement(React.StrictMode,null,React.createElement(AISettings,{onChanged:()=>changed++,...props}))); });
  const field = name => document.querySelector(`[aria-label="${name}"]`);
  const button = name => [...document.querySelectorAll('button')].find(item=>item.textContent===name||item.getAttribute('aria-label')===name);
  const click = async name => { const node=button(name); assert(node,`Missing ${name}`); assert(!node.disabled,`${name} should be enabled`); await act(async()=>node.click()); };
  const input = async (name,value) => act(async()=>{const node=field(name); assert(node&&!node.disabled); Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set.call(node,value); node.dispatchEvent(new window.Event('input',{bubbles:true}));});
  const select = async (name,value) => act(async()=>{const node=field(name); assert(node&&!node.disabled); node.value=value; node.dispatchEvent(new window.Event('change',{bubbles:true}));});
  await render(); assert(field('AI provider')); assert.equal(field('API key').type,'password'); assert.equal(button('Save and use').disabled,true);
  const secret='test-only-not-a-real-provider-key';
  await input('API key',secret); await click('Load models');
  assert.equal(calls.at(-1).body.api_key,secret); assert.equal(revision,0,'Loading models must not save settings');
  await select('AI model','test-model'); assert.equal(field('Model ID'),null,'Normal model selection does not show a duplicate manual field');
  await select('AI model','__custom__'); assert(field('Model ID')); await input('Model ID','custom-model');
  await select('AI model','test-model'); assert.equal(field('Model ID'),null);
  assert.equal(button('Save and use').disabled,false,'A key and valid model can be saved without a paid test request');
  assert.equal(calls.filter(item=>item.path.endsWith('/test')).length,0); await click('Save and use');
  assert.equal(revision,1); assert.equal(active,'gemini'); assert.equal(changed,1); assert.equal(field('API key').value,''); assert.match(document.body.textContent,/Key saved/);
  await click('Test connection'); assert(!('api_key' in calls.at(-1).body),'Blank input uses the saved server key');
  assert.equal(revision,1,'The optional test must not persist settings'); assert.match(document.body.textContent,/Connection successful/);
  await input('API key',secret); await select('AI provider','anthropic'); assert.equal(field('API key').value,''); assert.equal(field('Model ID').value,''); assert.match(field('AI provider').textContent,/Claude/);
  await input('API key',secret); await input('Model ID','manual-model'); await click('Test connection'); await click('Save and use');
  assert.equal(active,'anthropic'); assert.equal(saved.anthropic,'manual-model'); assert.equal(saved.gemini,'test-model');
  await click('Delete key'); assert.equal(saved.anthropic,undefined); assert.equal(field('API key').value,''); assert.equal(changed,3);

  // Failures never echo provider text/keys. CAS conflicts require an explicit reread.
  await select('AI provider','gemini'); await input('Model ID','new-model'); await click('Test connection');
  context.request = async (path,body) => { if(path==='account/ai-settings'&&body) throw problem(409,'ai_settings_conflict',secret); return defaultRequest(path,body); };
  await click('Save and use'); assert.match(document.body.textContent,/changed in another window/); assert.equal(button('Save and use').disabled,true); assert(!document.body.textContent.includes(secret));
  context.request=defaultRequest; await click('Reload settings'); assert.equal(field('Model ID').value,'test-model'); assert.equal(field('API key').value,'');
  await input('API key',secret);
  context.request=async()=>{throw problem(400,'provider_auth_failed',secret);}; await click('Test connection'); assert.match(document.body.textContent,/key was rejected/); assert(!document.body.textContent.includes(secret));

  // Account changes abort old in-flight requests and retire the previous password input.
  const response=deferred(); let oldSignal;
  context.request=(path,body,signal)=>{if(path.endsWith('/models')) {oldSignal=signal; return response.promise;} return defaultRequest(path,body);};
  const retiredPassword=field('API key'); await click('Load models'); const oldOwner=context.account.user.id;
  context.account={user:{id:'owner_b'},loading:false}; saved={};active=null;revision=0;await render();
  assert(oldSignal.aborted); assert.equal(field('API key').value,''); assert.equal(retiredPassword.value,'','The retired input clears its unsaved secret');
  await act(async()=>response.resolve({user_id:oldOwner,models:[{id:'old-account-model',name:'Old account model'}]}));
  assert(!document.body.textContent.includes('Old account model'));

  // Unavailable encryption still permits explicit deletion; no browser persistence.
  available=false;saved={openai:'saved-model'};active='openai';revision=3;
  await act(async()=>root.unmount());root=createRoot(document.querySelector('#root'));context.locale='zh';context.request=defaultRequest;await render();
  assert.match(document.body.textContent,/个人 AI 设置暂不可用/);assert.equal(field('API key').disabled,true);assert.equal(button('读取模型').disabled,true);
  await click('删除密钥'); assert.equal(saved.openai,undefined); assert.match(document.body.textContent,/密钥已删除/);
  assert.equal(storageWrites,0);assert.equal(window.localStorage.length,0);assert.equal(window.sessionStorage.length,0);
  assert.equal(document.querySelectorAll('input[type="password"]').length,1);

  const accountSource=await readFile('src/Account.tsx','utf8'), studioSource=await readFile('src/RebuttalStudio.tsx','utf8');
  assert.match(accountSource,/personal_ai_settings === true && <AISettings/);
  assert.match(studioSource,/aiSettings && hosted && <AISettings/);
  assert.match(studioSource,/setAiRevision\(value => value \+ 1\)/,'Hub refreshes AI status without remounting the paper');
  console.log('PASS personal AI settings: provider models/manual fallback, direct save and optional tests, encrypted-key metadata, preservation/deletion, CAS conflicts, safe errors, account retirement, unavailable deletion, bilingual UI and no browser persistence.');
} finally {
  if(root) await act(async()=>root.unmount()); await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();
  for(const [key,value] of previous) value?Object.defineProperty(globalThis,key,value):delete globalThis[key];
}
