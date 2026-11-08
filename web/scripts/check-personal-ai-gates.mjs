import assert from 'node:assert/strict';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';

const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: 'https://paperleet.test/#/rebuttal/paper_a', pretendToBeVisual:true });
const context = { account: { user:{id:'owner_a'}, loading:false, capabilities:{personal_ai_settings:true} }, locale:'en', request:null };
const globals = { window:dom.window,document:dom.window.document,navigator:dom.window.navigator,location:dom.window.location,history:dom.window.history,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,HTMLElement:dom.window.HTMLElement,Element:dom.window.Element,HTMLInputElement:dom.window.HTMLInputElement,HTMLTextAreaElement:dom.window.HTMLTextAreaElement,IS_REACT_ACT_ENVIRONMENT:true,__personalGate:context };
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals)) Object.defineProperty(globalThis,key,{value,configurable:true,writable:true});
const mocks={
 './social':'export const useAccount=()=>globalThis.__personalGate.account;export const socialApi=(...args)=>globalThis.__personalGate.request(...args);',
 './apiClient':'export const apiFetch=(...args)=>globalThis.__personalGate.request(...args);',
 './i18n':'export const useI18n=()=>({locale:globalThis.__personalGate.locale,t:(zh,en)=>globalThis.__personalGate.locale==="zh"?zh:en});',
};
const cacheDir=await mkdtemp(`${tmpdir()}/paperleet-personal-gates-`);
const server=await createServer({configFile:false,cacheDir,esbuild:{jsx:'automatic'},server:{middlewareMode:true,watch:null,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'personal-gate-test',enforce:'pre',resolveId(source){if(mocks[source])return '\0gate:'+source;},load(id){return id.startsWith('\0gate:')?mocks[id.slice(6)]:undefined;}}]});
let root;
try {
 const {createRoot}=await import('react-dom/client');
 const {default:Studio}=await server.ssrLoadModule('/src/RebuttalStudio.tsx');
 const {newProject}=await server.ssrLoadModule('/src/rebuttalModel.ts');
 const {readPersonalAIStatus}=await server.ssrLoadModule('/src/personalAI.ts');
 const project={...newProject(),id:'paper_a',title:'Personal AI gate fixture',reviews:[{id:'r1',name:'R1',text:'Please explain the ablation.'}]};
 const shared=[],posts=[];
 const state=(owner,ready=false,provider='kimi')=>({user_id:owner,configured:ready,provider:ready?provider:null,model:ready?'fixture-model':null,source:ready?'personal':null,available:true,region:ready?'international':null});
 let status=state('owner_a'),hold;
 context.request=async(path,options)=>{
  if(path==='/api/rebuttal/status') {if(hold)return hold;return new Response(JSON.stringify(status));}
  if(path==='account/ai-settings')return {user_id:context.account.user.id,available:true,active_provider:null,revision:0,providers:['gemini','openai','anthropic','kimi','qwen','deepseek'].map(id=>({id,configured:false,model:null,updated_at:null,region:['kimi','qwen'].includes(id)?'international':null}))};
  posts.push({path,options});throw Error('Unexpected live action');
 };
 root=createRoot(document.querySelector('#root'));
 const render=async(key='studio')=>act(async()=>root.render(React.createElement(Studio,{key,hosted:{project,role:'owner',readOnly:false,saveStatus:'saved',onChange:value=>shared.push(value),onBack(){},onRefresh(){},onShare(){}}})));
 const button=label=>[...document.querySelectorAll('button')].find(item=>(item.getAttribute('aria-label')||item.textContent.trim())===label);
 const click=async(label)=>{assert(button(label),label);await act(async()=>button(label).click());};
 const settle=async()=>{for(let i=0;i<4;i++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));};
 await render();assert(document.querySelector('.personal-ai-connection').textContent.includes('Connect your API key to use AI'));
 await click('Reviews');assert.equal(button('Extract concerns').title,'Connect your API key first');await click('Extract concerns');await settle();
 assert(document.querySelector('[aria-label="API key"]'),'AI action opens actual key settings');assert.equal(posts.length,0);assert.equal(shared.length,0,'Missing API never silently performs local extraction');
 await click('Close AI settings');await click('Local breakdown');assert.equal(shared.length,1,'Explicit local breakdown remains available');
 status=state('owner_a',true);await render('connected');assert(document.querySelector('.personal-ai-connection').textContent.includes('Kimi · fixture-model'));
 let resolveOld;hold=new Promise(resolve=>resolveOld=resolve);await render('pending');
 context.account={...context.account,user:{id:'owner_b'}};status=state('owner_b',false);hold=undefined;await render('pending');
 assert(!document.querySelector('.personal-ai-connection').textContent.includes('Kimi'));
 await act(async()=>resolveOld(new Response(JSON.stringify(state('owner_a',true)))));
 assert(document.querySelector('.personal-ai-connection').textContent.includes('Connect your API key to use AI'),'Late previous-account configuration cannot re-enable AI');
 await click('Reviews');const before=shared.length;await click('Extract concerns');await settle();assert.equal(shared.length,before);assert.equal(posts.length,0);
 await click('Close AI settings');context.locale='zh';await render('zh');assert(document.querySelector('.personal-ai-connection').textContent.includes('使用 AI 前请先配置个人 API key'));
 assert.throws(()=>readPersonalAIStatus({...state('owner_b',true),source:'server'},'owner_b'),'Global keys never count as personal configuration');
 assert.throws(()=>readPersonalAIStatus(state('owner_a',true),'owner_b'),'Status requires matching account echo');
 console.log('PASS personal AI Hub prompt, actual settings entry, no silent local fallback, explicit manual breakdown, active provider/model, bilingual labels and late account status isolation');
} finally {
 if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();
 for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
