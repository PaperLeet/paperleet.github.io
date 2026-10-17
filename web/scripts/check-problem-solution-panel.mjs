import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/1?from=collections%2Fllm-basics',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:true});
dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__problemSolutions:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const cache=fs.mkdtempSync(path.join(os.tmpdir(),'paperleet-problem-solutions-'));
const server=await createServer({cacheDir:cache,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
 name:'problem-solutions-fixture',enforce:'pre',resolveId(source,importer){
  if(source==='./social')return '\0problem-solutions-social';
  if(source==='./deferredPages'&&importer?.endsWith('/ProblemSolutionPanel.tsx'))return '\0problem-solutions-feed';
  if(source==='./SyntaxHighlight'&&importer?.endsWith('/CommunityCodeBlock.tsx'))return '\0problem-solutions-highlight';
 },load(name){
  if(name==='\0problem-solutions-feed')return `export {ProblemSolutions} from '/src/Community.tsx';`;
  if(name==='\0problem-solutions-highlight')return `import React from 'react';export function Highlight({code}){globalThis.__problemSolutions.highlights.push(code);return React.createElement('pre',{'data-reference-code':''},code);}`;
  if(name==='\0problem-solutions-social')return `import React from 'react';export const useAccount=()=>globalThis.__problemSolutions.account;export const socialApi=(...args)=>globalThis.__problemSolutions.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;
 }
}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {default:Panel}=await server.ssrLoadModule('/src/ProblemSolutionPanel.tsx');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {readCommunityFeed}=await server.ssrLoadModule('/src/communityNavigation.ts');
 const user={id:'alice',username:'alice',display_name:'Alice',can_publish_community:true,affiliation:{kind:'institution',domain:'test.edu',name:'Test University'}};
 const account={user,loading:false,capabilities:{community_solutions:true,community_feed_sorting:true,community_bookmarks:true,community_anonymous_posts:true},refresh:async()=>{}};
 const requests=[],mutations=[],highlights=[];
 let topic={id:'architecture/attention',number:1,title:'注意力',title_en:'Attention'},available=true,solution='private_reference_answer',callbacks=0;
 globalThis.__problemSolutions={account,highlights,request:async(url,body)=>{
  if(body){mutations.push({url,body});throw Error('Offline');}
  requests.push(url);return {items:[],total:0,solutions:true,sort:new URLSearchParams(url.split('?')[1]).get('sort')||'latest'};
 }};
 function Language(){const {locale,setLocale}=useI18n();return React.createElement('button',{id:'language',onClick:()=>setLocale(locale==='en'?'zh':'en')},'Language');}
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,
  React.createElement(Language),React.createElement('textarea',{'aria-label':'Practice code',defaultValue:'def my_solution(): return 42'}),
  React.createElement(Panel,{key:account.user?.id||'guest',topic,solution,communityAvailable:available,onCode:()=>{callbacks++;document.querySelector('[aria-label="Practice code"]').focus();}})));});
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
 const button=text=>[...document.querySelectorAll('button')].find(node=>node.textContent===text);
 const tab=kind=>document.querySelector(`[role="tab"][id$="-${kind}-tab"]`);
 const pane=kind=>document.querySelector(`[role="tabpanel"][id$="-${kind}"]`);
 const change=async(node,value)=>act(async()=>{Object.getOwnPropertyDescriptor(node.tagName==='TEXTAREA'?dom.window.HTMLTextAreaElement.prototype:node.tagName==='SELECT'?dom.window.HTMLSelectElement.prototype:dom.window.HTMLInputElement.prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event(node.tagName==='SELECT'?'change':'input',{bubbles:true}));});
 const key=async(node,key,extra={})=>act(async()=>{node.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key,bubbles:true,cancelable:true,...extra}));});
 const reveal=async()=>act(async()=>{const details=pane('reference').querySelector('details');details.open=true;details.dispatchEvent(new dom.window.Event('toggle'));});
 await render();
 assert.equal(tab('community').getAttribute('aria-selected'),'true');assert(!pane('community').hidden);assert(pane('reference').hidden);
 assert.equal(requests.length,1);assert(requests[0].includes('solutions=true'));
 assert.equal(new URLSearchParams(requests[0].split('?')[1]).get('problem_id'),topic.id);
 assert.equal(highlights.length,0);assert(!document.body.textContent.includes(solution));
 const editor=document.querySelector('[aria-label="Practice code"]');editor.setSelectionRange(4,13,'backward');editor.scrollTop=51;editor.scrollLeft=12;
 const href=location.href;
 await change(document.querySelector('.community-sort select'),'top');
 await click(button('Share a solution'));
 const composer=document.querySelector('.community-composer'),field=composer.querySelector('textarea');
 await change(composer.querySelector('input[type="text"],input:not([type])'),'My attention solution');await change(field,'My own derivation, not the reference or practice code.');
 field.setSelectionRange(3,10,'backward');field.scrollTop=73;
 const storage=JSON.stringify({...sessionStorage}),requestCount=requests.length;
 await click(tab('reference'));
 assert(pane('community').hidden);assert(!pane('reference').hidden);assert.equal(highlights.length,0,'Selecting the reference still requires spoiler consent');
 assert.equal(document.querySelector('.community-composer'),composer);assert.equal(field.value,'My own derivation, not the reference or practice code.');assert.equal(JSON.stringify({...sessionStorage}),storage);
 await reveal();assert(highlights.includes(solution));assert.equal(pane('reference').querySelector('[data-reference-code]').textContent,solution);
 await key(tab('reference'),'Home');
 assert.equal(document.activeElement,tab('community'));assert(!pane('community').hidden);assert.equal(requests.length,requestCount,'Switching sources must not refetch or reset sorting');
 assert.equal(document.querySelector('.community-sort select').value,'top');assert.equal(document.querySelector('.community-composer textarea'),field);
 assert.deepEqual([field.selectionStart,field.selectionEnd,field.selectionDirection,field.scrollTop],[3,10,'backward',73]);
 assert.equal(location.href,href);assert.equal(document.querySelector('[aria-label="Practice code"]'),editor);
 assert.deepEqual([editor.value,editor.selectionStart,editor.selectionEnd,editor.selectionDirection,editor.scrollTop,editor.scrollLeft],['def my_solution(): return 42',4,13,'backward',51,12]);
 await key(tab('community'),'ArrowRight',{isComposing:true});assert.equal(tab('community').getAttribute('aria-selected'),'true');
 await key(tab('community'),'End',{ctrlKey:true});assert.equal(tab('community').getAttribute('aria-selected'),'true');
 await key(tab('community'),'ArrowRight');assert.equal(tab('reference').getAttribute('aria-selected'),'true');assert.equal(document.activeElement,tab('reference'));
 await click(document.querySelector('.problem-solution-code'));assert.equal(callbacks,1);assert.equal(document.activeElement,editor);assert.equal(location.href,href);
 await click(tab('community'));
 await act(async()=>composer.querySelector('form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
 assert.equal(mutations.length,1);assert.equal(mutations[0].body.solution,true);assert.equal(mutations[0].body.problem_id,topic.id);assert.equal(mutations[0].body.body,field.value);
 assert(!JSON.stringify(mutations).includes(solution));assert(!JSON.stringify(mutations).includes(editor.value));assert(document.querySelector('.community-composer textarea').value===field.value);
 await click(document.querySelector('#language'));assert.equal(tab('community').textContent,'社区题解');assert.equal(requests.length,requestCount);assert.equal(document.querySelector('.community-composer textarea'),field);
 const allSolutions=readCommunityFeed(document.querySelector('.problem-solutions-heading a').getAttribute('href'));
 assert.equal(allSolutions.topicId,topic.id);assert.equal(allSolutions.state.solutions,true);assert.equal(allSolutions.state.sort,'top');
 topic={id:'finetuning/lora',number:22,title:'LoRA'};solution='new_reference_answer';await render();
 assert.equal(tab('community').getAttribute('aria-selected'),'true');assert(!document.querySelector('.community-composer'));assert(!pane('reference').querySelector('details').open);assert(!document.body.textContent.includes(solution));
 assert.equal(new URLSearchParams(requests.at(-1).split('?')[1]).get('problem_id'),topic.id);
 // Offline editions show the independent reference without any community request.
 available=false;const beforeOffline=requests.length;await render();assert(!tab('community'));assert(!pane('community'));assert(!pane('reference').hidden);assert.equal(requests.length,beforeOffline);
 solution=undefined;await render();await reveal();assert(pane('reference').textContent.includes('参考实现尚未提供'));assert(!pane('reference').querySelector('[data-reference-code]'));
 // Guests can read and have an explicit sign-in contribution link for this module.
 available=true;account.user=undefined;await render();assert(!pane('community').hidden);assert(document.querySelector('.problem-solutions-heading a[href*="account"]'));assert(!document.querySelector('.community-composer'));
 console.log('PASS problem solution panel: real scoped community feed and composer, separate spoiler-gated reference, stable draft/filter/editor state, publication isolation, keyboard/IME/locale, per-module reset, guest and offline editions');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();fs.rmSync(cache,{recursive:true,force:true});
 for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
