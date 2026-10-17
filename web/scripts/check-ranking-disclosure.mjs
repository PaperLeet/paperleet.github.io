import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {readFileSync,mkdtempSync,rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__rankingFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const cache=mkdtempSync(path.join(os.tmpdir(),'paperleet-ranking-disclosure-'));
const server=await createServer({cacheDir:cache,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0')},plugins:[{
 name:'ranking-account-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0ranking-account-fixture';},
 load(id){if(id==='\0ranking-account-fixture')return `import React from 'react';
 export const useAccount=()=>globalThis.__rankingFixture.account;
 export const socialApi=(...args)=>globalThis.__rankingFixture.request(...args);
 export const errorMessage=e=>e.message;export class SocialError extends Error{}
 export const PersonLink=({user})=>React.createElement('span',null,user.display_name);
 export const DateLabel=({value})=>React.createElement('time',null,value);
 export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);`;}
}]});
let root;
const scrolls=[],pageScrolls=[];
dom.window.HTMLElement.prototype.scrollIntoView=function(){};
dom.window.HTMLElement.prototype.scrollTo=function({top}){this.scrollTop=top;scrolls.push({node:this,top});};
dom.window.scrollTo=options=>pageScrolls.push(options);
const originalSet=window.setTimeout.bind(window),originalClear=window.clearTimeout.bind(window),leases=new Set();
window.setTimeout=(fn,ms,...args)=>{const id=originalSet(fn,ms,...args);if(ms===8000)leases.add(id);return id;};
window.clearTimeout=id=>{leases.delete(id);return originalClear(id);};
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Ranking}=await server.ssrLoadModule('/src/InterviewQuestionRanking.tsx');
 const {emptyInterviewFilters,interviewHref}=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const catalog=JSON.parse(readFileSync(new URL('../public/data/index.json',import.meta.url),'utf8')).problems;
 const problems=catalog.slice(0,16),requests=[],cancelled=[],readyReturns=[];
 const questions=problems.map((problem,index)=>({problem_id:problem.id,reports:20-index,contributors:16-index,companies:[{company:'google',reports:20-index,contributors:16-index,problem_id:problem.id,latest:'2026-10-17'}]}));
 const account={user:{id:'alice',username:'alice',display_name:'Alice'},loading:false,capabilities:{community_bookmarks:true}};
 globalThis.__rankingFixture={account,request:async(url,body)=>{assert.equal(body,undefined,'List display must not publish or change grades');requests.push(url);return {items:[],total:0,limit:3,offset:0};}};
 let owner='alice',filters={...emptyInterviewFilters(),view:'questions',company:'google',days:'90',queue:'1,4,9'},props={questions,problems,saved:{[problems[8].id]:{code:'private draft',updated:'2026-10-17T10:00:00Z'}},ready:true},sourceReturn;
 const originalSaved=JSON.stringify(props.saved);
 let setLocale;
 function Language(){const i18n=useI18n();setLocale=i18n.setLocale;return React.createElement('button',{id:'language',onClick:()=>i18n.setLocale(i18n.locale==='en'?'zh':'en')},'Language');}
 const render=async()=>act(async()=>{
  root??=createRoot(document.getElementById('root'));
  root.render(React.createElement(LocaleProvider,null,React.createElement('div',{className:'app'},React.createElement('header',{className:'topbar'},React.createElement(Language)),React.createElement('textarea',{id:'private-code',defaultValue:'private code\n😀'}),React.createElement('div',{className:'description-scroll',style:{overflowY:'auto'}},React.createElement(Ranking,{key:owner+interviewHref(filters),...props,owner,filters,sourceReturn,onSourceReturnCancel:id=>cancelled.push(id),onSourceReturnReady:id=>readyReturns.push(id),onClearSearch(){}})))));
 });
 const rows=()=>[...document.querySelectorAll('.interview-ranking-row')],visible=()=>rows().filter(row=>!row.hidden),top=()=>document.querySelector('.interview-ranking-display:not(.interview-ranking-footer)'),footer=()=>document.querySelector('.interview-ranking-footer'),all=()=>top().querySelector('button'),compact=()=>top().querySelector('button+button'),bottomCompact=()=>footer().querySelector('button:last-child'),more=()=>document.querySelector('.interview-ranking-more');
 const click=async node=>{assert(node,'Missing list control');await act(async()=>{node.focus();node.click();});};
 const toggle=async details=>act(async()=>{details.open=true;details.dispatchEvent(new dom.window.Event('toggle'));});
 const waitFor=async predicate=>{const deadline=Date.now()+5000;while(!predicate()&&Date.now()<deadline)await act(async()=>{await new Promise(resolve=>setTimeout(resolve,10));});assert(predicate(),'Study guide did not settle');};
 location.hash=interviewHref(filters);await render();
 const privateCode=document.getElementById('private-code');privateCode.setSelectionRange(2,9,'backward');privateCode.scrollTop=88;privateCode.scrollLeft=31;
 const originalHref=location.href,originalStorage=JSON.stringify({...sessionStorage}),originalLocal=JSON.stringify({...localStorage});
 assert.equal(visible().length,6);assert.equal(rows().length,6,'Only the first page is mounted initially');assert.equal(requests.length,0);
 assert.equal(compact().getAttribute('aria-disabled'),'true');assert.equal(bottomCompact().getAttribute('aria-disabled'),'true');assert.equal(all().getAttribute('aria-disabled'),'false');
 assert.equal(all().getAttribute('aria-controls'),document.querySelector('.interview-ranking-list').id);assert(document.getElementById(all().getAttribute('aria-describedby')));
 await click(compact());assert.equal(visible().length,6);assert.equal(leases.size,0);
 // Two clicks in one React batch reveal one bounded page, rather than racing past it.
 await act(async()=>{more().focus();more().click();more().click();});assert.equal(visible().length,12);assert(document.activeElement===visible()[6].querySelector('.interview-ranking-title a'));
 const readerRow=visible()[8],guide=readerRow.querySelector('.preparation-guide');assert(guide);
 await toggle(guide);await waitFor(()=>guide.querySelector('.preparation-hint'));const hint=guide.querySelector('.preparation-hint');await toggle(hint);
 const studyBody=guide.querySelector('.preparation-guide-body');studyBody.scrollTop=37;
 await click(readerRow.querySelector('.interview-source-preview-toggle'));assert.equal(requests.length,1);const sourceToggle=readerRow.querySelector('.interview-source-preview-toggle'),sourcePanel=readerRow.querySelector('.interview-source-panel');assert.equal(sourceToggle.getAttribute('aria-expanded'),'true');
 const style=document.createElement('style');style.textContent=readFileSync('src/interviews.css','utf8')+'\n'+readFileSync('src/interview-ranking.css','utf8');document.head.append(style);
 const pane=document.querySelector('.description-scroll');Object.defineProperty(pane,'clientHeight',{value:400,configurable:true});Object.defineProperty(pane,'clientTop',{value:2,configurable:true});pane.getBoundingClientRect=()=>({top:100,bottom:502,height:402});pane.scrollTop=250;
 compact().getBoundingClientRect=()=>({top:10,bottom:50,height:40});
 scrolls.length=0;await click(bottomCompact());assert.equal(visible().length,6);assert.equal(rows().length,12);assert(readerRow.hidden);assert.equal(window.getComputedStyle(readerRow).display,'none');
 assert(document.activeElement===compact());assert.equal(scrolls.length,1);assert(scrolls[0].node===pane);assert.equal(scrolls[0].top,158);assert.equal(pageScrolls.length,0);
 assert(guide.open&&hint.open);assert(readerRow.querySelector('.preparation-guide-body')===studyBody);assert.equal(studyBody.scrollTop,37);assert.equal(sourceToggle.getAttribute('aria-expanded'),'true');assert.equal(requests.length,1);assert.equal(leases.size,0);
 const stableTop=compact();await click(all());assert.equal(visible().length,16);assert.equal(rows().length,16);assert(visible()[8]===readerRow);assert(readerRow.querySelector('.interview-source-panel')===sourcePanel);assert(guide.open&&hint.open);assert.equal(requests.length,1);assert(compact()===stableTop);assert.equal(all().getAttribute('aria-disabled'),'true');assert(!more());assert(document.activeElement===visible()[6].querySelector('.interview-ranking-title a'));
 const ranks=visible().map(row=>row.querySelector('.interview-rank').textContent);assert.deepEqual(ranks,Array.from({length:16},(_,index)=>String(index+1).padStart(2,'0')));
 await click(all());assert.equal(visible().length,16);assert.equal(leases.size,0);assert.equal(requests.length,1);
 await act(async()=>setLocale('zh'));assert(guide.open&&hint.open);assert(compact()===stableTop);assert.equal(compact().textContent,'只显示前 6 个');assert(readerRow.querySelector('.interview-source-panel')===sourcePanel);assert.equal(requests.length,1);await act(async()=>setLocale('en'));
 assert.equal(location.href,originalHref);assert.equal(JSON.stringify({...sessionStorage}),originalStorage);assert.equal(JSON.stringify({...localStorage}),originalLocal);assert.equal(JSON.stringify(props.saved),originalSaved);
 assert(document.getElementById('private-code')===privateCode);assert.deepEqual([privateCode.value,privateCode.selectionStart,privateCode.selectionEnd,privateCode.selectionDirection,privateCode.scrollTop,privateCode.scrollLeft],['private code\n😀',2,9,'backward',88,31]);
 // A later focus, scroll gesture, navigation or hidden document retires pending reveal focus.
 scrolls.length=0;await act(async()=>{bottomCompact().focus();bottomCompact().click();privateCode.focus();});assert.equal(visible().length,6);assert(document.activeElement===privateCode);assert.equal(scrolls.length,0);
 await act(async()=>{all().focus();all().click();document.dispatchEvent(new dom.window.WheelEvent('wheel',{bubbles:true}));});assert.equal(visible().length,16);assert(document.activeElement===all());assert.equal(leases.size,0);
 await act(async()=>{bottomCompact().focus();bottomCompact().click();location.hash='#/community';window.dispatchEvent(new dom.window.HashChangeEvent('hashchange'));location.hash=originalHref.split('#')[1];window.dispatchEvent(new dom.window.HashChangeEvent('hashchange'));});assert.equal(visible().length,6);assert(document.activeElement!==compact());assert.equal(leases.size,0);
 Object.defineProperty(document,'visibilityState',{value:'hidden',configurable:true});await click(all());assert.equal(visible().length,16);assert(document.activeElement===all());Object.defineProperty(document,'visibilityState',{value:'visible',configurable:true});
 // A source-return origin beyond the compact range remains reachable until an explicit collapse cancels it.
 await click(compact());sourceReturn={id:101,concept:problems[13].id};await render();assert.equal(visible().length,14);assert.equal(visible()[13].querySelector('.interview-source-preview-toggle').getAttribute('aria-expanded'),'true');
 const heldReturn=sourceReturn;await act(async()=>{bottomCompact().focus();bottomCompact().click();bottomCompact().click();});assert.equal(visible().length,6);assert.deepEqual(cancelled,[101]);await render();assert.equal(visible().length,6,'The same pending return cannot reopen a deliberately compacted list');
 sourceReturn={id:102,concept:problems[15].id};await render();assert.equal(visible().length,16);assert.equal(visible()[15].querySelector('.interview-source-preview-toggle').getAttribute('aria-expanded'),'true');sourceReturn=undefined;await render();assert.equal(visible().length,16,'Consumed source origins stay mounted');assert(heldReturn.id!==102);
 // Filter/owner/source changes use a new scope, while display controls never rewrite filters or evidence.
 filters={...filters,q:'no matching concept'};location.hash=interviewHref(filters);await render();assert.equal(visible().length,0);assert(!top());
 filters={...filters,q:''};location.hash=interviewHref(filters);await render();assert.equal(visible().length,6);assert.equal(rows().length,6);assert(!document.querySelector('.preparation-guide[open]'));assert.equal(document.querySelector('.interview-source-preview-toggle').getAttribute('aria-expanded'),'false');
 await click(all());owner='bob';account.user={...account.user,id:'bob',username:'bob'};await render();assert.equal(visible().length,6);assert.equal(rows().length,6);assert(!document.querySelector('.preparation-guide[open]'));
 props={...props,questions:[],problems:catalog};filters={...filters,company:'',queue:''};location.hash=interviewHref(filters);await render();assert.equal(visible().length,6);await click(all());assert(visible().length>16);assert(visible().every(row=>row.querySelector('.interview-rank').getAttribute('aria-label')==='Curated exercise'));assert(!document.querySelector('.interview-ranking-evidence'));await click(bottomCompact());assert.equal(visible().length,6);
 props={...props,questions:questions.slice(0,6),problems};filters={...filters,company:'google'};location.hash=interviewHref(filters);await render();assert.equal(visible().length,6);assert(!top()&&!footer(),'Short lists need no redundant display controls');
 assert.equal(leases.size,0);style.remove();
 console.log('PASS ranking list disclosure: bounded progressive/full expansion and compact top/footer controls, stable hidden row/guide/hint/source DOM and native focus, measured pane-only recovery, preserved real ranks/filters/batch/private code/caret/both-scroll/storage, no extra reads or writes, locale and newer-focus/wheel/route-ABA/visibility guards, source-return reveal/cancel/dedup and scoped reset, truthful curated/short-list behavior and lease cleanup');
}finally{
 if(root)await act(async()=>root.unmount());assert.equal(leases.size,0);window.setTimeout=originalSet;window.clearTimeout=originalClear;await server.close();dom.window.close();rmSync(cache,{recursive:true,force:true});
 for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
