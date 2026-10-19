import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';

const dom=new JSDOM('<div id="root"></div>',{url:'https://paperleet.test/#/community',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,location:dom.window.location,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-feed-filter-continuity-');
const server=await createServer({cacheDir,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:CommunityFilters}=await server.ssrLoadModule('/src/CommunityFeedFilters.tsx');
 const {default:InterviewFilters}=await server.ssrLoadModule('/src/InterviewActiveFilters.tsx');
 const nav=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const query='长关键词 🧭 <script> literal '.repeat(7),changes=[];
 const initialCommunity={category:'career',tag:'interview',following:true,saved:true,solutions:true,search:query,sort:'top',days:'7',page:3};
 const initialInterview={...nav.emptyInterviewFilters(),company:'google',role:'ml',stage:'technical',outcome:'offer',q:query,days:'90',view:'questions',saved:true,study:'todo',difficulty:'M',max_minutes:'30',location:'Remote',month_from:'2026-01',month_to:'2026-03',track:'llm',page:3};
 let kind='community',filters={...initialCommunity},actor='alice',allow=true,setLocale,afterChange;
 const expectedHref=location.href;
 const tree=()=>React.createElement(React.StrictMode,null,React.createElement(LocaleProvider,null,React.createElement(Frame)));
 function Frame(){
  setLocale=useI18n().setLocale;
  const props={filters,onChange:patch=>{changes.push({kind,patch});filters={...filters,page:1,...patch};root.render(tree());afterChange?.();},canAct:()=>allow&&actor==='alice'&&location.href===expectedHref,focusFallback:()=>document.getElementById('filter-search')};
  return React.createElement(React.Fragment,null,React.createElement('textarea',{id:'filter-private-code',defaultValue:'PRIVATE_FILTER_EDITOR\n'}),React.createElement('input',{id:'filter-search','aria-label':'Filter search',defaultValue:''}),React.createElement('section',{id:'filter-surface'},kind==='community'?React.createElement(CommunityFilters,props):React.createElement(InterviewFilters,{...props,problems:[]})));
 }
 const render=async()=>act(async()=>{root??=createRoot(document.getElementById('root'));root.render(tree());});
 const chips=()=>[...document.querySelectorAll(kind==='community'?'.community-feed-filter':'.interview-active-filters button:not(.text-button)')];
 const reset=()=>document.querySelector(kind==='community'?'.community-feed-reset':'.interview-active-filters .text-button');
 const click=async button=>{assert(button,'Missing filter control');await act(async()=>{button.focus();button.click();});};
 for(const name of ['community','interview']){
  kind=name;filters={...(kind==='community'?initialCommunity:initialInterview)};await render();
  const code=document.getElementById('filter-private-code');code.setSelectionRange(2,12,'backward');code.scrollTop=57;code.scrollLeft=13;
  const snapshot=[code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],storage=JSON.stringify({...sessionStorage});
  let buttons=chips(),next=buttons[2];await click(buttons[1]);assert(document.activeElement===next,'Removing a filter must focus the next surviving filter');assert(chips().includes(next));
  buttons=chips();const previousButton=buttons.at(-2);await click(buttons.at(-1));assert(document.activeElement===previousButton,'Removing the last filter must focus the previous surviving filter');
  while(chips().length){buttons=chips();next=buttons[1]||document.getElementById('filter-search');await click(buttons[0]);assert(document.activeElement===next);}
  assert.equal(filters.page,1);assert.equal(filters.saved,kind==='interview');if(kind==='community')assert(filters.solutions);else assert.equal(filters.view,'questions');
  filters={...(kind==='community'?initialCommunity:initialInterview)};await render();const search=document.getElementById('filter-search');await click(reset());assert(document.activeElement===search);assert.equal(chips().length,0);if(kind==='interview'){assert(filters.saved);assert.equal(filters.view,'questions');}else assert(filters.solutions);
  filters={...(kind==='community'?initialCommunity:initialInterview)};await render();const long=chips().find(node=>node.textContent.includes(query));assert(long);assert(long.title.includes(query));assert(long.getAttribute('aria-label').includes(query));assert(!long.querySelector('script'));
  await act(async()=>{long.focus();setLocale('zh');});assert(chips().includes(long));assert(document.activeElement===long);assert(long.getAttribute('aria-label').includes('移除筛选'));await act(async()=>setLocale('en'));
  console.log(`PASS ${kind} filter continuity: actual StrictMode filters support next/previous/final-search focus, clear-to-search, retained page reset and reading scope, full escaped long-value names/tooltips and bilingual native button identity`);
  const count=changes.length;code.focus();for(const attr of ['hidden','aria-hidden','inert']){
   const surface=document.getElementById('filter-surface');surface.setAttribute(attr,attr==='aria-hidden'?'true':'');await act(async()=>{chips()[0].click();reset().click();});assert.equal(changes.length,count);assert(document.activeElement===code);surface.removeAttribute(attr);
  }
  allow=false;await act(async()=>{chips()[0].click();reset().click();});assert.equal(changes.length,count);allow=true;
  actor='bob';await act(async()=>{chips()[0].click();reset().click();});assert.equal(changes.length,count);actor='alice';
  location.hash='#/favorites';await act(async()=>{chips()[0].click();reset().click();});assert.equal(changes.length,count);location.hash='#/community';
  await act(async()=>chips()[0].click());assert(document.activeElement===code,'Programmatic filter actions must yield to newer editor focus');
  filters={...(kind==='community'?initialCommunity:initialInterview)};await render();afterChange=()=>code.focus();await click(chips()[0]);assert(document.activeElement===code);afterChange=undefined;
  const detached=chips()[0];await act(async()=>root.unmount());root=undefined;const unmountedCount=changes.length;detached.click();assert.equal(changes.length,unmountedCount);
  assert.deepEqual([code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],snapshot);assert.equal(JSON.stringify({...sessionStorage}),storage);assert(!JSON.stringify(changes).includes(code.value));
  console.log(`PASS ${kind} filter action ownership: hidden/aria-hidden/inert, retired actor, stale route, rejected action and detached controls do not mutate; active/newer private editor focus, source, caret, two scroll positions and storage remain intact`);
 }
}finally{
 if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();
 for(const[key,value]of previous)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];
}
