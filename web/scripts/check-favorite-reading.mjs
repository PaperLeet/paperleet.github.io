import assert from 'node:assert/strict';
import React,{act,useEffect,useState} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/favorites?q=a&state=todo',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__favoriteReading:undefined};
const old=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
let viewport=1250,base=1500,scrolls=[];
Object.defineProperty(window,'scrollY',{configurable:true,get:()=>viewport});
window.scrollTo=options=>{viewport=options.top;scrolls.push({kind:'row',y:viewport});};
dom.window.HTMLElement.prototype.scrollIntoView=function(){scrolls.push({kind:'list'});};
dom.window.HTMLElement.prototype.getBoundingClientRect=function(){
 const card=this.closest?.('[data-reading-post]'),index=card?[...document.querySelectorAll('[data-reading-post]')].indexOf(card):0;
 const top=card?base+index*180+(document.querySelector('.reading-return-notice')?42:0)-viewport:80;
 return {x:0,y:top,left:0,right:500,width:500,top,bottom:top+32,height:32,toJSON(){return this;}};
};
const server=await createServer({define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0')},server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'favorite-reading',enforce:'pre',resolveId(source){if(source==='./social')return '\0favorite-reading-account';},load(id){if(id==='\0favorite-reading-account')return "import React from 'react';export const useAccount=()=>globalThis.__favoriteReading.account;export const socialApi=(...args)=>globalThis.__favoriteReading.request(...args);export class SocialError extends Error{};export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);";}}]});
let root,model;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {ProblemFavoriteButton,ProblemFavoritesPage}=await server.ssrLoadModule('/src/ProblemFavorites.tsx');
 const {usePracticeReadingOrigin}=await server.ssrLoadModule('/src/usePracticeReadingOrigin.ts');
 const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 const nav=await server.ssrLoadModule('/src/libraryNavigation.ts');
 model=await server.ssrLoadModule('/src/readingReturn.ts');
 const owner='account:alice',scope='#/favorites?q=a&state=todo',originId='finetuning/lora';
 const problems=[{id:'architecture/attention',number:22,title:'Attention',difficulty:'M'},{id:originId,number:3,title:'LoRA',difficulty:'E'}];
 let ids=problems.map(row=>row.id),revision=1,delayRead=false,gates=[],codeState,filterChanges=[];
 const account={user:{id:'alice'},loading:false,capabilities:{problem_bookmarks:true},refresh:async()=>{}};
 globalThis.__favoriteReading={account,request:async(path,body)=>{
  assert.equal(path,'account/problem-bookmarks');assert.equal(body,undefined,'Reading navigation must not mutate favorites or progress');
  const value={owner_id:account.user.id,revision,items:(account.user.id==='alice'?ids:[]).map(id=>({problem_id:id,created_at:'2026-10-15T00:00:00Z'}))};
  if(delayRead)return new Promise(resolve=>gates.push(()=>resolve(value)));
  return value;
 }};
 function Host(){
  const [route,setRoute]=useState(()=>decodeProblemHash(location.hash)),[codes,setCodes]=useState({});codeState=codes;
  useEffect(()=>{const change=()=>setRoute(decodeProblemHash(location.hash));window.addEventListener('hashchange',change);return()=>window.removeEventListener('hashchange',change);},[]);
  const path=route.split('?')[0],problem=problems.find(row=>row.id===path||String(row.number)===path),back=nav.libraryReturnHref(route);
  usePracticeReadingOrigin(account.loading?undefined:account.user?'account:'+account.user.id:'guest',problem,back,route);
  if(path==='favorites')return React.createElement(ProblemFavoritesPage,{key:account.user?.id||'guest',route,problems,saved:{},progressReady:true,catalogReady:true,onRetryCatalog:()=>{},onFilters:href=>{filterChanges.push(href);window.history.replaceState(null,'',href);setRoute(decodeProblemHash(href));}});
  if(problem)return React.createElement('main',null,React.createElement(ProblemFavoriteButton,{problemId:problem.id}),React.createElement('textarea',{'aria-label':'Private code',value:codes[problem.id]||'Private learner code',onChange:event=>setCodes(previous=>({...previous,[problem.id]:event.target.value}))}),React.createElement('a',{id:'back',href:back},'Back'),React.createElement('a',{id:'next',href:nav.withLibraryContext(22,back)},'Next module'));
  return React.createElement('main',null,'Another route');
 }
 const render=async()=>act(async()=>{root??=createRoot(document.getElementById('root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Host)));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const settle=async predicate=>{for(let index=0;index<80&&!predicate();index++)await act(async()=>new Promise(resolve=>setTimeout(resolve,5)));assert(predicate(),'Favorite reading view did not settle');};
 async function navigate(href){await act(async()=>{const before=location.href;window.history.replaceState(null,'',href);window.dispatchEvent(new window.HashChangeEvent('hashchange',{oldURL:before,newURL:location.href}));});}
 const anchor=(id=originId)=>document.querySelector('[data-reading-post="'+id+'"] a[data-reading-link="practice"]');
 async function activate(element,{keyboard=true,...options}={}){
  assert(element);await act(async()=>{if(keyboard)element.focus();document.addEventListener('click',event=>event.preventDefault(),{once:true});element.dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true,cancelable:true,button:0,detail:keyboard?0:1,...options}));});
 }
 async function depart(keyboard=true){
  await stop();await act(async()=>model.discardReadingReturn());account.user={id:'alice'};account.loading=false;ids=problems.map(row=>row.id);revision++;delayRead=false;gates=[];
  await navigate(scope);base=1500;viewport=1250;scrolls=[];await render();await settle(()=>anchor()&&!document.querySelector('[aria-label="Refresh saved problems"]').disabled);
  const link=anchor(),top=link.getBoundingClientRect().top;await activate(link,{keyboard});
  const row=model.currentReadingReturn();assert(row);assert.equal(row.post,originId);assert.equal(row.practice,originId);assert.equal(row.href,scope);assert.equal(row.owner,owner);assert.equal(row.focus,keyboard);assert.equal(row.phase,'opening');
  await navigate(link.getAttribute('href'));await settle(()=>document.querySelector('textarea')&&model.currentReadingReturn()?.phase==='reading');
  return {row,top};
 }
 // The actual page captures its canonical row; next-module and numeric routes
 // keep the original row, and a notice committed above it cannot shift its return.
 const first=await depart();
 await act(async()=>{const input=document.querySelector('textarea');Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype,'value').set.call(input,'Keep this private code');input.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
 assert(!sessionStorage.getItem(model.READING_RETURN_KEY).includes('Keep this private code'));
 await navigate(document.getElementById('next').getAttribute('href'));assert.equal(model.currentReadingReturn().post,originId);assert(location.hash.startsWith('#/22?'));
 ids=[originId,'architecture/attention'];revision++;
 await act(async()=>window.dispatchEvent(new dom.window.Event('focus')));
 base=1900;scrolls=[];await navigate(scope);await settle(()=>document.querySelector('.reading-return-notice'));
 assert.equal(document.activeElement,anchor());assert.equal(anchor().getBoundingClientRect().top,first.top);assert.equal(scrolls.length,1);assert.equal(codeState[originId],'Keep this private code');assert(document.querySelector('.reading-return-notice').textContent.includes('saved problem'));assert.equal(model.currentReadingReturn(),undefined);
 await render();assert.equal(scrolls.length,1,'Routine rerenders do not repeat scrolling');
 // Pointer returns retain the viewport without taking keyboard focus.
 const pointer=await depart(false);base=2000;scrolls=[];await navigate(scope);await settle(()=>document.querySelector('.reading-return-notice'));assert.equal(anchor().getBoundingClientRect().top,pointer.top);assert.equal(document.activeElement,document.body);
 // A new focus/scroll intent while the account list loads keeps its place.
 await depart();delayRead=true;base=2100;scrolls=[];await navigate(scope);await settle(()=>gates.length===1);
 const search=document.querySelector('input[type=search]');await act(async()=>{search.focus();window.dispatchEvent(new dom.window.WheelEvent('wheel'));});
 delayRead=false;await act(async()=>gates.shift()());await settle(()=>document.querySelector('.reading-return-notice button'));
 assert.equal(scrolls.length,0);assert.equal(document.activeElement,search);await act(async()=>{const button=document.querySelector('.reading-return-notice button');button.focus();button.click();});assert.equal(scrolls.length,1);assert.equal(document.activeElement,anchor());
 // A confirmed removal while away returns to the list region, never another row.
 await depart();ids=['architecture/attention'];revision++;await act(async()=>window.dispatchEvent(new dom.window.Event('focus')));base=2200;scrolls=[];await navigate(scope);await settle(()=>document.querySelector('.reading-return-notice'));
 assert(!anchor());assert(document.querySelector('.reading-return-notice').textContent.includes('no longer in this view'));assert.equal(document.activeElement,document.querySelector('[data-reading-list]'));assert.deepEqual(scrolls,[{kind:'list'}]);
 // Filtering during a delayed return updates its full route and cancels auto-restore.
 await depart();delayRead=true;scrolls=[];await navigate(scope);await settle(()=>gates.length===1);
 await act(async()=>{const input=document.querySelector('input[type=search]');input.focus();Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype,'value').set.call(input,'Attention');input.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
 assert.equal(location.hash,'#/favorites?q=Attention&state=todo');assert.equal(filterChanges.at(-1),location.hash);delayRead=false;await act(async()=>gates.shift()());assert.equal(scrolls.length,0);assert(!document.querySelector('[data-reading-resumed]'));
 // Switching actors while Alice's response is delayed cannot reveal her row or return.
 await depart();delayRead=true;scrolls=[];await navigate(scope);await settle(()=>gates.length===1);const aliceRead=gates.shift();delayRead=false;account.user={id:'bob'};await render();await act(async()=>aliceRead());await settle(()=>document.querySelector('.favorites-empty'));
 assert.equal(document.querySelectorAll('[data-reading-post]').length,0);assert.equal(scrolls.length,0);assert.equal(model.currentReadingReturn(),undefined);
 // Modified and prevented activations never create a false return journey.
 await depart();await navigate(scope);await settle(()=>anchor());await act(async()=>model.discardReadingReturn());await activate(anchor(),{ctrlKey:true});assert.equal(model.currentReadingReturn(),undefined);
 document.addEventListener('click',event=>event.preventDefault(),{once:true,capture:true});await activate(anchor());assert.equal(model.currentReadingReturn(),undefined);
 await stop();model.discardReadingReturn();
 // Fresh module instances model reload validation of the bounded v1 metadata.
 const stored={version:1,id:90,owner,href:scope,post:originId,link:'practice',practice:originId,top:400,focus:true,at:Date.now(),phase:'reading'};
 let copy=0;
 async function hydrate(value){sessionStorage.setItem(model.READING_RETURN_KEY,JSON.stringify(value));return (await server.ssrLoadModule('/src/readingReturn.ts?favorite-reload='+(++copy))).currentReadingReturn();}
 assert.deepEqual(await hydrate(stored),stored);
 for(const patch of [{owner:'guest'},{post:'a'.repeat(32)},{practice:'architecture/attention'},{href:'#/favorites?user_id=bob'},{link:'title'},{code:'Do not accept code'},{at:Date.now()-7*60*60*1000}])assert.equal(await hydrate({...stored,...patch}),undefined);
 sessionStorage.removeItem(model.READING_RETURN_KEY);
 console.log('PASS actual favorite reading journey: canonical row capture, next/numeric practice continuity, private code retained without metadata persistence, notice-aware viewport/focus restoration, pointer behavior, delayed reads, explicit recovery after new user intent, removed-row list fallback, exact filter/App state, actor isolation and late responses, modified/prevented navigation, bounded owner-bound reload metadata');
}finally{if(root)await act(async()=>root.unmount());model?.discardReadingReturn();await server.close();dom.window.close();for(const [key,value] of old){if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}}
