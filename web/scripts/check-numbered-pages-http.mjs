import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';

const fixture=JSON.parse(process.env.PAPERLEET_NUMBERED_PAGES_HTTP||'null');
assert(fixture&&new URL(fixture.url).hostname==='127.0.0.1'&&fixture.ids.length===45,'A disposable loopback fixture with45 public posts is required');
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/community',pretendToBeVisual:true});
const reveals=[];dom.window.matchMedia=()=>({matches:false});dom.window.HTMLElement.prototype.scrollIntoView=function(){reveals.push(this);};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,location:dom.window.location,history:dom.window.history,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,requestAnimationFrame:fn=>setTimeout(fn,0),cancelAnimationFrame:clearTimeout,innerHeight:900,scrollY:0,scrollTo(){},IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');sessionStorage.setItem('private-numbered-pages-draft','KEEP');
const requests=[],mutations=[];
globalThis.__numberedPagesHTTP={account:{loading:false,capabilities:{community_solutions:true,community_profile_solutions:true,community_profile_interviews:true,community_feed_sorting:true,community_taxonomy:true,community_bookmarks:true,interview_outcome_filter:true,interview_context_filters:true},refresh:async()=>{}},async request(path,body,signal){
 if(body!==undefined){mutations.push({path});throw Error('Read-only paging must not write');}
 assert(!path.includes('PRIVATE_NUMBERED_CODE'));const row={path,aborted:false};requests.push(row);
 try{const response=await fetch(fixture.url+'/api/'+path,{signal,headers:{Accept:'application/json'}});const value=await response.json();assert.equal(response.status,200,JSON.stringify(value));return value;}
 catch(error){if(signal?.aborted)row.aborted=true;throw error;}
}};
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-numbered-http-'),server=await createServer({cacheDir,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0')},plugins:[{name:'numbered-pages-account',enforce:'pre',resolveId(source){if(source==='./social')return '\0numbered-pages-social';if(source==='./ProfileProgress'||source==='./ProfileAchievements')return '\0numbered-pages-above-fold';},load(id){
 if(id==='\0numbered-pages-above-fold')return `export default function UnrelatedPracticePanel(){return null;}`;
 if(id==='\0numbered-pages-social')return `import React from 'react';export const useAccount=()=>globalThis.__numberedPagesHTTP.account;export const socialApi=(...args)=>globalThis.__numberedPagesHTTP.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;
 }}]});
let root;
try{
 const {createRoot}=await import('react-dom/client'),{LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx'),{default:Community,UserProfile}=await server.ssrLoadModule('/src/Community.tsx'),{default:Interviews}=await server.ssrLoadModule('/src/Interviews.tsx');
 let mode='feed';
 const tree=()=>React.createElement(React.StrictMode,null,React.createElement(LocaleProvider,null,React.createElement('textarea',{id:'private-code',defaultValue:'PRIVATE_NUMBERED_CODE'}),mode==='feed'?React.createElement(Community,{route:location.hash}):mode==='profile'?React.createElement(UserProfile,{username:fixture.username,route:location.hash}):React.createElement(Interviews,{route:location.hash,problems:[],papers:[]})));
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(tree());});
 const wait=async predicate=>{for(let n=0;n<300&&!predicate();n++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),'Real HTTP paging did not settle');};
 const nav=()=>document.querySelector(mode==='interviews'?'.interview-pagination':'.community-pagination'),rows=()=>[...document.querySelectorAll(mode==='interviews'?'.interview-report-card[data-reading-post]':'.community-post-card[data-reading-post]')],ids=()=>rows().map(node=>node.dataset.readingPost),page=n=>nav()?.querySelector(`.numbered-pages button[aria-label="Page ${n}"]`);
 const result=()=>document.querySelector(mode==='interviews'?'.interview-results-heading':mode==='profile'?'#profile-community-panel':'.feed-items');
 await render();await wait(()=>rows().length===20&&!!page(3));const code=document.querySelector('#private-code');code.setSelectionRange(2,12,'backward');code.scrollTop=63;code.scrollLeft=14;
 const snapshot=[code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],stored=sessionStorage.getItem('private-numbered-pages-draft');
 const observed={};
 for(const kind of ['feed','profile','interviews']){
  mode=kind;history.replaceState(null,'',kind==='feed'?'#/community':kind==='profile'?'#/u/'+fixture.username:'#/interviews');await render();await wait(()=>rows().length===20&&!!page(3));assert.deepEqual(ids(),fixture.ids.slice(0,20));assert.equal(document.getElementById(page(3).getAttribute('aria-controls')),result());
  const reads=requests.length;await act(async()=>page(1).click());assert.equal(requests.length,reads,'Selecting the active page must not fetch');
  const destination=page(3);await act(async()=>{destination.focus();destination.click();});assert.equal(document.activeElement,result(),'Explicit paging focuses the persistent result region synchronously');
  code.focus();await render();await wait(()=>rows().length===5&&page(3)?.getAttribute('aria-current')==='page');assert.deepEqual(ids(),fixture.ids.slice(40,45));assert.equal(document.activeElement,code,'Late HTTP results yield to newer editor focus');
  assert.equal(nav().querySelector('[aria-label="Next page"]').disabled,true);assert.equal(document.querySelector('#private-code'),code);
  const retired=page(1),retiredReads=requests.length,retiredReveals=reveals.length;history.replaceState(null,'','#/favorites');await act(async()=>retired.click());assert.equal(location.hash,'#/favorites');assert.equal(requests.length,retiredReads);assert.equal(reveals.length,retiredReveals);assert.equal(document.activeElement,code);
  observed[kind]={page1:20,page3:5,last_ids:ids()};
 }
 assert.deepEqual([code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],snapshot);assert.equal(sessionStorage.getItem('private-numbered-pages-draft'),stored);assert.equal(mutations.length,0);
 console.log('NUMBERED_PAGES_HTTP '+JSON.stringify({views:observed,reads:requests.length,aborted_reads:requests.filter(row=>row.aborted).length,writes:mutations.length,private_code_unchanged:true,retired_routes_rejected:true,exact_cohorts:true}));
 console.log('PASS real HTTP numbered pages: actual community/profile/experience components, exact three-page SQLite-backed cohorts, selected/disabled states, result associations, synchronous focus and later editor priority, current-page no-op and retired-route rejection with no writes');
}finally{if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});delete globalThis.__numberedPagesHTTP;dom.window.close();for(const[key,value]of previous)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}
