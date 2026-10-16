import assert from "node:assert/strict";
import React, { act } from "react";
import { JSDOM } from "jsdom";
import { createServer } from "vite";

const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: 'http://paperleet.test/#/u/alice', pretendToBeVisual: true });
dom.window.matchMedia = () => ({ matches: false });
const scrolls = [];
dom.window.HTMLElement.prototype.scrollIntoView = function (options) { scrolls.push({ element: this, options }); };
const globals = {
  window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
  HTMLElement: dom.window.HTMLElement, localStorage: dom.window.localStorage,
  sessionStorage: dom.window.sessionStorage, location: dom.window.location,
  requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
  IS_REACT_ACT_ENVIRONMENT: true, __profileFixture: undefined,
};
const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
const timezone = process.env.TZ;
localStorage.setItem('paperleet.locale', 'en');
const errors = [];
dom.window.addEventListener('error', event => errors.push(event.error));
const server = await createServer({
  server: { middlewareMode: true, hmr: false }, appType: "custom",
  optimizeDeps: { noDiscovery: true, include: [] },
  plugins: [{
    name: 'profile-connections-fixture', enforce: 'pre',
    resolveId(source) {
      if (source === './social') return '\0profile-social';
      if (source === './ProfileProgress' || source === './ProfileAchievements') return '\0profile-above-fold';
    },
    load(id) {
      if (id === '\0profile-above-fold') return `import React from 'react';export default function Progress(){return React.createElement('section',{'data-profile-progress':true,style:{height:600}},'Practice progress');}`;
      if (id === '\0profile-social') return `import React from 'react';
        export const useAccount=()=>globalThis.__profileFixture.account;
        export const socialApi=(...args)=>globalThis.__profileFixture.request(...args);
        export const errorMessage=e=>e.message;export class SocialError extends Error{}
        export const Avatar=({user})=>React.createElement('span',null,user.display_name);
        export const profileHref=user=>'#/u/'+user.username;
        export const PersonLink=({user})=>React.createElement('a',{href:profileHref(user)},user.display_name);
        export const DateLabel=({value})=>React.createElement('time',null,value);
        export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);
        export const TestAccountBadge=()=>null;`;
    },
  }],
});
let root;
try {
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {UserProfile}=await server.ssrLoadModule('/src/Community.tsx');
 const alice={id:'alice',username:'alice',display_name:'Alice',created_at:'2026-01-01',avatar_color:'#123456'};
 const counts={posts:24,solutions:21,replies:0,likes:0,reposts:0,following:0,followers:0,likes_received:0,reposts_received:0};
 const topic={id:'architecture/attention',title:'注意力',title_en:'Attention',number:1};
 const post={id:'a'.repeat(32),title:'Attention derivation',body:'## Approach\n\nA **clear** explanation.\n\n```python\nreturn q @ k.T\n```',solution:true,category:'implementation',problem_id:topic.id,problem:topic,author:alice,counts:{likes:2,replies:0,reposts:0},viewer:{},created_at:'2026-10-16',updated_at:'2026-10-16'};
 const requests=[];let mode='normal',held;
 const account={user:alice,capabilities:{community_solutions:true},loading:false};
 globalThis.__profileFixture={account,request:async(path,payload,signal)=>{
  assert.equal(payload,undefined,'Reading solutions never publishes or edits');requests.push({path,signal});
  const url=new URL(path,'https://fixture.test/');
  if(url.pathname.endsWith('/activity')){
   if(url.searchParams.get('tab')!=='solutions')return {items:[],total:0};
   if(mode==='hold')return new Promise((resolve,reject)=>{held={resolve,reject};});
   if(mode==='fail')throw Error('Profile solutions unavailable');
   const second=url.searchParams.get('offset')==='20';
   const row={kind:mode==='wrong-kind'?'posts':'solutions',created_at:post.created_at,post:mode==='ordinary'?{...post,solution:false}:mode==='missing-module'?{...post,problem_id:undefined}:{...post,title:second?'Second-page solution':post.title}};
   return {items:mode==='empty'||mode==='shrink'&&second?[]:[row],total:mode==='empty'?0:mode==='shrink'?20:21,...(mode==='no-ack'?{}:{solutions:true})};
  }
  if(url.pathname.endsWith('/connections'))return {items:[],total:0};
  assert.equal(url.pathname,'/community/profiles/alice');
  return {...alice,counts,viewer:{is_self:account.user?.id==='alice',following:false}};
 }};
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(UserProfile,{username:'alice'})));});
 const stop=async()=>{await act(async()=>root.unmount());root=undefined;};
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
 const panel=()=>document.querySelector('#profile-community-panel');
 const tab=()=>document.querySelector('#profile-tab-solutions');
 const counter=()=>document.querySelector('[aria-label="View solutions"]');
 const lastParams=()=>new URL(requests.at(-1).path,'https://fixture.test/').searchParams;
 const key=async(key,extra={})=>act(async()=>document.activeElement.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key,bubbles:true,cancelable:true,...extra})));
 // Feed support alone must not expose a profile operation an older server lacks.
 await render();assert(!tab());assert(!counter());assert(!requests.some(row=>row.path.includes('tab=solutions')));
 account.capabilities.community_profile_solutions=true;await render();
 assert(tab());assert.equal(tab().getAttribute('aria-selected'),'false');assert.equal(counter().textContent,'21');
 assert(!requests.some(row=>row.path.includes('tab=solutions')),'Keep solution data deferred until selected');
 await click(counter());
 assert.equal(lastParams().get('tab'),'solutions');assert.equal(lastParams().get('offset'),'0');
 assert.equal(document.activeElement,tab());assert.equal(tab().getAttribute('aria-selected'),'true');assert.equal(panel().getAttribute('aria-labelledby'),tab().id);
 assert.equal(location.hash,'#/u/alice');assert(panel().textContent.includes('Anonymous solutions appear here only'));
 assert(document.querySelector('.solution-reading-card'));assert(document.querySelector('.community-card-interaction'));assert.equal(document.querySelectorAll('.profile-tabs [tabindex="0"]').length,1);
 const oldCount=requests.length;await click(counter());assert.equal(requests.length,oldCount);
 const disclosure=document.querySelector('.solution-reading-card>details');await act(async()=>{disclosure.open=true;disclosure.dispatchEvent(new dom.window.Event('toggle'));});
 assert(document.querySelector('.discussion-reading'));assert(panel().textContent.includes('clear'));assert.equal(location.hash,'#/u/alice');
 await click(document.querySelector('.solution-reading-collapse'));assert(!document.querySelector('.solution-reading-card>details').open);
 const selected=tab();await click(selected);const navigationCount=requests.length;
 await key('ArrowRight',{ctrlKey:true});await key('ArrowRight',{isComposing:true});await key('ArrowRight',{keyCode:229});assert.equal(document.activeElement,selected);assert.equal(requests.length,navigationCount);
 await key('ArrowRight');assert.equal(document.activeElement.id,'profile-tab-replies');
 await key('ArrowLeft');assert.equal(document.activeElement,tab());
 await click(document.querySelector('[aria-label="Next page"]'));assert.equal(lastParams().get('offset'),'20');assert(panel().textContent.includes('Second-page solution'));
 await click(counter());assert.equal(lastParams().get('offset'),'0');assert(!panel().textContent.includes('Second-page solution'));
 mode='shrink';await click(document.querySelector('[aria-label="Next page"]'));assert.equal(lastParams().get('offset'),'0');assert(!panel().textContent.includes('Second-page solution'));assert(!document.querySelector('[aria-label="Next page"]'));mode='normal';
 // Reject a silently unfiltered response rather than presenting ordinary posts as solutions.
 for(const failure of ['no-ack','wrong-kind','ordinary','missing-module']){
  await click(document.querySelector('#profile-tab-posts'));mode=failure;await click(tab());
  assert(panel().querySelector('[role="alert"]'),failure);assert(!panel().querySelector('.post-title'));
 }
 mode='fail';await click(panel().querySelector('.feed-error button'));assert(panel().textContent.includes('Profile solutions unavailable'));
 mode='normal';await click(panel().querySelector('.feed-error button'));assert(panel().querySelector('.solution-reading-card'));
 // A stalled activity read becomes recoverable, and late success cannot replace the retry.
 await click(document.querySelector('#profile-tab-posts'));
 const nativeSetTimeout=globalThis.setTimeout;let deadline;
 try{
  globalThis.setTimeout=(callback,delay,...args)=>{if(delay===20000)deadline=callback;return nativeSetTimeout(callback,delay,...args);};
  mode='hold';await click(tab());const stalled=held,stalledSignal=requests.at(-1).signal;assert(deadline);
  await act(async()=>deadline());assert(stalledSignal.aborted);assert.equal(panel().getAttribute('aria-busy'),'false');assert(panel().textContent.includes('timed out'));
  mode='normal';await click(panel().querySelector('.feed-error button'));assert(panel().querySelector('.solution-reading-card'));
  await act(async()=>stalled.resolve({items:[],total:0,solutions:true}));assert(panel().querySelector('.solution-reading-card'));
 }finally{globalThis.setTimeout=nativeSetTimeout;}
 // A pending request is cancelled on tab/account/capability changes and cannot fill the new view.
 await click(document.querySelector('#profile-tab-posts'));mode='hold';await click(tab());const firstHeld=held,firstSignal=requests.at(-1).signal;assert.equal(panel().getAttribute('aria-busy'),'true');assert(!document.querySelector('[aria-label="Next page"]'));
 mode='normal';await click(document.querySelector('#profile-tab-posts'));assert(firstSignal.aborted);
 await act(async()=>firstHeld.resolve({items:[{kind:'solutions',post}],total:1,solutions:true}));assert(!panel().querySelector('.solution-reading-card'));
 mode='hold';await click(tab());const secondHeld=held,secondSignal=requests.at(-1).signal;account.capabilities.community_profile_solutions=false;mode='normal';await render();assert(secondSignal.aborted);assert(!tab());
 await act(async()=>secondHeld.resolve({items:[{kind:'solutions',post}],total:1,solutions:true}));assert.equal(document.querySelector('#profile-tab-posts').getAttribute('aria-selected'),'true');assert(!panel().querySelector('.solution-reading-card'));
 account.capabilities.community_profile_solutions=true;await render();mode='hold';await click(tab());const thirdHeld=held,thirdSignal=requests.at(-1).signal;
 account.user={...alice,id:'bob',username:'bob'};mode='empty';await render();assert(thirdSignal.aborted);assert.equal(tab().getAttribute('aria-selected'),'false');
 await act(async()=>thirdHeld.resolve({items:[{kind:'solutions',post}],total:1,solutions:true}));assert(!panel().querySelector('.solution-reading-card'));
 await click(tab());assert(panel().textContent.includes('No solutions yet'));assert(!panel().querySelector('.profile-solutions-note'));assert(!panel().querySelector('a[href="#/community?solutions=1"]'));
 account.user=undefined;await render();await click(tab());assert(!panel().querySelector('.profile-solutions-note'));
 // Empty self view, bilingual counter/tab and contribution entry use the same canonical feed.
 await stop();localStorage.setItem('paperleet.locale','zh');account.user=alice;counts.solutions=0;await render();
 assert(document.querySelector('[aria-label="查看题解"]'));assert.equal(tab().textContent,'题解0');await click(tab());assert(panel().textContent.includes('还没有题解'));assert(panel().querySelector('a[href="#/community?solutions=1"]'));
 assert.equal(document.querySelectorAll('.profile-tabs [tabindex="0"]').length,1);assert.deepEqual(errors,[]);
 console.log('PASS profile solutions: deferred capability-gated contribution counts/tab, scoped server acknowledgement, native in-place reading and actions, first-page shortcuts/pagination, private anonymous notice, owner/visitor empty states, bilingual/keyboard/IME controls, bounded timeout/retry, request abort and stale tab/account/capability isolation');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
 if(timezone===undefined)delete process.env.TZ;else process.env.TZ=timezone;
 for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
