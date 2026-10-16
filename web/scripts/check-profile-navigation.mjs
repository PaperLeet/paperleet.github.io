import assert from "node:assert/strict";
import React, { act } from "react";
import { JSDOM } from "jsdom";
import { createServer } from "vite";

const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: 'http://paperleet.test/#/u/alice', pretendToBeVisual: true });
dom.window.matchMedia = () => ({ matches: false });
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
let root,model;
let viewport=0,base=1500,scrolls=[];
Object.defineProperty(window,'scrollY',{get:()=>viewport,configurable:true});window.scrollTo=(options,y)=>{viewport=typeof options==='object'?options.top:y;scrolls.push(viewport);};
dom.window.HTMLElement.prototype.scrollIntoView=function(){viewport=base-100;scrolls.push(viewport);};
dom.window.HTMLElement.prototype.getBoundingClientRect=function(){const notice=document.querySelector('.reading-return-notice');const top=base+(notice?(notice.querySelector('button')?76:42):0)+(this.dataset?.readingLink==='read'?180:0)-viewport;return {x:0,y:top,left:0,right:400,width:400,height:30,top,bottom:top+30,toJSON(){return this;}};};
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {UserProfile,default:Community}=await server.ssrLoadModule('/src/Community.tsx');
 const nav=await server.ssrLoadModule('/src/profileNavigation.ts'),community=await server.ssrLoadModule('/src/communityNavigation.ts'),accountNav=await server.ssrLoadModule('/src/accountNavigation.ts'),{decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 model=await server.ssrLoadModule('/src/readingReturn.ts');
 const id='a'.repeat(32),reply='b'.repeat(32),tabs=['posts','solutions','replies','likes','reposts','following','followers'];
 for(const tab of tabs)for(const page of [1,2,450359962737049]){
  const view={username:'alice',tab,page},href=nav.profileFeedHref('alice',view),thread=community.communityPostHref(id,href,reply);
  assert.deepEqual(nav.readProfileRoute(href),view);assert.deepEqual(nav.readProfileRoute(decodeProblemHash(href)),view);
  assert.equal(accountNav.accountReturnHref(decodeProblemHash(accountNav.signInHref(href))),href);
  assert.equal(community.communityPostReturn(thread,id),href);assert.equal(accountNav.accountReturnHref(decodeProblemHash(accountNav.signInHref(thread))),thread);
  assert.equal(community.communityReplyTarget(thread,id),reply);assert.equal(community.communityShareUrl('https://paperleet.test/?private=query'+thread,id),'https://paperleet.test/#/community/'+id);
 }
 assert.equal(nav.profileFeedHref('alice'),'#/u/alice');assert.deepEqual(nav.readProfileRoute('u/alice?tab=posts&page=1'),{username:'alice',tab:'posts',page:1});
 assert(nav.sameProfilePage('#/u/alice?tab=solutions&page=2','u/alice?tab=followers'));assert(!nav.sameProfilePage('#/u/alice','#/u/bobby'));assert(!nav.sameProfilePage('#/community','#/u/alice'));
 for(const bad of ['u/ab','u/ALICE','u/alice/extra','u/alice?tab=','u/alice?tab=unknown','u/alice?tab=solutions&tab=posts','u/alice?page=0','u/alice?page=01','u/alice?page=1.5','u/alice?page=450359962737050','u/alice?page=2&page=3','u/alice?from=community','u/alice?code=private','u/alice?tab=%GG','u/alice?tab=solutions%0A','https://evil.test/u/alice','u/%2561lice?tab=posts']){
  assert.equal(nav.readProfileRoute(bad),undefined,bad);assert.equal(accountNav.signInHref(bad),'#/account',bad);assert.equal(community.communityPostReturn('#/community/'+id+'?'+new URLSearchParams({from:bad}),id),undefined,bad);
 }
 console.log('PASS profile navigation contracts: seven canonical public categories, safe bounded pages, exact post/reply/sign-in returns, source-free share links and invalid/duplicate/recursive/external context rejection');
 const alice={id:'alice',username:'alice',display_name:'Alice',created_at:'2026-01-01',avatar_color:'#123456'},topic={id:'architecture/attention',number:1,title:'Attention'};
 const counts={posts:22,solutions:22,replies:1,likes:1,reposts:1,following:1,followers:1};
 const post={id,title:'Profile solution',body:'## Approach\nA complete derivation.',solution:true,category:'implementation',problem_id:topic.id,problem:topic,author:alice,counts:{likes:0,reposts:0,replies:0},viewer:{},created_at:'2026-10-16',updated_at:'2026-10-16'};
 const account={user:alice,loading:false,capabilities:{community_profile_solutions:true,community_solutions:true}},requests=[];let hold=false,held,anonymous=false;
 globalThis.__profileFixture={account,request:async(path,payload,signal)=>{
  assert.equal(payload,undefined);requests.push({path,signal});const url=new URL(path,'https://fixture.test/');
  const shown=anonymous?{...post,anonymous:true,author:{...alice,id:'anonymous:'+id,username:'',display_name:'Anonymous',anonymous:true}}:post;
  if(url.pathname==='/community/posts/'+id)return {...shown,replies:[]};
  if(url.pathname.endsWith('/activity')){
   const tab=url.searchParams.get('tab');const result={items:tab==='replies'?[reply,'c'.repeat(32)].map(replyId=>({kind:tab,post:shown,reply:{id:replyId,body:'A follow-up',created_at:post.created_at},created_at:post.created_at})):[{kind:tab,post:shown,created_at:post.created_at}],total:22,...(tab==='solutions'?{solutions:true}:{})};
   if(hold)return new Promise(resolve=>held={resolve,result,signal});return result;
  }
  if(url.pathname.endsWith('/connections'))return {items:[],total:0};
  assert.equal(url.pathname,'/community/profiles/alice');return {...alice,counts,viewer:{is_self:account.user?.id==='alice',following:false}};
 }};
 function Journey(){
  const [route,setRoute]=React.useState(location.hash);
  React.useEffect(()=>{let before=location.hash;const change=()=>{if(!nav.sameProfilePage(before,location.hash))window.scrollTo(0,0);before=location.hash;setRoute(location.hash);};window.addEventListener('hashchange',change);return()=>window.removeEventListener('hashchange',change);},[]);
  const profile=nav.readProfileRoute(route);return profile?React.createElement(UserProfile,{username:profile.username,route}):React.createElement(Community,{postId:id,route});
 }
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Journey)));});
 const stop=async()=>{await act(async()=>root.unmount());root=undefined;};
 const wait=async()=>act(async()=>new Promise(resolve=>setTimeout(resolve,10)));
 const waitFor=async(predicate,message)=>{const deadline=Date.now()+2000;while(!predicate()&&Date.now()<deadline)await wait();assert(predicate(),message);};
 const navigate=async href=>act(async()=>{location.hash=href;await new Promise(resolve=>setTimeout(resolve,10));});
 const click=async(node,{keyboard=false}={})=>act(async()=>{assert(node);if(keyboard)node.focus();node.dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true,cancelable:true,button:0,detail:keyboard?0:1}));await new Promise(resolve=>setTimeout(resolve,10));});
 const anchor=()=>document.querySelector('.solution-reading-page');
 // A copied/new-tab link and remount both load the intended category/page directly.
 const scope=nav.profileFeedHref('alice',{tab:'solutions',page:2});await navigate(scope);viewport=1250;await render();
 assert.equal(document.querySelector('#profile-tab-solutions').getAttribute('aria-selected'),'true');assert(requests.some(row=>row.path.includes('tab=solutions')&&row.path.includes('offset=20')));assert.equal(viewport,1250);
 await stop();await render();assert.equal(document.querySelector('#profile-tab-solutions').getAttribute('aria-selected'),'true');assert(requests.at(-1).path.includes('offset=20'));
 // Newer manual scrolling takes priority over a counter's deferred section reveal.
 await navigate('#/u/alice');viewport=1250;scrolls=[];
 await act(async()=>{const counter=document.querySelector('[aria-label="View solutions"]');counter.focus();counter.click();viewport=800;window.dispatchEvent(new dom.window.WheelEvent('wheel',{deltaY:40}));await new Promise(resolve=>setTimeout(resolve,20));});
 assert.equal(viewport,800);assert.equal(scrolls.length,0);assert.equal(document.querySelector('#profile-tab-solutions').getAttribute('aria-selected'),'true');
 await navigate(scope);viewport=1250;
 // Full thread navigation retains the exact source and restores the actual full-page link after delayed data.
 base=1500;viewport=1250;await click(anchor(),{keyboard:true});
 await waitFor(()=>model.currentReadingReturn()?.phase==='reading'&&!!document.querySelector('.social-back'),'Full thread navigation did not settle');
 const origin=model.currentReadingReturn();assert(origin);assert.equal(origin.href,scope);assert.equal(origin.link,'read');assert.equal(origin.phase,'reading');assert(document.querySelector('.social-back').textContent.includes('Back to profile'));
 assert.equal(document.querySelector('.social-back').getAttribute('href'),scope);
 hold=true;base=1800;await click(document.querySelector('.social-back'));
 await waitFor(()=>model.currentReadingReturn()?.phase==='returning'&&!!document.querySelector('#profile-community-panel'),'Profile return navigation did not settle');
 assert.equal(location.hash,scope);assert.equal(model.currentReadingReturn().phase,'returning');assert.equal(document.querySelector('#profile-community-panel').getAttribute('aria-busy'),'true');assert(!anchor());
 scrolls=[];hold=false;await act(async()=>held.resolve(held.result));assert.equal(anchor().getBoundingClientRect().top,origin.top);assert.equal(document.activeElement,anchor());assert(document.querySelector('[data-reading-resumed="true"]'));assert.equal(scrolls.length,1);
 // Anonymous self-profile context remains in the current account session, never a public post URL.
 anonymous=true;await stop();await render();const privateLink=anchor();assert.equal(privateLink.getAttribute('href'),'#/community/'+id);assert.equal(community.communityPostReturn(privateLink.getAttribute('href'),id),undefined);
 viewport=1250;await click(privateLink,{keyboard:true});assert.equal(location.hash,'#/community/'+id);assert.equal(model.currentReadingReturn().privateProfile,true);assert.equal(model.privateProfileReadingReturn('account:bob',id,location.hash),undefined);assert.equal(model.privateProfileReadingReturn(undefined,id,location.hash),undefined);
 assert.equal(document.querySelector('.social-back').getAttribute('href'),scope);await click(document.querySelector('.social-back'));assert.equal(location.hash,scope);assert.equal(document.activeElement,anchor());
 // The same source-free URL in a fresh session exposes no private profile return.
 await click(anchor());await stop();model.discardReadingReturn();await render();assert.notEqual(document.querySelector('.social-back').getAttribute('href'),scope);
 await stop();anonymous=false;await navigate(scope);await render();
 // Browser history changes the category/page without replacing the whole profile or scrolling to its top.
 const identity=document.querySelector('.profile-header');viewport=900;scrolls=[];await navigate('#/u/alice?tab=likes&page=2');assert.equal(document.querySelector('.profile-header'),identity);assert.equal(document.querySelector('#profile-tab-likes').getAttribute('aria-selected'),'true');assert.equal(scrolls.length,0);
 await act(async()=>{window.history.back();await new Promise(resolve=>setTimeout(resolve,20));});assert.equal(location.hash,scope);assert.equal(document.querySelector('#profile-tab-solutions').getAttribute('aria-selected'),'true');assert.equal(viewport,900);
 await navigate('#/u/alice?tab=replies&page=2');const replyLink=document.querySelector('.profile-reply-card .activity-label a');assert.equal(community.communityReplyTarget(replyLink.getAttribute('href'),id),reply);assert.equal(community.communityPostReturn(replyLink.getAttribute('href'),id),'#/u/alice?tab=replies&page=2');
 // Distinct reply IDs select the exact source row, including anonymous owner-only context.
 anonymous=true;await stop();await render();const privateReply=document.querySelectorAll('.profile-reply-card .activity-label a')[1];assert.equal(privateReply.getAttribute('href'),'#/community/'+id+'?reply='+'c'.repeat(32));
 await click(privateReply,{keyboard:true});assert.equal(model.currentReadingReturn().reply,'c'.repeat(32));assert.equal(model.currentReadingReturn().privateProfile,true);assert.equal(document.querySelector('.social-back').getAttribute('href'),'#/u/alice?tab=replies&page=2');await click(document.querySelector('.social-back'));assert.equal(document.activeElement,document.querySelectorAll('.profile-reply-card .activity-label a')[1]);assert.equal(document.querySelectorAll('.profile-reply-card[data-reading-resumed="true"]').length,1);anonymous=false;
 // A late empty page cannot redirect after a newer URL is already visible but before router cleanup.
 await stop();hold=true;await navigate(scope);await render();const stale=held;
  window.history.replaceState(null,'','#/settings');await act(async()=>stale.resolve({...stale.result,items:[],total:0}));assert.equal(location.hash,'#/settings','A stale profile response must not replace a newer destination');
 hold=false;
 // Resolve capabilities before a copied Solutions link; an unsupported server falls back canonically.
 await stop();await navigate(scope);account.loading=true;const before=requests.length;await render();assert(!requests.slice(before).some(row=>row.path.includes('/activity')));account.loading=false;await render();assert(requests.at(-1).path.includes('tab=solutions'));
 account.capabilities.community_profile_solutions=false;await render();await wait();assert.equal(location.hash,'#/u/alice');assert.equal(document.querySelector('#profile-tab-posts').getAttribute('aria-selected'),'true');assert(!document.querySelector('#profile-tab-solutions'));
 assert.deepEqual(errors,[]);
 console.log('PASS real profile reading journey: copied/reloaded category/page, controlled route updates, stable profile and browser Back, full thread/reply source, anonymous session-only returns and source-free URLs, delayed exact reading/focus restoration, capability hydration and canonical unsupported fallback without publication');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();if(timezone===undefined)delete process.env.TZ;else process.env.TZ=timezone;for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
