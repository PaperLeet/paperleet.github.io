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
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {UserProfile}=await server.ssrLoadModule('/src/Community.tsx');
 const nav=await server.ssrLoadModule('/src/profileNavigation.ts'),community=await server.ssrLoadModule('/src/communityNavigation.ts'),accountNav=await server.ssrLoadModule('/src/accountNavigation.ts');
 const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 const alice={id:'alice',username:'alice',display_name:'Alice',created_at:'2026-01-01',avatar_color:'#123456'};
 const account={user:alice,loading:false,capabilities:{community_profile_solutions:true}};
 const counts={posts:24,interviews:21,solutions:1,replies:1,likes:1,reposts:1,following:1,followers:1,likes_received:2,reposts_received:1};
 const details={company:'amazon',role:'ml',stage:'technical',outcome:'offer',interview_month:'2026-09',location:'Remote',questions:['architecture/attention']};
 const post={id:'a'.repeat(32),title:'Attention interview experience',body:'## Reflection\n\nMy **own** account of the interview.',category:'career',interview:details,problem_id:null,problem:null,solution:false,author:alice,counts:{likes:2,replies:0,reposts:0},viewer:{liked:false,reposted:false,can_delete:false},created_at:'2026-10-17',updated_at:'2026-10-17'};
 let mode='normal',hold=false;const requests=[],pending=[],mutations=[];
 function activity(path){const query=new URL(path,'https://fixture.test/').searchParams,offset=Number(query.get('offset'));
  if(query.get('tab')!=='interviews')return {items:[],total:0,limit:20,offset};
  const anonymous=mode==='anonymous'||mode==='public-anonymous',id=anonymous?'c'.repeat(32):offset?'b'.repeat(32):post.id;
  const row={kind:mode==='wrong-kind'?'posts':'interviews',created_at:post.created_at,post:{...post,id,title:offset?'Second-page experience':post.title,...(mode==='ordinary'?{interview:undefined}:{}),...(anonymous?{anonymous:true,author:{...alice,id:'anonymous:'+id,username:'',display_name:'Anonymous',anonymous:true,affiliation:{domain:'test.edu',kind:'education',name:'Test University'}}}:{})}};
  return {items:mode==='empty'||mode==='shrink'&&offset?[]:mode==='mixed'?[row,{...row,post:{...post,interview:undefined}}]:[row],total:mode==='empty'?0:mode==='shrink'?1:21,limit:mode==='bad-limit'?30:20,offset:mode==='bad-offset'?offset+1:offset,...(mode==='no-ack'?{}:{interviews:true})};
 }
 globalThis.__profileFixture={account,request(path,body,signal){if(body){mutations.push({path,body});throw Error('Reading experiences cannot publish or change grades');}requests.push({path,signal});if(path.includes('/activity?'))return hold?new Promise((resolve,reject)=>pending.push({path,signal,resolve,reject})):Promise.resolve(activity(path));if(path.includes('/connections?'))return Promise.resolve({items:[],total:0});assert.equal(path,'community/profiles/alice');return Promise.resolve({...alice,counts:{...counts},viewer:{is_self:account.user?.id==='alice',following:false}});}};
 function Locale(){const {locale,setLocale}=useI18n();return React.createElement('button',{id:'locale',onClick:()=>setLocale(locale==='en'?'zh':'en')},'locale');}
 function ProfileRoute(){const [route,setRoute]=React.useState(location.hash);React.useEffect(()=>{const changed=()=>setRoute(location.hash);window.addEventListener('hashchange',changed);return()=>window.removeEventListener('hashchange',changed);},[]);return React.createElement(UserProfile,{username:'alice',route});}
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Locale),React.createElement('textarea',{'aria-label':'Practice code',defaultValue:'def solve(): return 42'}),React.createElement(ProfileRoute)));});
 const navigate=async(tab,page=1)=>act(async()=>{location.hash=nav.profileFeedHref('alice',{tab,page});await new Promise(resolve=>setTimeout(resolve,5));});
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();await new Promise(resolve=>setTimeout(resolve,5));});};
 const key=async value=>act(async()=>{document.activeElement.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:value,bubbles:true,cancelable:true}));await new Promise(resolve=>setTimeout(resolve,5));});
 const tab=()=>document.querySelector('#profile-tab-interviews'),counter=()=>document.querySelector('[aria-label="View interview experiences"], [aria-label="查看面经"]'),panel=()=>document.querySelector('#profile-community-panel');
 await render();assert(!tab());assert(!counter());assert(!requests.some(row=>row.path.includes('tab=interviews')));
 account.capabilities.community_profile_interviews=true;await render();assert(tab(),'Profiles need a dedicated interview-experience tab');assert.equal(counter().textContent,'21');assert(!requests.some(row=>row.path.includes('tab=interviews')),'Interview rows stay deferred until chosen');
 const code=document.querySelector('[aria-label="Practice code"]');code.setSelectionRange(2,8,'backward');code.scrollTop=84;code.scrollLeft=14;sessionStorage.setItem('profile-private-draft','Unpublished report');const stored=JSON.stringify({...sessionStorage});
 await click(counter());assert.equal(location.hash,'#/u/alice?tab=interviews');assert.equal(document.activeElement,tab());assert.equal(tab().getAttribute('aria-selected'),'true');assert.equal(panel().getAttribute('aria-labelledby'),tab().id);assert.equal(document.querySelectorAll('.profile-tabs [tabindex="0"]').length,1);assert.equal(document.querySelectorAll('.community-post-card').length,1);assert(document.querySelector('.interview-meta'));assert(panel().textContent.includes('Anonymous experiences appear here only'));
 const named=document.querySelector('.post-title').getAttribute('href');assert.equal(community.communityPostReturn(named,post.id),'#/u/alice?tab=interviews');assert.equal(accountNav.accountReturnHref(decodeProblemHash(accountNav.signInHref(named))),named);
 const beforeRepeat=requests.length;await click(counter());assert.equal(requests.length,beforeRepeat);
 await key('ArrowRight');assert.equal(document.activeElement.id,'profile-tab-replies');await key('ArrowLeft');assert.equal(document.activeElement.id,'profile-tab-interviews');await key('Home');assert.equal(document.activeElement.id,'profile-tab-posts');await key('End');assert.equal(document.activeElement.id,'profile-tab-followers');await click(counter());
 await click(document.querySelector('[aria-label="Next page"]'));assert.equal(location.hash,'#/u/alice?tab=interviews&page=2');assert(panel().textContent.includes('Second-page experience'));assert.equal(community.communityPostReturn(document.querySelector('.post-title').getAttribute('href'),'b'.repeat(32)),location.hash);assert.equal(document.activeElement,panel());
 const beforeLocale=requests.length,href=location.hash;await click(document.querySelector('#locale'));assert.equal(location.hash,href);assert.equal(requests.length,beforeLocale);assert.equal(tab().textContent,'面经21');await click(document.querySelector('#locale'));
 mode='shrink';await navigate('posts');await navigate('interviews',2);await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert.equal(location.hash,'#/u/alice?tab=interviews','Expired pages recover to the last available interview page');assert(panel().textContent.includes(post.title));
 mode='empty';await navigate('posts');await navigate('interviews');assert(panel().textContent.includes('No interview experiences yet'));assert(panel().querySelector('a[href="#/interviews"]'));
 for(const bad of ['wrong-kind','no-ack','ordinary','mixed','bad-limit','bad-offset']){mode=bad;await navigate('posts');await navigate('interviews');assert(panel().querySelector('[role="alert"]'),'Unconfirmed interview rows require explicit retry: '+bad);assert.equal(document.querySelectorAll('.community-post-card').length,0);}
 mode='normal';await click([...panel().querySelectorAll('button')].find(node=>node.textContent==='Retry'));assert.equal(document.querySelectorAll('.community-post-card').length,1);
 mode='anonymous';await navigate('posts');await navigate('interviews');const privateCard=document.querySelector('.community-post-card');assert.equal(privateCard.getAttribute('data-reading-private-profile'),'true');assert.equal(privateCard.querySelector('.post-title').getAttribute('href'),'#/community/'+'c'.repeat(32));assert(privateCard.textContent.includes('Anonymous'));assert(privateCard.textContent.includes('Test University'));assert(!privateCard.querySelector('.post-author-name'));
 // Owner changes immediately retire the previous private cohort, including a transport that ignores abort.
 hold=true;await navigate('posts');const oldPosts=pending.splice(0);await navigate('interviews');const oldPrivate=pending.splice(0);account.user={...alice,id:'bob',username:'bob'};mode='normal';await render();const publicRows=pending.splice(0);assert.equal(document.querySelectorAll('.community-post-card').length,0);assert(oldPrivate.every(row=>row.signal.aborted));code.focus();const oldScrolls=scrolls.length;await act(async()=>[...oldPosts,...oldPrivate].forEach(row=>row.resolve({...activity(row.path),items:[{kind:'interviews',created_at:post.created_at,post:{...post,anonymous:true}}]})));assert.equal(document.querySelectorAll('.community-post-card').length,0);await act(async()=>publicRows.forEach(row=>row.resolve(activity(row.path))));assert.equal(document.activeElement,code);assert.equal(scrolls.length,oldScrolls);assert.equal(document.querySelectorAll('.community-post-card').length,1);assert(!panel().textContent.includes('Anonymous experiences appear here only'));
 hold=false;mode='public-anonymous';await navigate('posts');await navigate('interviews');assert(panel().querySelector('[role="alert"]'),'An anonymous row cannot be attributed through a public author profile');assert.equal(document.querySelectorAll('.community-post-card').length,0);
 mode='normal';counts.interviews=undefined;await click([...panel().querySelectorAll('button')].find(node=>node.textContent==='Retry'));assert.equal(counter().textContent,'—','Missing counts stay unknown, not zero');
 account.loading=true;await navigate('posts');const beforeHydration=requests.filter(row=>row.path.includes('tab=interviews')).length;await navigate('interviews');assert.equal(requests.filter(row=>row.path.includes('tab=interviews')).length,beforeHydration);assert.equal(panel().getAttribute('aria-busy'),'true');account.loading=false;account.capabilities.community_profile_interviews=false;await render();assert(panel().textContent.includes('Interview experiences are unavailable'));assert.equal(document.querySelectorAll('.community-post-card').length,0);assert.equal(panel().getAttribute('aria-busy'),'false');await click([...panel().querySelectorAll('button')].find(node=>node.textContent==='View all posts'));assert.equal(location.hash,'#/u/alice');assert(!tab());
 assert.equal(code,document.querySelector('[aria-label="Practice code"]'));assert.equal(code.value,'def solve(): return 42');assert.deepEqual([code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],[2,8,'backward',84,14]);assert.equal(JSON.stringify({...sessionStorage}),stored);assert.equal(mutations.length,0);assert.equal(errors.length,0);
 console.log('PASS profile interviews: real profile contribution shortcut, native keyboard tabs and bilingual counts, confirmed structured rows and scoped post/login returns, stable pagination and expiry recovery, owner-only anonymous affiliation without profile provenance, stale-account rejection, malformed/public-anonymous response recovery, capability hydration/legacy service handling and unchanged private code/draft');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();if(timezone===undefined)delete process.env.TZ;else process.env.TZ=timezone;for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
