import assert from 'node:assert/strict';
import React,{act} from 'react';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/community',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:true});dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__communityPeriod:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-community-period-');
const server=await createServer({cacheDir,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
 name:'community-period-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0community-period-social';},load(name){if(name==='\0community-period-social')return `import React from 'react';export const useAccount=()=>globalThis.__communityPeriod.account;export const socialApi=(...args)=>globalThis.__communityPeriod.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}
}]});
let root;const realSet=globalThis.setTimeout,realClear=globalThis.clearTimeout,timers=[];
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Community,ProblemSolutions,COMMUNITY_FEED_TIMEOUT}=await server.ssrLoadModule('/src/Community.tsx');
 const nav=await server.ssrLoadModule('/src/communityNavigation.ts');
 const accountNav=await server.ssrLoadModule('/src/accountNavigation.ts');
 const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 globalThis.setTimeout=(callback,delay,...args)=>{if(delay===COMMUNITY_FEED_TIMEOUT){const timer={callback,cancelled:false};timers.push(timer);return timer;}return realSet(callback,delay,...args);};
 globalThis.clearTimeout=timer=>timers.includes(timer)?timer.cancelled=true:realClear(timer);
 const topic={id:'architecture/attention',number:1,title:'注意力',title_en:'Attention'};
 const user={id:'alice',username:'alice',display_name:'Alice',can_publish_community:true,affiliation:{kind:'institution',domain:'research.edu',name:'测试机构',name_en:'Research Institute'}};
 const account={user,loading:false,capabilities:{community_feed_sorting:true,community_bookmarks:true,community_solutions:true},refresh:async()=>{}};
 const post={id:'a'.repeat(32),title:'A real recent explanation',body:'Derivation and boundary cases',category:'implementation',solution:true,problem_id:topic.id,problem:topic,author:user,created_at:'2026-10-17T10:00:00Z',updated_at:'2026-10-17T10:00:00Z',counts:{likes:2,replies:0,reposts:0},viewer:{bookmarked:true}};
 const requests=[],pending=[],mutations=[];let mode='normal',embedded=false;
 const response=(request,title=post.title)=>({items:mode==='empty'?[]:[{...post,title}],total:mode==='empty'?0:42,limit:20,offset:Number(request.params.get('offset')),sort:request.params.get('sort')||'latest',solutions:request.params.get('solutions')==='true',...(request.params.get('days')?{days:request.params.get('days')}:{})});
 globalThis.__communityPeriod={account,request(path,body,signal){
  if(body){mutations.push({path,body});throw Error('Filtering must not publish or react');}
  const request={path,params:new URLSearchParams(path.split('?')[1]),signal};requests.push(request);
  if(mode==='hold')return new Promise((resolve,reject)=>pending.push({...request,resolve,reject}));
  if(mode==='error')return Promise.reject(Error('Network unavailable'));
  const value=response(request);if(mode==='no-ack')delete value.days;if(mode==='wrong-ack')value.days='30';if(mode==='shrink')value.total=1;
  return Promise.resolve(value);
 }};
 function Locale(){const {locale,setLocale}=useI18n();return React.createElement('button',{id:'period-locale',onClick:()=>setLocale(locale==='en'?'zh':'en')},'Language');}
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Locale),React.createElement('textarea',{'aria-label':'Practice code',defaultValue:'def solve(): return 42'}),embedded?React.createElement(ProblemSolutions,{topic}):React.createElement(Community,{route:location.hash,topicId:topic.id,topic,topics:[topic]})));});
 const navigate=async href=>{await act(async()=>{location.hash=href;await new Promise(resolve=>realSet(resolve,5));});await render();};
 const click=async node=>{assert(node,'Missing period control');await act(async()=>{node.focus();node.click();});await render();};
 const button=name=>[...document.querySelectorAll('button')].find(node=>(node.getAttribute('aria-label')||node.textContent)===name);
 const control=()=>document.querySelector('.community-period select'),toolbar=()=>document.querySelector('.community-feed-toolbar');
 const state=()=>nav.readCommunityFeed(location.hash).state;
 const change=async(node,value)=>{assert(node);await act(async()=>{Object.getOwnPropertyDescriptor(node.tagName==='TEXTAREA'?dom.window.HTMLTextAreaElement.prototype:node.tagName==='SELECT'?dom.window.HTMLSelectElement.prototype:dom.window.HTMLInputElement.prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event(node.tagName==='SELECT'?'change':'input',{bubbles:true}));});await render();};
 const release=async(request,value)=>{await act(async()=>request.resolve(value));await render();};
 const initial={category:'implementation',following:false,saved:true,solutions:true,sort:'top',search:'mask & 中文 😀',page:3};
 for(const days of ['7','30']){
  const scope=nav.communityFeedHref(topic.id,{...initial,days});assert.equal(nav.readCommunityFeed(scope).state.days,days);
  const thread=nav.communityPostHref(post.id,scope);assert.equal(nav.communityPostReturn(thread,post.id),scope);
  for(const target of [scope,thread])assert.equal(accountNav.accountReturnHref(decodeProblemHash(accountNav.signInHref(target))),target);
 }
 assert.equal(nav.communityFeedHref(topic.id,nav.emptyCommunityFeed()),'#/community/problem/architecture%2Fattention');
 for(const query of ['days=','days=0','days=365','days=007','days=7.0','days=-7','days=week','days=7&days=30','days=&days=7','days=7%20OR%201=1'])assert.equal(nav.readCommunityFeed('community?'+query),undefined);
 assert.equal(nav.readCommunityFeed('community?days=all').state.days,undefined);
 await navigate(nav.communityFeedHref(topic.id,initial));assert(!control());const beforeCapability=requests.length;account.capabilities.community_feed_period=true;await render();assert(control());assert.equal(requests.length,beforeCapability,'Discovering the capability must not refetch all-time data');
 await change(control(),'7');assert.equal(state().page,1);assert.equal(state().days,'7');assert.equal(state().saved,true);assert.equal(state().search,initial.search);assert.equal(state().sort,'top');assert.equal(requests.at(-1).params.get('days'),'7');assert.equal(requests.at(-1).params.get('offset'),'0');assert(document.querySelector('.community-period-note').textContent.includes('lifetime totals'));
 await click(document.querySelector('[aria-label="Next page"]'));assert.equal(state().page,2);assert.equal(requests.at(-1).params.get('offset'),'20');assert.equal(requests.at(-1).params.get('days'),'7');
 const linked=document.querySelector('[data-reading-link="title"]');assert.equal(nav.communityPostReturn(linked.getAttribute('href'),post.id),nav.communityFeedHref(topic.id,state()));
 const beforeLocale=requests.length;await click(document.querySelector('#period-locale'));assert.equal(control().getAttribute('aria-label'),'帖子发布时间');assert(button('移除筛选：近 7 天发布'));assert.equal(requests.length,beforeLocale);await click(document.querySelector('#period-locale'));
 // Explicit recovery drops only the time filter; the other applied scopes survive.
 await click(button('Remove filter: Published in the past 7 days'));assert.equal(state().days,undefined);assert.equal(state().saved,true);assert.equal(state().search,initial.search);assert.equal(state().sort,'top');assert.equal(state().page,1);assert.equal(document.activeElement,toolbar());assert.equal(requests.at(-1).params.get('days'),null);
 mode='no-ack';await change(control(),'7');assert(document.querySelector('[role="alert"]'));assert(!document.querySelector('.post-title'));assert.equal(control().value,'7');
 mode='wrong-ack';await click(button('Retry'));assert(document.querySelector('[role="alert"]'));mode='normal';await click(button('Retry'));assert(document.querySelector('.post-title'));
 mode='hold';await change(control(),'30');const older=pending.shift();await change(control(),'7');const newer=pending.shift();assert(older.signal.aborted);await release(older,response(older,'Stale 30-day data'));assert(!document.body.textContent.includes('Stale 30-day data'));
 const code=document.querySelector('[aria-label="Practice code"]');await act(async()=>code.focus());await release(newer,response(newer,'Current 7-day data'));assert(document.body.textContent.includes('Current 7-day data'));assert.equal(document.activeElement,code);
 mode='hold';await change(control(),'30');const timeoutRequest=pending.shift(),timer=timers.at(-1);await act(async()=>timer.callback());assert(timeoutRequest.signal.aborted);assert(document.querySelector('[role="alert"]').textContent.includes('timed out'));await release(timeoutRequest,response(timeoutRequest,'Too late'));assert(!document.body.textContent.includes('Too late'));
 mode='normal';await click(button('Retry'));assert.equal(requests.at(-1).params.get('days'),'30');
 // Period context survives the real feed's last-page correction.
 mode='shrink';await navigate(nav.communityFeedHref(topic.id,{...initial,days:'7',page:9}));await act(async()=>new Promise(resolve=>realSet(resolve,10)));await render();assert.equal(state().page,1);assert.equal(state().days,'7');assert.equal(state().saved,true);assert.equal(state().sort,'top');
 mode='empty';await navigate(nav.communityFeedHref(topic.id,{...nav.emptyCommunityFeed(),days:'7'}));assert(document.body.textContent.includes('No posts published in the past 7 days'));await click(button('Browse all time'));assert.equal(state().days,undefined);assert.equal(document.activeElement,toolbar());
 account.capabilities.community_feed_period=false;mode='normal';const beforeLegacy=requests.length;await navigate(nav.communityFeedHref(topic.id,{...initial,days:'30',page:2}));assert.equal(requests.length,beforeLegacy);assert(document.body.textContent.includes('Publication periods are unavailable'));assert(control().disabled);await click(button('Browse all time'));assert.equal(state().days,undefined);assert.equal(state().saved,true);assert.equal(state().sort,'top');assert.equal(document.activeElement,toolbar());assert(!control());
 account.loading=true;const hydrationStart=requests.length;await navigate(nav.communityFeedHref(topic.id,{...initial,days:'7',page:1}));assert.equal(requests.length,hydrationStart);assert(control().disabled);account.loading=false;account.capabilities.community_feed_period=true;await render();assert.equal(requests.length,hydrationStart+1);assert.equal(requests.at(-1).params.get('days'),'7');
 account.user=undefined;const beforePrivate=requests.length;await render();assert.equal(requests.length,beforePrivate);assert(!document.querySelector('.post-title'));await navigate(nav.communityFeedHref(topic.id,{...nav.emptyCommunityFeed(),days:'30'}));assert.equal(requests.at(-1).params.get('days'),'30');assert(document.querySelector('.post-title'),'Guests can browse public recent posts');
 // In-problem solution filtering keeps the practice URL and both editors mounted.
 account.user=user;embedded=true;await navigate('#/22?from=collections%2Fllm-basics');const practiceHash=location.hash;
 const editor=document.querySelector('[aria-label="Practice code"]');editor.setSelectionRange(3,9,'backward');editor.scrollTop=51;editor.scrollLeft=17;
 await click(button('Share a solution'));const composer=document.querySelector('.community-composer'),field=composer.querySelector('textarea');await change(field,'My private unpublished derivation');field.setSelectionRange(2,8,'backward');field.scrollTop=73;field.scrollLeft=11;const storage=JSON.stringify({...sessionStorage});
 const search=document.querySelector('.community-search input');await change(search,'Unapplied search');const beforeTyping=requests.length;await change(control(),'7');assert.equal(requests.length,beforeTyping+1);assert.equal(requests.at(-1).params.get('q'),'');assert.equal(search.value,'Unapplied search');
 await change(document.querySelector('.community-sort select'),'top');await click(document.querySelector('[aria-label="Next page"]'));assert.equal(requests.at(-1).params.get('days'),'7');assert.equal(requests.at(-1).params.get('offset'),'20');
 await click(button('Remove filter: Published in the past 7 days'));assert.equal(control().value,'all');assert.equal(requests.at(-1).params.get('days'),null);assert.equal(requests.at(-1).params.get('sort'),'top');assert.equal(requests.at(-1).params.get('offset'),'0');assert.equal(location.hash,practiceHash);
 await change(control(),'30');const allLink=document.querySelector('.problem-solutions-heading a');const allScope=nav.readCommunityFeed(allLink.getAttribute('href'));assert.equal(allScope.state.days,'30');assert.equal(allScope.state.sort,'top');assert.equal(allScope.topicId,topic.id);assert.equal(allScope.state.solutions,true);
 await click(button('Clear filters'));assert.equal(control().value,'all');assert.equal(requests.at(-1).params.get('sort'),null);assert(!document.querySelector('.community-feed-filters'));assert.equal(document.querySelector('.community-composer'),composer);assert.equal(composer.querySelector('textarea'),field);assert.equal(field.value,'My private unpublished derivation');assert.equal(JSON.stringify({...sessionStorage}),storage);
 assert.equal(document.querySelector('[aria-label="Practice code"]'),editor);assert.deepEqual([editor.value,editor.selectionStart,editor.selectionEnd,editor.selectionDirection,editor.scrollTop,editor.scrollLeft],['def solve(): return 42',3,9,'backward',51,17]);assert.deepEqual([field.selectionStart,field.selectionEnd,field.selectionDirection,field.scrollTop,field.scrollLeft],[2,8,'backward',73,11]);assert.equal(location.hash,practiceHash);
 mode='hold';await change(control(),'7');const outgoing=pending.shift(),outgoingData=response(outgoing,'Old private cohort');assert(outgoingData.items.length);account.user={...user,id:'bob',username:'bob'};mode='empty';await render();assert(outgoing.signal.aborted);await release(outgoing,outgoingData);assert(!document.body.textContent.includes('Old private cohort'));assert(!document.querySelector('.community-composer'));assert.equal(location.hash,practiceHash);assert.equal(mutations.length,0);
 console.log('PASS community publication periods: real ordinary/solution feeds, scoped seven/thirty-day URLs and auth/post returns, retained sorting/query/saved/pagination and last-page recovery, default/legacy/hydration/guest guards, confirmed response/retry/deadline/stale cohorts, stable keyboard/locale and independent search draft, mounted practice/solution drafts and no social writes');
}finally{if(root)await act(async()=>root.unmount());globalThis.setTimeout=realSet;globalThis.clearTimeout=realClear;await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
