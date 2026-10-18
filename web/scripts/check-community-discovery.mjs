import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const id='a'.repeat(32);
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/community?category=papers&q=old&page=2',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:false});
dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const values={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,requestAnimationFrame:dom.window.requestAnimationFrame.bind(dom.window),IS_REACT_ACT_ENVIRONMENT:true,__feedFixture:undefined};
const previous=new Map(Object.keys(values).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(values))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'feed-account',enforce:'pre',resolveId(source){if(source==='./social')return '\0feed-account';},load(name){if(name==='\0feed-account')return `import React from 'react';export const useAccount=()=>globalThis.__feedFixture.account;export const socialApi=(...args)=>globalThis.__feedFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Community}=await server.ssrLoadModule('/src/Community.tsx');
 const nav=await server.ssrLoadModule('/src/communityNavigation.ts');
 const {accountReturnHref,signInHref}=await server.ssrLoadModule('/src/accountNavigation.ts');
 const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 function Locale(){const {locale,setLocale}=useI18n();return React.createElement('button',{id:'locale',onClick:()=>setLocale(locale==='en'?'zh':'en')},'locale');}
 const user={id:'alice',username:'alice',display_name:'Alice',avatar_color:'#48766b'};
 const account={user,loading:false,capabilities:{community_feed_sorting:true,community_bookmarks:true}};
 const post={id,title:'Old discussion',body:'A useful paper explanation',category:'papers',created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',last_reply_at:'2026-10-15T01:00:00Z',author:user,counts:{likes:4,replies:2,reposts:0},viewer:{},problem_id:null,problem:null};
 const requests=[];let response='normal',release;
 globalThis.__feedFixture={account,request:async(path,body,signal)=>{
   assert.equal(body,undefined);const params=new URLSearchParams(path.split('?')[1]);requests.push({path,params,signal});
   if(response==='delayed')return new Promise(resolve=>{release=resolve;});
   if(response==='mismatch')return {items:[post],total:42,sort:'latest'};
   if(response==='error')throw Error('Network unavailable');
   if(response==='last-page')return {items:[post],total:21,sort:params.get('sort')||'latest'};
   return {items:[post],total:42,sort:params.get('sort')||'latest'};
 }};
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Locale),React.createElement(Community,{route:location.hash})));});
 const navigate=async href=>{location.href='http://paperleet.test/'+href;await render();};
 const select=()=>document.querySelector('.community-sort select');
 const choose=async value=>{await act(async()=>{select().value=value;select().dispatchEvent(new dom.window.Event('change',{bubbles:true}));});await render();};
 const click=async node=>{assert(node);await act(async()=>node.click());await render();};
 const tab=text=>[...document.querySelectorAll('.community-feed-toolbar button')].find(node=>node.textContent===text);
 const compactFeed=()=>{
   assert(!document.querySelector('.community-result-summary'),'The discussion count must not take up a row above posts');
   assert(!document.querySelector('#community-sort-description'),'Sort descriptions are removed for every ordering');
   const controls=document.querySelector('.community-feed-toolbar .community-feed-controls');
   assert(controls?.contains(document.querySelector('.community-search'))&&controls.contains(select()),'Search and Sort share one control row in the feed toolbar');
   assert(!select().hasAttribute('aria-describedby'),'Sort must not reference its removed description');
   const list=document.querySelector('[data-reading-list]');
   assert(list&&list.tabIndex===-1,'Reading returns retain a focusable list fallback after removing the summary');
 };
 await render();assert.equal(select().value,'latest');compactFeed();assert.equal(requests.at(-1).params.get('sort'),null);assert.equal(requests.at(-1).params.get('offset'),'20');
 assert(!document.querySelector('[aria-label="Previous page"]').disabled&&!document.querySelector('[aria-label="Next page"]').disabled,'Removing the summary preserves middle-page pagination');
 const originalBody=post.body;post.body='## Full local discussion\n\n'+('Readable detail. '.repeat(40))+'\n\n```python\nanswer = 42\n```';await render();const readingCard=document.querySelector('.community-post-card'),readingControl=readingCard.querySelector('.inline-post-body .inline-comment-toggle'),readingLink=readingCard.querySelector('[data-reading-link="title"]'),readingUrl=location.href,readingRequests=requests.length,readingStorage=JSON.stringify({...localStorage});assert(readingControl,'Community feed cards need their own full-body reading control');assert(!readingCard.querySelector('.community-prose'));await click(readingControl);assert(readingCard.querySelector('.community-prose pre'));assert.equal(location.href,readingUrl);assert.equal(requests.length,readingRequests);assert.equal(JSON.stringify({...localStorage}),readingStorage);assert.equal(readingCard.querySelector('[data-reading-link="title"]'),readingLink);await click(readingCard.querySelector('.inline-comment-collapse'));assert.equal(document.activeElement,readingControl);assert.equal(requests.length,readingRequests);post.body=originalBody;await render();assert(!readingCard.querySelector('.inline-comment-toggle'),'A short complete post needs no disclosure');
 console.log('PASS community full-post reading: actual feed wiring, local rich body, untouched source URL/actions, no detail request or draft write and short-body simplification');
 select().focus();await choose('top');assert.equal(document.activeElement,select());let scope=nav.readCommunityFeed(location.hash);assert.deepEqual(scope.state,{category:'papers',following:false,sort:'top',search:'old',page:1});assert.equal(requests.at(-1).params.get('sort'),'top');assert.equal(requests.at(-1).params.get('offset'),'0');compactFeed();
 const link=document.querySelector('[data-reading-link="title"]');assert.equal(nav.communityPostReturn(link.getAttribute('href'),id),nav.communityFeedHref(undefined,scope.state));assert.equal(accountReturnHref(decodeProblemHash(signInHref(link.getAttribute('href')))),link.getAttribute('href'));
 await click(tab('Following'));assert.equal(nav.readCommunityFeed(location.hash).state.following,true);assert.equal(select().value,'top');
 await click(tab('Saved'));assert.equal(nav.readCommunityFeed(location.hash).state.saved,true);assert.equal(nav.readCommunityFeed(location.hash).state.following,false);
 await choose('latest');assert(select().textContent.includes('Recently saved'));compactFeed();
 await choose('active');assert(document.querySelector('.post-card-activity').textContent.includes('Last reply'));compactFeed();
 await click(document.querySelector('#locale'));assert(select().textContent.includes('最近活跃'));compactFeed();await click(document.querySelector('#locale'));
 response='last-page';await navigate('#/community?category=papers&saved=1&sort=active&q=old&page=9');assert.deepEqual(nav.readCommunityFeed(location.hash).state,{category:'papers',following:false,saved:true,sort:'active',search:'old',page:2});await render();assert.equal(requests.at(-1).params.get('offset'),'20');assert(document.querySelector('[aria-label="Next page"]').disabled&&!document.querySelector('[aria-label="Previous page"]').disabled,'The corrected last page retains accurate navigation without a discussion-count heading');response='normal';
 await click(document.querySelector('[aria-label="Previous page"]'));assert.equal(nav.readCommunityFeed(location.hash).state.page,1);assert.equal(requests.at(-1).params.get('offset'),'0');
 await click(document.querySelector('[aria-label="Next page"]'));assert.equal(nav.readCommunityFeed(location.hash).state.page,2);assert.equal(requests.at(-1).params.get('offset'),'20');
 const search=document.querySelector('.community-search input');
 await act(async()=>{Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype,'value').set.call(search,'  new query  ');search.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
 await act(async()=>document.querySelector('.community-search').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));await render();
 assert.deepEqual(nav.readCommunityFeed(location.hash).state,{category:'papers',following:false,saved:true,sort:'active',search:'new query',page:1},'Toolbar search preserves board/saved/sort scope and restarts pagination');
 assert.equal(requests.at(-1).params.get('q'),'new query');assert.equal(requests.at(-1).params.get('offset'),'0');compactFeed();
 // A new filter aborts the previous request; even a transport ignoring abort cannot replace the new result.
 response='delayed';await navigate('#/community?sort=top&q=first');const stale=requests.at(-1);response='normal';await navigate('#/community?sort=active&q=second');assert(stale.signal.aborted);await act(async()=>release({items:[{...post,title:'Stale private row'}],total:1,sort:'top'}));assert(!document.body.textContent.includes('Stale private row'));assert.equal(select().value,'active');
 response='mismatch';await navigate('#/community?sort=top');assert(document.body.textContent.includes('Could not confirm'));assert(!document.querySelector('[data-reading-post]'));response='normal';await click([...document.querySelectorAll('button')].find(node=>node.textContent==='Retry'));assert(document.querySelector('[data-reading-post]'));
 response='error';await navigate('#/community?sort=active');assert(document.body.textContent.includes('Network unavailable'));assert(select());response='normal';await choose('top');assert(document.querySelector('[data-reading-post]'));
 account.capabilities.community_feed_sorting=false;await render();assert(!select());assert(document.body.textContent.includes('This order is currently unavailable'));const before=requests.length;await render();assert.equal(requests.length,before);await click([...document.querySelectorAll('button')].find(node=>node.textContent==='Use default order'));assert(!nav.readCommunityFeed(location.hash).state.sort);assert(document.querySelector('[data-reading-post]'));
 account.loading=true;await navigate('#/community?sort=top');const waiting=requests.length;assert(!document.querySelector('[data-reading-post]'));account.capabilities.community_feed_sorting=true;account.loading=false;await render();assert.equal(requests.length,waiting+1);assert.equal(select().value,'top');
 for(const [category,label] of [['admissions','School'],['career','Job search']]){
  post.category=category;
  await click([...document.querySelectorAll('.community-sidebar button')].find(node=>node.textContent.trim()===label));
  assert.equal(nav.readCommunityFeed(location.hash).state.category,category);
  assert.equal(requests.at(-1).params.get('category'),category,'Board selection sends its distinct backend category');
  assert.equal(document.querySelector('.community-post-card .community-category').textContent,label,'Cards display the split board label');
  assert.equal(nav.communityPostReturn(document.querySelector('[data-reading-link="title"]').getAttribute('href'),id),location.hash,'Opening a discussion preserves its selected board');
 }
 post.category='papers';
 account.user=undefined;await navigate('#/community?saved=1&sort=top');assert(document.body.textContent.includes('Sign in to see your saved posts'));assert(!document.querySelector('[data-reading-post]'));
 console.log('PASS community discovery: compact shared search/sort toolbar without count or sort descriptions, pagination and reading fallback, scoped sort/search controls, saved semantics, server order confirmation, abort/stale rejection, errors/retry, capability rollout, guest privacy, bilingual labels, keyboard focus and exact reading/login returns');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
