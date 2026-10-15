import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/'});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__savedFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_API_ORIGIN':JSON.stringify('')},plugins:[{
 name:'interview-saved-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0interview-saved-fixture';},
 load(id){if(id==='\0interview-saved-fixture')return `import React from 'react';export const useAccount=()=>globalThis.__savedFixture.account;export const socialApi=(...args)=>globalThis.__savedFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);`;}
}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Hub}=await server.ssrLoadModule('/src/Interviews.tsx');
 const nav=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const accountNav=await server.ssrLoadModule('/src/accountNavigation.ts');
 const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 const {libraryReturnHref}=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const requests=[],pending=[];let refreshes=0;
 globalThis.__savedFixture={account:{},request(path,body,signal){return new Promise((resolve,reject)=>{const request={path,body,signal,resolve,reject};requests.push(request);pending.push(request);});}};
 const identity=(owner,capable=true,loading=false)=>{globalThis.__savedFixture.account={user:owner?{id:owner,username:owner,display_name:owner}:null,loading,capabilities:{community_bookmarks:capable},refresh:async()=>{refreshes++;}};};
 const modules=[{id:'architecture/attention',number:1,title:'Attention',starter:'',module:{optional:false}}];
 const details={company:'amazon',role:'ml',stage:'technical',outcome:'unknown',interview_month:'',location:'Remote',questions:['architecture/attention']};
 const post={id:'a'.repeat(32),title:'A useful experience',body:'Personal reflection without a module identifier in the text.',category:'career',problem_id:null,problem:null,created_at:'2026-10-15',updated_at:'2026-10-15',author:{id:'author',username:'author',display_name:'Author',avatar_color:'#527ad9'},counts:{likes:1,replies:0,reposts:0},viewer:{liked:false,reposted:false,can_delete:false,bookmarked:true},interview:details};
 const feed=(owner,rows=[post],total=rows.length)=>({owner_id:owner,items:rows,total,offset:0,limit:20});
 const summary={reports:1,contributors:1,days:'90',basis:'published-community-reports',generated_at:'2026-10-15',companies:[{company:'amazon',reports:1,contributors:1}],questions:[{problem_id:'architecture/attention',reports:1,contributors:1,companies:[{company:'amazon',reports:1,contributors:1,problem_id:'architecture/attention',latest:'2026-10-15'}]}]};
 const filters={...nav.emptyInterviewFilters(),saved:true,concept:'architecture/attention',company:'amazon',role:'ml',stage:'technical',days:'90',track:'llm'},href=nav.interviewHref(filters);
 const render=async(route=href)=>{location.hash=route;await act(async()=>{if(!root)root=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Hub,{route:route.replace(/^#\/?/,''),problems:modules})));});};
 const take=fragment=>{const index=pending.findIndex(request=>request.path.includes(fragment));assert(index>=0,'Missing request '+fragment);return pending.splice(index,1)[0];};
 const reply=async(request,value)=>act(async()=>request.resolve(value));
 const click=async element=>{assert(element,'Missing control');await act(async()=>element.click());};
 const button=text=>[...document.querySelectorAll('button')].find(button=>button.textContent.trim()===text);
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;pending.length=0;};
 identity(null);await render();await reply(take('/summary'),summary);assert(!pending.some(request=>request.path.includes('interviews?')),'An anonymous saved view must never fall back to a public feed');
 const signin=[...document.querySelectorAll('a')].find(anchor=>anchor.textContent==='Sign in to view saved experiences');assert(signin);assert.equal(accountNav.accountReturnHref(decodeProblemHash(signin.getAttribute('href'))),href);
 identity(null,true,true);await render();assert(document.body.textContent.includes('Checking your account'));assert(!pending.some(request=>request.path.includes('interviews?')));
 identity('alice',false);await render();assert(document.body.textContent.includes('Saved experiences are unavailable'));assert(!pending.some(request=>request.path.includes('interviews?')));
 identity('alice');await render();const first=take('interviews?');assert(first.path.includes('saved=true')&&first.path.includes('problem_id=architecture%2Fattention'));
 // A private list returned under a different cookie cannot render before identity is refreshed.
 const mismatch=first;await reply(mismatch,feed('bob',[{...post,title:'Other account private title'}],99));assert(!document.body.textContent.includes('Other account private title'));assert(!document.body.textContent.includes('99'));assert.equal(document.querySelectorAll('.interview-pagination').length,0);assert(document.querySelector('[role="alert"]').textContent.includes('does not match your account'));
 await click(button('Refresh account and retry'));assert.equal(refreshes,1);await reply(take('interviews?'),feed('alice'));
 let unsave=document.querySelector('.post-bookmark > button');unsave.focus();await click(unsave);assert(unsave.disabled);const uncertain=take('/bookmark');assert.deepEqual(uncertain.body,{active:false,expected_user_id:'alice'});await click(unsave);assert(!pending.some(request=>request.path.endsWith('/bookmark')),'Repeated clicks cannot duplicate a pending mutation');
 await act(async()=>uncertain.reject(Error('Lost confirmation')));assert(button('Retry unsave'));await click(button('Retry unsave'));const retried=take('/bookmark');assert.deepEqual(retried.body,uncertain.body);await reply(retried,{owner_id:'bob',post_id:post.id,bookmarked:false});assert.equal(refreshes,2);assert(button('Read saved state'));
 await click(button('Read saved state'));const read=take('/bookmark');assert.equal(read.body,undefined);await reply(read,{owner_id:'alice',post_id:post.id,bookmarked:true});assert.equal(document.querySelector('.post-bookmark > button').getAttribute('aria-pressed'),'true');
 await click(document.querySelector('.post-bookmark > button'));await reply(take('/bookmark'),{owner_id:'alice',post_id:post.id,bookmarked:false});assert(document.querySelector('.interview-unsaved-notice'));assert(document.querySelector('.interview-report-title'),'Confirmed unsave retains reading position and a route to undo');assert.equal(document.activeElement,document.querySelector('.post-bookmark > button'));
 await click(document.querySelector('.post-bookmark > button'));await reply(take('/bookmark'),{owner_id:'alice',post_id:post.id,bookmarked:true});assert(!document.querySelector('.interview-unsaved-notice'));
 await click(document.querySelector('.post-bookmark > button'));const oldOwner=take('/bookmark');identity('bob');await render();assert(oldOwner.signal.aborted);assert.equal(document.querySelectorAll('.interview-report-card').length,0,'Previous account content disappears before a new response');const bobFeed=take('interviews?');await reply(oldOwner,{owner_id:'alice',post_id:post.id,bookmarked:false});await reply(bobFeed,feed('bob',[]));assert(document.body.textContent.includes('No saved experiences in this view'));await click(button('Browse this filtered view'));const publicFilters=nav.readInterviewRoute(location.hash);assert.deepEqual(publicFilters,{...filters,saved:false});
 await render(location.hash);await reply(take('interviews?'),feed('bob',[{...post,viewer:{...post.viewer,bookmarked:false}}]));await click(document.querySelector('.post-bookmark > button'));const oldFilter=take('/bookmark');await render(nav.interviewHref({...filters,saved:false,company:'google'}));assert(oldFilter.signal.aborted);await reply(take('interviews?'),feed('bob',[]));await reply(take('/summary'),{...summary,reports:0,contributors:0,companies:[],questions:[]});await reply(oldFilter,{owner_id:'bob',post_id:post.id,bookmarked:true});assert.equal(document.querySelectorAll('.interview-report-card').length,0,'Late confirmation cannot repopulate a different company view');
 await stop();identity('alice');await render(nav.interviewHref({...filters,saved:false,view:'questions',q:'attention'}));await reply(take('/summary'),summary);assert(document.querySelector('.interview-evidence-context').textContent.includes('not interview probabilities'));const sources=document.querySelector('.interview-ranking-evidence a');assert(sources.textContent.includes('Read 1 source report'));const sourceFilters=nav.readInterviewRoute(sources.getAttribute('href'));assert.deepEqual(sourceFilters,{...filters,saved:false,view:'reports',q:''});
 await render(sources.getAttribute('href'));const sourceRequest=take('interviews?');assert(sourceRequest.path.includes('problem_id=architecture%2Fattention'));assert(!sourceRequest.path.includes('q='));await reply(sourceRequest,feed('alice'));await click(button('Saved experiences'));await render(location.hash);await reply(take('interviews?'),feed('alice'));const practice=document.querySelector('.interview-path-modules a');assert(nav.readInterviewRoute(libraryReturnHref(practice.getAttribute('href'))).saved,'The practice return keeps the personal reading list');
 await stop();identity('alice');await render(nav.interviewHref({...filters,page:3}));await reply(take('/summary'),summary);await reply(take('interviews?'),{...feed('alice',[],1),offset:40});assert(document.body.textContent.includes('No reports on this page'));await click(button('Return to first page'));assert.deepEqual(nav.readInterviewRoute(location.hash),filters,'Returning from an expired saved page preserves all personal and concept filters');
 console.log('PASS saved interview workflow: private login/capability gates, response owner binding, mismatched identity refresh, exact retry intent, read-back recovery, no duplicate mutations, unsave/undo focus, account/filter aborts, concept-source drilldown, publication scope and saved practice returns');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
