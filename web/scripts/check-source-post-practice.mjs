import assert from 'node:assert/strict';
import React,{act,useEffect,useState} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/community?category=career&q=attention&page=2',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__sourcePractice:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');window.scrollTo=()=>{};dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const server=await createServer({cacheDir:'node_modules/.vite-source-post-check',server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'source-post-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0source-post-fixture';},load(id){if(id==='\0source-post-fixture')return `import React from 'react';export const useAccount=()=>globalThis.__sourcePractice.account;export const socialApi=(...args)=>globalThis.__sourcePractice.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=()=>null;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}}]});
let root,model;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Community}=await server.ssrLoadModule('/src/Community.tsx');
 const {ModuleNavigator}=await server.ssrLoadModule('/src/ModuleNavigator.tsx');
 const {PracticeNextStep}=await server.ssrLoadModule('/src/PracticeNextStep.tsx');
 const {usePracticeReadingOrigin}=await server.ssrLoadModule('/src/usePracticeReadingOrigin.ts');
 const nav=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const communityNav=await server.ssrLoadModule('/src/communityNavigation.ts');
 model=await server.ssrLoadModule('/src/readingReturn.ts');
 const id='a'.repeat(32),sourcePost=communityNav.communityPostHref(id),scope=location.hash;
 const problems=['architecture/attention','rl/gae','diffusion/ddpm'].map((id,index)=>({id,number:index+1,paper_id:'paper',title:'中文知识点 '+index,translations:{en:{title:'Localized concept '+index}},difficulty:'M',estimated_minutes:20,module:{optional:false},tests:{T0:1}}));
 const topics=problems.map(problem=>({id:problem.id,number:problem.number,title:problem.title,title_en:problem.translations.en.title}));
 const retired={id:'foundations/old_concept',number:9,title:'Old concept',retired:true};
 const author={id:'anonymous:'+id,anonymous:true,username:'',display_name:'Anonymous',affiliation:{kind:'institution',domain:'research.edu',name:'研究机构',name_en:'Research Institute'}};
 const basePost={id,title:'An interview reflection',body:'## Approach\n\nHow we reasoned about attention.',category:'career',author,anonymous:true,created_at:'2026-10-16',updated_at:'2026-10-16',counts:{likes:2,replies:0,reposts:0},viewer:{liked:false,bookmarked:false},interview:{company:'amazon',role:'ml',stage:'technical',outcome:'unknown',interview_month:'',location:'',questions:problems.map(problem=>problem.id)},replies:[]};
 const account={user:{id:'alice',username:'alice',display_name:'Alice',can_publish_community:true},loading:false,capabilities:{community_feed_sorting:true},refresh:async()=>{}};
 let post=basePost,catalog=topics,catalogError,catalogLoading=false,requests=[],writes=0,retries=0,refresh,codeState;
 globalThis.__sourcePractice={account,request:async(path,body)=>{requests.push(path);if(body){writes++;throw Error('Reading must not publish');}if(path==='community/posts/'+id)return post;assert(path.startsWith('community/posts?'));return {items:[post],total:21};}};
 function Language(){const {locale,setLocale}=useI18n();return React.createElement('button',{id:'language',onClick:()=>setLocale(locale==='en'?'zh':'en')},'Language');}
 function Journey(){
  const [route,setRoute]=useState(location.hash),[revision,setRevision]=useState(0),[codes,setCodes]=useState({});refresh=()=>setRevision(value=>value+1);codeState=codes;
  useEffect(()=>{const update=()=>setRoute(location.hash);window.addEventListener('hashchange',update);return()=>window.removeEventListener('hashchange',update);},[]);
  const path=route.replace(/^#\/?/,'').split('?')[0],problem=problems.find(problem=>problem.id===decodeURIComponent(path)||String(problem.number)===path),back=nav.libraryReturnHref(route),owner=account.user?'account:'+account.user.id:'guest';
  usePracticeReadingOrigin(owner,problem,back,route);
  const codeKey=owner+':'+problem?.id;
  if(problem)return React.createElement('main',null,React.createElement(Language),React.createElement('textarea',{'aria-label':'Private code',value:codes[codeKey]||'Private implementation',onChange:event=>setCodes(previous=>({...previous,[codeKey]:event.target.value}))}),React.createElement('a',{id:'source-back',href:back},'Back to source post'),React.createElement(ModuleNavigator,{current:problem,modules:problems,saved:{},ready:true,returnTo:back}),React.createElement(PracticeNextStep,{next:problems[1],samePaper:true,returnTo:back,onNotes(){},onLab(){}}));
  return React.createElement(React.Fragment,null,React.createElement(Language),React.createElement(Community,{route,postId:path.startsWith('community/')?path.slice('community/'.length):undefined,topics:catalog,catalogLoading,catalogError,reloadCatalog:()=>{retries++;catalog=[...topics,retired];catalogError=undefined;refresh();}}));
 }
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Journey)));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;model.discardReadingReturn();};
 const settle=async predicate=>{for(let index=0;index<100&&!predicate();index++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),'Source-post journey did not settle');};
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
 const change=async(node,value)=>act(async()=>{Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
 const practice=()=>document.querySelector('.discussion-page .interview-report-practice');
 await render();await settle(()=>document.querySelector('.post-title'));
 await click(document.querySelector('.post-title'));await settle(()=>practice()?.querySelector('a'));
 assert.equal(model.currentReadingReturn()?.href,scope);assert(document.querySelector('.discussion-author').textContent.includes('Research Institute'));
 const reply=document.querySelector('.reply-composer textarea');assert(reply);await change(reply,'My private unpublished reflection');reply.setSelectionRange(2,9,'backward');reply.scrollTop=27;reply.scrollLeft=11;
 const beforeExpand=JSON.stringify({...sessionStorage});await click(practice().querySelector('.interview-report-practice-toggle'));assert.equal(practice().querySelectorAll('a').length,3);assert(document.activeElement===practice().querySelectorAll('a')[2]);assert(document.querySelector('.reply-composer textarea')===reply);assert.deepEqual([reply.selectionStart,reply.selectionEnd,reply.selectionDirection,reply.scrollTop,reply.scrollLeft],[2,9,'backward',27,11]);assert.equal(JSON.stringify({...sessionStorage}),beforeExpand);
 for(const link of practice().querySelectorAll('a')){assert.equal(nav.libraryReturnHref(link.getAttribute('href')),sourcePost);assert(!link.getAttribute('href').includes('attention%26'));assert(!link.getAttribute('href').includes('alice'));assert(!link.getAttribute('href').includes('private'));}
 await click(document.querySelector('#language'));assert(practice().querySelector('a').textContent.includes('中文知识点'));await click(document.querySelector('#language'));
 await click(practice().querySelector('a'));await settle(()=>document.querySelector('#source-back'));
 assert.equal(document.querySelector('#source-back').getAttribute('href'),sourcePost);assert.equal(model.currentReadingReturn()?.href,scope);
 await change(document.querySelector('[aria-label="Private code"]'),'Private source stays in the workspace');
 assert.equal(nav.libraryReturnHref(document.querySelector('.journey-next-link').getAttribute('href')),sourcePost);await click(document.querySelector('.journey-next-link'));await settle(()=>location.hash.startsWith('#/2?'));
 assert.equal(document.querySelector('#source-back').getAttribute('href'),sourcePost);assert.equal(model.currentReadingReturn()?.href,scope);
 await click(document.querySelector('#source-back'));await settle(()=>practice()?.querySelector('a'));
 assert.equal(document.querySelector('.discussion-navigation .social-back').getAttribute('href'),scope);assert.equal(document.querySelector('.reply-composer textarea').value,'My private unpublished reflection');
 await click(document.querySelector('.discussion-navigation .social-back'));await settle(()=>document.querySelector('.post-title')&&!document.querySelector('.discussion-page'));
 assert.equal(location.hash,scope);assert.equal(model.currentReadingReturn(),undefined);assert(Object.values(codeState).includes('Private source stays in the workspace'));assert.equal(writes,0);await stop();
 // Full-detail metadata never creates links to missing or explicitly retired modules.
 post={...basePost,interview:{...basePost.interview,questions:[problems[0].id,retired.id,problems[2].id]}};catalog=[topics[0],retired];catalogError='Catalog offline';location.hash=sourcePost;await new Promise(resolve=>setTimeout(resolve,10));await render();await settle(()=>practice());await click(practice().querySelector('.interview-report-practice-toggle'));
 assert.equal(practice().querySelectorAll('a').length,1);assert.equal(practice().querySelectorAll('.interview-report-practice-missing').length,2);assert(!practice().querySelector('a[href*="old_concept"]'));
 await click(practice().querySelector('.interview-report-practice-retry button'));assert.equal(retries,1);assert.equal(practice().querySelectorAll('a').length,2);assert.equal(practice().querySelectorAll('.interview-report-practice-missing').length,1);assert(document.activeElement===practice().querySelector('[data-reading-practice="'+problems[2].id+'"]'));assert.equal(writes,0);await stop();
 // A retired-only retry keeps a real focus target even though no practice can be restored.
 post={...basePost,interview:{...basePost.interview,questions:[retired.id]}};catalog=[retired];catalogError='Catalog offline';await render();await settle(()=>practice()?.querySelector('.interview-report-practice-retry button'));await click(practice().querySelector('.interview-report-practice-retry button'));
 assert(!practice().querySelector('a'));assert(document.activeElement===practice().querySelector('.interview-report-practice-missing'));await stop();
 // Hydrating public titles cannot replace or refocus the reader's current draft.
 post=basePost;catalog=[];catalogLoading=true;catalogError=undefined;await render();await settle(()=>practice());assert(!practice().querySelector('a'));assert(practice().textContent.includes('Details pending'));
 const pendingDraft=document.querySelector('.reply-composer textarea');pendingDraft.focus();pendingDraft.setSelectionRange(1,8,'backward');pendingDraft.scrollTop=18;const beforeHydration=requests.length;
 catalog=topics;catalogLoading=false;await act(async()=>refresh());assert.equal(practice().querySelectorAll('a').length,2);assert(document.activeElement===pendingDraft);assert(document.querySelector('.reply-composer textarea')===pendingDraft);assert.equal(pendingDraft.value,'My private unpublished reflection');assert.deepEqual([pendingDraft.selectionStart,pendingDraft.selectionEnd,pendingDraft.selectionDirection,pendingDraft.scrollTop],[1,8,'backward',18]);assert.equal(requests.length,beforeHydration);await stop();
 // An anonymous self-profile origin stays only in this owner's tab across practice.
 post=basePost;catalog=topics;catalogError=undefined;const privateScope='#/u/alice?page=2';location.hash=sourcePost;await new Promise(resolve=>setTimeout(resolve,10));
 model.rememberReadingReturn({owner:'account:alice',href:privateScope,post:id,link:'title',privateProfile:true,top:120,focus:true});assert.equal(model.currentReadingReturn()?.href,privateScope);await render();await settle(()=>practice()?.querySelector('a'));
 assert.equal(document.querySelector('.discussion-navigation .social-back').getAttribute('href'),privateScope);
 const privatePractice=practice().querySelector('a').getAttribute('href');assert.equal(nav.libraryReturnHref(privatePractice),sourcePost);assert(!privatePractice.includes('alice'));await click(practice().querySelector('a'));await settle(()=>document.querySelector('#source-back'));assert.equal(model.currentReadingReturn()?.href,privateScope);
 account.user={...account.user,id:'bob',username:'bob'};await act(async()=>refresh());assert.equal(model.currentReadingReturn(),undefined);assert.equal(document.querySelector('[aria-label="Private code"]').value,'Private implementation');
 await click(document.querySelector('#source-back'));await settle(()=>practice()?.querySelector('a'));assert.equal(document.querySelector('.discussion-navigation .social-back').getAttribute('href'),'#/community');assert.equal(document.querySelector('.reply-composer textarea').value,'');assert.equal(writes,0);await stop();
 console.log('PASS source-post practice: actual community feed/thread/metadata/practice/next/thread/list traversal, localized titles and retained source filters, canonical public-only URLs, anonymous affiliation/private-profile owner isolation, missing/retired/recovered catalog focus, retained writing/caret/scroll and workspace source, no publication or grade changes');
}finally{if(root)await act(async()=>root.unmount());model?.discardReadingReturn();await server.close();dom.window.close();for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
