import assert from 'node:assert/strict';
import React,{act} from 'react';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/interviews?company=amazon&saved=true&page=2',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,location:dom.window.location,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-report-practice-');
const server=await createServer({cacheDir,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
 name:'no-social-requests',enforce:'pre',resolveId(source){if(source==='./social')return '\0no-social-requests';},
 load(id){if(id==='\0no-social-requests')return `import React from 'react';export const useAccount=()=>({user:null,loading:false});export const socialApi=()=>{throw Error('Reading practice must not request social data');};export class SocialError extends Error{};export const errorMessage=e=>e.message;export const Avatar=()=>React.createElement('span',{'aria-hidden':true});export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',null,error);`;}
}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Card}=await server.ssrLoadModule('/src/InterviewReportCard.tsx');
 const {useReadingPosition}=await server.ssrLoadModule('/src/useReadingPosition.ts');
 const {usePracticeReadingOrigin}=await server.ssrLoadModule('/src/usePracticeReadingOrigin.ts');
 const readingModel=await server.ssrLoadModule('/src/readingReturn.ts');
 dom.window.scrollTo=()=>{};dom.window.HTMLElement.prototype.scrollIntoView=function(){};
 const {libraryReturnHref}=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const {interviewHref,emptyInterviewFilters}=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const scope=interviewHref({...emptyInterviewFilters(),company:'amazon',role:'ml',stage:'technical',q:'Attention & 多模态',saved:true,page:2});location.hash=scope;
 const problems=['architecture/attention','rl/gae','diffusion/ddpm','optimization/adam'].map((id,index)=>({id,number:index+1,title:'中文知识点 '+index,translations:{en:{title:'Practice concept '+index}},module:{optional:false}}));
 const anonymous={id:'anonymous:'+'a'.repeat(32),username:'',display_name:'Anonymous',anonymous:true,avatar_color:'#48766b',affiliation:{kind:'institution',domain:'research.edu',name:'测试大学',name_en:'Research University'}};
 const post={id:'a'.repeat(32),title:'A source experience',body:'Actual experience text',category:'career',created_at:'2026-10-15',author:anonymous,counts:{likes:1,replies:0,reposts:0},viewer:{liked:false},interview:{company:'amazon',role:'ml',stage:'technical',outcome:'unknown',interview_month:'',location:'Remote',questions:problems.map(p=>p.id)}};
 let retries=0;
 function Locale(){const {locale,setLocale}=useI18n();return React.createElement('button',{id:'locale',onClick:()=>setLocale(locale==='en'?'zh':'en')},'Locale');}
 const props={post,owner:'',canSave:false,returnTo:scope,savedView:false,problems,onReload(){},onRetryCatalog(){retries++;}};
 const render=async(change={},copies=1)=>act(async()=>{root??=createRoot(document.getElementById('root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Locale),React.createElement('input',{id:'outside'}),Array.from({length:copies},(_,index)=>React.createElement(Card,{...props,...change,key:index}))));});
 const area=()=>document.querySelector('.interview-report-practice');
 const button=()=>area()?.querySelector('.interview-report-practice-toggle');
 const activate=async(node,{keyboard=true,after}={})=>act(async()=>{assert(node);node.focus();node.dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true,cancelable:true,detail:keyboard?0:1}));after?.();});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 await render();const storage=JSON.stringify({...sessionStorage});assert.equal(area().querySelectorAll('a').length,2);assert.equal(button().getAttribute('aria-expanded'),'false');assert.equal(document.getElementById(button().getAttribute('aria-controls')),area().querySelector('[role=group]'));assert(document.querySelector('.interview-report-author').textContent.includes('Research University'));assert(!document.querySelector('.interview-report-author a'));
 await activate(button());assert.equal(location.hash,scope);assert.equal(area().querySelectorAll('a').length,4);assert.equal(document.activeElement,area().querySelectorAll('a')[2]);for(const link of area().querySelectorAll('a')){assert.equal(libraryReturnHref(link.getAttribute('href')),scope);assert.equal(link.dataset.readingPractice,problems[Number(link.textContent.at(-1))].id);}
 await activate(document.getElementById('locale'));assert(area().textContent.includes('收起配套练习'));assert.equal(area().querySelectorAll('a').length,4);assert(area().querySelector('a').textContent.includes('中文知识点'));await activate(document.getElementById('locale'));
 const removed=area().querySelectorAll('a')[3];removed.focus();await act(async()=>button().click());assert.equal(area().querySelectorAll('a').length,2);assert.equal(document.activeElement,button(),'Collapse replaces a removed focused practice with its surviving toggle');
 await activate(button(),{keyboard:false});assert.equal(document.activeElement,button(),'Pointer expansion keeps its control focused');await activate(button());
 await activate(button(),{after:()=>document.getElementById('outside').focus()});assert.equal(document.activeElement,document.getElementById('outside'),'Later input focus wins over expansion');await activate(button());
 await activate(button(),{after:()=>{location.hash='#/community';}});assert.equal(document.activeElement,button(),'A changed route cannot apply an older focus request');location.hash=scope;assert.equal(JSON.stringify({...sessionStorage}),storage,'Expanding never stores reading, workspace or draft data');await stop();
 // A partial catalog preserves unavailable source concepts without dead practice links.
 const mixed={...post,interview:{...post.interview,questions:[problems[0].id,problems[1].id,'foundations/retired_concept',problems[3].id]}};
 await render({post:mixed,catalogError:'Catalog failed'});await activate(button());assert(document.activeElement.classList.contains('interview-report-practice-missing'));assert(area().textContent.includes('foundations/retired_concept'));assert.equal(area().querySelectorAll('a').length,3);assert(!area().querySelector('a[href*="retired"]'));assert(area().textContent.includes('Practice unavailable'));assert(area().textContent.includes('experience and comments remain readable'));
 await activate(area().querySelector('.interview-report-practice-retry button'));assert.equal(retries,1);assert.equal(button().getAttribute('aria-expanded'),'true');await render({post:mixed,problems:[...problems,{id:'foundations/retired_concept',number:5,title:'Recovered concept',module:{}}]});assert.equal(area().querySelectorAll('a').length,4);assert(!area().querySelector('.interview-report-practice-missing'));assert(!area().querySelector('.interview-report-practice-retry'));assert.equal(document.activeElement.textContent,'Recovered concept','Explicit retry restores focus to the recovered practice when its control disappears');await stop();
 await render({post:mixed,catalogError:'Catalog failed'});await activate(button());await activate(area().querySelector('.interview-report-practice-retry button'));document.getElementById('outside').focus();await render({post:mixed,problems:[...problems,{id:'foundations/retired_concept',number:5,title:'Recovered again',module:{}}]});assert.equal(document.activeElement,document.getElementById('outside'),'A completed retry cannot steal newer input focus');await stop();
 await render({post:mixed});await activate(button());const missingFocus=document.activeElement;await render({post:mixed,problems:[...problems,{id:'foundations/retired_concept',number:5,title:'Loaded in place',module:{}}]});assert.equal(document.activeElement,missingFocus,'Catalog hydration retains the focused logical concept');assert(missingFocus.querySelector('a'));await stop();
 await render({problems:[]});assert(area().textContent.includes('Details pending'));assert.equal(area().querySelectorAll('a').length,0);await activate(button());assert.equal(area().querySelectorAll('.interview-report-practice-missing').length,4);await stop();
 await render({},2);const toggles=[...document.querySelectorAll('.interview-report-practice-toggle')];assert.notEqual(toggles[0].getAttribute('aria-controls'),toggles[1].getAttribute('aria-controls'));await activate(toggles[1]);assert.equal(document.querySelectorAll('.interview-report-practice')[0].querySelectorAll('a').length,2);assert.equal(document.querySelectorAll('.interview-report-practice')[1].querySelectorAll('a').length,4);assert(document.querySelectorAll('.interview-report-practice')[1].contains(document.activeElement));await stop();
 await render({post:{...post,interview:{...post.interview,questions:[]}}});assert(!area());await stop();
 await render({post:{...post,interview:{...post.interview,questions:[problems[0].id,problems[0].id]}}});assert.equal(area().querySelectorAll('a').length,1);assert(!button());await stop();
 // The actual list/practice reading controller must restore an expanded third link.
 function Workspace(){usePracticeReadingOrigin('guest',problems[2],scope);return React.createElement('a',{id:'reading-back',href:scope},'Back to source');}
 function ReadingList(){const reading=useReadingPosition('guest',scope,true);return React.createElement('main',{ref:reading.container},React.createElement('p',{'data-reading-list':'',tabIndex:-1},'Source results'),React.createElement(Card,{...props,resumed:reading.resumedPost===post.id,resumedPractice:reading.resumedPost===post.id?reading.resumedPractice:undefined}));}
 function Journey(){const [route,setRoute]=React.useState(location.hash);React.useEffect(()=>{const update=()=>setRoute(location.hash);window.addEventListener('hashchange',update);return()=>window.removeEventListener('hashchange',update);},[]);return route===scope?React.createElement(ReadingList):React.createElement(Workspace);}
 const settle=async predicate=>{for(let i=0;i<80&&!predicate();i++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),'Reading journey did not settle');};
 readingModel.discardReadingReturn();location.hash=scope;await new Promise(resolve=>setTimeout(resolve,10));await act(async()=>{root=createRoot(document.getElementById('root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Journey)));});
 await activate(button());const third=area().querySelectorAll('a')[2],practiceId=third.dataset.readingPractice;await activate(third);await settle(()=>document.getElementById('reading-back'));assert.equal(readingModel.currentReadingReturn()?.phase,'reading');await activate(document.getElementById('reading-back'));await settle(()=>area()?.querySelectorAll('a').length===4&&document.activeElement?.dataset.readingPractice===practiceId);assert.equal(button().getAttribute('aria-expanded'),'true');assert.equal(location.hash,scope);assert.equal(readingModel.currentReadingReturn(),undefined);await activate(button());assert.equal(area().querySelectorAll('a').length,2,'A restored expansion can still be explicitly collapsed');await stop();
 await render({resumedPractice:problems[0].id});assert.equal(area().querySelectorAll('a').length,2,'Returning to an initially visible concept does not expand unrelated practice');await stop();
 await render({resumedPractice:problems[2].id});assert.equal(area().querySelectorAll('a').length,4);const restored=area().querySelectorAll('a')[2];restored.focus();await render({resumedPractice:undefined});assert.equal(area().querySelectorAll('a').length,4,'Yielding parent navigation cannot remove a newly focused practice');assert.equal(document.activeElement,restored);await stop();
 console.log('PASS report practice: real card expansion/collapse, exact source return links and third-concept reading round trips, keyboard and pointer focus, newer intent, bilingual state, missing/loading/recovered catalog, explicit retry, anonymous affiliation, duplicate instances, no social reads or storage changes');
}finally{if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();for(const [key,value] of previous){if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}}
