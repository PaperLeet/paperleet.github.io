import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/community',pretendToBeVisual:true});
const reveals=[];dom.window.matchMedia=()=>({matches:false});dom.window.HTMLElement.prototype.scrollIntoView=function(options){reveals.push({node:this,options});};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__writingDisclosureFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_API_ORIGIN':JSON.stringify('')},plugins:[{name:'writing-disclosure-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0writing-disclosure-fixture';},load(id){if(id==='\0writing-disclosure-fixture')return `import React from 'react';export const useAccount=()=>globalThis.__writingDisclosureFixture.account;export const socialApi=(...args)=>globalThis.__writingDisclosureFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Community,ProblemSolutions}=await server.ssrLoadModule('/src/Community.tsx');
 const {default:Interviews}=await server.ssrLoadModule('/src/Interviews.tsx');
 const {useWritingDisclosure}=await server.ssrLoadModule('/src/useWritingDisclosure.ts');
 const drafts=await server.ssrLoadModule('/src/communityDrafts.ts');
 const interviewDrafts=await server.ssrLoadModule('/src/interviewDrafts.ts');
 const user=id=>({id,username:id,display_name:id,can_publish_community:true,affiliation:{kind:'institution',domain:'test.edu',name:'测试大学',name_en:'Test University'}});
 const account={user:user('alice'),loading:false,capabilities:{community_solutions:true,community_anonymous_posts:true,community_papers:true,community_interviews:true}};
 const requests=[],mutations=[];
 globalThis.__writingDisclosureFixture={account,request:async(path,body)=>{
  if(body){mutations.push({path,body});throw Error('Opening and closing cannot publish');}
  requests.push(path);const params=new URLSearchParams(path.split('?')[1]);return path.includes('/summary?')?{reports:0,contributors:0,basis:'published-community-reports',companies:[],questions:[]}:{items:[],total:0,limit:20,offset:Number(params.get('offset')||0),...(params.get('solutions')==='true'?{solutions:true}:{})};
 }};
 const topic={id:'architecture/attention',number:1,title:'注意力',title_en:'Attention'};
 let surface='community',scope='A',hook,showOpener=true;
 const bounds=(top,height)=>({top,bottom:top+height,height,left:0,right:600,width:600,x:0,y:top,toJSON(){}});
 const bindPane=node=>{if(node){Object.defineProperty(node,'clientHeight',{value:500,configurable:true});node.getBoundingClientRect=()=>bounds(80,500);node.scrollTo=options=>{node.scrollTop=options.top;reveals.push({node,options});};}};
 function Locale(){const {locale,setLocale}=useI18n();return React.createElement('button',{id:'locale',onClick:()=>setLocale(locale==='en'?'zh':'en')},'locale');}
 function Harness(){hook=useWritingDisclosure(JSON.stringify([scope,globalThis.location?.hash]));return React.createElement(React.Fragment,null,showOpener&&React.createElement('button',{id:'harness-open',...hook.triggerProps,ref:node=>{if(node&&surface==='pane')node.getBoundingClientRect=()=>bounds(20,44);}},'open'),hook.expanded&&React.createElement('section',hook.panelProps,React.createElement('input',{'data-writing-title':'',defaultValue:'Kept text',ref:node=>{if(node&&surface==='pane')node.getBoundingClientRect=()=>bounds(900,44);}}),React.createElement('button',{id:'harness-close',onClick:hook.close},'close')));}
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Locale),React.createElement('textarea',{'aria-label':'Practice code',defaultValue:'def solve(): return 42'}),surface==='community'?React.createElement(Community,{route:location.hash}):surface==='solutions'?React.createElement(ProblemSolutions,{topic}):surface==='interviews'?React.createElement(Interviews,{route:location.hash,problems:[],progressReady:true}):surface==='pane'?React.createElement('div',{className:'description-scroll',style:{overflowY:'auto'},ref:bindPane},React.createElement(Harness)):React.createElement(Harness)));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const click=async(node,focus=true)=>{assert(node);await act(async()=>{if(focus)node.focus();node.click();});};
 const title=()=>document.querySelector('[data-writing-title]');
 const panel=()=>document.querySelector('.community-composer,.interview-composer');
 const trigger=()=>document.querySelector('.writing-disclosure-trigger');
 const edit=async(node,value)=>act(async()=>{Object.getOwnPropertyDescriptor(dom.window[node.tagName==='TEXTAREA'?'HTMLTextAreaElement':'HTMLInputElement'].prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
 const {renderToStaticMarkup}=await import('react-dom/server'),locationDescriptor=Object.getOwnPropertyDescriptor(globalThis,'location');
 delete globalThis.location;
 try{assert(renderToStaticMarkup(React.createElement(Harness)).includes('aria-expanded="false"'),'A closed disclosure can render without browser location');}finally{Object.defineProperty(globalThis,'location',locationDescriptor);}
 for(const kind of ['community','solutions','interviews']){
  surface=kind;location.hash=kind==='interviews'?'#/interviews':kind==='solutions'?'#/1':'#/community';await render();
  const opener=trigger(),code=document.querySelector('[aria-label="Practice code"]'),href=location.href;
  code.setSelectionRange(3,10,'backward');code.scrollTop=121;code.scrollLeft=31;
  const originalDraft=JSON.stringify({...sessionStorage});assert.equal(opener.getAttribute('aria-expanded'),'false');
  code.focus();await click(opener,false);assert.equal(document.activeElement,title());assert.equal(reveals.at(-1).node,title());assert.equal(reveals.at(-1).options.block,'center');assert.equal(opener.getAttribute('aria-expanded'),'true');assert.equal(opener.getAttribute('aria-controls'),panel().id);assert.equal(JSON.stringify({...sessionStorage}),originalDraft);
  await edit(title(),'My own title 😀');await edit(panel().querySelector('textarea'),'Exact own source\n\n```python\nx = 1\n```');
  const stored=JSON.stringify({...sessionStorage}),readCount=requests.length,close=panel().querySelector('.icon-button');await click(close);assert.equal(panel(),null);assert.equal(document.activeElement,opener);assert.equal(opener.getAttribute('aria-expanded'),'false');assert.equal(JSON.stringify({...sessionStorage}),stored);assert.equal(requests.length,readCount);assert.equal(location.href,href);
  await click(opener);assert.equal(document.activeElement,title());assert.equal(title().value,'My own title 😀');assert.equal(panel().querySelector('textarea').value,'Exact own source\n\n```python\nx = 1\n```');
  title().setSelectionRange(3,7,'backward');const stableTitle=title(),beforeLocale=JSON.stringify({...sessionStorage});await click(document.querySelector('#locale'));assert.equal(title(),stableTitle);assert.equal(document.activeElement,document.querySelector('#locale'));assert.deepEqual([title().selectionStart,title().selectionEnd,title().selectionDirection],[3,7,'backward']);assert.equal(JSON.stringify({...sessionStorage}),beforeLocale);await click(document.querySelector('#locale'));
  assert.equal(document.querySelector('[aria-label="Practice code"]'),code);assert.deepEqual([code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],['def solve(): return 42',3,10,'backward',121,31]);
  await click(panel().querySelector('.icon-button'));
  for(const entry of document.querySelectorAll('.writing-disclosure-trigger')){await click(entry);assert.equal(document.activeElement,title());await click(panel().querySelector('.icon-button'));assert.equal(document.activeElement,entry,'Both interview entry points restore their own opener');}
  await click(opener);await click(document.querySelector('#locale'));await click(document.querySelector('#locale'));
  account.user=user('bob');await render();assert.equal(panel(),null);assert.equal(document.activeElement,document.querySelector('#locale'));account.user=user('alice');await render();assert.equal(panel(),null,'Returning to an owner cannot reopen a retired composer');assert.equal(JSON.stringify({...sessionStorage}),stored);
  assert(kind==='interviews'?interviewDrafts.readInterviewDraft(sessionStorage,'alice'):drafts.readPostDraft(sessionStorage,'alice',kind==='solutions'?topic.id+':solutions':'general'));await stop();
 }
 surface='harness';location.hash='#/community';await render();let opener=document.querySelector('#harness-open'),outside=document.querySelector('#locale');
 await act(async()=>{opener.focus();opener.click();outside.focus();});assert.equal(document.activeElement,outside);assert(title());
 const oldClose=hook.close;await click(document.querySelector('#harness-close'));assert.equal(document.activeElement,opener);await click(opener);const newerTitle=title();await act(async()=>oldClose());assert.equal(title(),newerTitle);assert.equal(document.activeElement,newerTitle);
 const retiredOpen=hook.triggerProps.onClick;scope='B';outside.focus();await render();assert.equal(title(),null);assert.equal(document.activeElement,outside);await act(async()=>retiredOpen({currentTarget:opener}));assert.equal(title(),null);scope='A';await render();assert.equal(title(),null);
 opener=document.querySelector('#harness-open');await act(async()=>{opener.focus();opener.click();location.hash='#/community?category=questions';outside.focus();});assert.equal(title(),null);assert.equal(document.activeElement,outside);
 await render();opener=document.querySelector('#harness-open');await click(opener);const closeFromRoute=hook.close;location.hash='#/interviews';outside.focus();await act(async()=>closeFromRoute());assert.equal(document.activeElement,outside);await render();assert.equal(title(),null);
 await click(document.querySelector('#harness-open'));outside.focus();const noReveal=reveals.length;await act(async()=>hook.close());assert.equal(title(),null);assert.equal(document.activeElement,outside);assert.equal(reveals.length,noReveal);
 await click(document.querySelector('#harness-open'));showOpener=false;await render();await click(document.querySelector('#harness-close'));assert.equal(title(),null);assert.equal(document.activeElement,document.body);showOpener=true;await render();
 opener=document.querySelector('#harness-open');await click(opener);opener.disabled=true;await click(document.querySelector('#harness-close'));assert.equal(title(),null);assert.notEqual(document.activeElement,opener);await stop();
 surface='pane';location.hash='#/1';await render();const pane=document.querySelector('.description-scroll');pane.scrollTop=120;await click(document.querySelector('#harness-open'));assert.equal(document.activeElement,title());assert.equal(reveals.at(-1).node,pane);assert.equal(pane.scrollTop,712);await click(document.querySelector('#harness-close'));assert.equal(document.activeElement,document.querySelector('#harness-open'));assert.equal(reveals.at(-1).node,pane);assert.equal(pane.scrollTop,652);
 assert.equal(mutations.length,0);await stop();
 console.log('PASS writing disclosure: actual community/embedded solutions/interview composers focus and reveal on explicit open, exact entry return, page/pane scroll surfaces, exact drafts, code/title selection and locale continuity, owner/route retirement, stale callbacks, removed/disabled entry and newer-focus guards, no publication');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,descriptor]of previous)if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];
}
