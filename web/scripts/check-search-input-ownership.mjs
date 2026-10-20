import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';

const dom=new JSDOM('<div id="root"></div>',{url:'https://paperleet.test/#/interviews?view=questions',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true};
const original=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-search-owner-');
const server=await createServer({root:process.cwd(),configFile:false,esbuild:{jsx:'automatic'},cacheDir,server:{watch:null,middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root;
try{
 const {createRoot}=await import('react-dom/client'),{LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx'),{default:Search}=await server.ssrLoadModule('/src/CommunitySearch.tsx');
 let locale,allowed=true,disabled=false,blockKey=false,blockSubmit=false,applied='Alpha';const calls=[],href=location.href;
 function Frame(){locale=useI18n();return React.createElement(React.Fragment,null,
  React.createElement('textarea',{id:'private-code',defaultValue:'PRIVATE_SEARCH_OWNER_CODE'}),
  React.createElement('dialog',{open:true},React.createElement('details',{open:true},React.createElement('summary',null,'Search filters'),
   React.createElement('section',{id:'search-surface',onKeyDownCapture:event=>{if(blockKey)event.preventDefault();},onSubmitCapture:event=>{if(blockSubmit)event.preventDefault();}},
    React.createElement(Search,{applied,disabled,canAct:()=>allowed&&location.href===href,onApply:value=>calls.push(value),label:'Search experiences'})))));
 }
 const tree=()=>React.createElement(React.StrictMode,null,React.createElement(LocaleProvider,null,React.createElement(Frame)));
 const render=async()=>act(async()=>{root??=createRoot(document.getElementById('root'));root.render(tree());});
 const field=()=>document.querySelector('input[type=search]'),form=()=>document.querySelector('form'),surface=()=>document.getElementById('search-surface');
 const edit=async value=>act(async()=>{Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set.call(field(),value);field().dispatchEvent(new window.Event('input',{bubbles:true}));});
 const key=async(value='Escape',extra={})=>{const event=new window.KeyboardEvent('keydown',{key:value,bubbles:true,cancelable:true,...extra});await act(async()=>field().dispatchEvent(event));return event;};
 const submit=async()=>act(async()=>form().dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true})));
 const click=async selector=>act(async()=>{const node=document.querySelector(selector);assert(node,selector);node.click();});
 await render();const code=document.getElementById('private-code'),input=field();code.setSelectionRange(2,12,'backward');code.scrollTop=57;code.scrollLeft=19;
 const privateState=()=>[code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft];
 const saved=privateState(),storage=JSON.stringify({...localStorage});
 await edit('UNAPPLIED_SEARCH');input.focus();input.setSelectionRange(2,8,'backward');input.scrollLeft=21;
 blockKey=true;const handled=await key();assert(handled.defaultPrevented);assert.equal(input.value,'UNAPPLIED_SEARCH','An already-handled Escape must preserve pending search text');assert.equal(calls.length,0);assert.equal(document.activeElement,input);
 blockKey=false;blockSubmit=true;await edit('  pending search  ');await submit();assert.equal(input.value,'  pending search  ','A parent-owned submit must not normalize or apply input');assert.equal(calls.length,0);blockSubmit=false;
 await edit('NEWER_EDITOR_FOCUS');code.focus();const unfocused=await key();assert(!unfocused.defaultPrevented);assert.equal(input.value,'NEWER_EDITOR_FOCUS');assert.equal(document.activeElement,code);assert.equal(calls.length,0);
 console.log('PASS shared search event ownership: already-handled Escape and submit preserve the draft, native input and caller-owned navigation');

 for(const mode of ['hidden','inert','aria-hidden','closed-details','closed-dialog','retired-owner','retired-route','disabled']){
  await edit('KEPT_'+mode);const before=input.value,count=calls.length;
  if(['hidden','inert','aria-hidden'].includes(mode))surface().setAttribute(mode,mode==='aria-hidden'?'true':'');
  if(mode==='closed-details')document.querySelector('details').open=false;
  if(mode==='closed-dialog')document.querySelector('dialog').open=false;
  if(mode==='retired-owner')allowed=false;
  if(mode==='retired-route')window.history.replaceState(null,'','#/favorites');
  if(mode==='disabled'){disabled=true;await render();}
  code.focus();const ignored=await key();assert(!ignored.defaultPrevented,mode+' search cannot consume Escape');
  await submit();await click('.community-search-clear');
  assert.equal(input.value,before,mode+' search cannot clear pending input');assert.equal(calls.length,count,mode+' search cannot change the result scope');assert.equal(document.activeElement,code,mode+' search cannot steal newer editor focus');
  await act(async()=>locale.setLocale('zh'));assert.equal(field(),input);assert.equal(input.value,before);await submit();assert.equal(calls.length,count,'Locale changes cannot reactivate '+mode+' controls');await act(async()=>locale.setLocale('en'));
  if(['hidden','inert','aria-hidden'].includes(mode))surface().removeAttribute(mode);
  if(mode==='closed-details')document.querySelector('details').open=true;
  if(mode==='closed-dialog')document.querySelector('dialog').open=true;
  allowed=true;window.history.replaceState(null,'',href);disabled=false;await render();
 }
 assert.deepEqual(privateState(),saved);assert.equal(JSON.stringify({...localStorage}),storage);
 console.log('PASS shared search availability: hidden/inert/aria-hidden, closed native disclosures/dialogs, retired owner/route and disabled controls retain input and never steal private-editor focus across locale changes');

 await edit('Beta');input.focus();await click('.community-search-restore');assert.equal(input.value,'Alpha');assert.equal(calls.length,0);assert.equal(document.activeElement,input);
 await edit('  Explicit search  ');await submit();assert.deepEqual(calls,['Explicit search']);assert.equal(input.value,'Explicit search');applied='Explicit search';await render();
 await edit('Unapplied replacement');input.focus();const normal=await key();assert(normal.defaultPrevented);assert.deepEqual(calls,['Explicit search','']);assert.equal(input.value,'');assert.equal(document.activeElement,input);
 assert.equal(field(),input);assert.deepEqual(privateState(),saved);assert.equal(location.href,href);assert.equal(JSON.stringify({...localStorage}),storage);
 console.log('PASS live shared search recovery: explicit restore/search/Escape still work, normalization and native field identity remain, and private source/caret/two-scroll/storage are unchanged');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();
 for(const[key,value]of original)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];
}
