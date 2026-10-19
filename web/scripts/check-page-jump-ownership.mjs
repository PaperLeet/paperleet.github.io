import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
const web=process.cwd();
const dom=new JSDOM('<!doctype html><div id="root"></div><button id="outside">Outside</button>',{url:'https://paperleet.test/#/?view=modules&page=2',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});localStorage.setItem('paperleet.locale','en');
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-jump-ownership-');
const server=await createServer({root:web,configFile:false,esbuild:{jsx:'automatic'},cacheDir,server:{watch:null,middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});let root;
try{
 const {createRoot}=await import('react-dom/client'),{default:PageJump}=await server.ssrLoadModule('/src/PageJump.tsx'),{LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 let changeLocale;const changes=[],route=location.href;
 function Frame(){changeLocale=useI18n().setLocale;return React.createElement(PageJump,{page:2,maxPage:8,onChange:p=>changes.push(p),label:'Jump to library page',controls:'results',canAct:()=>location.href===route});}
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(React.StrictMode,null,React.createElement(LocaleProvider,null,React.createElement(Frame))));});
 const form=()=>document.querySelector('form'),field=()=>form().querySelector('input'),outside=document.querySelector('#outside');
 const input=async value=>act(async()=>{Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set.call(field(),value);field().dispatchEvent(new window.Event('input',{bubbles:true}));});
 const submit=async()=>act(async()=>form().dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true})));
 const key=async()=>{const event=new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true});await act(async()=>field().dispatchEvent(event));return event;};
 await render();const node=field();await input('0');outside.focus();await submit();assert(document.activeElement===node);assert.equal(node.getAttribute('aria-invalid'),'true');assert(document.querySelector('[role=alert]'));await key();assert.equal(node.value,'2');assert(!node.hasAttribute('aria-invalid'));
 for(const attr of ['hidden','inert','aria-hidden']){
  await input('0');form().setAttribute(attr,attr==='aria-hidden'?'true':'');outside.focus();await submit();assert(document.activeElement===outside,`${attr} page jump must not steal focus`);assert(!node.hasAttribute('aria-invalid'),`${attr} page jump must not announce a new error`);assert(!(await key()).defaultPrevented,`${attr} page jump leaves Escape unhandled`);assert.equal(node.value,'0');
  await input('3');await submit();assert.deepEqual(changes,[],`${attr} page jump cannot navigate`);form().removeAttribute(attr);
 }
 await input('0');window.history.replaceState(null,'','#/favorites');outside.focus();await submit();assert(document.activeElement===outside,'A retired invalid jump cannot steal focus');assert(!node.hasAttribute('aria-invalid'));assert(!(await key()).defaultPrevented);await act(async()=>changeLocale('zh'));assert(field()===node);await submit();assert(document.activeElement===outside,'Locale changes do not reactivate retired controls');await input('3');await submit();assert.deepEqual(changes,[],'A retired valid jump cannot navigate');
 window.history.replaceState(null,'',route);await render();await input('４');await act(async()=>changeLocale('zh'));assert(field()===node);assert.equal(node.value,'４');assert.equal(node.getAttribute('aria-controls'),'results');await submit();assert.deepEqual(changes,[4]);assert.equal(node.value,'4');
 await input('5');await act(async()=>node.dispatchEvent(new window.CompositionEvent('compositionstart',{bubbles:true})));await submit();assert.deepEqual(changes,[4]);await act(async()=>node.dispatchEvent(new window.CompositionEvent('compositionend',{bubbles:true})));await submit();assert.deepEqual(changes,[4,5]);
 const staleForm=form(),staleField=node;await act(async()=>root.unmount());root=undefined;outside.focus();staleForm.dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true}));assert(document.activeElement===outside);assert.deepEqual(changes,[4,5]);assert(!staleField.isConnected);
 console.log('PASS page-jump ownership: actual shared native field, invalid input recovery, hidden/inert/aria-hidden and retired route guards, Escape ownership, locale-stable DOM, full-width digits, IME and disconnected controls');
}finally{if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();for(const[key,value]of previous)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}
