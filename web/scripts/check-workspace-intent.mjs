import assert from 'node:assert/strict';
import React,{act,useState} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

// Real intent hook and recovery editor; delayed rendering, events and timers are
// controlled locally. No browser, OS clipboard, live account or real draft is used.
const dom=new JSDOM('<div id="root"></div><button id="other">Newer work</button>',{url:'https://paperleet.test/#/22?from=collections%2Fllm-basics',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,location:dom.window.location,getComputedStyle:dom.window.getComputedStyle.bind(dom.window),IS_REACT_ACT_ENVIRONMENT:true};
const old=new Map(Object.keys(globals).map(k=>[k,Object.getOwnPropertyDescriptor(globalThis,k)]));
for(const[k,v]of Object.entries(globals))Object.defineProperty(globalThis,k,{value:v,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');window.HTMLElement.prototype.scrollIntoView=function(){};
window.HTMLElement.prototype.getClientRects=function(){return this.hasAttribute('data-test-hidden')?[]:[new window.DOMRect(0,0,120,40)];};
const nativeSet=window.setTimeout.bind(window),nativeClear=window.clearTimeout.bind(window),deadlines=new Map();let timerId=100000;
window.setTimeout=(fn,ms,...args)=>{if(ms===8000){const id=timerId++;deadlines.set(id,()=>fn(...args));return id;}return nativeSet(fn,ms,...args);};
window.clearTimeout=id=>{if(deadlines.has(id))deadlines.delete(id);else nativeClear(id);};
const watched=new Set(['pointerdown','click','keydown','beforeinput','input','compositionstart','wheel','touchstart','touchmove','focusin','visibilitychange','hashchange','popstate','blur']),listeners=new Map();
for(const target of [document,window]) {
  const add=target.addEventListener.bind(target),remove=target.removeEventListener.bind(target);
  target.addEventListener=(name,fn,options)=>{if(watched.has(name)){const key=(target===document?'d:':'w:')+name;let rows=listeners.get(key);if(!rows)listeners.set(key,rows=new Set());rows.add(fn);}return add(name,fn,options);};
  target.removeEventListener=(name,fn,options)=>{if(watched.has(name))listeners.get((target===document?'d:':'w:')+name)?.delete(fn);return remove(name,fn,options);};
}
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root;
try {
  const {createRoot}=await import('react-dom/client');
  const {useWorkspaceIntent}=await server.ssrLoadModule('/src/useWorkspaceIntent.ts');
  const {EditorFailure,EditorLoading}=await server.ssrLoadModule('/src/EditorFallback.tsx');
  const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
  const value='def solve(x):\r\n\treturn "中文 🧪"\r\n',normalized=value.replace(/\r\n/g,'\n');
  let scope='alice:22',ready=true,readOnly=false,latest,setLocale,issued,changes=0;
  const handled=[],allClaims=[];
  function Harness(){
    latest=useWorkspaceIntent(scope);setLocale=useI18n().setLocale;
    const [request,setRequest]=useState();
    function start(location){issued={source:value,location,claimFocus:latest.create()};allClaims.push(issued.claimFocus);setRequest(issued);}
    const props={value,readOnly,font:16,label:'Recovery code',onChange:()=>{changes++;},focusRequest:request,onFocusHandled:(request,focused)=>handled.push({request,focused}),retry:()=>{}};
    return React.createElement(React.Fragment,null,
      React.createElement('button',{id:'return',onClick:()=>start()},'Return to code'),
      React.createElement('button',{id:'locate',onClick:()=>start({line:2,column:2})},'Locate error'),
      ready?React.createElement(EditorFailure,{...props,key:scope}):React.createElement(EditorLoading,props));
  }
  root=createRoot(document.querySelector('#root'));
  const render=async()=>act(async()=>root.render(React.createElement(LocaleProvider,null,React.createElement(Harness))));
  const find=s=>document.querySelector(s),field=()=>find('textarea'),click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
  const timerCount=()=>deadlines.size;
  const activeListeners=()=>[...listeners.values()].reduce((n,rows)=>n+rows.size,0);
  await render();const baseListeners=activeListeners(),href=location.href,stored=JSON.stringify({...localStorage});
  const editor=field();editor.setSelectionRange(3,13,'backward');editor.scrollTop=47;editor.scrollLeft=19;
  const sameCode=()=>{assert(field()===editor);assert.deepEqual([editor.value,editor.selectionStart,editor.selectionEnd,editor.selectionDirection,editor.scrollTop,editor.scrollLeft],[normalized,3,13,'backward',47,19]);assert.equal(changes,0);};
  await click(find('#return'));assert(document.activeElement===editor);sameCode();assert.equal(handled.at(-1).focused,true);assert.equal(timerCount(),0);assert.equal(activeListeners(),baseListeners);
  editor.setSelectionRange(5,16,'backward');editor.scrollTop=53;
  await act(async()=>setLocale('zh'));assert.deepEqual([editor.selectionStart,editor.selectionEnd,editor.selectionDirection,editor.scrollTop],[5,16,'backward',53],'Locale/handler rerenders do not repeat an already handled recovery request');await act(async()=>setLocale('en'));
  await click(find('#locate'));assert(document.activeElement===editor);assert.equal(editor.selectionStart,normalized.indexOf('return'));assert.equal(editor.selectionEnd,editor.selectionStart);assert.equal(changes,0);
  // Intent waits for the actual editor, rather than focusing a loading pre.
  ready=false;await render();await click(find('#return'));const waiting=issued;assert.equal(timerCount(),1);assert.equal(document.activeElement.id,'return');ready=true;await render();assert(document.activeElement===field());assert.equal(handled.at(-1).request,waiting);assert.equal(handled.at(-1).focused,true);assert.equal(timerCount(),0);
  for(const event of ['pointerdown','click','keydown','beforeinput','input','compositionstart','wheel','touchstart','touchmove']) {
    ready=false;await render();await click(find('#locate'));const request=issued;assert.equal(timerCount(),1);
    await act(async()=>document.dispatchEvent(new window.Event(event,{bubbles:true,cancelable:true})));
    assert.equal(timerCount(),0,`New ${event} retires the pending request`);ready=true;await render();assert.equal(handled.at(-1).request,request);assert.equal(handled.at(-1).focused,false);assert(document.activeElement!==field());assert.equal(activeListeners(),baseListeners);
  }
  ready=false;await render();await click(find('#return'));find('#other').focus();assert.equal(timerCount(),0);ready=true;await render();assert(document.activeElement===find('#other'));assert.equal(handled.at(-1).focused,false);
  // A programmatic blur still fails the final active-element check without focusin.
  ready=false;await render();await click(find('#return'));find('#return').blur();ready=true;await render();assert(document.activeElement===document.body);assert.equal(handled.at(-1).focused,false);
  ready=false;await render();await click(find('#return'));const hiddenOpener=find('#return');hiddenOpener.setAttribute('data-test-hidden','');hiddenOpener.blur();await act(async()=>document.body.dispatchEvent(new window.FocusEvent('focusin',{bubbles:true})));assert.equal(timerCount(),1,'Neutral body focus after the originating pane hides keeps its explicit navigation alive');ready=true;await render();assert(document.activeElement===field());assert.equal(handled.at(-1).focused,true);hiddenOpener.removeAttribute('data-test-hidden');
  ready=false;await render();await click(find('#return'));await act(async()=>window.dispatchEvent(new window.Event('blur')));assert.equal(timerCount(),0);ready=true;await render();assert.equal(handled.at(-1).focused,false,'Leaving the window retires delayed focus');
  for(const routeEvent of ['hashchange','popstate','blur']) {
    ready=false;await render();await click(find('#return'));
    window.history.replaceState(null,'','#/interviews');await act(async()=>window.dispatchEvent(new window.Event(routeEvent)));
    assert.equal(timerCount(),0);ready=true;await render();assert(document.activeElement!==field());window.history.replaceState(null,'',href);
  }
  ready=false;await render();await click(find('#return'));window.history.replaceState(null,'','#/community');ready=true;await render();assert.equal(handled.at(-1).focused,false,'The URL is checked even before its history event arrives');window.history.replaceState(null,'',href);
  ready=false;await render();await click(find('#return'));await act(async()=>window.dispatchEvent(new window.HashChangeEvent('hashchange')));assert.equal(timerCount(),1,'An unchanged queued history notification does not cancel a current request');ready=true;await render();assert.equal(handled.at(-1).focused,true);
  ready=false;await render();await click(find('#return'));const staleCreate=latest.create,staleCancel=latest.cancel,oldClaim=issued.claimFocus;scope='bob:22';await render();assert.equal(timerCount(),0);const newClaim=latest.create();assert.equal(timerCount(),1);staleCancel();assert.equal(timerCount(),1,'An old workspace closure cannot cancel the current workspace');assert.equal(staleCreate()(),false);assert.equal(oldClaim(),false);assert.equal(newClaim(),true);assert.equal(timerCount(),0);scope='alice:22';await render();assert.equal(oldClaim(),false,'Owner A→B→A cannot revive a retired request');
  await click(find('#return'));const expired=issued;await act(async()=>[...deadlines.values()][0]());assert.equal(timerCount(),0);ready=true;await render();assert.equal(handled.at(-1).request,expired);assert.equal(handled.at(-1).focused,false);assert.equal(activeListeners(),baseListeners);
  // Read-only recovery is not focused, and a newer action still cancels while it waits.
  readOnly=true;await render();await click(find('#locate'));assert.equal(timerCount(),1);find('#other').focus();readOnly=false;await render();assert(document.activeElement===find('#other'));assert.equal(handled.at(-1).focused,false);
  ready=false;await render();await click(find('#return'));Object.defineProperty(document,'visibilityState',{configurable:true,value:'hidden'});await act(async()=>document.dispatchEvent(new window.Event('visibilitychange')));assert.equal(timerCount(),0);assert.equal(latest.create()(),false);assert.equal(timerCount(),0,'Background requests do not wait to focus on return');Object.defineProperty(document,'visibilityState',{configurable:true,value:'visible'});ready=true;await render();assert.equal(handled.at(-1).focused,false);
  // A detached originating control may yield neutral body focus while its replacement loads.
  ready=false;await render();const opener=find('#return');await click(opener);opener.remove();assert(document.activeElement===document.body);assert.equal(issued.claimFocus(),true);document.querySelector('#root').prepend(opener);ready=true;await render();
  ready=false;await render();await click(find('#return'));const unmounted=issued.claimFocus;assert.equal(timerCount(),1);await act(async()=>root.unmount());root=undefined;assert.equal(timerCount(),0);assert.equal(unmounted(),false);assert.equal(activeListeners(),baseListeners);
  assert.equal(location.href,href);assert.equal(JSON.stringify({...localStorage}),stored);assert.equal(changes,0);assert(allClaims.every(claim=>claim()===false),'Consumed, interrupted and retired requests remain single-use');
  console.log('PASS workspace navigation intent: ready/deferred recovery focus, exact selection/scroll and diagnostic offsets, newer focus/click/input/scroll/route/owner priority, stale closures and A→B→A isolation, history echoes, timeout/visibility/read-only/unmount cleanup and no source/storage mutation');
} finally {
  if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
  for(const[k,d]of old){if(d)Object.defineProperty(globalThis,k,d);else delete globalThis[k];}
}
