import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';

const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,HTMLElement:dom.window.HTMLElement,navigator:dom.window.navigator,localStorage:dom.window.localStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const originalFetch=globalThis.fetch;let requests=0;globalThis.fetch=()=>{requests++;throw Error('Navigation cannot request account or source data');};
let nextFrame=0,width=240,total=530,left=0,scrolls=0,active='papers',mode='ready',setLocale,outerScrolls=0;
const frames=new Map(),resizes=new Set();
window.requestAnimationFrame=callback=>{frames.set(++nextFrame,callback);return nextFrame;};window.cancelAnimationFrame=id=>frames.delete(id);
window.scrollTo=()=>outerScrolls++;window.HTMLElement.prototype.scrollIntoView=()=>outerScrolls++;
class Resize{constructor(callback){this.callback=callback;}observe(node){this.node=node;resizes.add(this);}disconnect(){resizes.delete(this);}}
window.ResizeObserver=Resize;
const NativeMutation=window.MutationObserver;let observers=0;
window.MutationObserver=class{constructor(callback){this.inner=new NativeMutation(callback);}observe(...args){observers++;this.inner.observe(...args);}disconnect(){observers--;this.inner.disconnect();}};
const starts={papers:[0,80],practice:[80,105],submit:[185,105],community:[290,140],interviews:[430,100]};
const nativeRect=window.HTMLElement.prototype.getBoundingClientRect;
window.HTMLElement.prototype.getBoundingClientRect=function(){
 if(this.id==='main-nav')return {left:20,right:22+width,top:100,bottom:144,width:width+2,height:44};
 if(this.dataset.tab){const[start,size]=starts[this.dataset.tab];return {left:21+start-left,right:21+start+size-left,top:100,bottom:144,width:size,height:44};}
 return nativeRect.call(this);
};
const nativeScroll=Object.getOwnPropertyDescriptor(window.Element.prototype,'scrollLeft');
Object.defineProperties(window.HTMLElement.prototype,{clientWidth:{get(){return this.id==='main-nav'?width:0;}},scrollWidth:{get(){return this.id==='main-nav'?total:0;}},clientLeft:{get(){return this.id==='main-nav'?1:0;}},scrollLeft:{get(){return this.id==='main-nav'?left:nativeScroll.get.call(this);},set(value){if(this.id==='main-nav'){left=value;scrolls++;}else nativeScroll.set.call(this,value);}}});
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-main-navigation-'),server=await createServer({root:process.cwd(),configFile:false,esbuild:{jsx:'automatic'},cacheDir,server:{watch:null,middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});let root;
try{
 const {createRoot}=await import('react-dom/client'),{LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx'),{BilingualText}=await server.ssrLoadModule('/src/BilingualText.tsx'),{default:Visibility}=await server.ssrLoadModule('/src/MainNavigationVisibility.tsx'),{default:Link}=await server.ssrLoadModule('/src/AppLink.tsx'),{deferredFeature}=await server.ssrLoadModule('/src/deferredFeature.tsx');
 const h=React.createElement,Pending=deferredFeature(()=>new Promise(()=>{}),['',''],{loading:()=>null,failure:()=>null});
 let failVisibility;
 const Failed=deferredFeature(()=>new Promise((_resolve,reject)=>{failVisibility=reject;}),['',''],{loading:()=>null,failure:()=>null});
 function Language(){setLocale=useI18n().setLocale;return null;}
 function Harness(){return h(React.Fragment,null,h(Language),h('header',{className:'topbar'},h('div',{className:'nav-inner'},h('nav',{id:'main-nav','aria-label':'Main navigation'},...Object.entries({papers:['论文题库','Papers'],practice:['工作区','Workspace'],submit:['提交论文','Submit paper'],community:['社区','Community'],interviews:['面经','Interviews']}).map(([id,[zh,en]])=>h(id==='practice'?'button':Link,{key:id,'data-tab':id,className:active===id?'active':'','aria-current':active===id?'page':undefined,...(id==='practice'?{type:'button'}:{href:id==='papers'?'#':'#/'+(id==='submit'?'submit':id)})},h(BilingualText,{zh,en}))),h(mode==='pending'?Pending:mode==='failed'?Failed:Visibility,{key:mode})))),h('textarea',{'aria-label':'Private editor',defaultValue:'PRIVATE_NAVIGATION_DRAFT'}));}
 const render=()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(h(React.StrictMode,null,h(LocaleProvider,null,h(Harness))));});
 const flush=()=>act(async()=>{await Promise.resolve();const pending=[...frames.values()];frames.clear();for(const callback of pending)callback(0);await Promise.resolve();});
 const resize=(next=width===240?241:240)=>{width=next;if(resizes.size){for(const entry of [...resizes])entry.callback();}else window.dispatchEvent(new window.Event('resize'));};
 const nav=()=>document.querySelector('#main-nav'),tab=id=>nav().querySelector(`[data-tab="${id}"]`),code=()=>document.querySelector('textarea');
 const source=JSON.stringify({...localStorage});await render();await flush();assert.equal(left,0);assert.equal(scrolls,0);assert.equal(resizes.size,1);assert.equal(observers,1,'StrictMode cleans its first observation and keeps one active subscription');
 const editor=code(),nativeLinks=[...nav().children].filter(node=>node.matches('a,button'));editor.setSelectionRange(2,14,'backward');editor.scrollTop=73;editor.scrollLeft=17;
 active='interviews';window.history.replaceState(null,'','#/interviews');await render();await flush();assert.equal(left,total-width);assert.notEqual(document.activeElement,tab('interviews'),'Route visibility never focuses a link');
 left=0;await act(async()=>tab('interviews').focus());await flush();assert.equal(left,total-width);assert.equal(document.activeElement,tab('interviews'));
 await act(async()=>tab('practice').focus());await flush();assert.equal(left,72);assert.equal(document.activeElement,tab('practice'));
 editor.focus();active='practice';await render();await flush();assert.equal(left,72);resize(180);await flush();assert.equal(left,72);assert.equal(document.activeElement,editor);width=240;
 console.log('PASS main navigation visibility: actual enhancer with native links/buttons, current-route and keyboard-focus reveal, bounded local horizontal movement, geometry changes, no document scroll or focus theft and one cleaned StrictMode observation');
 for(const type of ['pointerdown','touchstart','wheel','keydown','focusin']){
  left=0;resize();assert(frames.size);const event=new window.Event(type,{bubbles:true,cancelable:true});editor.dispatchEvent(event);await flush();assert.equal(left,0,type);assert(!event.defaultPrevented);
 }
 left=0;resize();window.history.replaceState(null,'','#/settings');await flush();assert.equal(left,0,'A route change retires a queued measurement');window.history.replaceState(null,'','#/interviews');
 for(const attr of ['hidden','inert','aria-hidden']){left=0;nav().setAttribute(attr,attr==='aria-hidden'?'true':'');await flush();resize();await flush();assert.equal(left,0);nav().removeAttribute(attr);await flush();}
 left=0;nav().parentElement.hidden=true;resize();await flush();assert.equal(left,0);nav().parentElement.hidden=false;
 tab('practice').disabled=true;resize();await flush();assert.equal(left,0);tab('practice').disabled=false;
 tab('practice').setAttribute('aria-disabled','true');resize();await flush();assert.equal(left,0);tab('practice').removeAttribute('aria-disabled');
 resize(0);await flush();assert.equal(left,0);resize(800);await flush();assert.equal(left,0);width=240;
 left=0;resize();await act(async()=>editor.focus());editor.dispatchEvent(new window.FocusEvent('focusin',{bubbles:true}));await flush();assert.equal(left,0);
 console.log('PASS main navigation intent: newer pointer/touch/wheel/key/focus and route changes cancel queued reveals; hidden/inert/aria-hidden ancestors, disabled targets, unknown geometry and non-overflowing rows never scroll');
 active='interviews';await render();left=0;await flush();assert.equal(left,total-width);const link=tab('interviews');await act(async()=>setLocale('zh'));await flush();assert.equal(tab('interviews'),link);assert.equal(link.querySelector('[lang="zh-CN"]').getAttribute('aria-hidden'),'false');assert.equal(link.querySelector('[lang="en"]').getAttribute('aria-hidden'),'true');
 for(let i=0;i<nativeLinks.length;i++)assert.equal(nav().querySelectorAll('a,button')[i],nativeLinks[i]);
 assert.equal(code(),editor);assert.deepEqual([editor.value,editor.selectionStart,editor.selectionEnd,editor.selectionDirection,editor.scrollTop,editor.scrollLeft],['PRIVATE_NAVIGATION_DRAFT',2,14,'backward',73,17]);assert.equal(JSON.stringify({...localStorage}).replace('"zh"','"en"'),source);assert.equal(requests,0);assert.equal(outerScrolls,0);
 await act(async()=>root.unmount());root=undefined;assert.equal(frames.size,0);assert.equal(resizes.size,0);assert.equal(observers,0);resize();assert.equal(frames.size,0);
 left=110;await render();await flush();assert.equal(left,110,'Late enhancement loading preserves a previously scrolled row');for(const entry of resizes)entry.callback();window.dispatchEvent(new window.Event('resize'));await flush();assert.equal(left,110,'Initial observations and height-only resize do not recenter manual reading');await act(async()=>root.unmount());root=undefined;
 window.ResizeObserver=undefined;active='interviews';left=0;await render();await flush();assert.equal(left,total-width);left=0;resize();await flush();assert.equal(left,total-width,'Window resize remains a fallback without ResizeObserver');
 const fallbackEditor=code();fallbackEditor.setSelectionRange(3,9,'forward');fallbackEditor.scrollTop=31;fallbackEditor.scrollLeft=12;
 mode='pending';await render();await flush();assert.equal(resizes.size,0);assert.equal(observers,0);assert.equal(nav().querySelectorAll('a,button').length,5);assert.equal(tab('community').getAttribute('href'),'#/community');
 const errors=[],originalError=console.error;console.error=(...args)=>{errors.push(args);};try{mode='failed';await render();assert(failVisibility);await act(async()=>failVisibility(Error('Synthetic visibility bundle failure')));await flush();}finally{console.error=originalError;}assert(errors.some(args=>args.some(value=>String(value).includes('Synthetic visibility bundle failure'))));assert.equal(nav().querySelectorAll('a,button').length,5);assert(!nav().querySelector('[role="alert"],[role="status"]'));assert.equal(code(),fallbackEditor);assert.deepEqual([fallbackEditor.value,fallbackEditor.selectionStart,fallbackEditor.selectionEnd,fallbackEditor.selectionDirection,fallbackEditor.scrollTop,fallbackEditor.scrollLeft],['PRIVATE_NAVIGATION_DRAFT',3,9,'forward',31,12]);
 console.log('PASS main navigation continuity: native bilingual links and private editor/caret/two-scroll survive locale changes and optional loading/failure; late loading/manual-scroll and height-only resize retention, resize fallback, unmount cleanup, zero data requests, zero document scrolling and no navigation-state persistence');
}finally{if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});globalThis.fetch=originalFetch;dom.window.close();for(const[key,value]of previous)value?Object.defineProperty(globalThis,key,value):delete globalThis[key];}
