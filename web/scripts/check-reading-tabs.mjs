import assert from 'node:assert/strict';
import React,{act,createRef,useState} from 'react';
import {JSDOM,VirtualConsole} from 'jsdom';
import {createServer} from 'vite';

const errors=[],consoleProxy=new VirtualConsole();
consoleProxy.on('jsdomError',error=>errors.push(error));
const dom=new JSDOM('<div id="root"></div>',{url:'https://paperleet.test/#/92?from=interviews%3Ftrack%3Dnormalization',virtualConsole:consoleProxy});
const observers=[];
class Observer {
  constructor(callback){this.callback=callback;this.nodes=new Set();observers.push(this);}
  observe(node){this.nodes.add(node);}
  disconnect(){this.disconnected=true;}
}
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,Element:dom.window.Element,localStorage:dom.window.localStorage,location:dom.window.location,ResizeObserver:Observer,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const writes=[],originalSet=dom.window.Storage.prototype.setItem;
dom.window.Storage.prototype.setItem=function(key,value){writes.push(key);originalSet.call(this,key,value);};
window.scrollTo=()=>{throw Error('Tab navigation must not scroll the page');};
window.HTMLElement.prototype.scrollIntoView=function(){};
const refs={container:createRef(),viewport:createRef(),content:createRef()};
const metrics={width:500,natural:700},offsets=new WeakMap();
const controls=()=>document.querySelector('.reading-tab-controls');
const visibleWidth=()=>Math.max(0,metrics.width-(controls()?64:0));
// JSDOM supplies no layout. Only geometry is simulated; actual React, native
// focus/selection, locale, keyboard activation and the controller remain real.
Object.defineProperties(window.HTMLElement.prototype,{
  clientWidth:{configurable:true,get(){return this.classList.contains('reading-tab-strip')?metrics.width:this.classList.contains('reading-tab-scroll')?visibleWidth():0;}},
  scrollWidth:{configurable:true,get(){return Math.max(metrics.natural,visibleWidth());}},
  scrollLeft:{configurable:true,get(){const value=offsets.get(this)||0;return this.classList.contains('reading-tab-scroll')?Math.max(0,Math.min(this.scrollWidth-this.clientWidth,value)):value;},set(value){offsets.set(this,value);}},
});
const originalRect=window.HTMLElement.prototype.getBoundingClientRect;
window.HTMLElement.prototype.getBoundingClientRect=function(){
  if(this.classList.contains('reading-tab-scroll'))return new window.DOMRect(0,0,this.clientWidth,44);
  if(this.hasAttribute('data-index'))return new window.DOMRect(Number(this.dataset.index)*100*metrics.natural/700-(refs.viewport.current?.scrollLeft||0),0,100*metrics.natural/700,44);
  return originalRect.call(this);
};
const server=await createServer({server:{middlewareMode:true},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root;
try{
  const {createRoot}=await import('react-dom/client');
  const {default:Controls}=await server.ssrLoadModule('/src/ReadingTabControls.tsx');
  const {navigateTabs}=await server.ssrLoadModule('/src/tabNavigation.ts');
  const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
  const h=React.createElement,keys=['description','result','explanation','lab','notes','solution'];
  function Frame(){
    const [selected,setSelected]=useState('description'),{locale,setLocale}=useI18n();
    return h(React.Fragment,null,
      h('button',{id:'show-result',onClick:()=>setSelected('result')},'Show result'),
      h('button',{id:'change-locale',onClick:()=>setLocale(locale==='en'?'zh':'en')},'Language'),
      h('div',{className:'reading-tab-strip',ref:refs.container},
        h('div',{className:'reading-tab-scroll',ref:refs.viewport},h('div',{id:'test-reading-tabs',className:'pane-tabs',ref:refs.content,role:'tablist',onKeyDown:navigateTabs},
          ...keys.map((key,index)=>h('button',{key,role:'tab','data-index':index,'data-key':key,tabIndex:selected===key?0:-1,'aria-selected':selected===key,onClick:()=>setSelected(key)},key)),
          h('a',{'data-index':6,href:'#/community/problem/architecture/group_norm'},'Discuss'))),
        h(Controls,{...refs,controlsId:'test-reading-tabs',selectedKey:selected})),
      h('textarea',{id:'editor',defaultValue:'original code — 保留 🧪\nsecond line'}));
  }
  const mount=async()=>{root=createRoot(document.querySelector('#root'));await act(async()=>root.render(h(LocaleProvider,null,h(Frame))));};
  const click=async node=>{assert(node);await act(async()=>node.click());};
  const resize=async()=>{await act(async()=>observers.at(-1).callback([]));};
  const selected=()=>document.querySelector('[role="tab"][aria-selected="true"]');
  const editorState=editor=>[editor.value,editor.selectionStart,editor.selectionEnd,editor.scrollTop];
  const href=location.href;
  await mount();
  const editor=document.querySelector('#editor');editor.focus();editor.setSelectionRange(3,13);editor.scrollTop=41;
  const original=editorState(editor),initialTabs=[...document.querySelectorAll('[role="tab"]')];
  assert.equal(observers.at(-1).nodes.size,3,'Track viewport, whole bar and intrinsic content dimensions');
  let [left,right]=controls().querySelectorAll('button');
  assert.equal(left.getAttribute('aria-disabled'),'true');assert.equal(right.getAttribute('aria-disabled'),'false');
  assert.equal(right.getAttribute('type'),'button');assert.equal(right.getAttribute('aria-controls'),'test-reading-tabs');
  right.focus();await click(right);
  assert.equal(document.activeElement,right,'Reaching the edge must not discard focus');
  assert.equal(right.getAttribute('aria-disabled'),'true');
  const link=document.querySelector('[data-index="6"]'),bounds=refs.viewport.current.getBoundingClientRect();
  assert(link.getBoundingClientRect().left>=bounds.left&&link.getBoundingClientRect().right<=bounds.right,'The previously clipped discussion link becomes fully reachable');
  assert.equal(selected().dataset.key,'description','Scrolling does not change the selected reading view');
  assert.deepEqual(editorState(editor),original);assert.equal(document.querySelector('#editor'),editor);
  const edge=refs.viewport.current.scrollLeft;await click(right);assert.equal(refs.viewport.current.scrollLeft,edge);
  selected().focus();await act(async()=>selected().dispatchEvent(new window.KeyboardEvent('keydown',{key:'End',bubbles:true})));
  assert.equal(selected().dataset.key,'solution');assert.equal(document.activeElement,selected());
  editor.focus();await click(document.querySelector('#show-result'));
  assert.equal(selected().dataset.key,'result');assert.equal(document.activeElement,editor,'Programmatic result selection reveals its tab without stealing editor focus');
  assert(selected().getBoundingClientRect().left>=0,'The selected result tab is revealed');
  assert.deepEqual([...document.querySelectorAll('[role="tab"]')],initialTabs,'Controls never replace tab nodes');
  refs.viewport.current.scrollLeft=999;
  await act(async()=>refs.viewport.current.dispatchEvent(new window.Event('scroll')));
  assert.equal(refs.viewport.current.scrollLeft,edge,'Manual browsing does not snap back to the selected tab');
  metrics.width=320;await resize();assert.equal(document.activeElement,editor);assert(selected().getBoundingClientRect().left>=0);
  right=controls().querySelectorAll('button')[1];right.focus();metrics.width=900;await resize();
  assert(!controls());assert.equal(document.activeElement,selected(),'Widening repairs focus only when a disappearing control owned it');
  assert.deepEqual(editorState(editor),original);assert.equal(location.href,href);assert.deepEqual(writes,[]);
  metrics.width=500;await resize();editor.focus();metrics.natural=400;await click(document.querySelector('#change-locale'));
  assert(!controls());assert.equal(document.activeElement,editor,'A newer editor focus survives a language/layout update');
  assert.deepEqual(writes,['paperleet.locale']);assert.deepEqual(editorState(editor),original);
  metrics.natural=900;await resize();assert(controls().querySelector('[aria-label*="向右"]'),'Localized labels survive intrinsic content changes');
  const retired=observers.at(-1);await act(async()=>root.unmount());assert(retired.disconnected);
  metrics.width=500;metrics.natural=700;globalThis.ResizeObserver=undefined;await mount();
  const fresh=document.querySelector('#editor');fresh.focus();refs.viewport.current.scrollLeft=999;
  const manual=refs.viewport.current.scrollLeft;await act(async()=>retired.callback([]));
  assert.equal(refs.viewport.current.scrollLeft,manual);assert.equal(document.activeElement,fresh,'A retired observer cannot act on a new workspace sharing the refs');
  metrics.width=950;await act(async()=>window.dispatchEvent(new window.Event('resize')));assert(!controls(),'Window resize remains a fallback without ResizeObserver');
  assert.equal(document.activeElement,fresh);assert.equal(location.href,href);assert.deepEqual(errors,[]);
  console.log('PASS reading tab navigation: hidden-link access, stable edge focus, keyboard activation, automatic result reveal, manual scroll intent, resize/locale focus, unchanged code/selection/scroll/route, cleanup and observer fallback');
}finally{
  if(root)await act(async()=>root.unmount());
  await server.close();dom.window.close();
  for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
