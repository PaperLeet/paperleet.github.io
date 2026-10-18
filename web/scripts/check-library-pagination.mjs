import assert from 'node:assert/strict';
import fs from 'node:fs';
import React, {act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom = new JSDOM('<div id="root"></div>', {url:'https://paperleet.test/#'});
const observers=[];
class Observer {
  constructor(callback){this.callback=callback;observers.push(this);}
  observe(target){this.target=target;}
  disconnect(){this.disconnected=true;}
  enter(){this.callback([{target:this.target,isIntersecting:true}]);}
}
const globals={window:dom.window,document:dom.window.document,Element:dom.window.Element,HTMLElement:dom.window.HTMLElement,location:dom.window.location,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,scrollY:0,innerHeight:900,scrollTo(x,y){globalThis.scrollY=y;},IntersectionObserver:Observer,ResizeObserver:class{observe(){} disconnect(){}},requestAnimationFrame:cb=>setTimeout(cb,0),cancelAnimationFrame:clearTimeout,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
window.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
window.HTMLElement.prototype.scrollIntoView=()=>{};
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root;
try {
 const {createRoot}=await import('react-dom/client');
 const {PaperLibrary}=await server.ssrLoadModule('/src/PaperViews.tsx');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const catalog=JSON.parse(fs.readFileSync('public/data/catalog.json','utf8'));
 const {readLibraryPosition}=await server.ssrLoadModule('/src/libraryPosition.ts');
 const ordered=[...catalog.papers].sort((a,b)=>a.number-b.number);
 const rows=()=>[...document.querySelectorAll('[data-paper-id]')];
 const flush=async()=>{await act(async()=>new Promise(resolve=>setTimeout(resolve,30)));};
 const waitFor=async predicate=>{for(let i=0;i<100&&!predicate();i++)await flush();assert(predicate());};
 const render=async(route='#',props={})=>{location.hash=route;await act(async()=>root.render(React.createElement(LocaleProvider,null,React.createElement(PaperLibrary,{...catalog,papers:[...catalog.papers].reverse(),saved:{},route,...props}))));};
 root=createRoot(document.querySelector('#root'));
 await render();await waitFor(()=>observers.at(-1)?.target);
 assert.equal(rows().length,20);
 assert.deepEqual(rows().map(row=>row.dataset.paperId),ordered.slice(0,20).map(p=>p.id));
 assert(document.querySelector('.library-results').textContent.includes(`${catalog.problems.length} modules`),'Totals include unloaded results');
 assert.deepEqual([...rows()[0].querySelectorAll('.paper-title-tags > span')].map(el=>el.textContent),ordered[0].tags);
 await act(async()=>{observers.at(-1).enter();observers.at(-1).enter();});
 assert.equal(rows().length,40,'Duplicate observer entries append only one batch');
 const old=observers.at(-1);
 const target=rows()[34];
 globalThis.scrollY=2400;
 target.getBoundingClientRect=()=>({top:350,bottom:430});
 target.addEventListener('click',event=>event.preventDefault());
 await act(async()=>target.click());
 assert.equal(readLibraryPosition('#').anchor.id,ordered[34].id);
 assert.equal(readLibraryPosition('#').count,40);
 await act(async()=>root.unmount());
 assert(old.disconnected);
 globalThis.scrollY=0;
 window.HTMLElement.prototype.getBoundingClientRect=function(){const i=rows().indexOf(this);return {top:300+80*i-globalThis.scrollY,bottom:380+80*i-globalThis.scrollY};};
 root=createRoot(document.querySelector('#root'));
 await render();await waitFor(()=>observers.at(-1)!==old&&observers.at(-1)?.target);
 assert.equal(rows().length,40,'Returning restores the batch before restoring its anchor');
 assert.equal(rows()[34].getBoundingClientRect().top,350);
 const beforeFilter=observers.at(-1);
 await render('#/?q=Dropout');await flush();
 assert.equal(rows().length,1,'Search includes results outside the first batch');
 assert.equal(rows()[0].dataset.paperId,'dropout');
 await act(async()=>beforeFilter.enter());assert.equal(rows().length,1,'Disposed observers cannot alter a new view');
 await render('#/?sort=newest');await waitFor(()=>observers.at(-1)!==beforeFilter);
 assert.equal(rows().length,20,'A new sort starts with one batch');
 assert.deepEqual(rows().map(row=>row.dataset.paperId),[...ordered].sort((a,b)=>!a.year||!b.year?a.year?-1:b.year?1:a.number-b.number:b.year-a.year||a.number-b.number).slice(0,20).map(p=>p.id));
 while(document.querySelector('.library-load-more'))await act(async()=>document.querySelector('.library-load-more').click());
 assert.equal(rows().length,catalog.papers.length,'Manual fallback reaches the exact end');
 assert.equal(new Set(rows().map(row=>row.dataset.paperId)).size,catalog.papers.length);
 await render('#/?state=done',{progressReady:false});await flush();
 assert.equal(rows().length,0);assert.equal(document.querySelector('.library-load-more'),null);
 console.log('Library pagination passed: batches, number order, complete search/totals, inline tags, scroll return and stale observer cleanup.');
} finally {
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
 for(const [key,value] of previous)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];
}
