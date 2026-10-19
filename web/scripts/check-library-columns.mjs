import assert from 'node:assert/strict';
import fs from 'node:fs';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
const dom=new JSDOM('<div id="root"></div>',{url:'https://paperleet.test/#/?view=modules',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,Element:dom.window.Element,location:dom.window.location,history:dom.window.history,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,innerHeight:900,scrollY:0,scrollTo(){},ResizeObserver:class{observe(){}disconnect(){}},requestAnimationFrame:callback=>setTimeout(callback,0),cancelAnimationFrame:clearTimeout,IS_REACT_ACT_ENVIRONMENT:true};
const original=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
window.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});window.scrollTo=()=>{};window.HTMLElement.prototype.scrollIntoView=function(){};localStorage.setItem('paperleet.locale','en');
const catalog=JSON.parse(fs.readFileSync('public/data/catalog.json','utf8'));let root;
try{
 const {createRoot}=await import('react-dom/client');
 for(const mode of ['loading','failure']){
  let release;globalThis.__libraryColumns={mode,ready:new Promise(resolve=>release=resolve)};
  const cacheDir=await mkdtemp(tmpdir()+'/paperleet-library-columns-'),server=await createServer({cacheDir,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'held-real-library-columns',enforce:'pre',resolveId(source){if(source==='./LibraryResultControls')return '\0held-library-controls';},load(id){if(id==='\0held-library-controls')return `import ResultControls,{LibraryActiveFilters,LibraryTableHeaders} from '/src/LibraryResultControls.tsx';await globalThis.__libraryColumns.ready;if(globalThis.__libraryColumns.mode==='failure')throw Error('Controlled optional library-controls failure');export default ResultControls;export {LibraryActiveFilters,LibraryTableHeaders};`;}}]});
  try{
   const {PaperLibrary}=await server.ssrLoadModule('/src/PaperViews.tsx'),{LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx'),nav=await server.ssrLoadModule('/src/libraryNavigation.ts');let route='#/?view=modules',setLocale;const changes=[];
   const tree=()=>React.createElement(React.StrictMode,null,React.createElement(LocaleProvider,null,React.createElement(Frame)));
   function Frame(){setLocale=useI18n().setLocale;const view=nav.readLibraryRoute(route);return React.createElement(React.Fragment,null,React.createElement('textarea',{id:'private-column-code',defaultValue:'PRIVATE_COLUMN_SOURCE'}),React.createElement(PaperLibrary,{...catalog,route,collectionId:view?.collectionId,saved:{},onFilters:href=>{changes.push(href);route=href;history.replaceState(null,'',href);root.render(tree());}}));}
   const render=async href=>act(async()=>{if(href){route=href;history.replaceState(null,'',href);}root??=createRoot(document.getElementById('root'));root.render(tree());});
   const wait=async predicate=>{for(let n=0;n<160&&!predicate();n++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),'Library column state did not settle');};
   const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};const column=field=>document.querySelector(`[data-sort-column="${field}"]`),rows=()=>[...document.querySelectorAll('[data-module-id]')],head=()=>document.querySelector('.table-head');
   await render(route);await wait(()=>rows().length===20&&document.querySelector('.collection-card')&&document.querySelector('.library-overview'));const code=document.getElementById('private-column-code');code.setSelectionRange(2,9,'backward');code.scrollTop=47;code.scrollLeft=13;const snapshot=[code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft];assert.equal(head().children.length,5);assert(!column('difficulty'),'Loading presents plain column labels, not dead controls');assert.equal(head().children[3].textContent,'Difficulty');
   await act(async()=>release());
   if(mode==='failure'){
    await wait(()=>document.querySelector('.feature-failure'));assert.equal(head().children.length,5);assert(!column('difficulty'));assert.equal(head().children[3].textContent,'Difficulty');assert.equal(rows().length,20);const search=document.querySelector('input[aria-label="Search papers or modules"]');await act(async()=>{Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set.call(search,String(catalog.problems[0].number));search.dispatchEvent(new window.Event('input',{bubbles:true}));});await wait(()=>rows().length===1);assert(document.activeElement!==code);assert.deepEqual([code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],snapshot);
    console.log('PASS library column fallback: held and rejected optional controls retain five labelled columns, usable real exercise/search rows and unchanged private source/caret/two-scroll');continue;
   }
   await wait(()=>column('difficulty')&&document.querySelector('select[aria-label="Sort exercises"]'));
   for(const [field,orders]of [['duration',['duration','duration-desc','default']],['difficulty',['difficulty','difficulty-desc','default']],['title',['title','default']]]){
    const control=column(field);assert.equal(control.tagName,'BUTTON');assert.equal(document.getElementById(control.getAttribute('aria-controls')),document.querySelector('.problem-rows'));
    for(const sort of orders){await click(control);assert.equal(nav.readLibraryRoute(route).filters.sort,sort==='default'?undefined:sort);assert.equal(document.querySelector('select[aria-label="Sort exercises"]').value,sort);assert(column(field)===control);assert(document.activeElement===control);assert.equal(control.getAttribute('aria-pressed'),sort==='default'?'false':'true');assert.equal(nav.readLibraryRoute(route).filters.page,undefined);}
   }
   await click(column('difficulty'));const level=column('difficulty');assert(level.getAttribute('aria-label').includes('Easy first'));assert(level.getAttribute('aria-label').includes('Hard first'));await act(async()=>setLocale('zh'));assert(column('difficulty')===level);assert(document.activeElement===level);assert(level.getAttribute('aria-label').includes('简单优先'));await act(async()=>setLocale('en'));
   const before=changes.length,source=location.hash;document.getElementById('paper-library').hidden=true;code.focus();await act(async()=>level.click());assert.equal(changes.length,before);assert(document.activeElement===code);document.getElementById('paper-library').hidden=false;history.replaceState(null,'','#/favorites');await act(async()=>level.click());assert.equal(changes.length,before);assert.equal(location.hash,'#/favorites');assert(document.activeElement===code);history.replaceState(null,'',source);
   await render('#/?page=2');await wait(()=>column('year'));const year=column('year');for(const sort of ['newest','oldest','default']){await click(year);assert.equal(nav.readLibraryRoute(route).filters.sort,sort==='default'?undefined:sort);assert.equal(nav.readLibraryRoute(route).filters.page,undefined);assert(column('year')===year);assert(document.activeElement===year);}
   await render('#/collections/llm-basics');await wait(()=>document.querySelector('[data-module-id]'));assert(!document.querySelector('.library-column-sort'),'Course columns retain their existing source order');assert.equal(head().children.length,5);assert.deepEqual([code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],snapshot);
   console.log('PASS actual library column sorting: native title/time/difficulty/year cycles, selected/direction descriptions and list association, synchronized select/page reset, stable locale/focus, hidden/stale guard and unchanged course/private editor state');
  }finally{if(root)await act(async()=>root.unmount());root=undefined;await server.close();await rm(cacheDir,{recursive:true,force:true});}
 }
}finally{delete globalThis.__libraryColumns;dom.window.close();for(const[key,value]of original)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}
