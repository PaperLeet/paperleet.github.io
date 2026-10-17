import assert from 'node:assert/strict';
import fs from 'node:fs';
import React,{act,useState} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/interviews',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');sessionStorage.setItem('private-draft','retained source');
const server=await createServer({cacheDir:'node_modules/.vite-related-module-check',server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Fields}=await server.ssrLoadModule('/src/InterviewFields.tsx');
 const {searchRelatedModules,relatedModuleQuery}=await server.ssrLoadModule('/src/relatedModuleMatching.ts');
 const published=JSON.parse(fs.readFileSync('public/data/index.json','utf8')).problems;
 assert(published.length>=78);
 for(const problem of published){
  for(const title of [problem.title,problem.translations?.en?.title||problem.title])assert(searchRelatedModules(published,`${problem.number}. ${title}`).some(row=>row.id===problem.id));
  assert(searchRelatedModules(published,`#${problem.number} · ${problem.title}`).some(row=>row.id===problem.id));
  assert(searchRelatedModules(published,problem.id).some(row=>row.id===problem.id));
 }
 assert.equal(relatedModuleQuery('😀'.repeat(120)).length,200);
 assert.equal(searchRelatedModules(published,'.* [not-a-module]').length,0);
 const modules=Array.from({length:20},(_,index)=>({id:'architecture/concept'+(index+1),number:index+1,title:'知识点 '+(index+1),translations:{en:{title:index===1?'LoRA rank update':'Concept '+(index+1)}}}));
 assert.deepEqual(searchRelatedModules(modules,'＃２ ＬＯＲＡ').map(row=>row.number),[2]);
 assert.deepEqual(searchRelatedModules(modules,'architecture/concept2 rank').map(row=>row.number),[2]);
 const initial={company:'amazon',role:'ml',stage:'technical',outcome:'unknown',interview_month:'',location:'',questions:[modules[1].id,modules[3].id]};
 let latest,submits=0,outerKeys=0,setLocale;
 function Language(){setLocale=useI18n().setLocale;return null;}
 function Frame({value=initial,problems=modules,disabled=false}){
  const [details,setDetails]=useState(value);latest=details;
  return React.createElement('form',{onSubmit:event=>{event.preventDefault();submits++;},onKeyDown:()=>outerKeys++},
   React.createElement(Fields,{value:details,onChange:setDetails,problems,disabled}),
   React.createElement('textarea',{'aria-label':'Existing reflection',defaultValue:'My private reflection and implementation'}),
   React.createElement('button',{type:'submit'},'Publish'));
 }
 let scope='initial';
 const render=async(props={})=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Language),React.createElement(Frame,{...props,key:scope})));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const search=()=>document.querySelector('.related-module-search input');
 const options=()=>[...document.querySelectorAll('.interview-module-options button')];
 const option=id=>options().find(node=>node.dataset.moduleId===id);
 const chip=id=>[...document.querySelectorAll('[data-selected-module]')].find(node=>node.dataset.selectedModule===id);
 const more=()=>document.querySelector('.interview-module-results button');
 const change=async(node,value)=>act(async()=>{Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
 const key=async(node,options)=>{const event=new dom.window.KeyboardEvent('keydown',{bubbles:true,cancelable:true,...options});await act(async()=>node.dispatchEvent(event));return event;};
 const storage=JSON.stringify({...sessionStorage}),href=location.href;
 await render();assert.equal(options().length,6);assert(more());
 const reflection=document.querySelector('textarea');reflection.setSelectionRange(3,11,'backward');reflection.scrollTop=34;reflection.scrollLeft=16;
 const before=JSON.stringify(latest),input=search();
 await change(input,'＃２ ＬＯＲＡ');assert.equal(options().length,1);assert.equal(option(modules[1].id).getAttribute('aria-pressed'),'true');assert.equal(JSON.stringify(latest),before);
 const enter=await key(input,{key:'Enter'});assert(enter.defaultPrevented);assert.equal(outerKeys,0);assert.equal(submits,0);
 const modifiedEnter=await key(input,{key:'Enter',ctrlKey:true});assert(modifiedEnter.defaultPrevented);assert.equal(submits,0);
 await change(input,'no matching module');assert.equal(options().length,0);assert.equal(document.querySelectorAll('[data-selected-module]').length,2);assert(document.querySelector('.interview-module-results [role=status]').textContent.includes('Showing 0 of 0'));
 const escape=await key(input,{key:'Escape'});assert(escape.defaultPrevented);assert(search()===input);assert.equal(input.value,'');assert(document.activeElement===input);assert.equal(options().length,6);
 assert.equal(JSON.stringify(latest),before);assert(document.querySelector('textarea')===reflection);assert.deepEqual([reflection.selectionStart,reflection.selectionEnd,reflection.selectionDirection,reflection.scrollTop,reflection.scrollLeft],[3,11,'backward',34,16]);
 await click(more());assert.equal(options().length,12);assert(document.activeElement===option(modules[6].id));
 await click(more());assert.equal(options().length,18);assert(document.activeElement===option(modules[12].id));
 await click(more());assert.equal(options().length,20);assert(!more());assert(document.activeElement===option(modules[18].id));
 await click(option(modules[19].id));assert(latest.questions.includes(modules[19].id));
 await change(search(),'Concept');assert.equal(options().length,12);await click(more());assert.equal(options().length,18);await click(more());assert.equal(options().length,20,'The module ID also matches Concept for the LoRA-titled row');
 // Composition stays editable; ordinary Enter is owned by search and Escape only clears it.
 await act(async()=>search().dispatchEvent(new dom.window.CompositionEvent('compositionstart',{bubbles:true})));
 await change(search(),'😀'.repeat(110));assert.equal(Array.from(search().value).length,110);
 const ime=await key(search(),{key:'Enter',isComposing:true});assert(!ime.defaultPrevented);
 const legacy=await key(search(),{key:'Escape',keyCode:229});assert(!legacy.defaultPrevented);assert.equal(Array.from(search().value).length,110);
 await act(async()=>search().dispatchEvent(new dom.window.CompositionEvent('compositionend',{bubbles:true})));
 assert.equal(Array.from(search().value).length,100);await click(document.querySelector('[aria-label="Clear module search"]'));assert.equal(search().value,'');assert(document.activeElement===search());
 // Removing a chip has an actual surviving focus target; newer focus/navigation wins.
 scope='remove';await render();await click(chip(modules[1].id));assert(document.activeElement===chip(modules[3].id));await click(chip(modules[3].id));assert(document.activeElement===search());assert.deepEqual(latest.questions,[]);
 scope='newer-focus';await render();await act(async()=>{const node=chip(modules[1].id);node.focus();node.click();document.querySelector('textarea').focus();});assert(document.activeElement===document.querySelector('textarea'));
 scope='route';await render();await act(async()=>{const node=chip(modules[1].id);node.focus();node.click();location.hash='#/settings';});assert(document.activeElement!==chip(modules[3].id));location.hash=href.split('#')[1];
 scope='missing-catalog';await render({problems:[],value:{...initial,questions:['retired/module']}});await click(chip('retired/module'));assert(document.activeElement===document.querySelector('.interview-question-selector>b'));assert.deepEqual(latest.questions,[]);
 // A full selection remains removable and every result can be inspected without focus on disabled choices.
 scope='full';await render({value:{...initial,questions:modules.slice(0,12).map(row=>row.id)}});assert(options().every(node=>!node.disabled));await click(more());assert.equal(options().length,12);assert(options().every(node=>!node.disabled));await click(more());assert(options().some(node=>node.disabled));await click(more());assert(document.activeElement===search());
 assert.equal(latest.questions.length,12);await click(option(modules[19].id));assert.equal(latest.questions.length,12);
 await click(chip(modules[0].id));await click(option(modules[19].id));assert(latest.questions.includes(modules[19].id));assert.equal(latest.questions.length,12);
 await change(search(),'concept');const disabledQuery=search().value;await render({disabled:true,value:{...initial,questions:modules.slice(0,12).map(row=>row.id)}});const disabledValue=JSON.stringify(latest);assert(search().disabled);await click(document.querySelector('[aria-label="Clear module search"]'));assert.equal(search().value,disabledQuery);assert.equal(JSON.stringify(latest),disabledValue);
 // Locale changes keep the same search and chosen concepts; new authoring scope retires local search.
 await render();await act(async()=>setLocale('zh'));assert.equal(search().value,disabledQuery);assert(document.querySelector('.interview-module-results [role=status]').textContent.includes('显示'));
 scope='new-owner';await render({value:{...initial,questions:[]}});assert.equal(search().value,'');assert.deepEqual(latest.questions,[]);
 assert.equal(location.href,href);assert.equal(JSON.stringify({...sessionStorage}),storage);assert.equal(submits,0);await stop();
 console.log(`PASS related module selection: all${published.length} bilingual/copied-title/ID Unicode matches, multi-term literal search, complete progressive browsing, native outer-form Enter isolation and IME/Escape/clear, real selection limits, missing-catalog/newer-focus/route recovery, disabled/locale/scope boundaries and retained reflection/caret/scroll/storage`);
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
