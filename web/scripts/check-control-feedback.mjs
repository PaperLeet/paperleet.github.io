import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root;
try{
 const {createRoot}=await import('react-dom/client'),{LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Custom}=await server.ssrLoadModule('/src/CustomCases.tsx'),{default:External}=await server.ssrLoadModule('/src/ExternalCompanyPractice.tsx');
 const {parseCustomCase,CustomCaseInputError}=await server.ssrLoadModule('/src/customCaseModel.ts');
 for(const [args,kwargs,expected,field,reason]of [['[','{}','','args','json'],['{}','{}','','args','array'],['[]','[','','kwargs','json'],['[]','[]','','kwargs','object'],['[]','{}','bad','expected','json'],[' '.repeat(33000),'{}','','args','size']])assert.throws(()=>parseCustomCase('solve',args,kwargs,expected,['solve']),error=>error instanceof CustomCaseInputError&&error.field===field&&error.reason===reason);
 let locale,ready=false,busy=false,company='',changes=[];
 const onBusyChange=value=>changes.push(value);
 function Locale(){locale=useI18n();return null;}
 const render=()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Locale),React.createElement(Custom,{problem:{functions:[{name:'solve'}],browser_judge:{}},source:'untouched code',ready,busy,onBusyChange}),React.createElement(External,{company,problems:[]})));});
 const edit=async(node,value)=>act(async()=>{Object.getOwnPropertyDescriptor(node.tagName==='TEXTAREA'?dom.window.HTMLTextAreaElement.prototype:dom.window.HTMLInputElement.prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
 const click=async node=>{assert(node);await act(async()=>node.click());};
 const field=name=>document.querySelector(`.custom-cases [data-field="${name}"]`),run=()=>document.querySelector('.custom-case-actions button');
 await render();document.querySelector('.custom-cases').open=true;document.querySelector('.external-company-practice').open=true;
 assert(run().disabled);assert(document.getElementById(run().getAttribute('aria-describedby'))?.textContent.includes('Preparing your workspace'));
 ready=true;busy=true;await render();assert(run().disabled);assert(document.querySelector('.custom-cases [role=status]').textContent.includes('current test'));
 busy=false;await render();await edit(field('args'),'{}');await click(run());assert.equal(document.activeElement,field('args'));assert.equal(field('args').getAttribute('aria-invalid'),'true');assert(document.querySelector('.custom-case-error').textContent.includes('JSON array'));assert.equal(changes.length,0);
 await act(async()=>locale.setLocale('zh'));assert.equal(field('args').value,'{}');assert(document.querySelector('.custom-case-error').textContent.includes('JSON 数组'));await act(async()=>locale.setLocale('en'));
 await edit(field('args'),'[]');assert(!document.querySelector('.custom-case-error'));await edit(field('kwargs'),'[');await click(run());assert.equal(document.activeElement,field('kwargs'));assert.equal(field('kwargs').getAttribute('aria-describedby'),document.querySelector('.custom-case-error').id);
 await edit(field('kwargs'),'{}');await edit(field('expected'),'bad');await click(run());assert.equal(document.activeElement,field('expected'));assert.equal(field('args').getAttribute('aria-invalid'),null);assert.equal(changes.length,0,'Invalid input never prepares or starts a runtime');
 const directory=()=>document.querySelector('.external-company-practice'),select=()=>directory().querySelector('select'),search=()=>directory().querySelector('input');
 await act(async()=>{select().value='AWS';select().dispatchEvent(new dom.window.Event('change',{bubbles:true}));});assert.equal(select().value,'AWS');assert(directory().querySelector('li'));assert([...directory().querySelectorAll('li small')].every(node=>node.textContent.startsWith('AWS ·')));
 company='google';await render();assert.equal(select().value,'','Removed company options cannot leave a hidden filter active');assert(directory().querySelector('li'));assert(!directory().querySelector('[role=status]').textContent.startsWith('No exercises'));
 await edit(search(),'no-match-query');assert(directory().querySelector('[role=status]').textContent.startsWith('No exercises'));await click([...directory().querySelectorAll('button')].find(node=>node.textContent==='Clear practice filters'));assert.equal(search().value,'');assert.equal(document.activeElement,search());assert(directory().querySelector('li'));
 console.log('PASS control feedback: localized field validation/focus, retained JSON, disabled reasons, zero invalid runtime calls, stale company selection recovery, empty-state reset and keyboard focus.');
}finally{if(root)await act(()=>root.unmount());await server.close();dom.window.close();for(const [key,value]of previous)value?Object.defineProperty(globalThis,key,value):delete globalThis[key];}
