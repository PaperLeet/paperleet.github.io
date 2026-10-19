import assert from 'node:assert/strict';
import fs from 'node:fs';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/community?solutions=1',pretendToBeVisual:true});
dom.window.HTMLElement.prototype.scrollIntoView=function(){};dom.window.matchMedia=()=>({matches:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});let root;
try{
 const {createRoot}=await import('react-dom/client'),{LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx'),{default:Browser}=await server.ssrLoadModule('/src/CommunityTopicBrowser.tsx');
 const nav=await server.ssrLoadModule('/src/communityNavigation.ts'),scope=await server.ssrLoadModule('/src/communityTopicScope.ts'),{topicFromProblem}=await server.ssrLoadModule('/src/communityTopics.ts');
 const catalog=JSON.parse(fs.readFileSync('public/data/index.json','utf8')).problems.map(topicFromProblem);
 const a={id:'architecture/attention',number:1,title:'注意力',title_en:'Attention'},c={id:'finetuning/lora',number:22,title:'低秩适配',title_en:'LoRA rank update'},retired={id:'retired/module',title:'Historical module',retired:true};
 const filters={category:'implementation',tag:'attention',following:false,saved:true,solutions:true,sort:'top',days:'30',search:'mask & gradient',page:3};
 const initial=nav.communityFeedHref(undefined,filters);let from=initial,owner='alice',topics=[a,c,retired],loading=false,error,retries=0,setLocale;
 function Language(){setLocale=useI18n().setLocale;return null;}
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(React.StrictMode,null,React.createElement(LocaleProvider,null,React.createElement(Language),React.createElement(Browser,{key:owner+from,from,topics,loading,error,reload:()=>retries++}),React.createElement('textarea',{'aria-label':'Private reflection',defaultValue:'My private reasoning and code'}))));});
 const change=async(node,value)=>act(async()=>{Object.getOwnPropertyDescriptor(node.tagName==='SELECT'?dom.window.HTMLSelectElement.prototype:dom.window.HTMLInputElement.prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event(node.tagName==='SELECT'?'change':'input',{bubbles:true}));});
 const search=()=>document.querySelector('.community-topic-panel input'),select=()=>document.querySelector('.community-topic-selection select'),apply=()=>document.querySelector('.community-topic-panel footer a'),details=()=>document.querySelector('details'),summary=()=>document.querySelector('summary');
 const open=async()=>act(async()=>{details().open=true;details().dispatchEvent(new dom.window.Event('toggle'));});
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
 const key=async(node,extra)=>{const event=new dom.window.KeyboardEvent('keydown',{bubbles:true,cancelable:true,...extra});await act(async()=>node.dispatchEvent(event));return event;};
 const status=()=>document.querySelector('.community-topic-status').textContent;
 location.hash=initial;const storage=JSON.stringify({...sessionStorage});await render();await open();
 const reflection=document.querySelector('textarea');reflection.setSelectionRange(3,12,'backward');reflection.scrollTop=52;reflection.scrollLeft=21;
 assert.equal(select().value,'');assert.equal(apply().getAttribute('aria-disabled'),'true');assert.equal(apply().getAttribute('href'),initial);assert(!select().querySelector('option[value="retired/module"]'));
 const input=search();await change(input,'＃２２ ＬＯＲＡ');assert.equal(select().querySelectorAll('option').length,2);assert.equal(select().value,'');assert.equal(location.hash,initial);
 assert((await key(input,{key:'Enter'})).defaultPrevented);await change(select(),c.id);const destination=apply().getAttribute('href');assert.equal(destination,scope.communityTopicScopeHref(initial,c.id));assert.deepEqual(nav.readCommunityFeed(destination),{topicId:c.id,state:{...filters,page:1}});assert.equal(apply().getAttribute('aria-disabled'),'false');
 // Search only narrows options; it never silently chooses a module or submits/navigates.
 await change(input,'no matching module');assert.equal(select().value,c.id);assert(select().querySelector('optgroup option'));assert(status().includes('No modules match'));assert.equal(apply().getAttribute('href'),destination);assert.equal(location.hash,initial);
 assert((await key(input,{key:'Escape'})).defaultPrevented);assert.equal(search(),input);assert.equal(input.value,'');assert.equal(select().value,c.id);assert(details().open);assert(document.activeElement===input);
 await act(async()=>input.dispatchEvent(new dom.window.CompositionEvent('compositionstart',{bubbles:true})));assert(!(await key(input,{key:'Escape'})).defaultPrevented);assert(details().open);await act(async()=>input.dispatchEvent(new dom.window.CompositionEvent('compositionend',{bubbles:true})));
 assert(!(await key(select(),{key:'Escape'})).defaultPrevented);assert(details().open);assert((await key(input,{key:'Escape'})).defaultPrevented);assert(!details().open);assert(document.activeElement===summary());await open();assert.equal(select().value,c.id);
 await act(async()=>setLocale('zh'));assert.equal(select().value,c.id);assert.equal(apply().getAttribute('href'),destination);assert.equal(select().selectedOptions[0].textContent,'#22 · 低秩适配');await act(async()=>setLocale('en'));
 assert.equal(reflection,document.querySelector('textarea'));assert.equal(reflection.value,'My private reasoning and code');assert.deepEqual([reflection.selectionStart,reflection.selectionEnd,reflection.selectionDirection,reflection.scrollTop,reflection.scrollLeft],[3,12,'backward',52,21]);assert.equal(JSON.stringify({...sessionStorage}),storage);
 // Catalog outages and removed choices never turn an old draft into a different module.
 topics=[a];await render();assert.equal(select().value,c.id);assert(status().includes('selected module is unavailable'));assert.equal(apply().getAttribute('aria-disabled'),'true');assert.equal(apply().getAttribute('href'),initial);
 topics=[a,c];loading=true;await render();assert(status().includes('Loading modules'));assert.equal(apply().getAttribute('aria-disabled'),'true');loading=false;error='Network unavailable';await render();assert(status().includes('catalog is unavailable'));await click(document.querySelector('footer button'));assert.equal(retries,1);assert.equal(select().value,c.id);error=undefined;await render();assert.equal(apply().getAttribute('href'),destination);
 // Actual anchors support modified navigation, but a departed route cannot be changed by stale UI.
 const modified=new dom.window.MouseEvent('click',{bubbles:true,cancelable:true,ctrlKey:true});let allowedModified;document.addEventListener('click',e=>{allowedModified=!e.defaultPrevented;e.preventDefault();},{once:true});await act(async()=>apply().dispatchEvent(modified));assert(allowedModified);assert.equal(location.hash,initial);
 location.hash='#/settings';const stale=new dom.window.MouseEvent('click',{bubbles:true,cancelable:true});await act(async()=>apply().dispatchEvent(stale));assert(stale.defaultPrevented);assert.equal(location.hash,'#/settings');
 // A new canonical feed/owner retires the draft, while unknown historical scope can be cleared.
 from=destination;location.hash=from;await render();assert.equal(search().value,'');assert.equal(select().value,c.id);assert(!details().open);assert.equal(apply().getAttribute('href'),from);
 from=nav.communityFeedHref(retired.id,filters);location.hash=from;topics=[a,c];await render();await open();assert.equal(select().value,retired.id);assert(summary().textContent.includes(retired.id));error='Offline';await render();await change(select(),'');assert.equal(apply().getAttribute('aria-disabled'),'false');assert.deepEqual(nav.readCommunityFeed(apply().getAttribute('href')),{topicId:undefined,state:{...filters,page:1}});
 await change(search(),'private local query');owner='bob';await render();assert.equal(search().value,'');assert.equal(select().value,retired.id);assert(!details().open);error=undefined;
 // Every published exercise is selectable; URLs are closed and keep the kind and all filters.
 for(const topic of catalog){assert(scope.validCommunityTopic(topic.id));const result=scope.communityTopicScopeHref(initial,topic.id);assert.deepEqual(nav.readCommunityFeed(result),{topicId:topic.id,state:{...filters,page:1}});}
 assert.equal(scope.communityTopicScopeHref(initial,''),initial);
 for(const bad of ['https://evil.test','#/settings','#/community?unknown=1'])assert.equal(scope.communityTopicScopeHref(bad,c.id),undefined);
 for(const bad of ['../x','bad/module/extra','https://evil.test','A/module','architecture/'+'a'.repeat(200)])assert.equal(scope.communityTopicScopeHref(initial,bad),undefined);
 assert.equal(scope.communityTopicScopeHref(destination,c.id),destination);assert.deepEqual(nav.readCommunityFeed(scope.communityTopicKindHref(nav.communityFeedHref(c.id,filters),false)),{topicId:c.id,state:Object.fromEntries(Object.entries({...filters,page:1}).filter(([key])=>key!=='solutions'))});
 console.log(`PASS community module scope: all ${catalog.length} catalog routes, Unicode number/title/ID discovery, explicit filter-preserving anchors, same-scope paging, kept selection and no-match recovery, native select Escape/IME isolation, collapse/locale/caret/scroll retention, stale route/catalog/owner boundaries and historical-scope recovery`);
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const[key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
