import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
const fixture=JSON.parse(process.env.PAPERLEET_INBOX_FILTER_HTTP||'null');
assert(fixture&&new URL(fixture.url).hostname==='127.0.0.1'&&fixture.replies.length===30,'A disposable authenticated loopback inbox is required');
const dom=new JSDOM('<!doctype html><textarea id="code">PRIVATE_INBOX_CODE</textarea><textarea id="draft">Private unsent reply</textarea><div id="root"></div>',{url:'https://paperleet.test/#/community',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,location:dom.window.location,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,CustomEvent:dom.window.CustomEvent,IS_REACT_ACT_ENVIRONMENT:true,__inboxHTTP:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');sessionStorage.setItem('private-inbox-draft','KEEP');
const requests=[],mutations=[];globalThis.__inboxHTTP={account:{user:fixture.user,loading:false,capabilities:fixture.capabilities},async request(path,body,signal){
 if(body!==undefined){mutations.push(path);throw Error('Filtering notifications must not acknowledge or send anything');}
 assert(!path.includes('PRIVATE_'));const entry={path,aborted:false};requests.push(entry);
 try{const response=await fetch(fixture.url+'/api/'+path,{signal,headers:{Accept:'application/json',Cookie:fixture.cookie}});const value=await response.json();assert.equal(response.status,200,'Local inbox read failed');return value;}catch(error){if(signal?.aborted)entry.aborted=true;throw error;}
}};
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-inbox-http-'),server=await createServer({cacheDir,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'inbox-http-account',enforce:'pre',resolveId(source){if(source==='./social')return '\0inbox-http-social';},load(id){if(id==='\0inbox-http-social')return `import React from 'react';export const useAccount=()=>globalThis.__inboxHTTP.account;export const socialApi=(...args)=>globalThis.__inboxHTTP.request(...args);export const errorMessage=e=>e.message;export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const profileHref=u=>'#/u/'+u.username;`;}}]});let root;
try{
 const {createRoot}=await import('react-dom/client'),{default:Inbox}=await server.ssrLoadModule('/src/InboxPanel.tsx'),{LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const render=()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(React.StrictMode,null,React.createElement(LocaleProvider,null,React.createElement(Inbox,{owner:fixture.user.id,notifications:true,messages:true,onCounts:()=>{}}))));});
 const wait=async predicate=>{for(let n=0;n<300&&!predicate();n++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),'Real HTTP inbox filter did not settle');};
 const rows=()=>[...document.querySelectorAll('.inbox-notification-row')],select=()=>document.querySelector('.inbox-notification-filters select'),replyIds=()=>rows().map(node=>new URL(node.querySelector('a').href).hash.split('reply=')[1]),tab=name=>[...document.querySelectorAll('[role=tab]')].find(node=>node.textContent.startsWith(name));
 const choose=async kind=>{await act(async()=>{select().focus();select().value=kind;select().dispatchEvent(new window.Event('change',{bubbles:true}));});};
 await render();await wait(()=>rows().length===20);assert(rows().every(node=>node.textContent.includes('replied to you')),'Unfiltered first page contains only newer replies');
 const code=document.querySelector('#code'),draft=document.querySelector('#draft');code.setSelectionRange(2,9,'backward');draft.setSelectionRange(1,8,'backward');code.scrollTop=40;code.scrollLeft=12;draft.scrollTop=18;draft.scrollLeft=6;
 await choose('like');await wait(()=>rows().length===1);assert(rows()[0].textContent.includes('liked your post'));assert(rows()[0].textContent.includes('other'),'Grouped likers retain distinct actor counts');assert.equal(tab('Notifications').querySelector('span').textContent,'33');assert.equal(tab('Messages').querySelector('span').textContent,'2');
 await choose('reply');await wait(()=>rows().length===20);const first=replyIds();assert.deepEqual(first,fixture.replies.slice(0,20));await act(async()=>document.querySelector('.inbox-more').click());await wait(()=>rows().length===30);assert.deepEqual(replyIds(),fixture.replies);assert.equal(new Set(replyIds()).size,30);assert(!document.querySelector('.inbox-more'));
 for(const kind of ['follow','repost']){await choose(kind);await wait(()=>rows().length===1);assert(rows()[0].textContent.includes(kind==='follow'?'followed you':'reposted your post'));}
 await choose('review');await wait(()=>document.body.textContent.includes('No notifications of this type'));assert.equal(rows().length,0);
 await act(async()=>document.querySelector('[aria-label="Clear notification type filter"]').click());await wait(()=>rows().length===20);assert.equal(document.activeElement,select());
 await act(async()=>tab('Messages').click());assert.equal(document.querySelectorAll('.inbox-tab-content .inbox-item').length,2);assert(document.querySelector('.inbox-tab-content').textContent.includes('Private fixture message'));
 const beforeRetired=requests.length;window.history.replaceState(null,'','#/favorites');await act(async()=>document.querySelector('.inbox-read-all-button').click());assert.equal(requests.length,beforeRetired);assert.equal(mutations.length,0);
 assert.deepEqual([code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],['PRIVATE_INBOX_CODE',2,9,'backward',40,12]);assert.deepEqual([draft.value,draft.selectionStart,draft.selectionEnd,draft.selectionDirection,draft.scrollTop,draft.scrollLeft],['Private unsent reply',1,8,'backward',18,6]);assert.equal(sessionStorage.getItem('private-inbox-draft'),'KEEP');
 console.log('INBOX_FILTER_HTTP '+JSON.stringify({reads:requests.length,aborted:requests.filter(row=>row.aborted).length,writes:mutations.length,full_reply_cohort:true,filtered_kinds:['like','reply','follow','repost','review'],reply_rows:30,global_counts_retained:true,older_likes_found:true,dm_rows:2,private_code_unchanged:true,retired_read_all_rejected:true}));
}finally{if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();for(const[key,value]of previous)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}
