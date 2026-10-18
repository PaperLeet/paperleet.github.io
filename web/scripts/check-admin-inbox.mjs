import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtempSync,rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
const dom=new JSDOM('<!doctype html><textarea id="code">PRIVATE CODE</textarea><div id="root"></div>',{url:'http://paperleet.test/#/community/admin',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:false});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,Element:dom.window.Element,Node:dom.window.Node,location:dom.window.location,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,requestAnimationFrame:dom.window.requestAnimationFrame.bind(dom.window),cancelAnimationFrame:dom.window.cancelAnimationFrame.bind(dom.window),IS_REACT_ACT_ENVIRONMENT:true,__inbox:undefined};
const previous=new Map(Object.keys(globals).map(k=>[k,Object.getOwnPropertyDescriptor(globalThis,k)]));for(const[k,value]of Object.entries(globals))Object.defineProperty(globalThis,k,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const cache=mkdtempSync(path.join(os.tmpdir(),'paperleet-inbox-'));
const server=await createServer({cacheDir:cache,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'inbox-fixture',enforce:'pre',resolveId(s){if(s==='./social')return '\0inbox-social';},load(id){if(id==='\0inbox-social')return `import React from 'react';export const useAccount=()=>globalThis.__inbox.account;export const socialApi=(...a)=>globalThis.__inbox.request(...a);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const profileHref=u=>'#/u/'+u.username;export const DateLabel=({value})=>React.createElement('time',null,value);export const Avatar=()=>null;`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client'),{LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx'),{default:Admin}=await server.ssrLoadModule('/src/AdminWorkspace.tsx'),model=await server.ssrLoadModule('/src/adminInboxModel.ts'),{readAdminRoute}=await server.ssrLoadModule('/src/adminNavigation.ts'),{adminHref}=await server.ssrLoadModule('/src/adminLinks.ts');
 const owner='a'.repeat(32),post='b'.repeat(32),paper='c'.repeat(32),candidate='d'.repeat(32),date='2026-10-18T00:00:00Z';
 const value={owner_id:owner,reports:{total:1,items:[{post_id:post,title:'A reported anonymous post',pending_reports:3,total_reports:5,last_report_at:date,post_available:false}]},intake:{total:1,items:[{id:paper,arxiv_id:'2601.12345',version:0,status:'pending_review',created_at:date}]},candidates:{total:1,items:[{id:candidate,arxiv_id:'2601.12346',version:2,status:'pool',created_at:date}]}};
 assert(model.isAdminReviewInbox(value));assert(model.isAdminReviewInbox({owner_id:owner,...Object.fromEntries(['reports','intake','candidates'].map(k=>[k,{total:0,items:[]}]))}));
 for(const change of [v=>v.reports.total=-1,v=>v.reports.total=4,v=>v.reports.items[0].pending_reports=0,v=>v.reports.items[0].total_reports=2,v=>v.reports.items[0].post_id='https://bad.test',v=>v.reports.items[0].last_report_at='invalid',v=>v.intake.items[0].status='published',v=>v.candidates.items[0].version=0,v=>v.candidates.items[0].id=paper,v=>v.candidates.items[0].arxiv_id='../bad']){const bad=structuredClone(value);change(bad);assert.equal(model.isAdminReviewInbox(bad),false);}
 const href=adminHref({section:'reports',page:1,item:post});assert.equal(readAdminRoute(href).item,post);for(const query of ['item=invalid','item='+post+'&item='+post,'page=0&item='+post,'create=1&item='+post,'item='+post.toUpperCase()])assert.equal(readAdminRoute('community/admin/reports?'+query),undefined);
 console.log('PASS administrator inbox model: bounded previews, exact pending/candidate status, unique safe identifiers, valid dates and closed report-review continuations');
 const account={user:{id:owner,username:'administrator',display_name:'Administrator',role:'admin',status:'active'},loading:false,capabilities:{admin_workspace:true,admin_review_navigation:true,admin_review_inbox:true,community_reports:true}};
 const requests=[];let fail=false,malformed=false,route='community/admin',locale,held,hold=false;
 __inbox={account,request:async(path,body,signal)=>{requests.push({path,body,signal});assert.equal(body,undefined,'The dashboard and navigation must not write');if(path==='admin/overview')return {owner_id:owner,counts:{pending_reports:1,pending_papers:1,candidate_papers:1,active_users:2,disabled_users:0,boards:4,tags:2}};
 if(path==='admin/inbox'){if(hold)return new Promise(resolve=>{held=resolve;});if(fail)throw Error('Review inbox offline');return malformed?{...value,owner_id:'different-admin'}:structuredClone(value);}
 if(path.startsWith('community/moderation?'))return {owner_id:owner,items:[],total:0,state:'pending',minimum:1,limit:20,offset:0};
 if(path==='community/moderation/'+post+'?offset=0')return {owner_id:owner,post_id:post,post:null,revision:1,state:'pending',total_reports:5,pending_reports:3,offset:0,limit:5,reports:[],reviews:[]};
 throw Error('Unexpected path '+path);
 }};
 function Locale(){locale=useI18n();return null;}
 function App(){return React.createElement(LocaleProvider,null,React.createElement(Locale),React.createElement(Admin,{route}));}
 root=createRoot(document.querySelector('#root'));
 const render=()=>act(async()=>root.render(React.createElement(App))),wait=async fn=>{for(let n=0;n<100;n++){if(fn())return;await act(async()=>new Promise(r=>setTimeout(r,15)));}throw Error('Inbox condition timed out');};
 const click=async node=>{assert(node);await act(async()=>node.dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true,cancelable:true})));};
 const button=name=>[...document.querySelectorAll('button')].find(n=>n.textContent.trim()===name);
 const go=async next=>{route=next;location.hash='#/'+next;await render();};
 await render();await wait(()=>document.querySelector('.admin-inbox-group li a'));
 assert.equal(document.querySelectorAll('.admin-inbox-group').length,3);assert(document.querySelector('.admin-review-inbox').textContent.includes('Post unavailable; report snapshots remain available'));assert.equal(document.querySelectorAll('.admin-inbox-group li a').length,3);
 assert.equal(document.querySelector('.admin-inbox-group li a').getAttribute('href'),href);
 assert.equal(document.querySelectorAll('.admin-inbox-group li a')[1].getAttribute('href'),'#/community/admin/papers?status=all&item='+paper);
 const code=document.querySelector('#code');code.focus();code.setSelectionRange(2,6,'backward');const reads=requests.length;await act(async()=>locale.setLocale('zh'));assert.equal(document.activeElement,code);assert.deepEqual([code.selectionStart,code.selectionEnd],[2,6]);assert.equal(requests.length,reads);assert(document.querySelector('.admin-review-inbox').textContent.includes('优先处理'));await act(async()=>locale.setLocale('en'));
 fail=true;await click(button('Refresh'));await wait(()=>document.querySelector('.admin-review-inbox [role=alert]'));assert.equal(document.querySelectorAll('.admin-review-inbox a').length,0);assert.equal(document.querySelectorAll('.admin-metrics a').length,7);fail=false;await click(button('Retry loading'));await wait(()=>document.querySelector('.admin-review-inbox a'));
 malformed=true;await click(button('Refresh'));await wait(()=>document.querySelector('.admin-review-inbox [role=alert]'));assert.equal(document.querySelectorAll('.admin-review-inbox a').length,0);malformed=false;await click(button('Retry loading'));await wait(()=>document.querySelector('.admin-review-inbox a'));
 await go('community/admin/reports?item='+post);await wait(()=>requests.some(r=>r.path==='community/moderation/'+post+'?offset=0'));await wait(()=>document.querySelector('.moderation-case'));assert(document.querySelector('.moderation-case').textContent.includes('snapshot'));
 await go('community/admin');await wait(()=>document.querySelector('.admin-review-inbox a'));hold=true;await click(button('Refresh'));await wait(()=>held);const pending=requests.filter(r=>r.path==='admin/inbox').at(-1);account.user={...account.user,id:'e'.repeat(32),role:'member'};await render();assert.equal(pending.signal.aborted,true);await act(async()=>held(structuredClone(value)));assert(!document.querySelector('.admin-review-inbox'));assert(document.body.textContent.includes('Administrator access required'));
 account.user={...account.user,id:owner,role:'admin'};hold=false;for(const capability of [undefined,false,'true',1]){account.capabilities.admin_review_inbox=capability;const count=requests.filter(r=>r.path==='admin/inbox').length;await render();await wait(()=>document.querySelector('.admin-metrics a'));assert(!document.querySelector('.admin-review-inbox'));assert.equal(requests.filter(r=>r.path==='admin/inbox').length,count);}
 assert.equal(code.value,'PRIVATE CODE');assert(requests.every(r=>!r.body));assert(!JSON.stringify(requests).includes('PRIVATE CODE'));
 console.log('PASS administrator review inbox: actual gated workspace, three actionable queues and selected report case, independent error/retry, owner mismatch and late account retirement, literal capability fallback, bilingual focus/caret continuity and no social writes');
}finally{if(root)await act(async()=>root.unmount());await server.close();rmSync(cache,{recursive:true,force:true});dom.window.close();for(const[k,d]of previous)if(d)Object.defineProperty(globalThis,k,d);else delete globalThis[k];}
