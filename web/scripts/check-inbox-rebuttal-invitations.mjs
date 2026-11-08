import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div><textarea id="draft">Private unsent text</textarea>',{url:'https://paperleet.test/#/community',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,CustomEvent:dom.window.CustomEvent,requestAnimationFrame:dom.window.requestAnimationFrame.bind(dom.window),cancelAnimationFrame:dom.window.cancelAnimationFrame.bind(dom.window),IS_REACT_ACT_ENVIRONMENT:true,__invitationInbox:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({cacheDir:'node_modules/.vite-invitation-inbox-check',server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'invitation-inbox-account',enforce:'pre',resolveId(source){if(source==='./social')return '\0invitation-inbox';},load(name){if(name==='\0invitation-inbox')return `import React from 'react';export const useAccount=()=>globalThis.__invitationInbox.account;export const socialApi=(...args)=>globalThis.__invitationInbox.request(...args);export const errorMessage=e=>e.message;export const profileHref=u=>'#/u/'+u.username;export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Panel}=await server.ssrLoadModule('/src/InboxPanel.tsx');
 const {default:Bell}=await server.ssrLoadModule('/src/InboxBell.tsx');
 const {rebuttalInvitationNotice}=await server.ssrLoadModule('/src/RebuttalInvitationNotification.tsx');
 const {inboxChanged,inboxReadCutoff,applyInboxReadAll}=await server.ssrLoadModule('/src/inbox.ts');
 const {confirmInboxFilter}=await server.ssrLoadModule('/src/inboxNotificationQuery.ts');
 const {inboxListUpdate}=await server.ssrLoadModule('/src/inboxListUpdate.ts');
 const notice={id:'invite-one',paper_id:'paper-one',title:'<script>Private paper</script>',venue:'ICLR',role:'viewer',status:'pending'};
 assert.deepEqual(rebuttalInvitationNotice(notice),notice);
 for(const value of [null,{}, {...notice,paper_id:'https://evil.test'}, {...notice,id:'../x'}, {...notice,role:'owner'}, {...notice,status:'unknown'}, {...notice,title:'x'.repeat(241)}])assert.equal(rebuttalInvitationNotice(value),undefined);
 const alice={id:'alice',username:'alice',display_name:'Alice'},bob={id:'bob',username:'bob',display_name:'Bob'};
 const row={id:'rebuttal_invite:invite-one',kind:'rebuttal_invite',latest_event_id:1,created_at:'2026-11-07',unread:true,actors:[alice],actor_count:1,post:null,reply_id:null,preview:'',rebuttal_invitation:notice};
 const account={user:bob,loading:false,capabilities:{inbox_notifications:true,direct_messages:true,inbox_read_all:true,inbox_notification_filters:true,inbox_rebuttal_invitations:true}};
 let state={user_id:'bob',unread_count:1,notification_unread_count:1,message_unread_count:0,rebuttal_invitations_included:true,observed_event_id:1,observed_message_id:0,notifications:[row],notifications_next_before:null,conversations:[],conversations_next_before:null};
 assert.deepEqual(inboxReadCutoff(state),{through_event_id:1,through_message_id:0,include_rebuttal_invitations:true});
 assert(applyInboxReadAll(state,state,{through_event_id:1,through_message_id:0}).notifications[0].unread,'Legacy read-all cannot acknowledge invitations');
 assert.throws(()=>confirmInboxFilter({...state,rebuttal_invitations_included:undefined},'all',false,false,true));
 const later={...row,id:'rebuttal_invite:later',latest_event_id:2};
 const after=applyInboxReadAll({...state,notifications:[row,later]},state,{through_event_id:1,through_message_id:0,include_rebuttal_invitations:true});
 assert(!after.notifications[0].unread&&after.notifications[1].unread,'New arrivals stay unread');
 const settled={...state,notifications:[{...row,unread:false,rebuttal_invitation:{...notice,status:'accepted'}}]};
 assert.equal(inboxListUpdate(state,settled).snapshot.notifications[0].rebuttal_invitation.status,'accepted');
 const calls=[],pending=[];let mode='panel',lose=false,hold=false;
 async function request(path,body,signal){
  calls.push({path,body:body?structuredClone(body):undefined,signal});
  if(!body&&hold)return new Promise(resolve=>pending.push({resolve,signal,owner:account.user.id}));
  const params=new URLSearchParams(path.split('?')[1]),include=params.get('rebuttal_invitations')==='1';
  if(body){
   assert.equal(body.expected_user_id,account.user.id);assert.equal(body.include_rebuttal_invitations,true);
   state={...state,notifications:state.notifications.map(item=>item.latest_event_id<=body.through_event_id&&(!body.notification_ids||body.notification_ids.includes(item.id))?{...item,unread:false}:item)};
   state={...state,unread_count:state.notifications.filter(item=>item.unread).length,notification_unread_count:state.notifications.filter(item=>item.unread).length};
   if(lose){lose=false;hold=true;throw Error('Read response lost');}
   return {user_id:account.user.id,unread_count:state.unread_count,notification_unread_count:state.notification_unread_count,message_unread_count:0};
  }
  const result=structuredClone({...state,user_id:account.user.id});
  if(!include){result.notifications=[];result.unread_count=0;result.notification_unread_count=0;result.observed_event_id=0;delete result.rebuttal_invitations_included;}
  if(params.has('notification_kind'))result.notification_kind=params.get('notification_kind');
  return result;
 }
 globalThis.__invitationInbox={account,request};
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(React.StrictMode,null,React.createElement(LocaleProvider,null,mode==='panel'?React.createElement(Panel,{owner:account.user.id,notifications:true,messages:true,onCounts:()=>{}}):React.createElement(Bell))));});
 const wait=async predicate=>{for(let n=0;n<150&&!predicate();n++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),document.body.textContent);};
 const click=async node=>{assert(node);await act(async()=>node.click());};
 const writes=()=>calls.filter(item=>item.body);
 await render();await wait(()=>document.querySelector('.inbox-rebuttal-invitation'));
 assert.equal(writes().length,0,'Opening inbox does not acknowledge invites');
 assert(calls.some(item=>item.path.includes('rebuttal_invitations=1')));
 assert(document.body.textContent.includes('<script>Private paper</script>'));assert(!document.querySelector('script,iframe,img'));
 assert.equal(document.querySelector('.inbox-rebuttal-invitation a').getAttribute('href'),'#/rebuttal/invitations');
 assert(document.body.textContent.includes('Viewer · Pending'));
 const select=document.querySelector('.inbox-notification-filters select');assert(select.querySelector('[value=rebuttal_invite]'));
 await act(async()=>{select.value='rebuttal_invite';select.dispatchEvent(new window.Event('change',{bubbles:true}));});
 await wait(()=>calls.some(item=>item.path.includes('notification_kind=rebuttal_invite')));
 await wait(()=>document.querySelector('.inbox-row-read')?.getAttribute('aria-disabled')==='false');
 await click(document.querySelector('.inbox-row-read'));
 await wait(()=>writes().length===1&&!document.querySelector('.inbox-row-dot'));
 assert.deepEqual(writes()[0].body,{expected_user_id:'bob',through_event_id:1,include_rebuttal_invitations:true,notification_ids:[row.id]});
 // Polling reflects acceptance in-place even when the notice was already read.
 state={...state,notifications:[{...state.notifications[0],rebuttal_invitation:{...notice,status:'accepted'}}]};
 await act(async()=>inboxChanged('bob'));await wait(()=>document.body.textContent.includes('Viewer · Accepted'));
 state={...state,unread_count:1,notification_unread_count:1,notifications:[{...state.notifications[0],unread:true}]};
 await act(async()=>inboxChanged('bob'));await wait(()=>document.querySelector('.inbox-read-all-button')&&!document.querySelector('.inbox-read-all-button').disabled);
 await click(document.querySelector('.inbox-read-all-button'));
 await wait(()=>writes().some(item=>item.path==='inbox/read-all'));
 assert.equal(writes().at(-1).body.include_rebuttal_invitations,true);
 // A lost response retries the same acknowledgement flags and cutoff.
 state={...state,unread_count:1,notification_unread_count:1,notifications:[{...state.notifications[0],unread:true}]};
 await act(async()=>inboxChanged('bob'));await wait(()=>document.querySelector('.inbox-row-read')?.getAttribute('aria-disabled')==='false');
 lose=true;await click(document.querySelector('.inbox-row-read'));await wait(()=>document.querySelector('.inbox-notification-read-error button'));
 const attempted=writes().at(-1).body;hold=false;await click(document.querySelector('.inbox-notification-read-error button'));await wait(()=>writes().length>=4);
 assert.deepEqual(writes().at(-1).body,attempted);
 // Older servers receive no new parameter, and lose the unsupported filter.
 account.capabilities.inbox_rebuttal_invitations=false;await render();await wait(()=>!document.querySelector('.inbox-rebuttal-invitation'));
 assert(!document.querySelector('[value=rebuttal_invite]'));assert(!calls.at(-1).path.includes('rebuttal_invitations'));
 mode='bell';await render();await wait(()=>document.querySelector('.inbox-bell-trigger'));
 assert(!document.querySelector('.inbox-unread-dot'));
 account.capabilities.inbox_rebuttal_invitations=true;state={...state,unread_count:1,notification_unread_count:1,notifications:[row]};await render();await act(async()=>inboxChanged('bob'));await wait(()=>document.querySelector('.inbox-unread-dot'));
 assert(calls.some(item=>item.path==='inbox?summary=1&rebuttal_invitations=1'));
 // A retired user's delayed snapshot cannot paint private invitations in a new inbox.
 mode='panel';hold=true;await render();await wait(()=>pending.length>0);account.user=alice;hold=false;state={...state,user_id:'alice',unread_count:0,notification_unread_count:0,notifications:[]};await render();await wait(()=>!document.querySelector('.inbox-rebuttal-invitation'));
 await act(async()=>{for(const entry of pending)entry.resolve({...state,user_id:'bob',notifications:[row]});});
 assert(!document.querySelector('.inbox-rebuttal-invitation'));assert(pending.every(item=>item.signal.aborted));
 assert.equal(document.querySelector('#draft').value,'Private unsent text');assert(!JSON.stringify(calls).includes('Private unsent text'));
 console.log('PASS rebuttal invitation inbox: opt-in count/list/filter/read contract, owner guards and late-response isolation, literal safe notices/link/status updates, no passive acceptance/read, legacy fallback, retry identity, snapshot-safe read-all, preserved private drafts');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const[key,value]of previous)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}
