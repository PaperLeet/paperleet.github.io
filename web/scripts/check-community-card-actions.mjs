import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const id='a'.repeat(32);
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/community?category=papers&q=old&page=2',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:false});
dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const values={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,requestAnimationFrame:dom.window.requestAnimationFrame.bind(dom.window),IS_REACT_ACT_ENVIRONMENT:true,__cardFixture:undefined};
const previous=new Map(Object.keys(values).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(values))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'card-account',enforce:'pre',resolveId(source){if(source==='./social')return '\0card-account';},load(name){if(name==='\0card-account')return `import React from 'react';export const useAccount=()=>globalThis.__cardFixture.account;export const socialApi=(...args)=>globalThis.__cardFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{constructor(message,status,code){super(message);this.status=status;this.code=code;}};export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {signInHref}=await server.ssrLoadModule('/src/accountNavigation.ts');
 const {default:Actions}=await server.ssrLoadModule('/src/CommunityCardActions.tsx');
 const {SocialError}=await server.ssrLoadModule('\0card-account');
 const drafts=await server.ssrLoadModule('/src/replyDrafts.ts');
 const publications=await server.ssrLoadModule('/src/publicationRequests.ts');
 const user={id:'alice',username:'alice',display_name:'Alice',avatar_color:'#48766b',affiliation:{kind:'institution',domain:'research.edu',name:'测试大学',name_en:'Research University'}};
 const anonymous={...user,id:'anonymous:'+id,username:'',display_name:'Anonymous',anonymous:true};
 let refreshes=0,refreshFailure=false,delayRefresh=false,refreshRelease;const account={user,loading:false,capabilities:{},refresh:async()=>{refreshes++;if(refreshFailure)throw Error('Account refresh unavailable');if(delayRefresh)return new Promise(resolve=>refreshRelease=resolve);}};
 let post={id,title:'Useful discussion',body:'A paper explanation',category:'papers',created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',author:anonymous,anonymous:true,counts:{likes:4,replies:8,reposts:2},viewer:{liked:false,can_edit:true},problem_id:null,problem:null};
 let replies=Array.from({length:8},(_,index)=>({id:(index+1).toString(16).repeat(32),post_id:id,parent_id:null,body:'Comment '+(index+1),created_at:`2026-10-15T0${index}:00:00Z`,author:index%2?user:anonymous}));
 const requests=[],mutations=[];let authLike=false,authReply=false,failLike=false,failRead=false,failReply=false,releaseLike,releaseRead,delayLike=false,delayRead=false;
 globalThis.__cardFixture={account,request:async(path,body,signal)=>{
  requests.push({path,body,signal});
  if(!body){if(delayRead)return new Promise(resolve=>releaseRead=resolve);if(failRead)throw Error('Comments unavailable');return {...post,replies:[...replies]};}
  mutations.push({path,body});assert.equal(body.expected_user_id,account.user.id);
  if(path.endsWith('/like')){if(authLike)throw new SocialError('Session expired',401);if(delayLike)return new Promise(resolve=>releaseLike=()=>{post={...post,counts:{...post.counts,likes:5},viewer:{...post.viewer,liked:body.active}};resolve(post);});if(failLike)throw Error('Like unavailable');post={...post,counts:{...post.counts,likes:body.active?5:4},viewer:{...post.viewer,liked:body.active}};return post;}
  assert(path.endsWith('/replies'));if(authReply)throw new SocialError('Account changed',403,'account_changed');if(failReply)throw Error('Reply connection lost');
  const existing=replies.find(row=>row.request===body.request_id);if(existing)return existing;
  const row={id:(replies.length+1).toString(16).padStart(32,'0'),post_id:id,parent_id:body.parent_id,body:body.body,created_at:'2026-10-15T22:00:00Z',author:anonymous,request:body.request_id};replies=[...replies,row];post={...post,counts:{...post.counts,replies:replies.length}};return row;
 }};
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Actions,{post,returnTo:'community?sort=active&q=paper'})));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
 const button=text=>[...document.querySelectorAll('button')].find(node=>node.textContent===text);
 const expand=()=>document.querySelector('button[aria-expanded]'),like=()=>document.querySelector('button[aria-pressed]');
 const frame=()=>act(async()=>{await new Promise(resolve=>requestAnimationFrame(resolve));});
 const edit=async value=>act(async()=>{const field=document.querySelector('textarea');Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype,'value').set.call(field,value);field.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
 const submit=async()=>act(async()=>document.querySelector('form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
 await render();assert.equal(requests.length,0,'Closed previews make no per-card requests');const original=location.href;
 await click(like());assert.equal(mutations.length,1);assert.equal(like().getAttribute('aria-pressed'),'true');assert(like().textContent.includes('5'));assert.equal(location.href,original);
 await click(like());assert.equal(like().getAttribute('aria-pressed'),'false');
 failLike=true;await click(like());assert(like().disabled);assert.equal(like().getAttribute('aria-pressed'),'false');failLike=false;await click(button('Retry like action'));assert.equal(mutations.at(-1).body.active,true);assert.equal(like().getAttribute('aria-pressed'),'true');
 await click(button('Write comment'));await frame();assert.equal(document.activeElement,document.querySelector('textarea'));assert.equal(location.href,original);assert.equal(document.querySelectorAll('.card-comment-list>li').length,3);assert(document.querySelector('.card-comment-list>li').textContent.includes('Comment 8'));assert(document.querySelector('.card-reply-form').textContent.includes('Anonymous'));assert(!document.querySelector('.card-reply-form a[href="#/u/alice"]'),'Anonymous OP identity must stay anonymous');assert(document.querySelector('.card-reply-form').textContent.includes('Research University'));
 // Writing from an open preview retains its draft and does not refetch or publish.
 await edit('Keep this draft');const beforeWriting=requests.length;await click(button('Preview'));assert(document.querySelector('.compose-preview'));await click(button('Write comment'));await frame();assert.equal(document.activeElement,document.querySelector('textarea'));assert.equal(document.querySelector('textarea').value,'Keep this draft');assert.equal(requests.length,beforeWriting);
 // A newer focus choice wins over the scheduled move into the composer.
 await click(button('Write comment'));like().focus();await frame();assert.equal(document.activeElement,like());
 await click(button('Show 3 more comments'));assert.equal(document.querySelectorAll('.card-comment-list>li').length,6);assert.equal(location.href,original);
 const full=document.querySelector('.card-comments>header>a');assert(full.getAttribute('href').includes('from=community%3Fsort%3Dactive%26q%3Dpaper'));
 await edit('Saved inline response');await click(document.querySelector('.card-comment-links button'));assert.equal(drafts.readReplyDraft(sessionStorage,'alice',id).parentId,'8'.repeat(32));
 await click(button('Collapse comments'));assert.equal(document.activeElement,expand());assert(!document.querySelector('textarea'));assert.equal(drafts.readReplyDraft(sessionStorage,'alice',id).body,'Saved inline response');
 // A discussion-page draft is restored on reopening, including the intended parent.
 drafts.writeReplyDraft(sessionStorage,'alice',id,{body:'Edited in full discussion',parentId:'7'.repeat(32)});await click(expand());assert.equal(document.querySelector('textarea').value,'Edited in full discussion');assert(document.querySelector('.reply-target'));
 failReply=true;await submit();const first=mutations.at(-1).body;assert.equal(first.parent_id,'7'.repeat(32));assert(document.querySelector('[role=alert]').textContent.includes('connection lost'));assert.equal(document.querySelector('textarea').value,'Edited in full discussion');
 failReply=false;await submit();assert.equal(mutations.at(-1).body.request_id,first.request_id);assert.equal(document.querySelector('textarea').value,'');assert.equal(drafts.readReplyDraft(sessionStorage,'alice',id),undefined);assert(document.querySelector('.card-comment-list>li').textContent.includes('Edited in full discussion'));assert(document.querySelector('footer').textContent.includes('Reply posted'));
 assert.equal(publications.readPublication(sessionStorage,'alice','reply',id),undefined);
 // Retrying an earlier uncertain publication must retain a newer unsent draft.
 await edit('Earlier uncertain response');failReply=true;await submit();const uncertain=mutations.at(-1).body;await edit('Newer unsent response');failReply=false;
 await click(button('Retry previous publication'));assert.equal(mutations.at(-1).body.request_id,uncertain.request_id);assert.equal(mutations.at(-1).body.body,'Earlier uncertain response');assert.equal(document.querySelector('textarea').value,'Newer unsent response');assert.equal(drafts.readReplyDraft(sessionStorage,'alice',id).body,'Newer unsent response');
 // Failed reads retain the editor and draft but cannot publish against missing data.
 await edit('Do not lose me');await click(expand());failRead=true;await click(expand());assert(document.querySelector('[role=alert]').textContent.includes('unavailable'));assert(button('Post comment').disabled);assert.equal(document.querySelector('textarea').value,'Do not lose me');failRead=false;await click(button('Retry loading comments'));assert(!button('Post comment').disabled);
 drafts.writeReplyDraft(sessionStorage,'alice',id,{body:'Parent removed',parentId:'f'.repeat(32)});await click(expand());await click(expand());assert(button('Post comment').disabled);await click(document.querySelector('[aria-label="Cancel reply target"]'));assert(!button('Post comment').disabled);
 // A routine same-account refresh hides private text without resetting the open card or draft.
 await edit('Private text across account checking');account.loading=true;await render();assert(!document.querySelector('textarea'));assert(document.querySelector('.card-account-checking'));assert(expand().disabled);account.loading=false;await render();assert.equal(document.querySelector('textarea').value,'Private text across account checking');assert.equal(expand().getAttribute('aria-expanded'),'true');
 // Session errors retain the durable reply request and expose recovery if account refresh also fails.
 authReply=true;refreshFailure=true;await submit();const denied=mutations.at(-1).body;assert.equal(refreshes,1);assert(document.querySelector('.card-account-recovery'));assert(document.querySelector('textarea').disabled);assert(button('Retry publication').disabled);assert.equal(document.querySelector('.card-account-recovery a').getAttribute('href'),signInHref('community?sort=active&q=paper'));
 refreshFailure=false;await click(button('Refresh account'));assert.equal(refreshes,2);assert(!document.querySelector('.card-account-recovery'));assert.equal(document.activeElement,document.querySelector('textarea'));authReply=false;await submit();assert.equal(mutations.at(-1).body.request_id,denied.request_id);assert.equal(document.querySelector('textarea').value,'');
 authLike=true;refreshFailure=true;await click(like());const intended=mutations.at(-1).body.active;assert.equal(refreshes,3);assert(button('Retry like action').disabled);refreshFailure=false;await click(button('Refresh account'));assert.equal(document.activeElement,button('Retry like action'));authLike=false;await click(button('Retry like action'));assert.equal(mutations.at(-1).body.active,intended);
 // A delayed explicit account check cannot take focus from a newer user action.
 authLike=true;refreshFailure=true;await click(like());refreshFailure=false;delayRefresh=true;await click(button('Refresh account'));const newerFocus=expand();newerFocus.focus();await act(async()=>refreshRelease());assert.equal(document.activeElement,newerFocus);delayRefresh=false;authLike=false;await click(button('Retry like action'));
 // An ordinary network failure never triggers an account refresh.
 const confirmedRefreshes=refreshes;failLike=true;await click(like());assert.equal(refreshes,confirmedRefreshes);failLike=false;await click(button('Retry like action'));
 // Owner changes remount private state and a late request cannot update the new actor.
 await click(expand());delayLike=true;await click(like());account.user={...user,id:'bob',username:'bob',display_name:'Bob'};await render();await act(async()=>releaseLike());delayLike=false;await click(expand());assert.equal(document.querySelector('textarea').value,'');await stop();
 // Unmount aborts lazy reads and ignores their late resolution.
 account.user=user;delayRead=true;await render();await click(expand());const late=requests.at(-1);await stop();assert(late.signal.aborted);await act(async()=>releaseRead({...post,replies}));delayRead=false;
 account.user=undefined;await render();await click(expand());assert(!document.querySelector('textarea'));assert.equal(document.querySelector('.card-comment-signin').getAttribute('href'),signInHref('community?sort=active&q=paper'));const before=mutations.length;await click(button('Write comment'));assert.equal(mutations.length,before);assert.equal(location.hash,signInHref('community?sort=active&q=paper'));location.hash=original.split('#')[1];await click(like());assert.equal(mutations.length,before);assert(location.hash.includes('account'));await stop();
 console.log('PASS inline community actions: direct write-comment entry with draft/focus guards, lazy latest-three previews, expansion/collapse, focus return, exact feed context, real like/undo with fixed-direction retry, anonymous affiliation, shared owner/thread drafts, parent guards, idempotent publication and confirmation, read failure recovery, account isolation, late response/unmount, same-owner account checking, expired-session recovery with exact retry/focus, guest sign-in');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,{...descriptor});else delete globalThis[key];}}
