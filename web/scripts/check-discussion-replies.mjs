import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const postId='a'.repeat(32),id=n=>n.toString(16).padStart(32,'0');
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/community/'+postId,pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:false});
let scrolls=[],errors=[];
dom.window.addEventListener('error',event=>errors.push(event.error));
dom.window.HTMLElement.prototype.scrollIntoView=function(){scrolls.push(this.id);};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,requestAnimationFrame:dom.window.requestAnimationFrame.bind(dom.window),cancelAnimationFrame:dom.window.cancelAnimationFrame.bind(dom.window),IS_REACT_ACT_ENVIRONMENT:true,__replyFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'reply-account-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0reply-account';},load(key){if(key==='\0reply-account')return `import React from 'react';export const useAccount=()=>globalThis.__replyFixture.account;export const socialApi=(...args)=>globalThis.__replyFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=()=>null;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;export const profileHref=user=>'#/profile/'+user.username;`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Community}=await server.ssrLoadModule('/src/Community.tsx');
 const {writeReplyDraft,readReplyDraft}=await server.ssrLoadModule('/src/replyDrafts.ts');
 const alice={id:'alice',username:'alice',display_name:'Alice',avatar_color:'#48766b'},bob={...alice,id:'bob',username:'bob',display_name:'Bob'};
 const replies=Array.from({length:53},(_,i)=>({id:id(i+1),post_id:postId,parent_id:i===52?id(1):null,body:`Reply ${i+1}\n\n**Markdown remains readable.**`,created_at:`2026-10-15T12:${String(Math.floor(i/2)).padStart(2,'0')}:00Z`,author:i%7===3||i===52?alice:bob,can_delete:false,can_edit:i===0,edit_version:'e'.repeat(64)}));
 const post={id:postId,title:'Long shared interview',body:'Actual shared experience.',category:'career',problem:null,problem_id:null,created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',author:alice,counts:{likes:0,reposts:0,replies:replies.length},viewer:{},replies};
 let requests=0,mutations=0,currentPost=post,allowPublication=false,failRefresh=false;
 globalThis.__replyFixture={account:{user:bob,loading:false,capabilities:{},refresh:async()=>{}},request:async(path,payload)=>{requests++;
  if(payload){assert(allowPublication);assert.equal(path,'community/posts/'+postId+'/replies');assert.equal(payload.expected_user_id,bob.id);assert.equal(payload.body,draft.body);assert.equal(payload.parent_id,draft.parentId);assert.match(payload.request_id,/^[a-f0-9]{32}$/);mutations++;const confirmed={...replies[0],id:id(54),created_at:'2026-10-15T12:59:00Z',body:payload.body,parent_id:payload.parent_id};currentPost={...currentPost,replies:[...currentPost.replies,confirmed],counts:{...currentPost.counts,replies:54}};failRefresh=true;return confirmed;}
  if(failRefresh){failRefresh=false;throw Error('Refresh temporarily unavailable');}
  return structuredClone(currentPost);
 }};
 const draft={body:'Private unpublished reflection 😀',parentId:id(12)};assert(writeReplyDraft(sessionStorage,bob.id,postId,draft));const savedDraft=readReplyDraft(sessionStorage,bob.id,postId);
 const render=async(route='community/'+postId)=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Community,{postId,route})));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const click=async node=>{assert(node);await act(async()=>node.click());};
 const rows=()=>[...document.querySelectorAll('.community-reply')];
 const controls=()=>document.querySelector('.discussion-reply-controls');
 const order=async value=>act(async()=>{const select=controls().querySelector('select');select.value=value;select.dispatchEvent(new dom.window.Event('change',{bubbles:true}));});
 const author=()=>controls().querySelector('button');
 await render();assert.equal(rows().length,20);assert.deepEqual(rows().map(row=>row.id),replies.slice(0,20).map(row=>'reply-'+row.id));assert(document.querySelector('.community-reply strong'));assert(controls().textContent.includes('20 of 53 replies shown'));
 const editor=document.querySelector('#reply-body');editor.setSelectionRange(3,11);editor.scrollTop=27;const originalHash=location.hash;
 const more=document.querySelector('.discussion-replies-more');more.focus();await click(more);assert.equal(rows().length,40);assert.equal(document.activeElement.id,'reply-'+id(21));assert.equal(scrolls.at(-1),'reply-'+id(21));assert.equal(document.querySelector('#reply-body'),editor);assert.equal(editor.selectionStart,3);assert.equal(editor.scrollTop,27);
 await order('newest');assert.equal(rows().length,20);assert.equal(rows()[0].id,'reply-'+id(53));assert.equal(rows()[1].id,'reply-'+id(52),'Timestamp ties follow reverse ID order without mutating server rows');
 await click(author());assert.equal(author().getAttribute('aria-pressed'),'true');assert(rows().every(row=>row.querySelector('.reply-author-badge')));assert.equal(rows().length,replies.filter(row=>row.author.id==='alice').length);
 const context=document.querySelector('#reply-'+id(53)+' .reply-context');context.focus();await click(context);assert.equal(author().getAttribute('aria-pressed'),'false','A parent outside the author filter must remain reachable');assert.equal(document.activeElement.id,'reply-'+id(1));assert.equal(rows().length,53);assert.equal(location.hash,originalHash);assert.deepEqual(readReplyDraft(sessionStorage,bob.id,postId),savedDraft);assert.equal(requests,1,'Sorting, batching and context jumps need no new request');assert.equal(mutations,0);
 await order('oldest');const edit=document.getElementById('edit-reply-'+id(1));await click(edit);const editArea=document.querySelector('[aria-label="Edit reply"] textarea');assert(editArea);assert(controls().querySelector('select').disabled);assert(author().disabled);assert(document.querySelector('.discussion-replies-more').disabled);await order('newest');assert.equal(rows()[0].id,'reply-'+id(1));assert.equal(document.querySelector('[aria-label="Edit reply"] textarea'),editArea,'Changing a disabled sort cannot unmount an active edit');await click([...document.querySelectorAll('button')].find(button=>button.textContent==='Close editor'));await act(async()=>new Promise(resolve=>setTimeout(resolve,25)));
 // A queued reveal cannot take focus from a newer input or another route.
 await act(async()=>{document.querySelector('.discussion-replies-more').focus();document.querySelector('.discussion-replies-more').click();editor.focus();});assert.equal(document.activeElement,editor);
 await order('oldest');scrolls=[];await act(async()=>{document.querySelector('.discussion-replies-more').focus();document.querySelector('.discussion-replies-more').click();location.hash='#/interviews';});assert.equal(scrolls.length,0);location.hash=originalHash;await stop();
 // Linked replies beyond the first batch are expanded and focused on first load.
 location.hash='#/community/'+postId+'?reply='+id(45);await render('community/'+postId+'?reply='+id(45));assert.equal(rows().length,53);assert.equal(document.activeElement.id,'reply-'+id(45));assert(document.querySelector('.linked-reply'));assert.equal(location.hash,'#/community/'+postId+'?reply='+id(45));await stop();
 // Empty author view is distinct from an empty discussion; no phantom reply rows.
 currentPost={...post,replies:replies.slice(0,2),counts:{...post.counts,replies:2}};location.hash=originalHash;await render();await click(author());assert.equal(rows().length,0);assert(document.querySelector('.discussion-reply-empty').textContent.includes('author has not replied'));await click(document.querySelector('.discussion-reply-empty button'));assert.equal(rows().length,2);await stop();
 // Deleted parent and target links leave the remaining discussion usable.
 currentPost={...post,replies:[{...replies[52],parent_id:id(999)}],counts:{...post.counts,replies:1}};await render('community/'+postId+'?reply='+id(999));assert(document.querySelector('.reply-context').disabled);assert(document.body.textContent.includes('linked reply is deleted'));assert.equal(rows().length,1);await stop();
 // New account/thread instances start independently; guest reading remains available.
 globalThis.__replyFixture.account={user:undefined,loading:false,capabilities:{}};currentPost=post;localStorage.setItem('paperleet.locale','zh');await render();assert(document.body.textContent.includes('只看楼主'));assert.equal(rows().length,20);assert(!document.querySelector('#reply-body'));assert.deepEqual(readReplyDraft(sessionStorage,bob.id,postId),savedDraft);await order('newest');assert.equal(rows()[0].id,'reply-'+id(53));assert.equal(mutations,0);await stop();
 // A confirmed reply remains reachable even when it is outside the current filter and refresh fails.
 globalThis.__replyFixture.account={user:bob,loading:false,capabilities:{}};localStorage.setItem('paperleet.locale','en');location.hash=originalHash;currentPost=post;allowPublication=true;await render();await click(author());assert.equal(author().getAttribute('aria-pressed'),'true');await click([...document.querySelectorAll('button')].find(button=>button.textContent==='Post reply'));assert.equal(mutations,1);assert(document.body.textContent.includes('Your reply is posted and visible'));assert(!document.getElementById('reply-'+id(54)));const ownReply=[...document.querySelectorAll('button')].find(button=>button.textContent==='View your posted reply');ownReply.focus();await click(ownReply);assert.equal(document.activeElement.id,'reply-'+id(54));assert.equal(rows().length,54);assert.equal(author().getAttribute('aria-pressed'),'false');assert.equal(readReplyDraft(sessionStorage,bob.id,postId),undefined);assert.equal(location.hash,originalHash);assert.deepEqual(errors,[]);await stop();
 console.log('PASS real discussion replies: bounded Markdown batches, deterministic order, author-only and empty view, parent/deep-link reveal, guarded focus, unchanged private composer/caret/scroll, edit protection, deleted targets, guest and bilingual controls');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
 for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,{...descriptor});else delete globalThis[key];}
}
