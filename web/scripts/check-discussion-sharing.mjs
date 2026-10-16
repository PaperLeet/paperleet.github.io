import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const postId='a'.repeat(32),replyId='b'.repeat(32);
const dom=new JSDOM('<!doctype html><div id="root"></div><button id="other-task">Other task</button>',{url:'https://paperleet.test/preview/?token=private#/u/alice?tab=solutions&page=2',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:false});const scrolls=[];let viewport=900;
Object.defineProperty(dom.window,'scrollY',{get:()=>viewport,configurable:true});
dom.window.scrollTo=(options,y)=>{viewport=typeof options==='object'?options.top:y;scrolls.push({top:viewport});};
dom.window.HTMLElement.prototype.scrollIntoView=function(){scrolls.push(this.id);};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__sharingFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const historyListeners=new Map([['hashchange',new Set()],['popstate',new Set()]]),nativeAdd=window.addEventListener.bind(window),nativeRemove=window.removeEventListener.bind(window);
window.addEventListener=(type,callback,options)=>{historyListeners.get(type)?.add(callback);nativeAdd(type,callback,options);};
window.removeEventListener=(type,callback,options)=>{historyListeners.get(type)?.delete(callback);nativeRemove(type,callback,options);};
const nativeTimeout=window.setTimeout.bind(window),nativeClear=window.clearTimeout.bind(window),deadlines=new Map();let timerId=100000;
window.setTimeout=(callback,delay,...args)=>{if(delay===8000){const id=timerId++;deadlines.set(id,()=>callback(...args));return id;}return nativeTimeout(callback,delay,...args);};
window.clearTimeout=id=>{if(deadlines.has(id))deadlines.delete(id);else nativeClear(id);};
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'sharing-account-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0sharing-account';},load(key){if(key==='\0sharing-account')return `import React from 'react';export const useAccount=()=>globalThis.__sharingFixture.account;export const socialApi=(...args)=>globalThis.__sharingFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=()=>null;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;export const profileHref=user=>'#/u/'+user.username;`;}}]});
let root;
try{
  const nav=await server.ssrLoadModule('/src/communityNavigation.ts'),accountNav=await server.ssrLoadModule('/src/accountNavigation.ts'),{decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
  for(const base of ['https://name:secret@paperleet.test/preview/?token=private#/u/alice?tab=solutions','http://127.0.0.1:5185/?code=secret#/community?q=private'])for(const reply of [undefined,replyId]){
    const page=new URL(base),url=nav.communityShareUrl(base,postId,reply);
    assert.equal(url,page.origin+page.pathname+nav.communityPostHref(postId,undefined,reply));assert(!url.includes('secret')&&!url.includes('private')&&!url.includes('alice'));
    assert.equal(accountNav.accountReturnHref(decodeProblemHash(accountNav.signInHref(new URL(url).hash))),new URL(url).hash);
  }
  for(const bad of ['', '../escape','A'.repeat(32),'a'.repeat(33),'https://outside.test','a'.repeat(32)+'?from=u/alice'])assert.throws(()=>nav.communityShareUrl(location.href,bad),undefined,'Invalid public post IDs must not produce shareable routes');
  for(const bad of ['file:///tmp/index.html','javascript:alert(1)','invalid URL'])assert.throws(()=>nav.communityShareUrl(bad,postId));
  const {default:DiscussionLink}=await server.ssrLoadModule('/src/DiscussionLink.tsx'),{default:Community}=await server.ssrLoadModule('/src/Community.tsx');
  const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx'),{writeReplyDraft,readReplyDraft}=await server.ssrLoadModule('/src/replyDrafts.ts');
  const {createRoot}=await import('react-dom/client');
  let props={postId,replyId,returnTo:'#/u/alice?tab=solutions&page=2'},setLocale,discussion=false,route;
  function Harness(){setLocale=useI18n().setLocale;return discussion?React.createElement(Community,{postId,route}):React.createElement(DiscussionLink,props);}
  root=createRoot(document.querySelector('#root'));
  const render=async patch=>{props={...props,...patch};await act(async()=>root.render(React.createElement(LocaleProvider,null,React.createElement(Harness))));};
  const share=()=>document.querySelector('.discussion-share'),button=()=>share().querySelector('.discussion-copy'),field=()=>share().querySelector('.discussion-link-fallback input'),status=()=>share().querySelector('[role="status"]');
  const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
  const pending=[],writes=[],clipboard={writeText(value){writes.push(value);return new Promise((resolve,reject)=>pending.push({value,resolve,reject}));}};
  Object.defineProperty(navigator,'clipboard',{value:clipboard,configurable:true});
  const begin=async()=>{await click(button());assert.equal(button().getAttribute('aria-busy'),'true');return pending.shift();};
  const settle=async(item,fail=false)=>act(async()=>fail?item.reject(Error('Copy denied')):item.resolve());
  await render();const url=location.href,stored=JSON.stringify({...localStorage}),session=JSON.stringify({...sessionStorage});assert.equal(writes.length,0);assert([...historyListeners.values()].every(listeners=>listeners.size===0),'Idle share controls install no history listeners');
  const localLink=share().querySelector('a');assert.equal(nav.communityPostReturn(localLink.getAttribute('href'),postId),props.returnTo);assert.equal(nav.communityReplyTarget(localLink.getAttribute('href'),postId),replyId);
  const nativeButton=button();let first;
  await act(async()=>{nativeButton.focus();nativeButton.click();nativeButton.click();first=pending.shift();});assert.equal(writes.length,1);assert.equal(first.value,nav.communityShareUrl(url,postId,replyId));assert.equal(deadlines.size,1);
  assert.equal(nativeButton.disabled,false);assert.equal(document.activeElement,nativeButton);assert.equal(nativeButton.getAttribute('aria-disabled'),'true');
  assert([...historyListeners.values()].every(listeners=>listeners.size===1));
  await act(async()=>setLocale('zh'));assert.equal(button(),nativeButton);assert.equal(document.activeElement,nativeButton);assert(status().textContent.includes('正在复制'));await settle(first);assert(status().textContent.includes('链接已复制'));assert(!field());assert.equal(deadlines.size,0);
  await act(async()=>setLocale('en'));const denied=await begin(),other=document.querySelector('#other-task');other.focus();await settle(denied,true);assert.equal(document.activeElement,other);assert.equal(field().value,first.value);assert(field().readOnly);
  field().focus();assert.equal(field().selectionStart,0);assert.equal(field().selectionEnd,field().value.length);const manual=field();manual.setSelectionRange(5,12,'backward');manual.scrollLeft=37;
  await act(async()=>setLocale('zh'));assert.equal(field(),manual);assert.equal(document.activeElement,manual);assert.equal(manual.selectionStart,5);assert.equal(manual.selectionEnd,12);assert.equal(manual.selectionDirection,'backward');assert.equal(manual.scrollLeft,37);
  await act(async()=>setLocale('en'));const timed=await begin();await act(async()=>[...deadlines.values()][0]());assert(field());assert.equal(button().getAttribute('aria-busy'),'false');assert.equal(deadlines.size,0);
  const retry=await begin();await settle(timed);assert.equal(button().getAttribute('aria-busy'),'true');await settle(retry);assert(status().textContent.includes('Link copied.'));
  const oldReply=await begin();await render({replyId:'c'.repeat(32)});assert.equal(deadlines.size,0);const newer=await begin();await settle(oldReply,true);assert.equal(button().getAttribute('aria-busy'),'true');assert(!field());await settle(newer);assert(writes.at(-1).endsWith('?reply='+'c'.repeat(32)));
  const aba=await begin();await render({postId:'d'.repeat(32)});await render({postId});await settle(aba,true);assert.equal(status().textContent,'');assert(!field());
  const context=await begin();await render({returnTo:nav.communityFeedHref(undefined,{...nav.emptyCommunityFeed(),solutions:true,search:'中文 & mask'})});assert.equal(deadlines.size,0);await settle(context,true);assert(!field());assert.equal(nav.communityPostReturn(share().querySelector('a').getAttribute('href'),postId),props.returnTo);
  assert.equal(location.href,url);assert.equal(JSON.stringify({...localStorage}),stored);assert.equal(JSON.stringify({...sessionStorage}),session);
  const leaving=await begin();await act(async()=>{window.history.replaceState(null,'','#/community?solutions=1');window.dispatchEvent(new dom.window.PopStateEvent('popstate'));});assert.equal(deadlines.size,0);await settle(leaving);assert.equal(status().textContent,'');
  assert([...historyListeners.values()].every(listeners=>listeners.size===0),'Retired feedback removes its history listeners');
  const current=await begin();await act(async()=>window.dispatchEvent(new dom.window.HashChangeEvent('hashchange')));assert.equal(button().getAttribute('aria-busy'),'true','A delayed notification at the current URL does not cancel a new copy');await settle(current);
  const beforeEvent=await begin();window.history.replaceState(null,'','#/u/alice?tab=solutions');await settle(beforeEvent,true);assert(!field());assert.equal(status().textContent,'');
  Object.defineProperty(navigator,'clipboard',{value:undefined,configurable:true});await click(button());assert(field());assert.equal(deadlines.size,0);
  Object.defineProperty(navigator,'clipboard',{value:{writeText(){throw Error('Blocked');}},configurable:true});await click(button());assert(field());assert.equal(button().getAttribute('aria-busy'),'false');
  const realLocation=globalThis.location;Object.defineProperty(globalThis,'location',{value:{href:'file:///tmp/index.html'},configurable:true});await click(button());assert(!field());assert(share().querySelector('[role="alert"]'));Object.defineProperty(globalThis,'location',{value:realLocation,configurable:true});
  await render({postId:'invalid'});await click(button());assert(!field());assert(!share().querySelector('a'));assert(share().querySelector('[role="alert"]'));
  await render({postId,replyId:'invalid'});await click(button());assert(!field());assert(!share().querySelector('a'));assert(share().querySelector('[role="alert"]'));
  Object.defineProperty(navigator,'clipboard',{value:clipboard,configurable:true});await render({replyId:undefined});assert(!share().querySelector('a'));const plain=await begin();assert.equal(plain.value,nav.communityShareUrl(location.href,postId));await settle(plain);
  // The real discussion keeps drafts, reading context and replies intact while sharing.
  const alice={can_publish_community:true,id:'alice',username:'alice',display_name:'Alice',avatar_color:'#48766b'},bob={...alice,id:'bob',username:'bob',display_name:'Bob'};
  const replies=[{id:replyId,post_id:postId,parent_id:null,body:'A useful reply',created_at:'2026-10-15T12:00:00Z',author:bob}];
  const post={id:postId,title:'A real solution discussion',body:'## Approach\n\nA derivation.',solution:true,category:'implementation',problem:{id:'architecture/attention',number:1,title:'Attention'},problem_id:'architecture/attention',created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',author:alice,counts:{likes:0,reposts:0,replies:1},viewer:{},replies};
  let reads=0;globalThis.__sharingFixture={account:{user:bob,loading:false,capabilities:{},refresh:async()=>{}},request:async(path,payload)=>{assert.equal(payload,undefined,'Sharing must never publish or mutate social data');assert.equal(path,'community/posts/'+postId);reads++;return structuredClone(post);}};
  const draft={body:'Private unpublished reply 😀',parentId:replyId};assert(writeReplyDraft(sessionStorage,bob.id,postId,draft));const saved=readReplyDraft(sessionStorage,bob.id,postId);
  route=nav.communityPostHref(postId,'#/community?solutions=1&sort=top&q=mask+%26+gradient').slice(2);window.history.replaceState(null,'','#/'+route);discussion=true;await render();
  const editor=document.querySelector('#reply-body');assert(editor);editor.setSelectionRange(4,13,'backward');editor.scrollTop=24;viewport=900;const pageBefore=location.href,readsBefore=reads,scrollsBefore=scrolls.length;
  const realReplyLink=document.querySelector('#reply-'+replyId+' .discussion-share a');assert.equal(nav.communityPostReturn(realReplyLink.getAttribute('href'),postId),'#/community?solutions=1&sort=top&q=mask+%26+gradient');assert.equal(nav.communityReplyTarget(realReplyLink.getAttribute('href'),postId),replyId);
  await click(document.querySelector('#reply-'+replyId+' .discussion-copy'));const replyCopy=pending.shift();assert.equal(replyCopy.value,nav.communityShareUrl(pageBefore,postId,replyId));await settle(replyCopy,true);
  assert.equal(document.querySelector('#reply-body'),editor);assert.equal(editor.value,draft.body);assert.equal(editor.selectionStart,4);assert.equal(editor.selectionEnd,13);assert.equal(editor.selectionDirection,'backward');assert.equal(editor.scrollTop,24);assert.deepEqual(readReplyDraft(sessionStorage,bob.id,postId),saved);assert.equal(location.href,pageBefore);assert.equal(reads,readsBefore);assert.equal(scrolls.length,scrollsBefore);assert.equal(viewport,900);
  const unmounted=await begin();await act(async()=>root.unmount());root=undefined;assert.equal(deadlines.size,0);assert([...historyListeners.values()].every(listeners=>listeners.size===0));await settle(unmounted,true);assert.equal(document.querySelector('#root').textContent,'');
  assert(writes.every(value=>!value.includes('private')&&!value.includes('alice')&&!value.includes('q=')&&!value.includes('from=')));
  console.log('PASS discussion sharing: canonical public-only thread/reply URLs, strict IDs/origins and login continuation, preserved local source context, deduplicated focused copy, bilingual manual fallback/selection, timeout/retry and stale reply/post/ABA/context/route/unmount isolation, real discussion draft/caret/scroll continuity and no social mutation');
}finally{
  if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
  for(const [key,descriptor]of previous)if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];
}
