import assert from 'node:assert/strict';
import React,{act} from 'react';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/community',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:false});dom.window.HTMLElement.prototype.scrollIntoView=function(){};dom.window.scrollTo=()=>{};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__inlineEdit:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-inline-edits-');
const server=await createServer({cacheDir,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'inline-edit-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0inline-edit-social';},load(id){if(id==='\0inline-edit-social')return `import React from 'react';export const useAccount=()=>globalThis.__inlineEdit.account;export const socialApi=(...args)=>globalThis.__inlineEdit.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{constructor(message,status,code){super(message);this.status=status;this.code=code;}};export const Avatar=({user})=>React.createElement('span',null,user.anonymous?'Anonymous':user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.anonymous?'Anonymous':user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Actions}=await server.ssrLoadModule('/src/CommunityCardActions.tsx');
 const {default:Community,ProblemSolutions}=await server.ssrLoadModule('/src/Community.tsx');
 const edits=await server.ssrLoadModule('/src/replyEdits.ts');
 const drafts=await server.ssrLoadModule('/src/replyDrafts.ts');
 const {withConfirmedReplyEdit}=await server.ssrLoadModule('/src/communityDiscussion.ts');
 const {SocialError}=await server.ssrLoadModule('\0inline-edit-social');
 const first='a'.repeat(32),second='b'.repeat(32),one='1'.repeat(32),two='2'.repeat(32),three='3'.repeat(32),v1='a'.repeat(64),v2='b'.repeat(64),v3='c'.repeat(64);
 const alice={id:'alice',username:'alice',display_name:'Alice',can_publish_community:true,affiliation:{kind:'institution',domain:'research.edu',name:'测试研究院',name_en:'Research Institute'}};
 const bob={...alice,id:'bob',username:'bob',display_name:'Bob'},anonymous={...alice,id:'anonymous:'+first,username:'',anonymous:true,display_name:'Private name'};
 const original={id:one,post_id:first,parent_id:null,body:'Original reasoning.',author:alice,created_at:'2026-10-17T03:00:00Z',updated_at:'2026-10-17T03:00:00Z',can_edit:true,edit_version:v1};
 const initialReplies=()=>[{...original},{...original,id:two,author:anonymous,parent_id:one,body:'Anonymous reasoning.',created_at:'2026-10-17T02:00:00Z'},{...original,id:three,author:bob,can_edit:false,body:'Other person.',created_at:'2026-10-17T01:00:00Z'}];
 const topic={id:'architecture/attention',number:22,title:'Attention'};
 const createPost=id=>({id,title:'Attention discussion',body:'A public post.',author:alice,category:'implementation',problem_id:topic.id,problem:topic,created_at:'2026-10-17T00:00:00Z',updated_at:'2026-10-17T00:00:00Z',counts:{likes:2,replies:3,reposts:0},viewer:{liked:false}});
 let replies=initialReplies(),mode='',surface='cards',failRefresh=false,held,heldRead,holdRead=false;
 const requests=[],writes=[],account={user:alice,loading:false,capabilities:{community_solutions:true,community_anonymous_posts:true},refresh:async()=>{if(failRefresh)throw Error('Account refresh unavailable');}};
 const confirmed=(row,payload)=>({...row,body:payload.body.trim(),edit_version:row.body===payload.body.trim()?row.edit_version:row.edit_version===v1?v2:v3,updated_at:'2026-10-17T05:00:00Z'});
 globalThis.__inlineEdit={account,request:async(path,payload,signal)=>{
  requests.push({path,payload,signal});
  if(!payload){
   if(path.startsWith('community/posts?')){const solution=new URLSearchParams(path.split('?')[1]).get('solutions')==='true';return {items:[{...createPost(first),solution}],total:1,...(solution?{solutions:true}:{})};}
   const post={...createPost(path.split('/')[2]),replies:replies.map(row=>({...row}))};
   if(holdRead)return new Promise(resolve=>heldRead={signal,resolve:()=>resolve(post)});
   return post;
  }
  writes.push({path,payload,signal});assert(path.endsWith('/edit'));assert.equal(payload.expected_user_id,'alice');assert.equal(signal.aborted,false);
  const row=replies.find(row=>path.includes(row.id)),result=confirmed(row,payload);
  if(mode==='auth')throw new SocialError('Session expired',401,'account_changed');
  if(mode==='timeout')throw new DOMException('Expired','TimeoutError');
  if(mode==='conflict'||mode==='deleted'){
   replies=mode==='deleted'?replies.filter(item=>item.id!==row.id):replies.map(item=>item.id===row.id?{...item,body:'Updated in another tab.',edit_version:v2}:item);
   throw new SocialError('Changed',409,'edit_conflict');
  }
  if(mode==='malformed')return {...result,post_id:second};
  const complete=()=>{replies=replies.map(item=>item.id===row.id?result:item);return result;};
  if(mode==='hold')return new Promise(resolve=>held={resolve:()=>resolve(complete()),result});
  return complete();
 }};
 const h=React.createElement;
 function Locale(){const {locale,setLocale}=useI18n();return h('button',{id:'locale',onClick:()=>setLocale(locale==='en'?'zh':'en')},'Locale');}
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(h(LocaleProvider,null,h(Locale),h('textarea',{'aria-label':'Private code',defaultValue:'PRIVATE_CODE'}),surface==='cards'?h(React.Fragment,null,...[first,second].map(id=>h('article',{'data-card':id,key:id},h(Actions,{post:createPost(id),returnTo:'#/community'})))):surface==='solutions'?h(ProblemSolutions,{topic}):h(Community,{...(surface==='discussion'?{postId:first}:{}),route:location.hash})));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const reset=async()=>{await stop();sessionStorage.clear();replies=initialReplies();mode='';holdRead=false;failRefresh=false;surface='cards';account.user=alice;account.loading=false;location.hash='#/community';localStorage.setItem('paperleet.locale','en');await render();};
 const card=()=>surface==='cards'?document.querySelector(`[data-card="${first}"]`):surface==='discussion'?document.querySelector('.discussion-page'):document.querySelector('.community-post-card');
 const named=(text,scope=document)=>[...scope.querySelectorAll('button')].find(node=>node.textContent===text);
 const click=async node=>{assert(node,'Missing comment editing control');await act(async()=>{node.focus();node.click();});await render();};
 const change=async(node,value)=>{assert(node);await act(async()=>{Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});await render();};
 const editor=()=>document.querySelector('.reply-editor'),field=()=>editor()?.querySelector('textarea'),editButton=(id=one)=>document.querySelector(`[data-card-edit-reply="${id}"]`);
 const toggle=()=>card().querySelector('.card-actions>button');
 const open=async()=>click(toggle());
 const submit=async()=>{await act(async()=>editor().querySelector('form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));await render();};
 // A response must acknowledge the exact row, source and owner before any draft clears.
 console.log('CHECK A response must acknowledge the exact row, source and owner before any draft clears.');
 const submitted={body:'  New reasoning.  ',baseVersion:v1},ack=confirmed(original,submitted);
 assert.equal(edits.confirmedReplyEdit(original,'alice',submitted,ack),ack);
 for(const invalid of [null,[],{}, {...ack,id:two},{...ack,post_id:second},{...ack,parent_id:two},{...ack,created_at:'other'},{...ack,body:'Different text'},{...ack,author:bob},{...ack,author:{...alice,anonymous:true}},{...ack,can_edit:false},{...ack,edit_version:v1},{...ack,edit_version:'bad'}])assert.equal(edits.confirmedReplyEdit(original,'alice',submitted,invalid),undefined);
 assert(edits.confirmedReplyEdit(original,'alice',{body:original.body,baseVersion:v1},original),'An idempotent no-op acknowledgement is valid');
 const detail={...createPost(first),replies:initialReplies()},merged=withConfirmedReplyEdit(detail,ack,v1);
 assert.equal(merged.replies[0],ack);assert.equal(merged.replies[1],detail.replies[1]);assert.equal(merged.counts,detail.counts);assert.equal(merged.updated_at,detail.updated_at);
 assert.equal(withConfirmedReplyEdit(merged,ack,v1),merged);
 const newer={...detail,replies:[{...ack,body:'Newest cache',edit_version:v3}]};assert.equal(withConfirmedReplyEdit(newer,ack,v1),newer);assert.equal(withConfirmedReplyEdit(undefined,ack,v1),undefined);
 // Editing a published reply never replaces the independent new-comment draft.
 console.log('CHECK Editing a published reply never replaces the independent new-comment draft.');
 await reset();await open();assert(editButton());assert(editButton(two));assert(!editButton(three));
 await click(named('Write comment',card()));const newField=card().querySelector('.card-reply-form textarea');await change(newField,'PRIVATE_NEW_REPLY');newField.setSelectionRange(2,8,'backward');newField.scrollTop=27;newField.scrollLeft=11;
 const requestsBefore=requests.length;await click(editButton());assert.equal(requests.length,requestsBefore);assert(document.activeElement===field(),"DOM identity: document.activeElement === field()");assert(card().querySelector('.card-reply-form').hidden);
 await change(field(),'  Revised approach with $x^2$.  ');const editField=field();editField.setSelectionRange(2,9,'backward');editField.scrollTop=35;editField.scrollLeft=14;
 assert.equal(edits.readReplyEdit(sessionStorage,'alice',one).body,editField.value);assert.equal(drafts.readReplyDraft(sessionStorage,'alice',first).body,'PRIVATE_NEW_REPLY');
 assert(card().querySelector('.card-reply-form textarea')===newField,'Opening an edit keeps the new-comment editor mounted');account.loading=true;await render();assert(field().disabled);assert(field()===editField,"DOM identity: field() === editField");account.loading=false;await render();
 await click(toggle());assert(document.querySelector('.card-comment-editor').hidden);assert(field()===editField,"DOM identity: field() === editField");const beforeHidden=writes.length;await submit();assert.equal(writes.length,beforeHidden);
 await open();assert(field()===editField,"DOM identity: field() === editField");assert.deepEqual([field().selectionStart,field().selectionEnd,field().selectionDirection,field().scrollTop,field().scrollLeft],[2,9,'backward',35,14]);
 await click(named('Save changes',editor()));assert(!editor());assert(document.activeElement===editButton(),"DOM identity: document.activeElement === editButton()");assert(card().textContent.includes('Comment changes saved.'));assert(card().querySelector('.card-comment-edited'));assert(card().textContent.includes('3 comments'));
 assert.equal(edits.readReplyEdit(sessionStorage,'alice',one),undefined);assert.equal(drafts.readReplyDraft(sessionStorage,'alice',first).body,'PRIVATE_NEW_REPLY');const resumedNewField=card().querySelector('.card-reply-form textarea');assert.equal(resumedNewField.value,'PRIVATE_NEW_REPLY');assert.deepEqual([resumedNewField.selectionStart,resumedNewField.selectionEnd,resumedNewField.selectionDirection,resumedNewField.scrollTop,resumedNewField.scrollLeft],[2,8,'backward',27,11]);
 assert.equal(writes.at(-1).payload.expected_version,v1);assert(!JSON.stringify(writes).includes('PRIVATE_CODE'));assert(!JSON.stringify(writes).includes('PRIVATE_NEW_REPLY'));
 // Close/reopen keeps an unsaved per-reply draft, including anonymous authorship.
 console.log('CHECK Close/reopen keeps an unsaved per-reply draft, including anonymous authorship.');
 await click(editButton(two));assert(field().value.includes('Anonymous reasoning'));await change(field(),'Anonymous correction');await click(named('Close editor',editor()));assert(document.activeElement===editButton(two),"DOM identity: document.activeElement === editButton(two)");await click(editButton(two));assert.equal(field().value,'Anonymous correction');await click(named('Save changes',editor()));assert(!document.body.textContent.includes('Private name'));assert(document.body.textContent.includes('Research Institute'));
 // Entry focus respects a newer reader choice; preview and collapse retain native edit state.
 await reset();await open();const newerFocus=document.querySelector('[aria-label="Private code"]');await act(async()=>{editButton().focus();editButton().click();newerFocus.focus();});await render();assert(document.activeElement===newerFocus,'Opening an edit must respect a newer focus');await change(field(),'Preview correction');field().setSelectionRange(1,7,'backward');field().scrollTop=31;field().scrollLeft=9;await click(named('Preview',editor()));await click(toggle());await open();assert(named('Preview',editor()).getAttribute('aria-pressed')==='true');await click(named('Write',editor()));assert.deepEqual([field().value,field().selectionStart,field().selectionEnd,field().selectionDirection,field().scrollTop,field().scrollLeft],['Preview correction',1,7,'backward',31,9]);await click(named('Close editor',editor()));
 // Malformed acknowledgements and timeouts preserve the submitted version for an exact retry.
 console.log('CHECK Malformed acknowledgements and timeouts preserve the submitted version for an exact retry.');
 for(const failure of ['malformed','timeout']){await reset();await open();await click(editButton());await change(field(),'Retry exact change');mode=failure;await submit();assert(editor());assert.equal(edits.readReplyEdit(sessionStorage,'alice',one).baseVersion,v1);assert(!card().textContent.includes('Comment changes saved.'));mode='';await submit();assert(!editor());assert.equal(writes.at(-1).payload.expected_version,v1);}
 // Conflict refresh retains the live editor; review is explicit before using the new version.
 console.log('CHECK Conflict refresh retains the live editor; review is explicit before using the new version.');
 await reset();await open();await click(editButton());await change(field(),'Keep my local edit');const conflictField=field();conflictField.setSelectionRange(2,7,'backward');conflictField.scrollTop=44;mode='conflict';await submit();assert(field()===conflictField,"DOM identity: field() === conflictField");assert.equal(field().value,'Keep my local edit');assert(editor().textContent.includes('Updated in another tab.'));assert(named('Save changes',editor()).disabled);assert.equal(field().selectionDirection,'backward');
 mode='';await click(named('Reviewed; continue from this version',editor()));await submit();assert(!editor());assert.equal(writes.at(-1).payload.expected_version,v2);
 // Saving cannot clear newer private work or steal a newer focus or navigation intent.
 console.log('CHECK Saving cannot clear newer private work or steal a newer focus or navigation intent.');
 await reset();await open();await click(editButton());await change(field(),'Submitted correction');mode='hold';await click(named('Save changes',editor()));assert(held);edits.writeReplyEdit(sessionStorage,'alice',one,{body:'Later private edit',baseVersion:v1});const code=document.querySelector('[aria-label="Private code"]');code.focus();await act(async()=>held.resolve());await render();assert(document.activeElement===code,"DOM identity: document.activeElement === code");assert.equal(edits.readReplyEdit(sessionStorage,'alice',one).body,'Later private edit');assert(!editor());
 await click(editButton());assert.equal(field().value,'Later private edit');assert(named('Save changes',editor()).disabled);await click(named('Close editor',editor()));
 await reset();await open();await click(editButton());await change(field(),'Save before leaving');mode='hold';await submit();location.hash='#/interviews';document.querySelector('#locale').focus();await act(async()=>held.resolve());await render();assert.equal(location.hash,'#/interviews');assert.equal(document.activeElement.id,'locale');
 // A hidden edit may finish without reopening or jumping back into its card.
 console.log('CHECK A hidden edit may finish without reopening or jumping back into its card.');
 await reset();await open();await click(editButton());await change(field(),'Save then collapse');mode='hold';await submit();await click(toggle());assert(document.querySelector('.card-comment-editor').hidden);await act(async()=>held.resolve());await render();assert(!editor());assert(!card().querySelector('.card-comments'));
 // An obsolete comment read must not overwrite a confirmed save after reopening.
 console.log('CHECK An obsolete comment read must not overwrite a confirmed save after reopening.');
 await reset();await open();await click(editButton());await change(field(),'Latest acknowledged text');mode='hold';await submit();await click(toggle());holdRead=true;await open();assert(heldRead);await act(async()=>held.resolve());await render();assert(heldRead.signal.aborted);await act(async()=>heldRead.resolve());await render();assert(card().textContent.includes('Latest acknowledged text'));assert(!card().textContent.includes('Original reasoning.'));
 // Expired identity has explicit recovery and retains the old edit until confirmed.
 console.log('CHECK Expired identity has explicit recovery and retains the old edit until confirmed.');
 await reset();await open();await click(editButton());await change(field(),'Retained during account recovery');mode='auth';failRefresh=true;await submit();assert(editor().textContent.includes('Your account is unconfirmed'));assert(field().disabled);assert(editor().querySelector('a[href*="account"]'));assert(edits.readReplyEdit(sessionStorage,'alice',one));mode='';failRefresh=false;await click(named('Refresh account',editor()));assert(!field().disabled);await submit();assert(!editor());
 // A deleted reply releases the edit lock without silently discarding its draft.
 console.log('CHECK A deleted reply releases the edit lock without silently discarding its draft.');
 await reset();await open();await click(editButton());await change(field(),'Deleted target edit');mode='deleted';await submit();assert(!editor());assert(document.querySelector('.card-comment-editor').textContent.includes('deleted'));assert(edits.readReplyEdit(sessionStorage,'alice',one));await click(named('Close editor',document.querySelector('.card-comment-editor')));assert(!named('Write comment',card()).disabled);
 // Owner switches retire the rendered private state; guests and legacy rows never gain edit controls.
 console.log('CHECK Owner switches retire the rendered private state; guests and legacy rows never gain edit controls.');
 await reset();await open();await click(editButton());await change(field(),'Alice only');account.user=bob;await render();assert(!editor());assert(!document.body.textContent.includes('Alice only'));account.user=undefined;await render();await open();assert(!editButton());account.user=alice;replies=replies.map(row=>({...row,edit_version:undefined}));await render();await open();assert(!editButton());
 // Quoting an edited source with the same id must include its new body, with exact undo.
 await reset();await open();await click(document.querySelector(`[data-card-reply="${one}"] [aria-label^="Quote comment"]`));const oldQuote=drafts.readReplyDraft(sessionStorage,'alice',first);await click(editButton());await change(field(),'New source after editing');await submit();await click(document.querySelector(`[data-card-reply="${one}"] [aria-label^="Quote comment"]`));assert(card().querySelector('.card-reply-form textarea').value.includes('New source after editing'));await click(named('Undo quote',card()));assert.deepEqual(drafts.readReplyDraft(sessionStorage,'alice',first),oldQuote);
 // Real feed, embedded solutions and complete discussions share the guarded editor.
 console.log('CHECK Real feed, embedded solutions and complete discussions share the guarded editor.');
 for(const view of ['community','solutions','discussion']){
  await reset();surface=view;location.hash=view==='discussion'?`#/community/${first}`:view==='solutions'?'#/22':'#/community';await render();
  if(view==='discussion')await click(document.getElementById(`edit-reply-${one}`));else {await open();await click(editButton());}
  assert(editor());assert(document.activeElement===field(),"DOM identity: document.activeElement === field()");await change(field(),'Correction from '+view);await submit();assert(!editor());assert(document.body.textContent.includes('Correction from '+view));
 }
 console.log('PASS inline reply editing: exact acknowledgement and version-aware cache merge; own/anonymous permission, independent comment drafts, hidden editor/caret/scroll retention, guarded focus/navigation, malformed/timeout retry, explicit conflict review, stale-read retirement, expired-account recovery, deleted/legacy/guest/owner isolation and real feed/solution/full-discussion integration');
}finally{if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();for(const[key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
