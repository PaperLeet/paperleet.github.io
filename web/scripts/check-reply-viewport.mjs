import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const id='a'.repeat(32);
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/community?category=papers&q=old&page=2',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:false});
let viewport=240;
Object.defineProperty(dom.window,'scrollY',{get:()=>viewport});
const values={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,requestAnimationFrame:dom.window.requestAnimationFrame.bind(dom.window),IS_REACT_ACT_ENVIRONMENT:true,__replyViewportFixture:undefined};
const previous=new Map(Object.keys(values).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(values))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'reply-viewport-account',enforce:'pre',resolveId(source){if(source==='./social')return '\0reply-viewport-account';},load(name){if(name==='\0reply-viewport-account')return `import React from 'react';export const useAccount=()=>globalThis.__replyViewportFixture.account;export const socialApi=(...args)=>globalThis.__replyViewportFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{constructor(message,status,code){super(message);this.status=status;this.code=code;}};export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Actions}=await server.ssrLoadModule('/src/CommunityCardActions.tsx');
 const {captureReadingAnchor,restoreReadingAnchor}=await server.ssrLoadModule('/src/readingViewport.ts');
 const drafts=await server.ssrLoadModule('/src/replyDrafts.ts');
 const user={id:'alice',username:'alice',display_name:'Alice',can_publish_community:true,avatar_color:'#48766b',affiliation:{kind:'institution',domain:'test.edu',name:'测试大学',name_en:'Test University'}};
 const account={user,loading:false,capabilities:{},refresh:async()=>{}};
 const replyId=n=>n.toString(16).padStart(32,'0');
 const replies=Array.from({length:9},(_,i)=>({id:replyId(i+1),post_id:id,parent_id:null,body:'Comment '+(i+1),created_at:`2026-10-16T00:00:0${i}Z`,author:user}));
 replies[8].parent_id=replyId(1);replies[0].parent_id=replyId(2);replies[0].body='## A detailed explanation\n\n'+('Boundary cases and reasoning. '.repeat(35))+'\n\n```python\nx = 2\n```';
 const post={id,title:'An attention solution',body:'Reasoning',solution:true,category:'implementation',problem_id:'architecture/attention',author:user,counts:{likes:2,replies:9,reposts:0},viewer:{},created_at:'2026-10-16T00:00:00Z',updated_at:'2026-10-16T00:00:00Z'};
 const reads=[],writes=[];globalThis.__replyViewportFixture={account,request:async(path,body)=>{if(body){writes.push({path,body});throw Error('No publication expected');}reads.push(path);return {...post,replies:[...replies]};}};
 let paneTop=100,paneHeight=560;const windowCalls=[],nativeCalls=[],paneCalls=[];
 dom.window.scrollTo=options=>{windowCalls.push(options);viewport=options.top;};
 dom.window.HTMLElement.prototype.scrollTo=function(options){paneCalls.push({node:this,...options});if(options.top!==undefined)this.scrollTop=options.top;if(options.left!==undefined)this.scrollLeft=options.left;};
 dom.window.HTMLElement.prototype.scrollIntoView=function(options){nativeCalls.push({node:this,options});};
 const commentHeight=node=>190+(node?.querySelector('.inline-comment-body.is-open')?1100:0)+(node?.querySelector('.reply-reading-return')?32:0);
 const offset=node=>{
  if(node.matches('.inline-comment-collapse'))return offset(node.closest('[data-discussion-reply]'))+commentHeight(node.closest('[data-discussion-reply]'))-44;
  const row=node.closest('[data-discussion-reply]'),list=[...document.querySelectorAll('[data-discussion-reply]')],notice=document.querySelector('.reply-reading-notice')?32:0;
  if(row){const index=list.indexOf(row);return 320+notice+list.slice(0,index).reduce((sum,n)=>sum+commentHeight(n),0)+(node===row?0:70);}
  if(node.closest('.card-reply-form')||node.matches('.card-comment-compose-prompt'))return 320+notice+list.reduce((sum,n)=>sum+commentHeight(n),0)+170;
  if(node.matches('.card-collapse-comments'))return 320+notice+list.reduce((sum,n)=>sum+commentHeight(n),0)+360;
  if(node.matches('.card-more-comments'))return 320+notice+list.reduce((sum,n)=>sum+commentHeight(n),0)+30;
  return node.matches('.card-comments')?220:140;
 };
 dom.window.HTMLElement.prototype.getBoundingClientRect=function(){
  const pane=this.closest('.description-scroll'),base=pane?paneTop:0;
  if(this.matches('.description-scroll'))return {top:paneTop,bottom:paneTop+paneHeight+2,height:paneHeight+2,left:0,right:600,width:600};
  const height=this.matches('[data-discussion-reply]')?commentHeight(this):this.tagName==='TEXTAREA'?120:40,top=base+offset(this)-(pane?pane.scrollTop:viewport);
  return {top,bottom:top+height,height,left:0,right:580,width:580};
 };
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement('textarea',{'aria-label':'Practice code',defaultValue:'def solve(x): return x'}),React.createElement('section',{className:'description-scroll'},React.createElement(Actions,{post,returnTo:'community/problem/architecture/attention?solutions=1'}))));});
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
 const pane=()=>document.querySelector('.description-scroll'),row=n=>document.querySelector(`[data-discussion-reply="${replyId(n)}"]`),button=text=>[...document.querySelectorAll('button')].find(node=>node.textContent===text);
 const relative=node=>node.getBoundingClientRect().top-pane().getBoundingClientRect().top-pane().clientTop;
 const edit=async(node,value)=>act(async()=>{Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
 location.hash='#/22?from=collections%2Fllm-basics';const href=location.href;
 await render();Object.defineProperty(pane(),'clientTop',{value:2});Object.defineProperty(pane(),'clientHeight',{get:()=>paneHeight});
 pane().scrollTop=90;pane().scrollLeft=27;const code=document.querySelector('[aria-label="Practice code"]');code.setSelectionRange(4,12,'backward');code.scrollTop=88;code.scrollLeft=13;
 await click(document.querySelector('.card-actions>button[aria-expanded]'));assert(document.querySelector('.card-comments'));assert.equal(reads.length,1);assert.equal(document.querySelectorAll('[data-discussion-reply]').length,3);
 const origin=row(9),reference=origin.querySelector('.card-comment-context'),oldPosition=relative(origin);await click(reference);assert(row(1));assert.equal(document.activeElement,row(1));assert.equal(relative(row(1)),24);assert.equal(windowCalls.length,0);assert.equal(nativeCalls.length,0);
 const parentPosition=relative(row(1)),secondReference=row(1).querySelector('.card-comment-context');await click(secondReference);assert.equal(document.activeElement,row(2));assert.equal(relative(row(2)),24);
 paneTop=180;await click(document.querySelector('.reply-reading-return'));assert.equal(document.activeElement,secondReference);assert.equal(relative(row(1)),parentPosition);assert.equal(document.querySelectorAll('[data-discussion-reply]').length,9);
 await click(document.querySelector('.reply-reading-return'));assert.equal(document.activeElement,reference);assert.equal(relative(row(9)),oldPosition);assert.equal(document.querySelectorAll('[data-discussion-reply]').length,3);assert(document.querySelector('.reply-reading-notice').textContent.includes('Returned'));
 assert.equal(location.href,href);assert.equal(reads.length,1);assert.equal(pane().scrollLeft,27);assert.equal(code.value,'def solve(x): return x');assert.equal(code.selectionStart,4);assert.equal(code.selectionEnd,12);assert.equal(code.selectionDirection,'backward');assert.equal(code.scrollTop,88);assert.equal(code.scrollLeft,13);assert.equal(windowCalls.length,0);
 // More-comments focus and expanded-body collapse use the same pane.
 await click(document.querySelector('.card-more-comments'));assert.equal(document.querySelectorAll('[data-discussion-reply]').length,6);assert.equal(document.activeElement,row(6));assert.equal(relative(row(6)),24);
 await click(row(9).querySelector('.card-comment-context'));const expand=row(1).querySelector('.inline-comment-toggle');await click(expand);assert(row(1).querySelector('.inline-comment-body.is-open'));
 pane().scrollTop+=900;await click(row(1).querySelector('.inline-comment-collapse'));assert.equal(document.activeElement,expand);assert(!row(1).querySelector('.inline-comment-body.is-open'));assert(relative(expand)>=0&&relative(expand)+40<=paneHeight);assert.equal(windowCalls.length,0);assert.equal(nativeCalls.length,0);
 // Comment writing restores its own selection/scroll while retaining practice code.
 await click(document.querySelector('.card-comment-compose-prompt'));const field=document.querySelector('.card-reply-form textarea');assert.equal(document.activeElement,field);assert(relative(field)>=0&&relative(field)+120<=paneHeight);
 await edit(field,'Unpublished thought about the proof');field.setSelectionRange(3,18,'backward');field.scrollTop=51;field.scrollLeft=22;const draft=JSON.stringify(drafts.readReplyDraft(sessionStorage,'alice',id));
 const currentReference=row(9).querySelector('.card-comment-context');await click(currentReference);const back=document.querySelector('.reply-reading-return'),beforeWheel=pane().scrollTop;
 await act(async()=>{back.focus();back.click();window.dispatchEvent(new dom.window.WheelEvent('wheel',{deltaY:30}));});assert.equal(pane().scrollTop,beforeWheel);assert(document.querySelector('.reply-reading-return'));
 await click(document.querySelector('.reply-reading-return'));assert.equal(document.activeElement,currentReference);assert.equal(document.querySelector('.card-reply-form textarea'),field);assert.equal(field.value,'Unpublished thought about the proof');assert.equal(field.selectionStart,3);assert.equal(field.selectionEnd,18);assert.equal(field.selectionDirection,'backward');assert.equal(field.scrollTop,51);assert.equal(field.scrollLeft,22);assert.equal(JSON.stringify(drafts.readReplyDraft(sessionStorage,'alice',id)),draft);
 const untouched=pane().scrollTop;await act(async()=>{currentReference.focus();currentReference.click();code.focus();});assert.equal(document.activeElement,code);assert.equal(pane().scrollTop,untouched);
 // A shorter pane must reveal the focused return control, not just its row title.
 paneHeight=560;pane().scrollTop=0;const resizeReference=row(9).querySelector('.card-comment-context');await click(resizeReference);paneHeight=160;await click(document.querySelector('.reply-reading-return'));assert.equal(document.activeElement,resizeReference);assert(relative(resizeReference)>=0&&relative(resizeReference)+40<=paneHeight,'The restored comment control remains visible after shrinking its pane');paneHeight=560;
 // Folding a long preview returns to the card control within the pane.
 await click(document.querySelector('.card-collapse-comments'));assert(!document.querySelector('.card-comments'));const toggle=document.querySelector('.card-actions>button[aria-expanded]');assert.equal(document.activeElement,toggle);assert(relative(toggle)>=0&&relative(toggle)+40<=paneHeight);assert.equal(JSON.stringify(drafts.readReplyDraft(sessionStorage,'alice',id)),draft);
 assert.equal(windowCalls.length,0);assert.equal(nativeCalls.length,0);assert.equal(viewport,240);assert.equal(pane().scrollLeft,27);assert.equal(writes.length,0);
 // Anchor primitives protect surface identity, resize and partially visible origins.
 const target=document.createElement('button');pane().append(target);pane().scrollTop=0;const anchor=captureReadingAnchor(target);paneHeight=100;pane().scrollTop=500;assert(restoreReadingAnchor(target,anchor));assert.equal(relative(target),56);
 pane().scrollTop=300;const partial=captureReadingAnchor(target);assert(partial.top<0);pane().scrollTop=500;assert(restoreReadingAnchor(target,partial));assert.equal(relative(target),partial.top);
 const other=document.createElement('section');other.className='description-scroll';document.body.append(other);other.append(target);const scrollCount=paneCalls.length;assert.equal(restoreReadingAnchor(target,partial),false);assert.equal(paneCalls.length,scrollCount);other.remove();assert.equal(restoreReadingAnchor(target,partial),false);
 await act(async()=>root.unmount());root=undefined;
 console.log('PASS nested reply viewport: actual card parent/multi-hop/return/batch reading, moving-pane offsets and notices, pane-only body/editor/collapse visibility, retained comment/practice source/caret/both scrolls, newer wheel/focus priority, surface/resize/partial-origin guards and zero navigation/publication');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
