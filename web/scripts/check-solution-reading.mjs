import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/community',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:true});dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__solutionReaderFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
 name:'solution-reader-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0solution-reader-fixture';},load(name){if(name==='\0solution-reader-fixture')return `import React from 'react';export const useAccount=()=>globalThis.__solutionReaderFixture.account;export const socialApi=(...args)=>globalThis.__solutionReaderFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}
}]});
let root;const priorResizeDescriptor=Object.getOwnPropertyDescriptor(globalThis,"ResizeObserver");
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {ProblemSolutions}=await server.ssrLoadModule('/src/Community.tsx');
 const {default:Card}=await server.ssrLoadModule('/src/SolutionReadingCard.tsx');
 const nav=await server.ssrLoadModule('/src/communityNavigation.ts');
 const topic={id:'architecture/attention',number:1,title:'注意力',title_en:'Attention'};
 const user={id:'alice',username:'alice',display_name:'Alice',can_publish_community:true,affiliation:{kind:'institution',domain:'test.edu',name:'测试大学',name_en:'Test University'}};
 const account={user,loading:false,capabilities:{community_solutions:true,community_feed_sorting:true,community_bookmarks:true},refresh:async()=>{}};
 const id='a'.repeat(32);let source='## Approach\n\nA careful explanation.\n\n## Implementation\n\n```python\ndef solve(x):\n    return x * 2\n```\n\n## Complexity\n\nLinear time and constant extra space.';
 const requests=[],windowScrolls=[],nativeScrolls=[],observers=[];let fetchedBody=source,solo=false,codeCallbacks=0;
 globalThis.__solutionReaderFixture={account,request:async(path,body)=>{
  assert.equal(body,undefined,'Reading and folding do not publish or react');requests.push(path);
  return {items:[{id,title:'A complete attention solution',body:fetchedBody,solution:true,category:'implementation',problem_id:topic.id,problem:topic,author:user,counts:{likes:3,replies:0,reposts:0},viewer:{},created_at:'2026-10-16T00:00:00Z',updated_at:'2026-10-16T00:00:00Z'}],total:1,solutions:true,sort:new URLSearchParams(path.split("?")[1]).get("sort")||"latest"};
 }};
 globalThis.ResizeObserver=class{constructor(callback){this.callback=callback;this.targets=[];this.disconnected=false;observers.push(this);}observe(node){this.targets.push(node);}disconnect(){this.disconnected=true;}};
 dom.window.matchMedia=()=>({matches:true});
 dom.window.scrollTo=options=>windowScrolls.push(options);
 dom.window.HTMLElement.prototype.scrollTo=function(options){if(options.top!==undefined)this.scrollTop=options.top;if(options.left!==undefined)this.scrollLeft=options.left;this.dispatchEvent(new dom.window.Event('scroll'));};
 dom.window.HTMLElement.prototype.scrollIntoView=function(options){nativeScrolls.push({node:this,options});};
 dom.window.HTMLElement.prototype.getBoundingClientRect=function(){
  if(this.matches('.description-scroll'))return {top:100,bottom:662,height:562,left:0,right:600,width:600};
  const pane=this.closest('.description-scroll'),scroll=pane?.scrollTop||0;
  const headings=[...document.querySelectorAll('.discussion-section-heading')],index=headings.indexOf(this);
  const offset=index>=0?500+index*400:140;
  return {top:100+offset-scroll,bottom:144+offset-scroll,height:44,left:0,right:580,width:580};
 };
 function Locale(){const {setLocale}=useI18n();return React.createElement(React.Fragment,null,React.createElement('button',{onClick:()=>setLocale('zh')},'Chinese'),React.createElement('button',{onClick:()=>setLocale('en')},'English'));}
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Locale),React.createElement('textarea',{'aria-label':'Practice code',defaultValue:'def my_solution(): return 42'}),solo?React.createElement(Card,{id,title:'Standalone solution',body:source}):React.createElement('section',{className:'description-scroll',id:'problem-reading-content'},React.createElement(ProblemSolutions,{topic,onCode:()=>{codeCallbacks++;document.querySelector('[aria-label="Practice code"]').focus({preventScroll:true});}}))));});
 const pane=()=>document.querySelector('.description-scroll'),details=()=>document.querySelector('.solution-reading-card>details'),summary=()=>details().querySelector('summary');
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();await new Promise(resolve=>setTimeout(resolve,5));});};
 const frame=()=>act(async()=>new Promise(resolve=>dom.window.requestAnimationFrame(resolve)));
 const button=text=>[...document.querySelectorAll('button')].find(node=>node.textContent===text);
 const shortcut=async(target,init={})=>act(async()=>{target.focus();target.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true,...init}));});
 const expanded=()=>summary().getAttribute('aria-expanded')==='true';
 location.hash='#/22?from=collections%2Fllm-basics';const href=location.href;
 await render();Object.defineProperty(pane(),'clientHeight',{value:560});Object.defineProperty(pane(),'clientTop',{value:2});
 const code=document.querySelector('[aria-label="Practice code"]');code.focus();code.setSelectionRange(4,13,'backward');code.scrollTop=91;code.scrollLeft=37;pane().scrollTop=72;pane().scrollLeft=19;
 const calls=requests.length;assert.equal(calls,1);assert(!expanded());assert(!document.querySelector('.discussion-reading-body'));
 const actions=document.querySelector('.community-card-interaction'),title=summary();assert(!title.querySelector('a'));
 await click(title);assert(expanded());assert(document.querySelector('.discussion-reading-body').textContent.includes('def solve'));assert.equal(requests.length,calls);assert.equal(location.href,href);assert.equal(code,document.querySelector('[aria-label="Practice code"]'));assert.equal(actions,document.querySelector('.community-card-interaction'));assert.equal(code.value,'def my_solution(): return 42');assert.equal(code.selectionStart,4);assert.equal(code.selectionEnd,13);assert.equal(code.selectionDirection,'backward');assert.equal(code.scrollTop,91);assert.equal(code.scrollLeft,37);assert.equal(pane().scrollLeft,19);
 await click(button('Back to code'));assert.equal(codeCallbacks,1);assert.equal(document.activeElement,code);assert(expanded());assert.equal(code.selectionStart,4);assert.equal(code.selectionEnd,13);assert.equal(code.scrollTop,91);assert.equal(code.scrollLeft,37);assert(document.querySelector('[aria-label="Copy python code"]'));
 assert.equal(document.querySelector('.discussion-reading').classList.contains('has-outline'),true);assert(!document.querySelector('.discussion-outline').open);await click(document.querySelector('.discussion-outline>summary'));assert.equal(document.querySelectorAll('.discussion-outline button').length,3);
 const latestObserver=observers.at(-1);assert(latestObserver.targets.includes(pane()));
 await click([...document.querySelectorAll('.discussion-outline button')][1]);assert.equal(document.activeElement.textContent,'Implementation');assert.equal(pane().scrollTop,874);assert.equal(pane().scrollLeft,19);assert.equal(windowScrolls.length,0);assert.equal(nativeScrolls.length,0);await frame();
 assert(!document.querySelector('.discussion-outline').open);assert(latestObserver.disconnected);await click(document.querySelector('.discussion-outline>summary'));assert.equal(document.querySelector('.discussion-outline [aria-current="location"]').textContent,'Implementation');
 pane().scrollTop=1274;pane().dispatchEvent(new dom.window.Event('scroll'));await frame();assert.equal(document.querySelector('.discussion-outline [aria-current="location"]').textContent,'Complexity');
 await click(button('Collapse solution'));assert(!expanded());assert.equal(document.activeElement,title);assert.equal(pane().scrollTop,138);assert.equal(windowScrolls.length,0);assert.equal(actions,document.querySelector('.community-card-interaction'));assert(latestObserver.disconnected);const oldTop=pane().scrollTop;latestObserver.callback();await frame();assert.equal(pane().scrollTop,oldTop);
 // Locale and code edits do not close an exact body. Explicit closing cannot steal newer focus.
 await click(title);await click(button('Chinese'));assert(expanded());assert(button('收起题解'));assert.equal(document.querySelector('.solution-reading-content').textContent.includes('def solve'),true);await click(button('English'));assert(expanded());
 const close=button('Collapse solution');pane().scrollTop=700;await act(async()=>{close.focus();close.click();code.focus();});assert.equal(document.activeElement,code);assert.equal(pane().scrollTop,700);assert(!expanded());
 await click(title);await shortcut(title,{isComposing:true});assert(expanded());await shortcut(title,{keyCode:229});assert(expanded());await shortcut(title,{ctrlKey:true});assert(expanded());await shortcut(title,{repeat:true});assert(expanded());
 // An editable descendant belongs to its own keyboard flow.
 const field=document.createElement('textarea');details().append(field);await shortcut(field);assert(expanded());field.remove();
 await shortcut(title);assert(!expanded());assert.equal(document.activeElement,title);assert.equal(requests.length,calls);
 await click(title);await act(async()=>{button('Collapse solution').focus();button('Collapse solution').click();location.hash='#/interviews';});assert(!expanded());assert.notEqual(document.activeElement,title);location.href=href;
 // A changed body closes the previous reader without reviving it on a late toggle.
 await click(title);assert(expanded());fetchedBody='## Updated derivation\n\nNew explanation.';
 await act(async()=>{const select=document.querySelector('.community-sort select');Object.getOwnPropertyDescriptor(dom.window.HTMLSelectElement.prototype,'value').set.call(select,'top');select.dispatchEvent(new dom.window.Event('change',{bubbles:true}));await new Promise(resolve=>setTimeout(resolve,5));});
 assert(!expanded());assert(!document.querySelector('.discussion-reading-body'));assert.equal(location.href,href);assert.equal(document.activeElement===code,false);await click(summary());assert(document.querySelector('.discussion-reading-body').textContent.includes('New explanation'));
 const link=document.querySelector('.solution-reading-page');assert.equal(nav.communityPostReturn(link.getAttribute('href'),id).includes('solutions=1'),true);assert.equal(requests.length,2);
 account.user={...user,id:'bob',username:'bob'};await render();assert(!expanded());assert.equal(document.querySelector('[aria-label="Practice code"]'),code);assert.equal(code.selectionDirection,'backward');
 // A standalone reader retains native page-scrolling semantics.
 solo=true;await render();await click(summary());await click(document.querySelector('.discussion-outline>summary'));await click([...document.querySelectorAll('.discussion-outline button')][0]);assert.equal(nativeScrolls.at(-1).options.block,'start');assert.equal(windowScrolls.length,0);
 source='## Replacement body\n\nUpdated reader content';await render();assert(!expanded());assert(!document.querySelector('.discussion-reading-body'));await click(summary());assert(document.querySelector('.discussion-reading-body').textContent.includes('Updated reader content'));
 await act(async()=>root.unmount());root=undefined;assert(observers.every(observer=>observer.disconnected));
 console.log('PASS in-place solution reading: native title disclosure/full Markdown/outline, retained practice URL/code/caret/both-scroll/card actions, pane-only jumps/section tracking, guarded collapse/Escape focus, IME/editable/route isolation, locale/body/account continuity, canonical standalone links, resize/scroll cleanup and standalone page fallback');
}finally{if(priorResizeDescriptor)Object.defineProperty(globalThis,'ResizeObserver',priorResizeDescriptor);else delete globalThis.ResizeObserver;if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
