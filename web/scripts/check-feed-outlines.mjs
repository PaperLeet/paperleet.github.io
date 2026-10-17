import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/community',pretendToBeVisual:true});
const observers=[],nativeScrolls=[],paneScrolls=[];let pageTop=0;
dom.window.matchMedia=()=>({matches:true});
dom.window.scrollTo=options=>{pageTop=options.top??pageTop;Object.defineProperty(dom.window,'scrollY',{value:pageTop,writable:true,configurable:true});};
dom.window.HTMLElement.prototype.scrollIntoView=function(options){nativeScrolls.push({node:this,options});};
dom.window.HTMLElement.prototype.scrollTo=function(options){paneScrolls.push({node:this,options});if(options.top!==undefined)this.scrollTop=options.top;this.dispatchEvent(new dom.window.Event('scroll'));};
dom.window.HTMLElement.prototype.getBoundingClientRect=function(){
 if(this.matches('.description-scroll'))return {top:100,bottom:662,height:562,left:0,right:600,width:600};
 const pane=this.closest('.description-scroll'),reader=this.closest('.discussion-reading');
 const headings=reader?[...reader.querySelectorAll('.discussion-section-heading')]:[],index=headings.indexOf(this);
 const top=pane?100+(index>=0?500+index*400:140)-pane.scrollTop:(index>=0?200+index*400:140)-pageTop;
 return {top,bottom:top+44,height:44,left:0,right:580,width:580};
};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__feedOutlineFixture:undefined,ResizeObserver:class{constructor(callback){this.callback=callback;this.targets=[];this.disconnected=false;observers.push(this);}observe(node){this.targets.push(node);}disconnect(){this.disconnected=true;}}};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
 name:'feed-outline-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0feed-outline-social';},load(id){if(id==='\0feed-outline-social')return `import React from 'react';export const useAccount=()=>globalThis.__feedOutlineFixture.account;export const socialApi=(...args)=>globalThis.__feedOutlineFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=()=>null;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;export const profileHref=user=>'#/profile/'+user.username;`;}
}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Community}=await server.ssrLoadModule('/src/Community.tsx');
 const {default:Card}=await server.ssrLoadModule('/src/InterviewReportCard.tsx');
 const {InlinePostBody,default:InlineReply}=await server.ssrLoadModule('/src/InlineCommentBody.tsx');
 const {default:Reading}=await server.ssrLoadModule('/src/DiscussionReading.tsx');
 const {headingAnchor}=await server.ssrLoadModule('/src/markdownOutline.ts');
 const {interviewHref,emptyInterviewFilters}=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const id='a'.repeat(32),user={id:'alice',username:'alice',display_name:'Alice',can_publish_community:true,affiliation:{kind:'institution',domain:'test.edu',name:'测试大学',name_en:'Test University'}};
 const body='## Approach\n\nClarify the dimensions and preserve $x^2$.\n\n## Implementation\n\n```python\n# Not a section\ndef solve(x):\n    return x * 2\n```\n\n## Approach\n\nReview the boundary cases.';
 let currentBody=body,postId=id,surface='experience',inPane=true,mutations=0;const requests=[];
 const post=()=>({id:postId,title:'A complete reflection',body:currentBody,category:'career',solution:surface==='community',problem_id:surface==='community'?'architecture/attention':null,problem:surface==='community'?{id:'architecture/attention',number:1,title:'注意力',title_en:'Attention'}:null,author:user,created_at:'2026-10-16T00:00:00Z',updated_at:'2026-10-16T00:00:00Z',counts:{likes:2,replies:1,reposts:0},viewer:{liked:false,reposted:false},interview:surface==='experience'?{company:'google',role:'llm',stage:'technical',outcome:'unknown',interview_month:'',location:'',questions:[]}:undefined});
 globalThis.__feedOutlineFixture={account:{user,loading:false,capabilities:{community_solutions:true},refresh:async()=>{}},request:async(path,payload)=>{
  if(payload){mutations++;throw Error('Reading must not publish');}requests.push(path);
  if(path.startsWith('community/posts?'))return {items:[post()],total:1,solutions:surface==='community'};
  assert.equal(path,'community/posts/'+postId);
  return {...post(),replies:[{id:'b'.repeat(32),post_id:postId,parent_id:null,body:'A useful response.',author:user,created_at:'2026-10-16T00:01:00Z'}]};
 }};
 function Locale(){const {locale,setLocale}=useI18n();return React.createElement('button',{id:'locale',type:'button',onClick:()=>setLocale(locale==='en'?'zh':'en')},'Language');}
 const filters={...emptyInterviewFilters(),company:'google',role:'llm',stage:'technical',saved:true,q:'reflection',page:2};
 const render=async()=>act(async()=>{
  root??=createRoot(document.querySelector('#root'));
  const main=surface==='experience'?React.createElement(Card,{post:post(),owner:user.id,canSave:false,returnTo:interviewHref(filters),savedView:true,onReload(){}}):React.createElement(Community,{route:'community?solutions=1&q=attention'});
  root.render(React.createElement(LocaleProvider,null,React.createElement(Locale),React.createElement('input',{id:'newer-focus',defaultValue:'Keep this input'}),React.createElement('section',{key:surface,className:inPane?'description-scroll':'feed-surface'},main),React.createElement('aside',{id:'duplicate'},React.createElement(InlinePostBody,{postId,body:currentBody})),React.createElement('aside',{id:'full-article'},React.createElement(Reading,{postId,body:currentBody}))));
 });
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const click=async node=>{assert(node,'Missing reader control');await act(async()=>{node.focus();node.click();await new Promise(resolve=>setTimeout(resolve,5));});};
 const frame=()=>act(async()=>new Promise(resolve=>window.requestAnimationFrame(resolve)));
 const card=()=>document.querySelector(surface==='experience'?'.interview-report-card':'.community-post-card');
 const inline=()=>card().querySelector('.inline-post-body');
 const toggle=()=>inline().querySelector('.inline-comment-toggle');
 const headings=reader=>[...reader.querySelectorAll('.discussion-section-heading')];
 const state=()=>JSON.stringify({local:{...localStorage},session:{...sessionStorage}});
 const edit=async(field,value)=>act(async()=>{Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype,'value').set.call(field,value);field.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
 assert.equal(headingAnchor('reading-'+id+'-instance-003a00720030003a',12),'reading-'+id+'-instance-003a00720030003a-12');
 for(const invalid of ['reading-'+id+'-instance-x','reading-'+id+'-instance-','reading-'+id+'-instance-abc','reading-'+id+'-instance-'+ 'a'.repeat(260),'reading-'+id+'-instance-003a" onclick="evil','reading-'+id+'-other-003a'])assert.equal(headingAnchor(invalid,0),undefined);
 for(surface of ['experience','community']){
  inPane=surface==='experience';location.hash=surface==='experience'?interviewHref(filters):'#/community?solutions=1&q=attention';await render();
  const pane=document.querySelector('.description-scroll');if(pane){Object.defineProperty(pane,'clientHeight',{value:560});Object.defineProperty(pane,'clientTop',{value:2});pane.scrollTop=72;pane.scrollLeft=19;}
  await click(card().querySelector('.card-actions>button[aria-expanded]'));await click(card().querySelector('.card-comment-compose-prompt'));
  const field=card().querySelector('.card-reply-form textarea'),actions=card().querySelector('.community-card-interaction');await edit(field,'Unpublished reflection 😀');field.focus();field.setSelectionRange(2,11,'backward');field.scrollTop=27;field.scrollLeft=13;
  const reads=requests.length,stored=state(),href=location.href,title=card().querySelector(surface==='experience'?'.interview-report-title':'.post-title');
  await click(toggle());const reader=inline().querySelector('.discussion-reading');assert(reader.classList.contains('is-compact'));assert(reader.classList.contains('has-outline'));assert(!reader.querySelector('.discussion-outline').open,'Feed outlines start folded even on a wide viewport');assert(reader.querySelector('pre')&&reader.querySelector('.katex'));
  await click(reader.querySelector('.discussion-outline>summary'));const nav=[...reader.querySelectorAll('.discussion-outline nav button')];assert.deepEqual(nav.map(node=>node.textContent),['Approach','Implementation','Approach']);const observer=observers.at(-1);
  if(pane)assert(observer.targets.includes(pane));
  nativeScrolls.length=0;paneScrolls.length=0;await click(nav[1]);assert.equal(document.activeElement,headings(reader)[1]);assert(!reader.querySelector('.discussion-outline').open);assert.equal(toggle().getAttribute('aria-expanded'),'true');assert(observer.disconnected);
  if(pane){assert.equal(pane.scrollTop,874);assert.equal(pane.scrollLeft,19);assert.equal(nativeScrolls.length,0);}else{assert.equal(nativeScrolls.at(-1).node,headings(reader)[1]);assert.equal(nativeScrolls.at(-1).options.block,'start');assert.equal(paneScrolls.length,0);}
  // Reopening follows the actual current section. Duplicate post instances cannot steal the target.
  await click(reader.querySelector('.discussion-outline>summary'));if(pane){pane.scrollTop=1274;pane.dispatchEvent(new dom.window.Event('scroll'));}else{pageTop=950;window.dispatchEvent(new dom.window.Event('scroll'));}await frame();assert.equal(reader.querySelector('[aria-current="location"]').textContent,'Approach');assert.equal([...reader.querySelectorAll('.discussion-outline button')].indexOf(reader.querySelector('[aria-current="location"]')),2);
  const duplicate=document.querySelector('#duplicate');await click(duplicate.querySelector('.inline-comment-toggle'));await click(duplicate.querySelector('.discussion-outline>summary'));const duplicateReader=duplicate.querySelector('.discussion-reading');await click(duplicateReader.querySelectorAll('.discussion-outline button')[1]);assert.equal(document.activeElement,headings(duplicateReader)[1]);assert(reader.querySelector('.discussion-outline').open);assert(!duplicateReader.querySelector('.discussion-outline').open);
  const ids=[...document.querySelectorAll('.discussion-section-heading')].map(node=>node.id);assert.equal(new Set(ids).size,ids.length,'Two feed instances and a standalone article with the same post use distinct heading IDs');assert(headings(reader).every(node=>node.id.includes('-instance-')));assert(headings(document.querySelector('#full-article'))[0].id.startsWith('reading-'+id+'-'));
  assert.equal(card().querySelector('.card-reply-form textarea'),field);assert.equal(field.value,'Unpublished reflection 😀');assert.equal(field.selectionStart,2);assert.equal(field.selectionEnd,11);assert.equal(field.selectionDirection,'backward');assert.equal(field.scrollTop,27);assert.equal(field.scrollLeft,13);assert.equal(card().querySelector('.community-card-interaction'),actions);assert.equal(card().querySelector(surface==='experience'?'.interview-report-title':'.post-title'),title);assert.equal(requests.length,reads);assert.equal(mutations,0);assert.equal(state(),stored);assert.equal(location.href,href);
  await click(document.querySelector('#locale'));assert.equal(toggle().getAttribute('aria-expanded'),'true');assert.equal(reader.querySelector('.discussion-outline strong').textContent,'文章目录');assert.equal(headings(reader)[1].id,ids[1]);await click(document.querySelector('#locale'));assert.equal(reader.querySelector('.discussion-outline strong').textContent,'On this page');
  // Compact navigation still yields to a newer focus, route or body revision.
  let jump=reader.querySelectorAll('.discussion-outline button')[1];nativeScrolls.length=0;paneScrolls.length=0;await act(async()=>{jump.focus();jump.click();document.querySelector('#newer-focus').focus();});assert.equal(document.activeElement,document.querySelector('#newer-focus'));assert.equal(nativeScrolls.length+paneScrolls.length,0);
  await click(reader.querySelector('.discussion-outline>summary'));jump=reader.querySelectorAll('.discussion-outline button')[1];await act(async()=>{jump.focus();jump.click();location.hash='#/different-reading-intent';});assert.equal(nativeScrolls.length+paneScrolls.length,0);location.href=href;
  await click(reader.querySelector('.discussion-outline>summary'));const currentSection=reader.querySelector('.discussion-outline button');await act(async()=>{currentSection.focus();currentSection.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true}));});assert.equal(document.activeElement,reader.querySelector('.discussion-outline>summary'));assert(!reader.querySelector('.discussion-outline').open);assert(inline().querySelector('.discussion-reading'),'First Escape closes only the local directory');await act(async()=>document.activeElement.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true})));assert.equal(document.activeElement,toggle());assert(!inline().querySelector('.discussion-reading'));assert.equal(card().querySelector('.card-reply-form textarea'),field);assert.equal(duplicate.querySelector('.inline-comment-toggle').getAttribute('aria-expanded'),'true','Local Escape leaves a sibling reader open');
  if(surface==='experience'){
   await click(toggle());currentBody='## Updated\n\nNew body.\n\n## Follow-up\n\nA revision.';await render();assert.equal(toggle().getAttribute('aria-expanded'),'false');assert(!inline().querySelector('.discussion-reading'));assert.equal(card().querySelector('.card-reply-form textarea'),field);
   currentBody=body;await render();assert.equal(toggle().getAttribute('aria-expanded'),'false','A→B→A revisions cannot reopen an old full reader');assert.equal(card().querySelector('.card-reply-form textarea'),field);assert.equal(field.value,'Unpublished reflection 😀');
  }
  await stop();pageTop=0;sessionStorage.clear();
 }
 // Reply readers keep their compact prose, and their old opened revision also retires.
 let replyBody=body;const renderReply=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(InlineReply,{body:replyBody})));});
 await renderReply();await click(document.querySelector('.inline-comment-toggle'));assert(!document.querySelector('.discussion-outline'));replyBody='Changed **reply**';await renderReply();replyBody=body;await renderReply();assert.equal(document.querySelector('.inline-comment-toggle').getAttribute('aria-expanded'),'false');await stop();
 const renderIdentity=async identifier=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(InlinePostBody,{postId:identifier,body})));});
 await renderIdentity(id);await click(document.querySelector('.inline-comment-toggle'));await click(document.querySelector('.discussion-outline>summary'));const oldAnchors=headings(document.querySelector('.discussion-reading')).map(node=>node.id),oldObserver=observers.at(-1);await renderIdentity('c'.repeat(32));assert.equal(document.querySelector('.inline-comment-toggle').getAttribute('aria-expanded'),'false','Changing post identity retires the old reader even for the same body');assert(oldObserver.disconnected);assert(oldAnchors.every(anchor=>!document.getElementById(anchor)));await click(document.querySelector('.inline-comment-toggle'));assert(headings(document.querySelector('.discussion-reading')).every(node=>node.id.startsWith('reading-'+ 'c'.repeat(32)+'-instance-')));await stop();
 assert(observers.every(observer=>observer.disconnected),'All expanded outline observers retire on close/revision/unmount');
 console.log('PASS feed article outlines: real community solution and experience cards, folded compact layout, repeated-heading anchor parity, unique same-post instances beside standalone reading, page/pane-only jumps and current-section tracking, local collapse/newer focus/route/body guards, locale continuity, A-B-A revision retirement, unchanged comments/drafts/caret/both-scroll/actions/filters/storage and zero reading mutations or extra detail fetches');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const[key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
