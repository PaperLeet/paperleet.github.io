import assert from 'node:assert/strict';
import React,{act} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/community/'+ 'a'.repeat(32)+'?from=interviews%3Fcompany%3Dgoogle',pretendToBeVisual:true});
let wide=false,viewport=0,scrolls=[];
dom.window.matchMedia=()=>({matches:wide});
dom.window.HTMLElement.prototype.scrollIntoView=function(options){scrolls.push({id:this.id,options});};
dom.window.HTMLElement.prototype.getBoundingClientRect=function(){const headings=[...document.querySelectorAll('.discussion-section-heading')];return {top:200+headings.indexOf(this)*400-viewport};};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__outlineFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'outline-account-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0outline-account';},load(id){if(id==='\0outline-account')return `import React from 'react';export const useAccount=()=>globalThis.__outlineFixture.account;export const socialApi=(...args)=>globalThis.__outlineFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=()=>null;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;export const profileHref=user=>'#/profile/'+user.username;`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {markdownOutline,headingAnchor}=await server.ssrLoadModule('/src/markdownOutline.ts');
 const {default:Markdown}=await server.ssrLoadModule('/src/CommunityMarkdown.tsx'),{default:Reading}=await server.ssrLoadModule('/src/DiscussionReading.tsx');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const postId='a'.repeat(32),scope='reading-'+postId;
 const body='# 中文 😀 `code`\n\nIntro.\n\n## Repeated\n\n```python\n# This is code, not a section\n```\n\nRepeated\n--------\n\n### [Paper](https://tracker.invalid) and $\\alpha$\n\n> #### Quoted section\n\n## ![diagram](https://tracker.invalid/pixel)\n';
 const sections=markdownOutline(body,scope);assert.equal(sections.length,6);assert.equal(sections[0].label,'中文 😀 code');assert.equal(sections[3].label,'Paper and \\alpha');assert.equal(new Set(sections.map(row=>row.id)).size,6);
 const html=renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(Markdown,{body,headingScope:scope})));
 const markup=new JSDOM(html);const rendered=[...markup.window.document.querySelectorAll('.discussion-section-heading')];assert.deepEqual(rendered.map(node=>node.id),sections.map(row=>row.id),'Outline IDs match real rendered ATX, Setext, nested, GFM, Unicode and math headings');assert(!html.includes('<img'));markup.window.close();
 assert.equal(headingAnchor('window',0),undefined);assert.equal(headingAnchor(scope,-1),undefined);assert.equal(headingAnchor(scope,NaN),undefined);assert.deepEqual(markdownOutline('No headings',scope),[]);
 const crlf='## First\r\n\r\nText\r\n\r\n## Second\r\n';assert.equal(markdownOutline(crlf,scope)[1].id,scope+'-'+crlf.indexOf('## Second'));
 const render=async(Component,props={})=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Component,props)));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const click=async element=>{assert(element);await act(async()=>element.click());};
 const toggle=async()=>act(async()=>{const details=document.querySelector('.discussion-outline');details.open=!details.open;details.dispatchEvent(new dom.window.Event('toggle'));});
 const tick=()=>act(async()=>new Promise(resolve=>setTimeout(resolve,30)));
 let props={body,postId};
 const beforeHash=location.hash;
 await render(Reading,props);assert(!document.querySelector('.discussion-outline nav'),'Narrow screens start collapsed without rendering duplicate navigation rows');await toggle();assert.equal(document.querySelectorAll('.discussion-outline nav button').length,6);
 const target=()=>document.getElementById(sections[2].id);const sectionButton=()=>document.querySelectorAll('.discussion-outline nav button')[2];
 const button=sectionButton();button.focus();await click(button);assert(!document.querySelector('.discussion-outline').open);assert.equal(document.activeElement,target());assert.equal(scrolls.at(-1).id,sections[2].id);assert.equal(location.hash,beforeHash,'Local section navigation must not change the discussion or return route');
 await toggle();const newest=sectionButton();newest.focus();scrolls=[];await act(async()=>{newest.click();document.querySelector('.discussion-outline > summary').focus();});assert.equal(scrolls.length,0,'A queued jump cannot steal newer external focus');
 await toggle();const moving=sectionButton();moving.focus();scrolls=[];await act(async()=>{moving.click();location.hash='#/community';});assert.equal(scrolls.length,0,'A route change cancels a queued section jump');location.hash=beforeHash;
 await toggle();const stale=sectionButton();stale.focus();scrolls=[];await act(async()=>{stale.click();root.render(React.createElement(LocaleProvider,null,React.createElement(Reading,{body:'## Changed\n\n## New section',postId})));});assert.equal(scrolls.length,0,'A newer article revision cancels old navigation intent');await stop();
 wide=true;viewport=0;scrolls=[];await render(Reading,props);assert(document.querySelector('.discussion-outline').open);assert.equal(document.querySelector('[aria-current="location"]').textContent,sections[0].label);
 viewport=950;await act(async()=>window.dispatchEvent(new dom.window.Event('scroll')));await tick();assert.equal(document.querySelector('[aria-current="location"]'),document.querySelectorAll('.discussion-outline nav button')[2],'Duplicate titles are tracked by their distinct anchors');
 sectionButton().focus();await click(sectionButton());assert(document.querySelector('.discussion-outline').open);assert.equal(document.activeElement,target());
 await render(Reading,{body:'Only ordinary text.',postId});assert(!document.querySelector('.discussion-outline'));assert(document.querySelector('.community-prose').textContent.includes('ordinary text'));await stop();
 localStorage.setItem('paperleet.locale','zh');await render(Reading,props);assert(document.body.textContent.includes('文章目录'));assert.equal(document.querySelectorAll('.discussion-outline nav button').length,6);await stop();localStorage.setItem('paperleet.locale','en');wide=false;
 // Real discussion integration: section navigation never mutates posts or private reply drafts.
 const user={id:'alice',username:'alice',display_name:'Alice',avatar_color:'#48766b'};
 const post={id:postId,title:'A real report',body,category:'career',problem:null,problem_id:null,created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',author:user,counts:{likes:0,reposts:0,replies:0},viewer:{},replies:[]};
 let mutations=0;globalThis.__outlineFixture={account:{user,loading:false,capabilities:{},refresh:async()=>{}},request:async(path,payload)=>{assert.equal(payload,undefined);if(payload)mutations++;return post;}};
 const {default:Community}=await server.ssrLoadModule('/src/Community.tsx'),{writeReplyDraft,readReplyDraft}=await server.ssrLoadModule('/src/replyDrafts.ts');
 const draft={body:'Private unpublished reflection 😀',parentId:undefined};assert(writeReplyDraft(sessionStorage,user.id,postId,draft));const saved=readReplyDraft(sessionStorage,user.id,postId);
 await render(Community,{postId,route:'community/'+postId});assert(document.querySelector('.discussion-outline'));await toggle();await click(sectionButton());assert.equal(document.activeElement,target());assert.deepEqual(readReplyDraft(sessionStorage,user.id,postId),saved);assert.equal(mutations,0);assert.equal(location.hash,beforeHash);await stop();
 const failing=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'outline-failure-fixture',enforce:'pre',resolveId(source,importer){if(source==='./postExcerpt'&&importer?.endsWith('/markdownOutline.ts'))return '\\0outline-failure';},load(id){if(id==='\\0outline-failure')return 'export const parsePostMarkdown=()=>{throw new Error("Outline parser unavailable")};';}}]});
 try{const {default:FailedReading}=await failing.ssrLoadModule('/src/DiscussionReading.tsx');const failed=renderToStaticMarkup(React.createElement(FailedReading,{body:'## Still readable\n\nActual post text.\n\n## Another section',postId}));assert(!failed.includes('discussion-outline'));assert(failed.includes('Still readable')&&failed.includes('Actual post text.'));}finally{await failing.close();}
 console.log('PASS discussion reading: parser/renderer anchor parity, duplicate/Unicode/Setext/nested/code/math headings, mobile collapse, desktop current section, scoped revision-safe focus, route preservation, bilingual controls and unchanged private reply drafts');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
 for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,{...descriptor});else delete globalThis[key];}
}
