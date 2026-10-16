import assert from 'node:assert/strict';
import React,{act} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/community?category=questions',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:false});dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,requestAnimationFrame:dom.window.requestAnimationFrame.bind(dom.window),IS_REACT_ACT_ENVIRONMENT:true,__communityWriting:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');let loads=0;
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0')},plugins:[{name:'community-writing-fixture',enforce:'pre',resolveId(source,importer){if(source==='./social')return '\0community-writing-account';if(source==='./CommunityWritingGuide'&&importer?.endsWith('/Community.tsx'))return '\0community-writing-guide';},load(id){
 if(id==='\0community-writing-account')return `import React from 'react';export const useAccount=()=>globalThis.__communityWriting.account;export const socialApi=(...args)=>globalThis.__communityWriting.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;
 if(id==='\0community-writing-guide'){loads++;return `import React from 'react';import Guide from '/src/CommunityWritingGuide.tsx';export default function FixtureGuide(props){globalThis.__communityWriting.guide=props;if(globalThis.__communityWriting.fail)throw Error('Writing prompt fixture failure');return React.createElement(Guide,props);}`;}
}}]});
let root,caught=0;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Community}=await server.ssrLoadModule('/src/Community.tsx');
 const {default:Markdown}=await server.ssrLoadModule('/src/CommunityMarkdown.tsx');
 const model=await server.ssrLoadModule('/src/communityWriting.ts');
 const drafts=await server.ssrLoadModule('/src/communityDrafts.ts');
 const receipts=await server.ssrLoadModule('/src/publicationRequests.ts');
 assert.equal(loads,0,'Reading the feed must not load optional writing prompts');
 assert.deepEqual(Object.keys(model.communityOutlines).sort(),["career","implementation","lounge","papers","questions","solution"]);
 for(const id of Object.keys(model.communityOutlines))for(const en of [true,false]){
  const text=model.communityOutlineText(id,en);assert(!model.hasCommunityDraftContent(text));assert(model.hasCommunityDraftContent(text+'My actual observations.'));
  const result=model.insertCommunityOutline('',id,en);assert(result.ok);
  const html=renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(Markdown,{body:result.edit.after})));
  const view=new JSDOM(html);assert.equal(view.window.document.querySelectorAll('h2').length,4,'Inserted outlines are actual reader headings');view.window.close();
 }
 for(const text of ['## My own heading','---','```python\nx=1\n```','$$x^2$$'])assert(model.hasCommunityDraftContent(text));
 assert.equal(model.hasCommunityDraftContent('  \n'),false);
 assert.deepEqual(model.insertCommunityOutline('x'.repeat(20000),'questions',true),{ok:false,reason:'length'});
 for(const text of ['```python\nx=1','$$\nx^2','<!-- not closed'])assert.deepEqual(model.insertCommunityOutline(text,'questions',true),{ok:false,reason:'position'});
 const closed='```python\nx=1\n```';assert(model.insertCommunityOutline(closed,'questions',true).ok);
 for(const [text,position,prefix] of [['a😀b',2,'a😀'],['éx',1,'é'],['👩‍💻x',2,'👩‍💻']]){const result=model.insertCommunityOutline(text,'questions',true,position);assert(result.ok);assert(result.edit.after.startsWith(prefix+'\n\n## '));}
 let owner='writer-alice',topicId;
 const user=()=>({can_publish_community:true,id:owner,username:owner,display_name:owner,avatar_color:'#48766b',affiliation:{kind:'institution',domain:'research.edu',name:'测试研究院',name_en:'Research Institute'}});
 const account={user:user(),loading:false,capabilities:{community_anonymous_posts:true,community_papers:true},refresh:async()=>{}};
 const requests=[];
 globalThis.__communityWriting={account,request(path,body){if(!body)return Promise.resolve({items:[],total:0});return new Promise((resolve,reject)=>requests.push({path,body,resolve,reject}));}};
 function Locale(){const {setLocale}=useI18n();return React.createElement(React.Fragment,null,React.createElement('button',{className:'outside',onClick:()=>setLocale('en')},'English'),React.createElement('button',{onClick:()=>setLocale('zh')},'Chinese'));}
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'),{onCaughtError(error){assert.equal(error.message,'Writing prompt fixture failure');caught++;}});root.render(React.createElement(LocaleProvider,null,React.createElement(Locale),React.createElement(Community,{route:location.hash,topicId,topic:topicId?{id:topicId,title:'Attention'}:undefined})));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const click=async node=>{assert(node,'Missing control');await act(async()=>node.click());};
 const button=text=>[...document.querySelectorAll('button')].find(node=>node.textContent.trim()===text);
 const field=label=>[...document.querySelectorAll('.community-composer label')].find(node=>[...node.childNodes].filter(child=>child.nodeType===3).map(child=>child.textContent).join('').trim()===label)?.querySelector('input,select,textarea');
 const body=()=>document.querySelector('.community-composer textarea'),publish=()=>document.querySelector('.community-composer .form-footer > .button');
 const edit=async(node,value)=>{assert(node,'Missing field');await act(async()=>{const type=node.tagName==='SELECT'?'HTMLSelectElement':node.tagName==='TEXTAREA'?'HTMLTextAreaElement':'HTMLInputElement';Object.getOwnPropertyDescriptor(dom.window[type].prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event(node.tagName==='SELECT'?'change':'input',{bubbles:true}));});};
 const settle=async predicate=>{for(let attempt=0;attempt<100&&!predicate();attempt++)await act(async()=>new Promise(resolve=>setTimeout(resolve,20)));assert(predicate(),'Writing UI did not settle');};
 const open=async()=>{await act(async()=>{const panel=document.querySelector('.community-writing-guide');panel.open=true;panel.dispatchEvent(new dom.window.Event('toggle'));});await settle(()=>document.querySelector('.community-writing-panel')||document.querySelector('.community-writing-failure'));};
 const close=async()=>act(async()=>{const panel=document.querySelector('.community-writing-guide');panel.open=false;panel.dispatchEvent(new dom.window.Event('toggle'));});
 const compose=async()=>{await render();await click(document.querySelector('.community-compose-button'));};
 const submit=async()=>act(async()=>document.querySelector('.community-composer form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
 await compose();assert.equal(loads,0);await edit(field('Title'),'A real question');const before=drafts.readPostDraft(sessionStorage,owner,'general');await open();assert.equal(loads,1);assert.deepEqual(drafts.readPostDraft(sessionStorage,owner,'general'),before);assert.equal(requests.length,0);
 for(const language of ['en','zh']){
  await click(button(language==='en'?'English':'Chinese'));
  for(const category of ["papers","implementation","questions","career","lounge"]){
   await edit(field(language==='en'?'Category':'版块'),category);const items=[...document.querySelectorAll('.community-writing-panel li')];assert.equal(items.length,4);assert.deepEqual(items.map(node=>node.querySelector('b').textContent.replace(/^\d\. /,'')),model.communityOutlines[category].sections.map(section=>section[language]));assert(items.every(node=>node.querySelector('p').textContent.length>15));
  }
 }
 await click(button('English'));await edit(field('Category'),'questions');const original='My own notes 😀\nKeep the second paragraph.';await edit(body(),original);const originalField=body();body().setSelectionRange(0,8);body().scrollTop=27;const prompt=document.querySelector('.community-writing-panel li p').textContent;await click(button('Insert outline at cursor'));const inserted=body().value;assert(inserted.startsWith(original.slice(0,8))&&inserted.endsWith(original.slice(8)));assert(!inserted.includes(prompt));assert.equal(body(),originalField);assert.equal(document.activeElement,body());assert.equal(body().scrollTop,27,'Inserting an outline must not force the editor back to the top');
 await click(button('Insert outline at cursor'));assert.equal(body().value,inserted,'Repeat taps cannot duplicate an unchanged outline');await close();await open();assert(!button('Undo outline').disabled);await edit(field('Title'),'A refined title');await click(document.querySelector('.publishing-identity input'));await click(button('Undo outline'));assert.equal(body().value,original);assert.equal(body().selectionStart,0);assert.equal(body().selectionEnd,8);assert.equal(body().scrollTop,27);assert.equal(drafts.readPostDraft(sessionStorage,owner,'general').title,'A refined title');assert.equal(drafts.readPostDraft(sessionStorage,owner,'general').anonymous,true);
 await click(button('Insert outline at cursor'));await edit(body(),body().value+'\nLater writing');const later=body().value;assert(button('Undo outline').disabled);await click(button('Undo outline'));assert.equal(body().value,later);
 await click(button('Preview'));assert(body().closest('label').hidden);await click(button('Insert outline at cursor'));assert.equal(body(),originalField);assert(!body().closest('label').hidden);const englishBody=body().value;await click(button('Chinese'));assert.equal(body().value,englishBody);await click(button('English'));assert.equal(body().value,englishBody);
 await edit(body(),'```python\nx=1');body().setSelectionRange(body().value.length,body().value.length);await click(button('Insert outline at cursor'));assert.equal(body().value,'```python\nx=1');assert(document.body.textContent.includes('Move the cursor outside'));
 await edit(body(),'x'.repeat(20000));await click(button('Insert outline at cursor'));assert.equal(body().value.length,20000);assert(document.body.textContent.includes('near its length limit'));
 await edit(body(),'Actual notes');const newerFocus=document.querySelector('.outside');await act(async()=>{button('Insert outline at cursor').click();newerFocus.focus();});assert.equal(document.activeElement,newerFocus);
 const routeBefore=location.href;await edit(body(),'Route-specific notes');await act(async()=>{button('Insert outline at cursor').click();location.hash='#/interviews';});assert.notEqual(document.activeElement,body());location.href=routeBefore;
 const stale=globalThis.__communityWriting.guide,opener=document.querySelector('.community-writing-actions .button'),aliceDraft=drafts.readPostDraft(sessionStorage,owner,'general');owner='writer-bob';account.user=user();await compose();assert.equal(body().value,'');await act(async()=>stale.onInsert('papers',opener));assert.deepEqual(drafts.readPostDraft(sessionStorage,'writer-alice','general'),aliceDraft);assert.equal(body().value,'');
 await edit(field('Title'),'An empty outline');await open();await click(button('Insert outline'));assert(publish().disabled);assert(document.body.textContent.includes('Your outline is ready'));await submit();assert.equal(requests.length,0);assert.equal(receipts.readPublication(sessionStorage,owner,'post','general'),undefined);
 const blank=body().value;await stop();await compose();assert.equal(body().value,blank);assert(publish().disabled);assert(!document.querySelector('.community-writing-panel'));
 await edit(body(),blank+'My own question and a reproducible observation.');await submit();assert.equal(requests.length,1);const first=requests[0];assert.equal(first.body.expected_user_id,owner);assert.equal(first.path,'community/posts');await open();assert([...document.querySelectorAll('.community-writing-actions button')].every(node=>node.disabled));const busyBody=body().value;await act(async()=>globalThis.__communityWriting.guide.onInsert('career',document.querySelector('.community-writing-actions .button')));assert.equal(body().value,busyBody);
 await act(async()=>first.reject(Error('Lost confirmation')));await click(button('Insert outline at cursor'));const newer=body().value;await click(button('Retry previous publication'));assert.equal(requests.length,2);assert.equal(requests[1].body.request_id,first.body.request_id);assert.equal(requests[1].body.body,first.body.body);await act(async()=>requests[1].resolve({id:'a'.repeat(32),author:{id:owner}}));assert.equal(drafts.readPostDraft(sessionStorage,owner,'general').body,newer);
 await stop();location.hash='#/community?category=questions';owner='writer-legacy';account.user=user();const legacy={title:'Legacy pending post',body:model.communityOutlineText('questions',true),category:'questions'};drafts.writePostDraft(sessionStorage,owner,'general',legacy);const receipt=receipts.preparePublication(sessionStorage,owner,'post','general',receipts.postPublication(legacy));await compose();assert(!publish().disabled);await submit();assert.equal(requests.at(-1).body.request_id,receipt.receipt.requestId);await act(async()=>requests.at(-1).resolve({id:'b'.repeat(32),author:{id:owner}}));
 await stop();location.hash='#/community?category=questions';owner='writer-failure';account.user=user();await compose();await edit(body(),'Keep this exact draft');const saved=body(),savedBody=saved.value;globalThis.__communityWriting.fail=true;await open();assert(document.querySelector('.community-writing-failure'));assert(caught>0);assert.equal(body(),saved);assert.equal(body().value,savedBody);globalThis.__communityWriting.fail=false;await click(button('Retry writing prompts'));await settle(()=>document.querySelector('.community-writing-panel'));assert.equal(body(),saved);assert.equal(body().value,savedBody);
 await stop();topicId='architecture/attention';location.hash='#/community/problem/architecture/attention';await compose();assert.equal(body().value,'');assert.equal(drafts.readPostDraft(sessionStorage,owner,'general').body,savedBody);
 await open();const storageDescriptor=Object.getOwnPropertyDescriptor(dom.window.Storage.prototype,'setItem');try{Object.defineProperty(dom.window.Storage.prototype,'setItem',{...storageDescriptor,value(){throw Error('Storage fixture unavailable');}});await click(button('Insert outline'));assert(body().value.includes('## '));assert(document.body.textContent.includes('This tab cannot save your draft'));}finally{Object.defineProperty(dom.window.Storage.prototype,'setItem',storageDescriptor);}
 console.log('PASS community writing: six bilingual deferred guides, real Markdown/Unicode/block safety, explicit insertion, exact guarded undo/focus, repeat protection, mounted preview, blank-outline publication guard, account/topic drafts, captured retries, legacy receipts and local prompt failure recovery');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
