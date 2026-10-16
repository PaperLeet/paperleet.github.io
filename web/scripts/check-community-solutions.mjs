import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/community',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:true});dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__solutionFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
 name:'solution-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0solution-fixture';},load(name){if(name==='\0solution-fixture')return `import React from 'react';export const useAccount=()=>globalThis.__solutionFixture.account;export const socialApi=(...args)=>globalThis.__solutionFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}
}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Community,ProblemSolutions}=await server.ssrLoadModule('/src/Community.tsx');
 const nav=await server.ssrLoadModule('/src/communityNavigation.ts');
 const drafts=await server.ssrLoadModule('/src/communityDrafts.ts');
 const pub=await server.ssrLoadModule('/src/publicationRequests.ts');
 const writing=await server.ssrLoadModule('/src/communityWriting.ts');
 const topic={id:'architecture/attention',number:1,title:'注意力',title_en:'Attention'};
 const other={id:'finetuning/lora',number:22,title:'LoRA'};
 const user={id:'alice',username:'alice',display_name:'Alice',can_publish_community:true,affiliation:{kind:'institution',domain:'test.edu',name:'测试大学',name_en:'Test University'}};
 const account={user,loading:false,capabilities:{community_solutions:true,community_feed_sorting:true,community_bookmarks:true,community_anonymous_posts:true},refresh:async()=>{}};
 const id='a'.repeat(32),post={id,title:'Masked attention explained',body:'A derivation with key implementation and boundary cases',solution:true,category:'implementation',problem_id:topic.id,problem:topic,author:user,counts:{likes:3,replies:0,reposts:0},viewer:{},created_at:'2026-10-16T00:00:00Z',updated_at:'2026-10-16T00:00:00Z'};
 const requests=[],mutations=[];let failPublish=false,answer='normal',release;
 globalThis.__solutionFixture={account,request:async(path,body,signal)=>{
  if(body){mutations.push({path,body});if(failPublish)throw Error('Connection lost');return {...post,title:body.title,body:body.body,problem_id:body.problem_id,anonymous:body.anonymous};}
  const params=new URLSearchParams(path.split('?')[1]);requests.push({path,params,signal});
  if(answer==='hold')return new Promise(resolve=>release=resolve);
  if(answer==='fail')throw Error('Solutions unavailable');
  return {items:answer==='empty'?[]:[answer==='wrong-module'?{...post,problem_id:other.id}:post],total:answer==='empty'?0:25,sort:params.get('sort')||'latest',...(answer==='no-ack'?{}:{solutions:true})};
 }};
 let embedded=true;
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(React.Fragment,null,React.createElement('textarea',{'aria-label':'Practice code',defaultValue:'def solve(): return 1'}),embedded?React.createElement(ProblemSolutions,{topic}):React.createElement(Community,{route:location.hash,topics:[topic,other]}))));});
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
 const button=text=>[...document.querySelectorAll('button')].find(node=>node.textContent===text);
 const change=async(node,value)=>act(async()=>{Object.getOwnPropertyDescriptor(node.tagName==='TEXTAREA'?dom.window.HTMLTextAreaElement.prototype:node.tagName==='SELECT'?dom.window.HTMLSelectElement.prototype:dom.window.HTMLInputElement.prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event(node.tagName==='SELECT'?'change':'input',{bubbles:true}));});
 const form=()=>document.querySelector('.community-composer form');
 const submit=async()=>act(async()=>form().dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
 // Route, sign-in and publication contracts do not expose unrelated query data.
 const href=nav.solutionsHref(topic.id);assert.equal(nav.readCommunityFeed(href).state.solutions,true);
 assert(nav.communityContinuation(nav.communityPostHref(id,href)).includes('solutions%3D1'));
 for(const route of ['community?solutions=yes','community?solutions=1&solutions=0'])assert.equal(nav.readCommunityFeed(route),undefined);
 const draft={title:'My solution',body:'Detailed reasoning',category:'implementation',problemId:other.id};
 const receipt=pub.preparePublication(sessionStorage,'alice','post','general:solutions',pub.postPublication(draft,undefined,true));assert.equal(receipt.receipt.payload.problem_id,other.id);
 assert.throws(()=>pub.preparePublication(sessionStorage,'alice','post','general',pub.postPublication(draft,undefined,true)));
 pub.clearPublication(sessionStorage,receipt.receipt);
 for(const en of [false,true]){const outline=writing.communityOutlineText('solution',en);assert(!writing.hasCommunityDraftContent(outline));assert(writing.hasCommunityDraftContent(outline+'My own reasoning'));}
 // Local controls keep the practice URL, code, selection and scroll intact.
 location.hash='#/22?from=collections%2Fllm-basics';const initialHash=location.hash;
 await render();assert.equal(requests.length,1);assert.equal(requests[0].params.get('problem_id'),topic.id);assert.equal(requests[0].params.get('solutions'),'true');assert(!document.querySelector('main'));assert(!document.querySelector('.community-sidebar'));
 assert(document.body.textContent.includes('Community solutions'));assert(document.querySelector('.post-card-meta').textContent.includes('Solution'));assert(document.querySelector('.community-card-interaction'));
 const code=document.querySelector('[aria-label="Practice code"]');code.focus();code.setSelectionRange(3,10,'backward');code.scrollTop=111;code.scrollLeft=25;
 await change(document.querySelector('.community-sort select'),'top');assert.equal(requests.at(-1).params.get('sort'),'top');assert.equal(location.hash,initialHash);
 await click(document.querySelector('[aria-label="Next page"]'));assert.equal(requests.at(-1).params.get('offset'),'20');
 const search=document.querySelector('input[type="search"]');await change(search,'mask gradient');await act(async()=>search.form.dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));assert.equal(requests.at(-1).params.get('offset'),'0');assert.equal(requests.at(-1).params.get('q'),'mask gradient');assert.equal(location.hash,initialHash);
 assert.equal(code,document.querySelector('[aria-label="Practice code"]'));assert.equal(code.value,'def solve(): return 1');assert.equal(code.selectionDirection,'backward');assert.equal(code.selectionStart,3);assert.equal(code.selectionEnd,10);assert.equal(code.scrollTop,111);assert.equal(code.scrollLeft,25);
 assert(document.querySelector('.solution-reading-card>details>summary .post-title'));
 assert(nav.communityPostReturn(document.querySelector('.solution-reading-page').getAttribute('href'),id).includes('solutions=1'));
 await click(button('Share a solution'));assert(document.querySelector('.community-composer'));assert.equal(document.querySelector('.community-composer select').disabled,true);assert(!form().textContent.includes('Choose a practice module'));
 const guide=document.querySelector('.community-writing-guide');await act(async()=>{guide.open=true;guide.dispatchEvent(new dom.window.Event('toggle'));});
 for(let attempt=0;attempt<50&&!document.querySelector('.community-writing-panel');attempt++)await act(async()=>new Promise(resolve=>setTimeout(resolve,20)));
 assert(document.querySelector('.community-writing-panel').textContent.includes('Time and space complexity'));await click(button('Insert outline'));assert(form().querySelector('textarea').value.includes('## Approach and derivation'));assert(document.querySelector('.form-footer button').disabled);await submit();assert.equal(mutations.length,0);
 await change(form().querySelector('input[type="text"],input:not([type])'),'My solution');await change(form().querySelector('textarea'),'Only this written text should be published');
 failPublish=true;await submit();assert.equal(mutations.length,1);assert.equal(mutations[0].body.solution,true);assert.equal(mutations[0].body.problem_id,topic.id);assert.equal(mutations[0].body.category,'implementation');assert.equal(mutations[0].body.body,'Only this written text should be published');assert(!JSON.stringify(mutations[0]).includes(code.value));
 assert(drafts.readPostDraft(sessionStorage,'alice',topic.id+':solutions'));assert(!drafts.readPostDraft(sessionStorage,'alice',topic.id));
 failPublish=false;await submit();assert.equal(mutations.length,2);assert.equal(mutations[0].body.request_id,mutations[1].body.request_id);assert(nav.communityPostReturn(location.hash,id).includes('solutions=1'));assert(!drafts.readPostDraft(sessionStorage,'alice',topic.id+':solutions'));
 // The general community can select a module without overwriting an ordinary draft.
 embedded=false;location.hash=nav.solutionsHref();drafts.writePostDraft(sessionStorage,'alice','general',{title:'Ordinary draft',body:'Keep this',category:'questions'});await render();await click(button('Share a solution'));
 const picker=form().querySelector('select');assert.equal(picker.required,true);await submit();assert.equal(mutations.length,2);
 await change(picker,other.id);await change(form().querySelector('input[type="text"],input:not([type])'),'LoRA reasoning');await change(form().querySelector('textarea'),'Factorization and rank constraints');
 await submit();assert.equal(mutations.at(-1).body.problem_id,other.id);assert.equal(drafts.readPostDraft(sessionStorage,'alice','general').body,'Keep this');
 // Old servers cannot silently return an unfiltered discussion feed.
 embedded=true;account.capabilities.community_solutions=false;const before=requests.length;await render();assert.equal(requests.length,before);assert(document.body.textContent.includes('Community solutions are unavailable'));
 account.loading=true;await render();assert.equal(requests.length,before);
 account.loading=false;account.capabilities.community_solutions=true;answer='no-ack';await render();assert(document.querySelector('[role="alert"]'));assert(!document.querySelector('.post-title'));
 answer='wrong-module';await click(button('Retry'));assert(document.querySelector('[role="alert"]'));assert(!document.querySelector('.post-title'));
 answer='hold';await click(button('Retry'));const old=requests.at(-1);account.user={...user,id:'bob',username:'bob'};answer='empty';await render();assert(old.signal.aborted);await act(async()=>release({items:[post],total:1,solutions:true}));assert(!document.querySelector('.post-title'));
 account.user=undefined;await render();assert(document.querySelector('.problem-solutions-heading a[href*="account"]'));assert(!document.querySelector('.community-composer'));
 console.log('PASS community solutions: explicit scoped discovery, embedded sorting/search/pagination with practice URL/code/caret/scroll retained, real composer publication/receipt retry, selected-module and draft isolation, type badges/card actions, capability/response/account guards and guest contribution return');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
