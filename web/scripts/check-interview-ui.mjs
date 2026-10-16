import assert from 'node:assert/strict';
import React,{act,useState} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/'});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__interviewFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_API_ORIGIN':JSON.stringify('')},plugins:[{
 name:'interview-account-fixture',enforce:'pre',
 resolveId(source){if(source==='./social')return '\0interview-account-fixture';},
 load(id){if(id==='\0interview-account-fixture')return `import React from 'react';
 export const useAccount=()=>globalThis.__interviewFixture.account;
 export const socialApi=(...args)=>globalThis.__interviewFixture.request(...args);
 export const errorMessage=e=>e.message;export class SocialError extends Error{}
 export const PersonLink=({user})=>React.createElement('span',null,user.display_name);
 export const DateLabel=({value})=>React.createElement('time',null,value);
 export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);`;}
}]});
let root;
try {
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Interviews}=await server.ssrLoadModule('/src/Interviews.tsx');
 const {default:Composer}=await server.ssrLoadModule('/src/InterviewComposer.tsx');
 const {default:Fields}=await server.ssrLoadModule('/src/InterviewFields.tsx');
 const {default:CompanyTags}=await server.ssrLoadModule('/src/ProblemCompanyTags.tsx');
 const {default:PostEditor}=await server.ssrLoadModule('/src/PostEditor.tsx');
 const {libraryReturnHref}=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const drafts=await server.ssrLoadModule('/src/interviewDrafts.ts'),publications=await server.ssrLoadModule('/src/publicationRequests.ts');
 const pending=[],requests=[];
 globalThis.__interviewFixture={account:{user:null},request(path,body,signal){return new Promise((resolve,reject)=>{const item={path,body,signal,resolve,reject};pending.push(item);requests.push(item);});}};
 const user=id=>({id,username:'researcher',display_name:'Researcher',avatar_color:'#48766b'});
 const account=id=>{globalThis.__interviewFixture.account={user:id?user(id):null};};
 const modules=[{id:'architecture/attention',number:1,title:'Attention',translations:{en:{title:'Attention'}}},{id:'finetuning/lora',number:2,title:'LoRA'}];
 const details={company:'amazon',role:'ml',stage:'technical',outcome:'unknown',interview_month:'',location:'',questions:['architecture/attention']};
 const report={id:'a'.repeat(32),title:'A real interview reflection',body:'Our approach and reflection.',category:'career',interview:details,author:user('ui-alice'),created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',counts:{likes:2,replies:1,reposts:0},viewer:{liked:false,bookmarked:false,can_edit:true},edit_version:'d'.repeat(64)};
 const empty={reports:0,contributors:0,days:'365',basis:'published-community-reports',generated_at:'2026-10-15',companies:[],questions:[]};
 const summary={...empty,reports:3,contributors:2,companies:[{company:'amazon',reports:3,contributors:2}],questions:[{problem_id:'architecture/attention',reports:3,contributors:2,companies:[{company:'amazon',reports:3,contributors:2,problem_id:'architecture/attention',latest:'2026-10-15'}]}]};
 const take=(pattern,method='GET')=>{const i=pending.findIndex(item=>item.path.includes(pattern)&&(item.body?'POST':'GET')===method);assert(i>=0,'Missing '+pattern+' in '+pending.map(p=>p.path));return pending.splice(i,1)[0];};
 const reply=async(item,value)=>act(async()=>item.resolve(value));
 const fail=async(item,message)=>act(async()=>item.reject(Error(message)));
 const button=text=>[...document.querySelectorAll('button')].find(el=>el.textContent.trim()===text);
 const click=async el=>{assert(el,'Missing button');await act(async()=>el.click());};
 const field=label=>[...document.querySelectorAll('label')].find(el=>[...el.childNodes].filter(node=>node.nodeType===3).map(node=>node.textContent).join('').trim()===label)?.querySelector('input,textarea,select');
 const edit=async(el,value)=>{assert(el,'Missing field');await act(async()=>{const proto=el.tagName==='TEXTAREA'?'HTMLTextAreaElement':el.tagName==='SELECT'?'HTMLSelectElement':'HTMLInputElement';Object.getOwnPropertyDescriptor(dom.window[proto].prototype,'value').set.call(el,value);el.dispatchEvent(new dom.window.Event(el.tagName==='SELECT'?'change':'input',{bubbles:true}));});};
 const render=async(component)=>{if(!root)root=createRoot(document.querySelector('#root'));await act(async()=>root.render(React.createElement(LocaleProvider,null,component)));};
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;pending.length=0;};
 const hub=async(route='interviews',props={})=>render(React.createElement(Interviews,{route,problems:modules,...props}));
 account();await hub();assert(document.querySelector('.interview-hero-stats').textContent.includes('—'),'Unknown counts are never shown as zero');
 await reply(take('interviews?'),{items:[],total:0,limit:20,offset:0});await reply(take('interviews/summary'),empty);
 assert(document.body.textContent.includes('Start with the first shared experience'));assert(document.querySelector('a[href*="account/continue/"]'));
 assert.equal(document.querySelectorAll('.interview-resource-panel a').length,3);assert.equal(document.querySelectorAll('.interview-path-grid article').length,11);assert(document.getElementById('interview-path-transformer').textContent.includes('Transformer core & efficiency'));assert(document.getElementById('interview-path-few-step-generation').textContent.includes('Few-step generation'));assert(document.getElementById('interview-path-world-models').textContent.includes('Latent world models & planning'));assert(document.getElementById('interview-path-representation-shape').textContent.includes('Representation alignment & distribution'));
 await hub('interviews?company=amazon&view=questions');await reply(take('summary'),summary);
 assert.equal(document.querySelectorAll('.interview-ranking-row').length,1);const practice=document.querySelector('.interview-ranking-row .button');assert.equal(libraryReturnHref(practice.getAttribute('href')),'#/interviews?company=amazon&view=questions&queue=1');
 await hub('interviews?company=amazon&view=questions&q=missing');assert(document.body.textContent.includes('No matching concepts'));assert(button('Clear search'));
 await hub('interviews?view=companies');await fail(take('summary'),'Statistics temporarily unavailable');assert(document.querySelector('[role="alert"]').textContent.includes('Statistics temporarily unavailable'));await click(button('Retry loading'));await reply(take('summary'),empty);
 assert.equal(document.querySelectorAll('.interview-company-card').length,18);assert(!document.querySelector('.interview-company-grid').textContent.includes('reports'),'No fabricated company counts');
 await hub('interviews?view=companies&q=no-such-company');assert(document.body.textContent.includes('No matching companies'));
 await hub('interviews?company=does-not-exist');assert(document.body.textContent.includes('Invalid interview filters'));assert.equal(pending.length,0);
 await hub('interviews');const late=take('interviews?');take('summary');await hub('interviews?company=google');assert(late.signal.aborted);await reply(late,{items:[report],total:1,limit:20,offset:0});assert(!document.querySelector('.interview-report-title'),'Late filter response cannot repopulate a new view');
 await reply(take('interviews?'),{items:[{interview:details}],total:1,limit:20,offset:0});await reply(take('summary'),empty);assert(document.body.textContent.includes('The reports could not be read'));
 account('ui-alice');await hub('interviews?company=google');const oldOwner=take('interviews?');account('ui-bob');await hub('interviews?company=google');assert(oldOwner.signal.aborted);await reply(oldOwner,{items:[report],total:1,limit:20,offset:0});assert(!document.querySelector('.interview-report-title'));await reply(take('interviews?'),{items:[],total:0,limit:20,offset:0});
 await stop();account('ui-alice');let published=[];const compose=()=>render(React.createElement(Composer,{owner:'ui-alice',company:'amazon',problems:modules,onClose(){},onPublished:post=>published.push(post)}));
 await compose();await edit(field('Title'),'My interview 😀');await edit(field('Process, approach and reflection'),'## Reflection\n\nDiscussed attention and **testing**.');await edit(field('Company'),'google');
 await click(button('Preview'));assert(document.querySelector('.compose-preview strong'));await click(button('Write'));await stop();await compose();assert.equal(field('Title').value,'My interview 😀');assert.equal(field('Company').value,'google');assert.equal(pending.length,0,'Draft recovery never publishes automatically');
 const submit=async()=>act(async()=>document.querySelector('.interview-composer form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
 await submit();const first=take('community/posts','POST');assert.equal(first.body.expected_user_id,'ui-alice');assert.equal(first.body.interview.company,'google');assert.equal(first.body.category,'career');
 await fail(first,'Connection lost');assert(document.querySelector('[role="alert"]'));await submit();const retry=take('community/posts','POST');assert.equal(retry.body.request_id,first.body.request_id);
 await reply(retry,{...report,interview:{...details,company:'google'}});assert.equal(published.length,1);assert.equal(drafts.readInterviewDraft(dom.window.sessionStorage,'ui-alice'),undefined);
 await stop();await compose();await edit(field('Title'),'Owner-bound pending draft');await edit(field('Process, approach and reflection'),'Pending reflection');await submit();const exiting=take('community/posts','POST');await stop();account('ui-bob');await reply(exiting,report);assert.equal(published.length,1);assert(drafts.readInterviewDraft(dom.window.sessionStorage,'ui-alice'));assert(publications.readPublication(dom.window.sessionStorage,'ui-alice','interview','general'),'Unmounted acknowledgement retains safe retry receipt');
 account('ui-alice');await compose();assert.equal(field('Title').value,'Owner-bound pending draft');await submit();const resumed=take('community/posts','POST');assert.equal(resumed.body.request_id,exiting.body.request_id);await reply(resumed,report);await stop();

 account('ui-alice');await compose();await edit(field('Title'),'Earlier report');await edit(field('Process, approach and reflection'),'Earlier reflection');await submit();const earlier=take('community/posts','POST');await fail(earlier,'Lost confirmation');await edit(field('Title'),'Newer unsent reflection');await click(button('Retry previous publication'));const priorRetry=take('community/posts','POST');assert.equal(priorRetry.body.request_id,earlier.body.request_id);await reply(priorRetry,report);assert.equal(field('Title').value,'Newer unsent reflection');assert(document.querySelector('[role="status"] a'),'Confirmed older report is discoverable without leaving the newer draft');assert(!document.querySelector('[role="alert"]'),'Success is not an error banner');await stop();drafts.discardInterviewDraft(dom.window.sessionStorage,'ui-alice');
 await compose();await edit(field('Title'),'Cleanup retry');await edit(field('Process, approach and reflection'),'Stored reflection');const remove=dom.window.Storage.prototype.removeItem;dom.window.Storage.prototype.removeItem=function(key){if(key.startsWith('paperleet.interview.draft.'))throw Error('Cleanup blocked');return remove.call(this,key);};
 let cleanupRequest;
 try{await submit();cleanupRequest=take('community/posts','POST');await reply(cleanupRequest,report);assert(publications.readPublication(dom.window.sessionStorage,'ui-alice','interview','general'),'Failed cleanup must retain its confirmed request ID');assert(document.body.textContent.includes('draft could not be cleared'));}finally{dom.window.Storage.prototype.removeItem=remove;}
 await submit();const cleanupRetry=take('community/posts','POST');assert.equal(cleanupRetry.body.request_id,cleanupRequest.body.request_id);await reply(cleanupRetry,report);await stop();
 function LimitFixture(){const [value,setValue]=useState({...details,questions:Array.from({length:12},(_,i)=>'architecture/q'+i)});return React.createElement(Fields,{value,onChange:setValue,problems:[...modules,...Array.from({length:12},(_,i)=>({id:'architecture/q'+i,title:'Question '+i,number:100+i}))]});}
 await render(React.createElement(LimitFixture));assert([...document.querySelectorAll('.interview-module-options button')].some(el=>el.disabled),'Thirteenth concept is disabled');await stop();
 await render(React.createElement(CompanyTags,{problemId:'architecture/attention'}));await reply(take('summary'),summary);assert(document.querySelector('.problem-company-evidence').textContent.includes('Amazon'));assert(document.querySelector('.problem-company-evidence a').getAttribute('href').includes('concept=architecture%2Fattention'));await stop();
 const originalFetch=globalThis.fetch;let catalogCalls=0;globalThis.fetch=async()=>{catalogCalls++;return Response.json({problems:modules});};
 try {
  let saved;await render(React.createElement(PostEditor,{post:report,owner:'ui-alice',disabled:false,onSaved:post=>saved=post,onClose(){},onRefresh:async()=>true}));
  await act(async()=>{});await edit(field('Company'),'microsoft');await click(button('Save changes'));const mutation=take('/edit','POST');assert.equal(mutation.body.interview.company,'microsoft');assert.equal(mutation.body.expected_version,report.edit_version);await reply(mutation,{...report,interview:{...details,company:'microsoft'}});assert.equal(saved.interview.company,'microsoft');await stop();
  const edits=await server.ssrLoadModule('/src/communityEdits.ts');edits.writePostEdit(dom.window.sessionStorage,'ui-alice',report.id,{...edits.newPostEdit(report),interview:{...details,company:'google'}});
  const newer={...report,edit_version:'e'.repeat(64),interview:{...details,company:'microsoft'}};await render(React.createElement(PostEditor,{post:newer,owner:'ui-alice',disabled:false,onSaved:post=>saved=post,onClose(){},onRefresh:async()=>true}));assert(button('Save changes').disabled);assert.equal(field('Company').value,'google');assert(document.querySelector('.publication-recovery').textContent.includes('Microsoft'),'Conflict review includes latest structured metadata');await click(button('Reviewed; continue from this version'));await click(button('Save changes'));const conflictSave=take('/edit','POST');assert.equal(conflictSave.body.expected_version,newer.edit_version);assert.equal(conflictSave.body.interview.company,'google');await reply(conflictSave,{...newer,interview:{...details,company:'google'}});await stop();
  const plain={...report,interview:undefined,category:'questions'};await render(React.createElement(PostEditor,{post:plain,owner:'ui-alice',disabled:false,onSaved(){},onClose(){},onRefresh:async()=>true}));assert.equal(catalogCalls,1,'Ordinary post edits never fetch the interview catalog');await stop();
 } finally {globalThis.fetch=originalFetch;}
 console.log('PASS interview UI: honest empty/loading/error states, official resources and curated paths, ranked evidence, filter/company search, contextual practice links, malformed payloads, stale owner/filter aborts, draft restoration/preview, exact idempotent retries, unmounted acknowledgements, concept limit, problem company tags and versioned metadata edits');
} finally {
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
 for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
