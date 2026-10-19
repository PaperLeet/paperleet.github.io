import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/interviews',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:false});dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__interviewSearchFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_API_ORIGIN':JSON.stringify('')},plugins:[{name:'interview-search-account',enforce:'pre',resolveId(source){if(source==='./social')return '\0interview-search-account';},load(name){if(name==='\0interview-search-account')return `import React from 'react';export const useAccount=()=>globalThis.__interviewSearchFixture.account;export const socialApi=(...args)=>globalThis.__interviewSearchFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Interviews}=await server.ssrLoadModule('/src/Interviews.tsx');
 const nav=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const drafts=await server.ssrLoadModule('/src/interviewDrafts.ts');
 const {signInHref,accountReturnHref}=await server.ssrLoadModule('/src/accountNavigation.ts');
 const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 const user={id:'alice',username:'alice',display_name:'Alice',can_publish_community:true};
 const account={user,loading:false,capabilities:{community_bookmarks:true}};
 const problems=[{id:'architecture/attention',number:1,title:'注意力',translations:{en:{title:'Attention'}},difficulty:'E',estimated_minutes:20,paper:{title:'Attention Is All You Need'},module:{goal:'Scale the dot product'}},{id:'finetuning/lora',number:2,title:'LoRA',difficulty:'M',estimated_minutes:30}];
 const saved={'architecture/attention':{code:'Private implementation',updated:'2026-10-16T00:00:00Z'}};
 const draft={title:'Private unfinished report',body:'Keep my own observations',interview:{company:'google',role:'ml',stage:'technical',outcome:'unknown',interview_month:'',location:'',questions:['architecture/attention']}};assert(drafts.writeInterviewDraft(sessionStorage,user.id,draft));const originalDraft=JSON.stringify({...sessionStorage}),originalSaved=JSON.stringify(saved);
 const requests=[],mutations=[];
 globalThis.__interviewSearchFixture={account,request:async(path,body,signal)=>{
  if(body){mutations.push({path,body});throw Error('Searching cannot publish, react or change learning progress');}
  const params=new URLSearchParams(path.split('?')[1]);requests.push({path,params,signal});
  return path.includes('/summary?')?{reports:0,contributors:0,days:params.get('days'),basis:'published-community-reports',generated_at:'2026-10-17',companies:[],questions:[]}:{items:[],total:0,limit:20,offset:Number(params.get('offset')),owner_id:account.user?.id};
 }};
 function Locale(){const {locale,setLocale}=useI18n();return React.createElement('button',{id:'locale',onClick:()=>setLocale(locale==='en'?'zh':'en')},'locale');}
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Locale),React.createElement('textarea',{'aria-label':'Practice code',defaultValue:'def solve(): return 42'}),React.createElement(Interviews,{route:location.hash,problems,saved,progressReady:true})));});
 const navigate=async filters=>{location.hash=nav.interviewHref(filters);await render();};
 const field=()=>document.querySelector('.interview-search input'),form=()=>document.querySelector('.interview-search');
 const edit=async value=>act(async()=>{Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype,'value').set.call(field(),value);field().dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
 const key=async(type,value,extra={})=>{const event=new dom.window.KeyboardEvent(type,{key:value,bubbles:true,cancelable:true,...extra});await act(async()=>field().dispatchEvent(event));return event;};
 const composition=async type=>act(async()=>field().dispatchEvent(new dom.window.CompositionEvent(type,{bubbles:true,data:'中'})));
 const submit=async()=>{await act(async()=>form().dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));await render();};
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});await render();};
 const state=()=>nav.readInterviewRoute(location.hash);
 const clear=()=>form().querySelector('button[type=button]');
 const baseline={...nav.emptyInterviewFilters(),company:'google',role:'ml',stage:'technical',days:'90',sort:'helpful',page:3,saved:true,concept:'architecture/attention',track:'llm'};
 await navigate(baseline);const original=field(),originalHref=location.href,calls=requests.length;
 assert.equal(original.type,'search');assert.equal(original.maxLength,400);assert.equal(original.getAttribute('enterkeyhint'),'search');assert.equal(form().getAttribute('role'),'search');assert.equal(original.getAttribute('aria-label'),'Search experiences');assert.equal(document.querySelectorAll('.interview-search').length,1);assert(!document.querySelector('.community-search'));
 assert(document.getElementById(original.getAttribute('aria-describedby')).textContent.includes('Enter to search; Escape to clear'));
 const code=document.querySelector('[aria-label="Practice code"]');code.setSelectionRange(3,9,'backward');code.scrollTop=91;code.scrollLeft=17;
 await edit('An unapplied query 😀');assert.equal(requests.length,calls);field().focus();let escaped=0;const escape=event=>{if(event.key==='Escape')escaped++;};window.addEventListener('keydown',escape);
 for(const extra of [{ctrlKey:true},{metaKey:true},{altKey:true},{shiftKey:true},{repeat:true}]){const event=await key('keydown','Escape',extra);assert(!event.defaultPrevented);assert.equal(field().value,'An unapplied query 😀');assert.equal(requests.length,calls);assert.equal(location.href,originalHref);}
 escaped=0;const cancel=await key('keydown','Escape');assert(cancel.defaultPrevented);assert.equal(escaped,0);assert.equal(field().value,'');assert.equal(document.activeElement,original);assert.equal(location.href,originalHref);assert.equal(requests.length,calls);assert.equal(state().page,3);
 const emptyEscape=await key('keydown','Escape');assert(!emptyEscape.defaultPrevented);assert.equal(escaped,1);window.removeEventListener('keydown',escape);
 await edit('Do not apply this');await click(clear());assert.equal(location.href,originalHref);assert.equal(requests.length,calls);assert.equal(document.activeElement,original);
 // IME confirmation remains native and cannot also submit a feed query.
 await composition('compositionstart');await edit('中'.repeat(201));assert.equal(field().value.length,201);
 const ime=await key('keydown','Enter',{isComposing:true,keyCode:229});assert(!ime.defaultPrevented);await submit();assert.equal(location.href,originalHref);assert.equal(requests.length,calls);
 const imeEscape=await key('keydown','Escape',{isComposing:true});assert(!imeEscape.defaultPrevented);assert.equal(field().value.length,201);
 await composition('compositionend');assert.equal(field().value.length,200);await submit();assert.equal(requests.length,calls);assert.equal(location.href,originalHref);
 await key('keyup','Enter');await key('keydown','Enter',{keyCode:13});await submit();assert.equal(requests.length,calls+1);assert.deepEqual(state(),{...baseline,page:1,q:'中'.repeat(200)});assert.equal(requests.at(-1).params.get('q'),'中'.repeat(200));assert.equal(field(),original);
 await key('keyup','Enter');const appliedEscape=await key('keydown','Escape');assert(appliedEscape.defaultPrevented);await render();assert.equal(field().value,'');assert.equal(document.activeElement,original);assert.deepEqual(state(),{...baseline,page:1});
 await edit('Safari candidate');await key('keydown','Enter',{keyCode:229});const legacyHref=location.href,legacyCalls=requests.length;await submit();assert.equal(location.href,legacyHref);assert.equal(requests.length,legacyCalls);
 await act(async()=>form().querySelector('[type=submit]').dispatchEvent(new dom.window.Event('pointerdown',{bubbles:true})));await submit();assert.equal(state().q,'Safari candidate');assert.equal(requests.length,legacyCalls+1);await key('keyup','Enter');
 await edit('  a\u0000b\t 中文😀\u007f  ');await submit();assert.equal(state().q,'a b  中文😀');assert.equal(field().value,'a b  中文😀');
 await edit('😀'.repeat(200)+'中');assert.equal(Array.from(field().value).length,200);await submit();assert.equal(state().q,'😀'.repeat(200));assert.equal(accountReturnHref(decodeProblemHash(signInHref(location.hash))),location.hash);
 const restorable={...baseline,q:'Kept applied query',page:3};await navigate(restorable);const keptHref=location.href,keptRequests=requests.length;await edit('Unsubmitted replacement');assert(form().querySelector('.community-search-pending [role=status]').textContent.includes('Not applied'));assert.equal(form().querySelector('[type=submit]').textContent,'Search');await click(form().querySelector('.community-search-restore'));assert.equal(field(),original);assert.equal(field().value,restorable.q);assert.equal(document.activeElement,original);assert.equal(form().querySelector('.community-search-pending'),null);assert.equal(location.href,keptHref);assert.equal(requests.length,keptRequests);assert.deepEqual(state(),restorable,'Restoring a draft search preserves the saved view, page and every applied report filter');

 await edit('Pending report query');original.setSelectionRange(2,9,'backward');original.scrollLeft=16;await navigate({...restorable,sort:'recent',page:2});assert.equal(field(),original);assert.deepEqual([field().value,field().selectionStart,field().selectionEnd,field().selectionDirection,field().scrollLeft],['Pending report query',2,9,'backward',16],'Report ordering and pagination retain unsubmitted search text');
 // Local concept/company search reuses the same input contract without searching the public feed.
 const concept={...baseline,view:'questions',page:2,saved:false,q:'',study:'all',queue:'1,2'};await navigate(concept);const conceptInput=field(),conceptCalls=requests.length;
 const descriptions=conceptInput.getAttribute('aria-describedby').split(/\s+/).map(id=>document.getElementById(id));assert.equal(descriptions.length,2);assert(descriptions.every(Boolean));assert(descriptions.some(node=>node.textContent.includes('200 characters')));assert(descriptions.some(node=>node.textContent.includes('Separate keywords')));
 const practice=document.querySelector('.interview-ranking-row .button');assert(practice);const practiceHref=practice.getAttribute('href');await edit('not applied');assert.equal(requests.length,conceptCalls);assert.equal(practice.getAttribute('href'),practiceHref);assert.deepEqual(state(),concept);
 conceptInput.setSelectionRange(2,7,'backward');conceptInput.scrollLeft=19;await click(document.querySelector('#locale'));assert.equal(field(),conceptInput);assert.equal(field().value,'not applied');assert.deepEqual([field().selectionStart,field().selectionEnd,field().selectionDirection,field().scrollLeft],[2,7,'backward',19]);assert.equal(field().getAttribute('aria-label'),'搜索知识点或来源公司');assert.equal(requests.length,conceptCalls);await click(document.querySelector('#locale'));
 await edit('Ｇｏｏｇｌｅ attention');await submit();assert.equal(requests.length,conceptCalls);assert.deepEqual(state(),{...concept,page:1,queue:'',q:'Ｇｏｏｇｌｅ attention'});assert.equal(field(),conceptInput);
 await edit('No matching private query');await navigate({...state(),days:'30'});assert.equal(field(),conceptInput);assert.equal(field().value,'Ｇｏｏｇｌｅ attention','A route change discards unsubmitted text and resets IME state to its applied query');
 await edit('No matching concept');await submit();const rankingClear=[...document.querySelectorAll('.interview-question-workbench button')].find(node=>node.textContent==='Clear search');assert(rankingClear);await click(rankingClear);assert.equal(field().value,'');assert.equal(document.activeElement,conceptInput);assert.equal(state().company,'google');assert.equal(state().days,'30');
 await composition('compositionstart');await edit('unfinished candidate');const outgoing=field();await navigate({...state(),view:'companies',company:'',q:'',concept:'',saved:false});assert.notEqual(field(),outgoing);assert.equal(field().value,'');assert.equal(field().getAttribute('aria-label'),'Search companies');await act(async()=>outgoing.dispatchEvent(new dom.window.CompositionEvent('compositionend',{bubbles:true,data:'stale candidate'})));assert.equal(field().value,'');
 const companyInput=field(),companyCalls=requests.length;await edit('Ｇｏｏｇｌｅ DeepMind');await submit();assert.equal(field(),companyInput);assert.equal(requests.length,companyCalls);assert.equal(document.querySelectorAll('.interview-company-card').length,1);assert(document.querySelector('.interview-company-card').textContent.includes('Google'));
 await edit('Private unsubmitted owner text');account.user={...user,id:'bob',username:'bob'};await render();assert.notEqual(field(),companyInput);assert.equal(field().value,'Ｇｏｏｇｌｅ DeepMind');assert.equal(state().q,'Ｇｏｏｇｌｅ DeepMind');
 assert.equal(code,document.querySelector('[aria-label="Practice code"]'));assert.equal(code.value,'def solve(): return 42');assert.deepEqual([code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],[3,9,'backward',91,17]);assert.equal(JSON.stringify({...sessionStorage}),originalDraft);assert.equal(JSON.stringify(saved),originalSaved);assert.equal(mutations.length,0);assert.deepEqual(drafts.readInterviewDraft(sessionStorage,'alice'),draft);
 console.log('PASS interview search: actual report/concept/company views share native IME/legacy Enter and Unicode200 contracts, Escape/clear preserve unapplied pagination and applied filter/login context, no typing requests or social writes, combined descriptions, locale/caret continuity, explicit route/view/owner retirement and unchanged practice code/progress/private report draft');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,descriptor]of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
