import assert from 'node:assert/strict';
import fs from 'node:fs';
import {mkdtempSync,rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

// Render the actual App and split panels. Isolate account/runtime boundaries so
// these interactions never contact a server, run Python, or use real drafts.
const dom = new JSDOM('<!doctype html><div id="root"></div>', {url:'https://paperleet.test/#/paper-registry'});
const names = ['window','document','navigator','HTMLElement','CustomEvent','localStorage','sessionStorage','location','history','requestAnimationFrame','cancelAnimationFrame','ResizeObserver','fetch','IS_REACT_ACT_ENVIRONMENT','__paperleetWorkspaceActionTest'];
const originals = new Map(names.map(name => [name,Object.getOwnPropertyDescriptor(globalThis,name)]));
for (const name of ['window','document','navigator','HTMLElement','CustomEvent','localStorage','sessionStorage','location','history']) Object.defineProperty(globalThis,name,{value:dom.window[name],configurable:true,writable:true});
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
globalThis.requestAnimationFrame = callback => setTimeout(callback,0);
globalThis.cancelAnimationFrame = clearTimeout;
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.fetch = async url => { throw Error(`Unexpected network request: ${url}`); };
window.matchMedia = () => ({matches:false,addEventListener(){},removeEventListener(){}});
window.scrollTo = () => {};
window.HTMLElement.prototype.scrollTo = function(x,y) { this.scrollLeft=x;this.scrollTop=y; };
window.HTMLElement.prototype.scrollIntoView = () => {};
Object.defineProperties(window.HTMLElement.prototype,{clientWidth:{get:()=>1200},clientHeight:{get:()=>700}});
localStorage.setItem('paperleet.locale','en');
const problem = {
  id:'architecture/attention',number:1,paper_id:'attention',title:'Attention',difficulty:'E',
  module:{goal:'Implement attention.',goal_en:'Implement attention.',scope:'Return a tensor.',scope_en:'Return a tensor.',required_functions:['attention'],provided_functions:[],optional:false},
  paper:{title:'Attention',url:'https://example.test/paper'},functions:[],
  readme:'# Attention\nReturn the attention output.',notes:'Notes',starter:'def attention(x):\n    return x\n',tests:{T0:1},
  translations:{en:{title:'Attention',readme:'# Attention\nReturn the attention output.',notes:'Notes'}},
  browser_judge:{bundle_url:'/test-fixture.json',backend:'numpy-wasm'},
};
const noop = () => {};
let revision = 0, runResolve, runReject;
const listeners = new Set();
const notify = () => { revision++; for (const listener of listeners) listener(); };
const calls = {runs:[],submits:[],appliedReports:[],stops:0,flushes:0,editorMounts:0,focusRequests:[],detailsLoads:0};
const fixture = {
  problem,calls,subscribe:listener => {listeners.add(listener);return()=>listeners.delete(listener);},snapshot:()=>revision,
  account:{user:{id:'fixture-user',username:'fixture-user',display_name:'Fixture'},health:{mode:'hosted',judge:{ready:true}},healthError:'',refreshHealth:noop,capabilities:{}},
  practice:{owner:'user:fixture-user',ready:true,loading:false,saved:{},conflicts:{},backups:[],owns:owner=>owner==='user:fixture-user',
    update(id,patch){this.saved={...this.saved,[id]:{...this.saved[id],...patch}};notify();},
    async flush(){calls.flushes++;},
    applyReport(entry,report,owner){calls.appliedReports.push({report,owner});this.saved={...this.saved,[entry.id]:{...this.saved[entry.id],report}};notify();},
    retry:noop,resolveConflict:noop},
  browser:{active:false,reports:{},syntaxErrors:{},reasons:{},calibrations:{},diagnostics:{},measurements:{},
    clear(id){for(const name of ['reports','syntaxErrors','reasons','diagnostics','calibrations','measurements'])delete this[name][id];notify();},
    run(entry,source){calls.runs.push({id:entry.id,source});this.active=true;this.problem=entry.id;this.stage='running';delete this.syntaxErrors[entry.id];this.reasons[entry.id]='';this.diagnostics[entry.id]='';notify();return new Promise((resolve,reject)=>{runResolve=resolve;runReject=reject;});},
    stop(){calls.stops++;fixture.browser.active=false;fixture.browser.problem=undefined;notify();runReject?.(new DOMException('Stopped','AbortError'));runResolve=undefined;runReject=undefined;}},
  judge:{active:false,async submit(id,source){calls.submits.push({id,source});if(this.submitException)throw this.submitException;if(this.submitError)throw Error(this.submitError);this.error='';this.active=true;this.pending={problem:id,source};this.job={problem:id,status:'queued',ahead:2};notify();},stopWaiting:noop,
    cancel(){this.active=false;this.pending=undefined;this.job={...this.job,status:'cancelled'};notify();}},
  learning:{ready:true,records:{},retry:noop,dismiss:noop,record:noop},
};
// App passes callbacks as event handlers; use bound fixtures just like the hooks.
fixture.practice.update=fixture.practice.update.bind(fixture.practice);
fixture.browser.run=fixture.browser.run.bind(fixture.browser);
globalThis.__paperleetWorkspaceActionTest = fixture;
const mocks = {
  './social': `import {useSyncExternalStore} from 'react';
    export {profileHref} from '/src/social.tsx';
    export const AccountProvider=({children})=>children;
    export const AccountNav=()=>null;
    export function useAccount(){const f=globalThis.__paperleetWorkspaceActionTest;useSyncExternalStore(f.subscribe,f.snapshot,f.snapshot);return f.account;}`,
  './PracticeWorkspace': `export const usePracticeWorkspace=()=>globalThis.__paperleetWorkspaceActionTest.practice;`,
  './useBrowserJudge': `export const useBrowserJudge=()=>globalThis.__paperleetWorkspaceActionTest.browser;`,
  './useQueuedJudge': `export function useQueuedJudge(owner,ready,onReport){const f=globalThis.__paperleetWorkspaceActionTest;f.onServerReport=onReport;return f.judge;}`,
  './useBrowserLearning': `export const useBrowserLearning=()=>globalThis.__paperleetWorkspaceActionTest.learning;`,
  './catalogIndex': `export const loadCatalogIndex=async()=>globalThis.__paperleetWorkspaceActionTest.catalog||({problems:[globalThis.__paperleetWorkspaceActionTest.problem],papers:[],categories:[]});`,
  './ProblemStats': `export const useProblemStats=()=>({stats:{},totalSubmissions:0,refresh:()=>{},loading:false});export const getProblemStats=()=>({});export const ProblemStats=()=>null;`,
  './PaperViews': `export {PaperLibrary,paperModules} from '/src/PaperViews.tsx';`,
  './ReportView': `import React from 'react';import Report from '/src/ReportView.tsx?fixture_actual=1';globalThis.__paperleetWorkspaceActionTest.calls.detailsLoads++;export default function FixtureReport(props){if(globalThis.__paperleetWorkspaceActionTest.failDetails)throw Error('Fixture detail failure');return React.createElement(Report,props);}`,
  './deferredPages': `import React,{useEffect} from 'react';
    import {deferredFeature} from '/src/deferredFeature.tsx';
    import {ProblemOverviewFallback} from '/src/ProblemOverviewFallback.tsx';
    import Overview from '/src/ProblemOverview.tsx';
    export const ProblemOverview=deferredFeature(()=>new Promise((resolve,reject)=>{const f=globalThis.__paperleetWorkspaceActionTest;f.resolveOverview=()=>resolve({default:Overview});f.rejectOverview=()=>reject(Error('Synthetic optional details failure'));}),['',''],{loading:p=>React.createElement(ProblemOverviewFallback,p),failure:(p,retry)=>React.createElement(ProblemOverviewFallback,{...p,retry})});
    export {PaperDetail} from '/src/PaperDetail.tsx';
    export {default as PaperRegistry} from '/src/PaperRegistry.tsx';
    export const ProblemFavoriteButton=props=>React.createElement('button',{'data-test-favorite':props.problemId,'data-return-to':props.returnTo},'Save fixture problem');
    export const ProblemFavoritesPage=props=>React.createElement('section',{'data-test-favorites':props.route},React.createElement('button',{'data-test-favorite-filters':true,onClick:()=>props.onFilters('#/favorites?q=LoRA&state=todo')},'Change favorite filters'));
    export const InboxBell=()=>null,Messages=()=>null;
    export {default as SubmissionCompletionNotice} from '/src/SubmissionCompletionNotice.tsx';
    export const Achievements=props=>React.createElement('a',{'data-test-achievement-return':true,href:props.returnTo},'Back to where you left off');
    export {default as ResizableSplit} from '/src/ResizableSplit.tsx';
    export {default as WorkspaceTestResults} from '/src/WorkspaceTestResults.tsx';
    export {default as JudgeRuntimeDetails} from '/src/JudgeRuntimeDetails.tsx';
    export {default as WorkspaceLayoutMenu} from '/src/WorkspaceLayoutMenu.tsx';
    export {default as ReadingTabControls} from '/src/ReadingTabControls.tsx';
    export function PythonEditor({value,onChange,label,readOnly,focusRequest,onFocusHandled}){useEffect(()=>{globalThis.__paperleetWorkspaceActionTest.calls.editorMounts++;},[]);useEffect(()=>{if(focusRequest){globalThis.__paperleetWorkspaceActionTest.calls.focusRequests.push(focusRequest);const focused=!focusRequest.claimFocus||focusRequest.claimFocus();if(focused)document.querySelector('#problem-editor-panel textarea')?.focus();onFocusHandled?.(focusRequest,focused);}},[focusRequest]);return React.createElement('textarea',{'aria-label':label,value,readOnly,onChange:event=>onChange(event.target.value)});}
    export const Prose=({content})=>React.createElement('article',null,content);
    export const ReportView=({report})=>React.createElement('div',{'data-report':true},report ? 'Report available' : 'No report');
    ${['PreparationGuide','InterviewPracticeNextStep','Interviews','ProblemCompanyTags','Community','UserProfile','Account','SolutionHistory','SubmissionResultPage','Submissions','Visuals','WorkspaceBackup','QueueAdmissionNotice','ModuleNavigator','ResetCodeButton','BrowserDraftBackups','SettingsPage','AchievementObserver','BrowserResultActions','ReportWorkflow','OnlinePresence','ArtifactCharts'].map(name=>`export const ${name}=()=>null;`).join('\n')}`,
};
const cache=mkdtempSync(path.join(os.tmpdir(),'paperleet-registry-browsing-'));
const catalog=JSON.parse(fs.readFileSync('public/data/catalog.json','utf8'));
fixture.catalog={...catalog,problems:catalog.problems.map(({detail_url,...entry})=>entry)};
fixture.account.loading=false;fixture.account.refresh=async()=>{};
fixture.practice.saved={'representation/graphsage_mean':{code:'PRIVATE REGISTRY PRACTICE 😀'}};
const savedBefore=JSON.stringify(fixture.practice.saved),network=[],starReads=[];
globalThis.fetch=async(url,options)=>{if(String(url)==='https://api.github.com/repos/Hongyang-Du/PaperLeet'){assert.equal(options?.credentials,'omit');assert.equal(options?.referrerPolicy,'no-referrer');assert(options?.signal instanceof AbortSignal);assert.equal(options.body,undefined);assert.equal(options.headers,undefined);starReads.push(String(url));return Response.json({stargazers_count:42});}network.push(String(url));throw Error('Unexpected registry request');};
const scrolls=[];dom.window.HTMLElement.prototype.scrollIntoView=function(){scrolls.push(this);};
const server=await createServer({cacheDir:cache,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0')},plugins:[{name:'registry-runtime-boundaries',enforce:'pre',resolveId(source,importer){if(source==='./social')return '\0registry:'+source;if(importer?.endsWith('/src/App.tsx')&&source in mocks)return '\0registry:'+source;},load(id){if(id.startsWith('\0registry:'))return mocks[id.slice('\0registry:'.length)];}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client'),{LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx'),{default:Registry}=await server.ssrLoadModule('/src/PaperRegistry.tsx');
 const nav=await server.ssrLoadModule('/src/libraryNavigation.ts'),account=await server.ssrLoadModule('/src/accountNavigation.ts'),{decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts'),{searchPaperRegistry}=await server.ssrLoadModule('/src/paperRegistrySearch.ts');
 const initial={query:'',category:'all',sort:'default',page:1},href=view=>nav.paperRegistryHref({...initial,...view});
 const valid=['paper-registry','paper-registry?','paper-registry?page=1','paper-registry?category=representation&sort=newest&page=501','paper-registry?category=all&sort=default&q=',href({query:'论文 & 100% _ + / # 😀',sort:'title',page:3}),href({query:'😀'.repeat(200),category:'future-category',sort:'oldest'})];
 for(const value of valid){const view=nav.readPaperRegistryRoute(value);assert(view,value);const canonical=href(view);assert.deepEqual(nav.readPaperRegistryRoute(canonical),view);for(const target of ['100','representation/graphsage_mean','papers/graphsage']){const linked=nav.withLibraryContext(target,canonical);assert.equal(nav.libraryReturnHref(linked),canonical);assert.equal(account.accountReturnHref(decodeProblemHash(account.signInHref(linked))),linked);}assert.equal(account.accountReturnHref(decodeProblemHash(account.signInHref(canonical))),canonical);}
 const invalid=['paper-registry/','paper-registry/../account','Paper-registry','https://example.test/#/paper-registry','//example.test/paper-registry',...['from=22','owner=alice','code=private','tag=all','state=done','runtime=browser','sort=down','constructor=x','__proto__=x','q=a&q=b','q=a&%71=b','sort=title&sort=title','q=%00','q=%7f','q=%','q=%ZZ','q=%E0%A4%A','q=%ED%A0%80','q=raw?from=22','?q=x','category=..%2Faccount','category=Architecture','category='+'a'.repeat(81),'q='+encodeURIComponent('😀'.repeat(201)),...['','0','01','-1','1.0','502','1000','1&page=2'].map(p=>'page='+p)].map(q=>'paper-registry?'+q)];
 for(const value of invalid){assert.equal(nav.readPaperRegistryRoute(value),undefined,value);assert.equal(nav.libraryReturnHref('100?'+new URLSearchParams({from:value})),undefined,value);assert.equal(account.signInHref(value),'#/account',value);}
 for(const view of [{page:0},{page:502},{page:1.5},{sort:'down'},{category:'../escape'},{query:'\0'},{query:'😀'.repeat(201)}])assert.throws(()=>href(view));
 const catalogBefore=JSON.stringify(catalog),published=catalog.papers.filter(p=>p.kind!=='course');
 for(const entry of catalog.problems){const expected=published.filter(p=>p.module_ids.includes(entry.id)).sort((a,b)=>a.number-b.number).map(p=>p.id);assert.deepEqual(searchPaperRegistry(catalog.papers,catalog.problems,'#'+entry.number).map(p=>p.id),expected,'Exercise number '+entry.number);}
 assert.deepEqual(searchPaperRegistry(catalog.papers,catalog.problems,'＃１００').map(p=>p.id),['graphsage']);assert.deepEqual(searchPaperRegistry(catalog.papers,catalog.problems,'ＧｒａｐｈＳＡＧＥ').map(p=>p.id),['graphsage']);assert.equal(JSON.stringify(catalog),catalogBefore);
 console.log('PASS paper registry navigation: closed bounded public routes, canonical Unicode/percent/hash query round trips, paper/module/sign-in context, malformed/external/private/recursive rejection; exact exercise numbers across the full catalog and NFKC search without catalog mutation');
 let controls,locale,applied='paper-registry',rows=catalog,error='',retries=0;
 function Harness(){const [route,setRoute]=React.useState(applied),[data,setData]=React.useState(rows),[failure,setFailure]=React.useState(error);controls={setRoute,setData,setFailure};locale=useI18n();return React.createElement(Registry,{...data,route,error:failure,onRetry:()=>{retries++;},onFilters:next=>{applied=decodeProblemHash(next);setRoute(applied);}});}
 root=createRoot(document.querySelector('#root'));await act(async()=>root.render(React.createElement(LocaleProvider,null,React.createElement(Harness))));
 const wait=async predicate=>{for(let n=0;n<160;n++){if(predicate())return;await act(async()=>new Promise(r=>setTimeout(r,12)));}assert(predicate(),document.body.textContent.slice(-1200));};
 const button=(text,scope=document)=>[...scope.querySelectorAll('button')].find(n=>n.textContent.trim()===text),click=async node=>{assert(node);assert(!node.disabled);await act(async()=>{node.focus();node.click();await new Promise(r=>setTimeout(r,20));});};
 const field=async(node,value)=>{assert(node);await act(async()=>{const prototype=node.tagName==='SELECT'?dom.window.HTMLSelectElement.prototype:dom.window.HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event(node.tagName==='SELECT'?'change':'input',{bubbles:true}));});};
 const search=()=>document.querySelector('.paper-registry-search input'),cards=()=>[...document.querySelectorAll('[data-paper-registry-id]')],category=()=>document.querySelector('.paper-registry-toolbar select'),sort=()=>document.querySelectorAll('.paper-registry-toolbar select')[1];
 const setRoute=async view=>{applied=decodeProblemHash(href(view));await act(async()=>controls.setRoute(applied));};
 assert.equal(cards().length,20);const first=cards()[0].dataset.paperRegistryId;await click(button('Next'));assert.equal(nav.readPaperRegistryRoute(applied).page,2);assert.equal(document.activeElement,document.querySelector('.paper-registry-count'));assert.equal(scrolls.at(-1),document.activeElement);assert.notEqual(cards()[0].dataset.paperRegistryId,first);
 await field(category(),'representation');assert.equal(nav.readPaperRegistryRoute(applied).page,1);assert(cards().every(card=>catalog.papers.find(p=>p.id===card.dataset.paperRegistryId).category==='representation'));await field(sort(),'newest');assert.equal(nav.readPaperRegistryRoute(applied).sort,'newest');
 await field(search(),'GraphSAGE');assert.equal(cards().length,1);const expected=href({query:'GraphSAGE',category:'representation',sort:'newest'});assert.equal('#/'+applied,expected);assert.equal(nav.libraryReturnHref(cards()[0].querySelector('h2 a').getAttribute('href')),expected);assert.equal(nav.libraryReturnHref(cards()[0].querySelector('a[href^="#/100?"]').getAttribute('href')),expected);
 search().focus();search().setSelectionRange(2,7);const input=search(),scrollCount=scrolls.length;await act(async()=>locale.setLocale('zh'));assert.equal(search(),input);assert.equal(document.activeElement,input);assert.deepEqual([input.selectionStart,input.selectionEnd],[2,7]);assert.equal(scrolls.length,scrollCount);await act(async()=>locale.setLocale('en'));
 const key=async options=>act(async()=>search().dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true,...options})));
 await key({isComposing:true});assert.equal(search().value,'GraphSAGE');await key({keyCode:229});assert.equal(search().value,'GraphSAGE');await key({ctrlKey:true});assert.equal(search().value,'GraphSAGE');await key({});assert.equal(search().value,'');assert.equal(category().value,'representation');assert.equal(sort().value,'newest');assert.equal(document.activeElement,input);
 await field(search(),'#100');await act(async()=>controls.setFailure('offline'));assert.equal(search().value,'#100');await click(button('Retry'));assert.equal(retries,1);assert.equal(document.activeElement,input);const retryScrolls=scrolls.length;await act(async()=>{controls.setFailure('');controls.setData({papers:[],problems:[],categories:[]});});assert(document.body.textContent.includes('Loading papers'));await act(async()=>controls.setData(catalog));assert.equal(cards().length,1);assert.equal(document.activeElement,input);assert.equal(scrolls.length,retryScrolls);
 await setRoute({query:'GraphSAGE',page:501});assert.equal(cards().length,0);assert(button('Go to the last page'));await click(button('Go to the last page'));assert.equal(nav.readPaperRegistryRoute(applied).page,1);assert.equal(search().value,'GraphSAGE');assert.equal(cards().length,1);
 await setRoute({category:'future-category'});assert.equal(category().value,'future-category');assert(category().selectedOptions[0].textContent.includes('Unavailable'));assert(button('Clear filters'));await click(button('Clear filters'));assert.equal(document.activeElement,input);assert.equal(cards().length,20);
 await act(async()=>controls.setRoute('paper-registry?owner=alice'));assert(document.querySelector('[role=alert]'));assert.equal(document.querySelector('.paper-registry-toolbar'),null);assert.equal(document.querySelector('a[href="#/paper-registry"]').textContent,'Open the full paper registry');
 assert.equal(JSON.stringify(fixture.practice.saved),savedBefore);assert.equal(network.length,0);
 assert.equal(starReads.length,0);console.log('PASS paper registry browsing: bounded pages and category/sort/search state, real contextual exercise links, explicit results focus, locale/caret preservation, composition/modifier-safe Escape, retained error/retry/late catalog controls, unavailable category and explicit empty-page recovery; no network or practice writes');
 await act(async()=>root.unmount());root=undefined;
 // Traverse the actual App: keep the registry, paper detail and workspace routes real.
 const {default:App}=await server.ssrLoadModule('/src/App.tsx');
 localStorage.setItem('paperleet.locale','en');location.hash='#/paper-registry';
 root=createRoot(document.querySelector('#root'));await act(async()=>{root.render(React.createElement(App));await new Promise(r=>setTimeout(r,30));});await wait(()=>cards().length===20);
 await click(button('Next'));assert.equal(nav.readPaperRegistryRoute(location.hash).page,2);const secondPage=location.hash;await click(cards()[0].querySelector('h2 a'));await wait(()=>document.querySelector('.paper-library-link'));assert.equal(document.querySelector('.paper-library-link').getAttribute('href'),secondPage);assert.equal(document.querySelector('.paper-library-link').textContent,'Back to paper registry');await click(document.querySelector('.paper-library-link'));await wait(()=>cards().length===20);assert.equal(location.hash,secondPage);
 await field(search(),'GraphSAGE');await field(sort(),'title');const source=location.hash;assert.equal(nav.readPaperRegistryRoute(source).page,1);assert.equal(cards().length,1);
 await click(cards()[0].querySelector('h2 a'));await wait(()=>document.querySelector('.paper-library-link'));const practice=[...document.querySelectorAll('a')].find(n=>n.getAttribute('href')?.startsWith('#/100?'));assert(practice);assert.equal(nav.libraryReturnHref(practice.getAttribute('href')),source);await click(practice);await wait(()=>document.querySelector('#problem-editor-panel textarea'));assert.equal(document.querySelector('#problem-editor-panel textarea').value,'PRIVATE REGISTRY PRACTICE 😀');const back=[...document.querySelectorAll('a')].find(n=>n.textContent.includes('Back to paper registry'));assert(back);assert.equal(back.getAttribute('href'),source);const login=account.signInHref(location.hash);assert.equal(account.accountReturnHref(decodeProblemHash(login)),location.hash);await click(back);await wait(()=>cards().length===1);assert.equal(location.hash,source);
 // History can be unavailable; controlled UI and explicit links must still carry the newest view.
 const replace=window.history.replaceState;window.history.replaceState=()=>{throw Error('History blocked');};await field(search(),'#100');assert.equal(cards().length,1);const recent=cards()[0].querySelector('h2 a').getAttribute('href');assert.equal(nav.readPaperRegistryRoute(nav.libraryReturnHref(recent)).query,'#100');window.history.replaceState=replace;
 assert.equal(JSON.stringify(fixture.practice.saved),savedBefore);assert.deepEqual(calls.runs,[]);assert.deepEqual(calls.submits,[]);assert.equal(calls.flushes,0);assert.equal(network.length,0);
 await wait(()=>starReads.length===1);assert.equal(starReads.length,1);console.log('PASS actual App registry flow: live pagination and filters, paper detail and practice breadcrumbs, exact list return and sign-in continuation, blocked-history explicit-link recovery; original private source retained without judge, submission, flush or registry/private requests; one controlled anonymous bounded GitHub star read');
 // Revisit an actual App workspace with geometry that changes when the return notice mounts.
 let registryY=0;const registryScrolls=[];Object.defineProperty(window,'scrollY',{get:()=>registryY,configurable:true});Object.defineProperty(document,'hidden',{get:()=>false,configurable:true});window.requestAnimationFrame=globalThis.requestAnimationFrame;window.cancelAnimationFrame=globalThis.cancelAnimationFrame;window.scrollTo=(value,next)=>{registryY=typeof value==='number'?next:value.top;registryScrolls.push(registryY);};
 const box=(top,height)=>({top,bottom:top+height,height,left:0,right:1200,width:1200,x:0,y:top,toJSON(){return this;}}),originalBounds=window.HTMLElement.prototype.getBoundingClientRect;
 window.HTMLElement.prototype.getBoundingClientRect=function(){if(this.classList.contains('topbar'))return box(0,64);const row=this.closest('[data-paper-registry-id]');if(!row)return originalBounds.call(this);const list=row.closest('.paper-registry-page'),index=[...list.querySelectorAll('[data-paper-registry-id]')].indexOf(row),top=500+index*260+(list.querySelector('.paper-registry-position')?60:0)-registryY;return this===row?box(top,240):this.matches('h2 a')?box(top+32,40):this.closest('.paper-registry-links')?box(top+160,28):box(top,32);};
 document.querySelector('.topbar').style.position='sticky';const readingSource=nav.libraryReturnHref(cards()[0].querySelector('h2 a').getAttribute('href'));await act(async()=>{location.hash=readingSource;await new Promise(r=>setTimeout(r,20));});await wait(()=>cards().length===1&&location.hash===readingSource);registryY=420;
 const originalModule=cards()[0].querySelector('a[href^="#/100?"]'),readingTop=originalModule.getBoundingClientRect().top;await click(originalModule);await wait(()=>document.querySelector('#problem-editor-panel textarea'));assert.equal(document.querySelector('#problem-editor-panel textarea').value,'PRIVATE REGISTRY PRACTICE 😀');const readingBack=[...document.querySelectorAll('a')].find(n=>n.textContent.includes('Back to paper registry'));await click(readingBack);await wait(()=>document.querySelector('[data-registry-resumed=true]')&&document.activeElement?.getAttribute('href')?.startsWith('#/100?'));
 const returnedModule=cards()[0].querySelector('a[href^="#/100?"]');assert.equal(returnedModule.getBoundingClientRect().top,readingTop);assert.equal(location.hash,readingSource);assert.equal(document.activeElement,returnedModule);assert(registryScrolls.length>0);
 const refreshed=await server.ssrLoadModule('/src/registryPosition.ts?fresh=app-reload');assert.equal(refreshed.readRegistryPosition(readingSource).paper,'graphsage');assert.equal(refreshed.readRegistryPosition(readingSource).top,readingTop);const beforeClose=returnedModule.getBoundingClientRect().top;await click(document.querySelector('[aria-label="Dismiss position notice"]'));await wait(()=>!document.querySelector('.paper-registry-position')&&document.activeElement===returnedModule);assert.equal(returnedModule.getBoundingClientRect().top,beforeClose);
 assert.equal(JSON.stringify(fixture.practice.saved),savedBefore);assert.equal(calls.flushes,0);assert.deepEqual(calls.runs,[]);assert.deepEqual(calls.submits,[]);assert.equal(network.length,0);
 assert.equal(starReads.length,1);console.log('PASS actual App registry reading return: captured exercise entry and exact source view, committed-notice geometry and keyboard focus after workspace return, reloadable public position, dismissal without a reading jump, unchanged private source, coalesced public star read and no judge/flush/registry or private requests');
}finally{await act(async()=>{if(root)root.unmount();await server.close();});rmSync(cache,{recursive:true,force:true});dom.window.close();for(const[name,value]of originals)if(value)Object.defineProperty(globalThis,name,value);else delete globalThis[name];}
