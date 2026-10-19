import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import {mkdtempSync,rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

// Render the actual App and split panels. Isolate account/runtime boundaries so
// these interactions never contact a server, run Python, or use real drafts.
const dom = new JSDOM('<!doctype html><div id="root"></div>', {url:'https://paperleet.test/#/?company=google&company_days=90'});
const failures=[];dom.window.addEventListener('error',event=>failures.push(event.error));
const names = ['window','document','navigator','HTMLElement','Element','CustomEvent','localStorage','sessionStorage','location','history','requestAnimationFrame','cancelAnimationFrame','ResizeObserver','scrollY','innerHeight','scrollTo','fetch','IS_REACT_ACT_ENVIRONMENT','__paperleetWorkspaceActionTest'];
const originals = new Map(names.map(name => [name,Object.getOwnPropertyDescriptor(globalThis,name)]));
for (const name of ['window','document','navigator','HTMLElement','Element','CustomEvent','localStorage','sessionStorage','location','history']) Object.defineProperty(globalThis,name,{value:dom.window[name],configurable:true,writable:true});
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
globalThis.requestAnimationFrame = callback => setTimeout(callback,0);
globalThis.cancelAnimationFrame = clearTimeout;
globalThis.scrollY=0;globalThis.innerHeight=900;globalThis.scrollTo=()=>{};
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
    export const AccountNav=()=>null;export const socialApi=(...args)=>globalThis.__paperleetWorkspaceActionTest.request(...args);export const errorMessage=e=>e.message;
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
    export {ModuleNavigator} from '/src/ModuleNavigator.tsx';export {default as InterviewPracticeNextStep} from '/src/InterviewPracticeNextStep.tsx';
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
    ${['PreparationGuide','Interviews','ProblemCompanyTags','Community','UserProfile','Account','SolutionHistory','SubmissionResultPage','Submissions','Visuals','WorkspaceBackup','QueueAdmissionNotice','ResetCodeButton','BrowserDraftBackups','SettingsPage','AchievementObserver','BrowserResultActions','ReportWorkflow','OnlinePresence','ArtifactCharts'].map(name=>`export const ${name}=()=>null;`).join('\n')}`,
};

const cache=mkdtempSync(path.join(os.tmpdir(),'paperleet-library-company-'));
const catalog=JSON.parse(fs.readFileSync('public/data/catalog.json','utf8'));
fixture.catalog={...catalog,problems:catalog.problems.map(({detail_url,...entry})=>entry)};
const [a,b]=fixture.catalog.problems,c=fixture.catalog.problems.find(p=>p.paper_id!==a.paper_id&&p.id!==b.id),ids=[b.id,c.id,a.id];
fixture.account.loading=false;fixture.account.refresh=async()=>{};
fixture.practice.saved={[b.id]:{code:'PRIVATE COMPANY PRACTICE 😀'}};
const pass=p=>({problem:p.id,summary:Object.fromEntries(Object.entries(p.tests).map(([layer,count])=>[layer,[count,count]])),tests:Object.entries(p.tests).flatMap(([layer,count])=>Array.from({length:count},(_,i)=>({id:layer+'-'+i,layer,passed:true})))});
const snapshot=JSON.stringify(fixture.practice.saved),requests=[],pending=[];
const summary=(company='google',days='90',linked=ids)=>({days,basis:'published-community-reports',generated_at:'2026-10-18T00:00:00Z',reports:4,contributors:3,companies:[{company,reports:4,contributors:3}],questions:linked.map((problem_id,index)=>({problem_id,reports:4-index,contributors:3-index,companies:[{company,problem_id,reports:4-index,contributors:3-index,latest:'2026-10-18T00:00:00Z'}]}))});
fixture.request=(url,body,signal)=>{assert.equal(body,undefined);requests.push({url,signal});return new Promise((resolve,reject)=>pending.push({url,signal,resolve,reject}));};
const server=await createServer({cacheDir:cache,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_HOSTED_APP_AVAILABLE':JSON.stringify('1')},plugins:[{name:'library-company-boundaries',enforce:'pre',resolveId(source,importer){if(source==='./social')return '\0company:'+source;if(importer?.endsWith('/src/App.tsx')&&source in mocks)return '\0company:'+source;},load(id){if(id.startsWith('\0company:'))return mocks[id.slice('\0company:'.length)];}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client'),h=React.createElement;
 const nav=await server.ssrLoadModule('/src/libraryNavigation.ts'),account=await server.ssrLoadModule('/src/accountNavigation.ts'),{decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 const {selectLibraryPractice}=await server.ssrLoadModule('/src/libraryPractice.ts'),{readLibraryCompanyEvidence:project}=await server.ssrLoadModule('/src/LibraryCompanyEvidence.tsx');
 const {recommendPractice,nextPractice}=await server.ssrLoadModule('/src/practice-recommendation.ts');
 for(const company of ['google','future-company'])for(const days of ['30','90','365','all'])for(const collection of [undefined,'llm-basics']){
  const view=nav.libraryHref(collection,{...nav.emptyLibraryFilters(),company,company_days:days,query:'注意力 & 100% _ 😀',difficulty:'M',state:'started',sort:'title'});
  assert.equal(nav.readLibraryRoute(view).filters.company,company);assert.equal(nav.readLibraryRoute(view).filters.company_days||'365',days);
  const target=nav.withLibraryContext(a.number,view);assert.equal(nav.libraryReturnHref(target),view);assert.equal(account.accountReturnHref(decodeProblemHash(account.signInHref(target))),target);
 }
 for(const query of ['company=','company=GOOGLE','company=google_team','company=../escape','company=google&company=meta','company_days=all','company=google&company_days=7','company=google&company_days=all&company_days=30','company=google&owner=alice','company=google&code=secret'])assert.equal(nav.readLibraryRoute('#/?'+query),undefined);
 const css=fs.readFileSync('src/library-company.css','utf8'),tokens=new Set([...fs.readFileSync('src/style.css','utf8').matchAll(/(--[a-z-]+)\s*:/g)].map(match=>match[1]));for(const [,name]of css.matchAll(/var\((--[a-z-]+)\)/g))assert(tokens.has(name),'Company UI must use a declared theme token: '+name);
 const cohort=summary(),original=JSON.stringify(cohort),projected=project(cohort,catalog.problems,'google','90');
 assert.deepEqual(projected.questions.map(row=>row.problem.id),ids);assert.equal(projected.missing,0);
 for(const bad of [{...cohort,days:'30'},{...cohort,companies:[{company:'meta',reports:4,contributors:3}]},{...cohort,contributors:2},{...cohort,questions:[cohort.questions[0],cohort.questions[0]]},{...cohort,questions:[{...cohort.questions[0],companies:[{...cohort.questions[0].companies[0],company:'meta'}]}]}])assert.equal(project(bad,catalog.problems,'google','90'),undefined);
 assert.equal(project(cohort,[],'google','90').missing,3);assert.equal(project(cohort,catalog.problems,'future-company','90'),undefined);assert.equal(JSON.stringify(cohort),original);
 const scope=nav.readLibraryRoute('#/?company=google&company_days=90'),set=new Set(ids);
 assert.deepEqual(selectLibraryPractice(catalog.papers,catalog.problems,{},scope).problems,[]);
 assert.deepEqual(selectLibraryPractice(catalog.papers,catalog.problems,{},scope,set).problems.map(row=>row.id),ids);
 const saved={[b.id]:{acceptedReport:pass(b)}};
 assert.equal(recommendPractice(selectLibraryPractice(catalog.papers,catalog.problems,saved,scope,set).problems,saved).id,c.id);
 assert.equal(nextPractice(b,ids,selectLibraryPractice(catalog.papers,catalog.problems,saved,scope,set).problems,saved).id,c.id);
 assert.deepEqual(selectLibraryPractice(catalog.papers,catalog.problems,saved,{...scope,filters:{...scope.filters,state:'done'}},set).problems.map(row=>row.id),[b.id]);
 console.log('PASS library company contracts: closed URL/auth round trips, exact company and period cohorts, strict inconsistent/missing evidence, frequency order and real module grades across paper boundaries, no catalog mutation');
 const {default:App}=await server.ssrLoadModule('/src/App.tsx');
 const wait=async fn=>{for(let i=0;i<240;i++){if(fn())return;await act(async()=>new Promise(r=>setTimeout(r,10)));}throw Error('Company flow timed out: '+document.body.textContent.slice(-1600));};
 const render=()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(h(App));});
 const route=async href=>{await act(async()=>{history.replaceState(null,'',href);window.dispatchEvent(new dom.window.HashChangeEvent('hashchange'));});};
 const field=async(node,value)=>{assert(node);await act(async()=>{node.focus();Object.getOwnPropertyDescriptor(node.tagName==='SELECT'?window.HTMLSelectElement.prototype:window.HTMLInputElement.prototype,'value').set.call(node,value);node.dispatchEvent(new window.Event(node.tagName==='SELECT'?'change':'input',{bubbles:true}));});};
 const click=async node=>{assert(node);await act(async()=>{node.focus();const href=node.getAttribute('href');if(href?.startsWith('#')){history.replaceState(null,'',href);window.dispatchEvent(new dom.window.HashChangeEvent('hashchange'));}else node.click();});};
 const take=()=>{const row=pending.shift();assert(row,'A requested scope is pending');return row;},answer=(row,value)=>act(async()=>row.resolve(value));
 const rows=()=>[...document.querySelectorAll('[data-module-id]')],company=()=>document.querySelector('[aria-label="Filter library by company"]'),period=()=>document.querySelector('[aria-label="Library report publication period"]'),editor=()=>document.querySelector('#problem-editor-panel textarea'),button=text=>[...document.querySelectorAll('button')].find(node=>node.textContent===text);
 sessionStorage.setItem('private-company-sentinel','DRAFT UNCHANGED');
 await render();await wait(()=>company()&&pending.length);assert.equal(document.querySelector('.brand-symbol').tagName,'IMG');assert.equal(document.querySelector('.brand-symbol').getAttribute('alt'),'');assert.equal(document.querySelector('.brand-symbol').getAttribute('aria-hidden'),'true');assert.equal(document.querySelector('.brand').getAttribute('aria-label'),'PaperLeet home');const first=take(),nativeCompany=company(),nativePeriod=period();assert.equal(requests.length,1);assert.equal(rows().length,0);assert(document.querySelector('.random-problem').disabled);assert(document.querySelector('.continue-link').disabled);assert(!document.querySelector('.no-results'));assert(!document.body.textContent.includes('Core modules complete'));
 await answer(first,summary());await wait(()=>rows().length===3&&document.querySelector('.library-interview-guide'));assert.deepEqual(rows().map(row=>row.dataset.moduleId),ids);assert.equal(requests.length,1,'The external sidebar reuses the same confirmed company cohort');assert(document.querySelector('.library-interview-guide').textContent.includes('3 linked exercises'));
 const initial=location.hash;await field(document.querySelector('[aria-label="Search papers or modules"]'),'UNMATCHABLE PUBLIC QUERY');assert.equal(rows().length,0);assert.equal(requests.length,1);assert.equal(nav.readLibraryRoute(location.hash).filters.company,'google');await field(document.querySelector('[aria-label="Search papers or modules"]'),'');assert.equal(rows().length,3);assert.equal(company(),nativeCompany);assert.equal(period(),nativePeriod);
 const started=rows().find(row=>row.dataset.moduleId===b.id);await click(started);await wait(()=>editor());const source=editor();assert.equal(source.value,'PRIVATE COMPANY PRACTICE 😀');source.focus();source.setSelectionRange(2,12,'backward');source.scrollTop=52;source.scrollLeft=13;const mounts=calls.editorMounts,storage=sessionStorage.getItem('private-company-sentinel');
 const from=nav.libraryReturnHref(location.hash);assert.equal(nav.readLibraryRoute(from).filters.company,'google');assert.equal(nav.readLibraryRoute(from).filters.company_days,'90');
 const picker=document.querySelector('.module-picker-trigger');await click(picker);assert.deepEqual([...document.querySelectorAll('.module-picker-links>a')].map(row=>nav.routePath(decodeProblemHash(row.getAttribute('href')))),ids.map(id=>String(catalog.problems.find(p=>p.id===id).number)));
 fixture.browser.reports[b.id]={...pass(b),runtime:'browser-numpy-wasm',source_sha256:createHash('sha256').update(source.value).digest('hex')};await act(async()=>notify());await wait(()=>document.querySelector('.journey-next-link'));assert.equal(nav.routePath(decodeProblemHash(document.querySelector('.journey-next-link').getAttribute('href'))),String(c.number));assert.equal(document.querySelector('.journey-next-link small').textContent,'Continue current company practice');
 await route(nav.withLibraryContext(b.number,nav.libraryHref(undefined,{...scope.filters,company_days:'30'})));await wait(()=>pending.length);const old=take();assert(editor()===source);assert(document.querySelector('.module-batch-unavailable'));assert(!document.querySelector('.module-picker-trigger'));assert(document.querySelector('.journey-next-link').textContent.includes('Company scope is unconfirmed'));assert(!document.body.textContent.includes('Core modules complete'));
 await route(nav.withLibraryContext(b.number,nav.libraryHref(undefined,{...scope.filters,company_days:'all'})));await wait(()=>pending.length);const latest=take();assert(old.signal.aborted);await answer(old,summary('google','30',[a.id]));assert(!document.querySelector('.module-picker-trigger'));await answer(latest,{...summary('google','all'),days:'90'});await wait(()=>document.querySelector('.library-company-status[data-phase="error"]'));
 await click(button('Retry company scope'));await wait(()=>pending.length);const retried=take();source.focus();await answer(retried,summary('google','all'));await wait(()=>document.querySelector('.module-picker-trigger'));assert.equal(document.activeElement,source);assert.equal(editor(),source);assert.equal(calls.editorMounts,mounts);assert.deepEqual([source.value,source.selectionStart,source.selectionEnd,source.selectionDirection,source.scrollTop,source.scrollLeft],['PRIVATE COMPANY PRACTICE 😀',2,12,'backward',52,13]);assert.equal(sessionStorage.getItem('private-company-sentinel'),storage);
 // A→B→A cannot revive the prior confirmed cohort, even with the same company and URL.
 fixture.account.loading=true;await act(async()=>notify());await wait(()=>document.querySelector('.module-batch-unavailable'));fixture.account.loading=false;await act(async()=>notify());await wait(()=>pending.length);const revalidation=take();assert(!document.querySelector('.module-picker-trigger'));source.focus();await answer(revalidation,summary('google','all'));await wait(()=>document.querySelector('.module-picker-trigger'));assert.equal(document.activeElement,source);assert.equal(editor(),source);
 await route(nav.libraryReturnHref(location.hash));await wait(()=>rows().length===3);assert(period().value==='all');const reads=requests.length;
 await field(company(),'meta');await wait(()=>pending.length);const meta=take();assert.equal(rows().length,0);assert(!document.querySelector('.no-results'));await answer(meta,{...summary('meta','all',[]),reports:0,contributors:0,companies:[]});await wait(()=>document.querySelector('.no-results'));assert.equal(rows().length,0);assert(document.querySelector('.continue-link').disabled);
 await click(button('Clear company'));await wait(()=>document.querySelectorAll('[data-paper-id]').length>0);assert(!nav.readLibraryRoute(location.hash).filters.company);assert.equal(requests.length,reads+2,'Returning to the ordinary library requests only its existing sidebar evidence');
 if(pending.length)await answer(take(),summary('google','365'));
 assert.deepEqual(failures,[]);assert.equal(JSON.stringify(fixture.practice.saved),snapshot);assert.deepEqual(calls.runs,[]);assert.deepEqual(calls.submits,[]);assert.equal(calls.flushes,0);assert(requests.every(row=>row.url.startsWith('community/interviews/summary?')));
 console.log('PASS actual App company library journey: one shared summary, ranked exercise rows, local search without refetch, exact practice/auth returns and scoped navigator, period cancellation and malformed retry, owner revalidation, genuine empty and explicit clear recovery; unchanged editor DOM/source/caret/scroll, drafts and zero judge/social writes');
 // Ordinary exercise browsing is a real App route, including practice and history returns.
 await route('#/?view=modules&page=2');await wait(()=>rows().length===20&&document.querySelector('.library-view-switch'));assert.equal(document.querySelector('.library-view-switch [aria-pressed=true]').textContent,'Exercises');assert.equal(nav.libraryReturnHref(rows()[0].getAttribute('href')),location.hash);
 const secondPageOrigin=location.hash,firstOnPage=catalog.problems.find(p=>p.id===rows()[0].dataset.moduleId);await click(rows()[0]);await wait(()=>editor()&&document.querySelector('.module-picker-trigger'));assert.equal(document.querySelector('.workspace-toolbar .back-link').getAttribute('href'),secondPageOrigin);assert.equal(document.querySelector('.workspace-toolbar .back-link').textContent,'Exercise list');await click(document.querySelector('.module-picker-trigger'));assert.equal(document.querySelector('.module-picker').getAttribute('aria-label'),'Current exercise list');assert.deepEqual([...document.querySelectorAll('.module-picker-links>a')].map(node=>nav.routePath(decodeProblemHash(node.getAttribute('href')))),[...catalog.problems].sort((a,b)=>a.number-b.number).map(p=>String(p.number)),'The picker retains the entire filtered exercise scope across page boundaries');await click(document.querySelector('[aria-label="Next module"]'));await wait(()=>nav.routePath(decodeProblemHash(location.hash))===String(firstOnPage.number+1));assert.equal(nav.libraryReturnHref(location.hash),secondPageOrigin);await click(document.querySelector('.workspace-toolbar .back-link'));await wait(()=>rows().length===20);assert.equal(location.hash,secondPageOrigin);assert.equal(rows()[0].dataset.moduleId,firstOnPage.id);
 await route('#/?q='+b.number+'&view=modules&page_size=50');await wait(()=>rows().length===1);assert.equal(rows()[0].dataset.moduleId,b.id);const directOrigin=location.hash;await click(rows()[0]);await wait(()=>editor());assert.equal(nav.libraryReturnHref(location.hash),directOrigin);assert.equal(editor().value,'PRIVATE COMPANY PRACTICE 😀');assert.equal(nav.readLibraryRoute(nav.libraryReturnHref(location.hash)).filters.view,'modules');const directEditor=editor();directEditor.setSelectionRange(2,9,'backward');directEditor.scrollTop=41;directEditor.scrollLeft=15;await route(nav.withLibraryContext(b.number,'#/?state=todo&view=modules'));await wait(()=>document.querySelector('.module-batch-unavailable'));assert(!document.querySelector('.module-picker-trigger'));assert.equal(editor(),directEditor);assert.deepEqual([directEditor.value,directEditor.selectionStart,directEditor.selectionEnd,directEditor.selectionDirection,directEditor.scrollTop,directEditor.scrollLeft],['PRIVATE COMPANY PRACTICE 😀',2,9,'backward',41,15]);fixture.practice.ready=false;await act(async()=>notify());await wait(()=>document.querySelector('.module-batch-unavailable')?.textContent.includes('Checking list progress'));assert.equal(editor(),directEditor);fixture.practice.ready=true;await act(async()=>notify());await route(nav.withLibraryContext(b.number,directOrigin));await wait(()=>document.querySelector('.module-picker-trigger'));await click(document.querySelector('.workspace-toolbar .back-link'));await wait(()=>rows().length===1&&document.querySelector('.library-view-switch'));
 await field(document.querySelector('[aria-label="Search papers or modules"]'),'');await wait(()=>rows().length===50);const beforeViewHistory=history.length,paperView=[...document.querySelectorAll('.library-view-switch button')].find(node=>node.textContent==='Papers');await click(paperView);await wait(()=>document.querySelectorAll('[data-paper-id]').length===50);assert.equal(history.length,beforeViewHistory+1);assert.equal(nav.readLibraryRoute(location.hash).filters.page_size,50);assert.equal(nav.readLibraryRoute(location.hash).filters.view,undefined);await act(async()=>history.back());await wait(()=>rows().length===50);assert.equal(nav.readLibraryRoute(location.hash).filters.view,'modules');assert.equal(nav.readLibraryRoute(location.hash).filters.page_size,50);assert.equal(JSON.stringify(fixture.practice.saved),snapshot);assert.deepEqual(calls.runs,[]);assert.deepEqual(calls.submits,[]);assert.equal(calls.flushes,0);assert.equal(sessionStorage.getItem('private-company-sentinel'),storage);
 console.log('PASS actual App direct exercise journey: canonical paged exercise route, numeric question lookup, exact practice/source return, retained page size, whole-scope navigator and next-module order across pages, direct toolbar return, native view-switch history Back and unchanged saved source/drafts with zero judge or publication');
 if(process.env.PAPERLEET_LIBRARY_COMPANY_HTTP){
  await act(async()=>root.unmount());root=undefined;pending.length=0;
  const proof=JSON.parse(process.env.PAPERLEET_LIBRARY_COMPANY_HTTP),reads=[],fetchHttp=originals.get('fetch').value;
  fixture.request=async(url,body,signal)=>{assert.equal(body,undefined);const response=await fetchHttp(proof.url+'/api/'+url,{signal});assert.equal(response.status,200);const value=await response.json();reads.push({url,value});return value;};
  history.replaceState(null,'','#/?view=modules&company=google&company_days=90');await render();await wait(()=>rows().length===2);assert.deepEqual(new Set(rows().map(row=>row.dataset.moduleId)),new Set([b.id,c.id]));assert(reads.some(row=>row.value.reports===1&&row.value.contributors===1));assert.equal(reads.length,1);
  assert(!document.querySelector('.library-view-switch'));assert.equal(nav.readLibraryRoute(location.hash).filters.view,'modules');const nativeCompany=company(),nativePeriod=period();await field(nativePeriod,'all');await wait(()=>rows().length===3);assert.deepEqual(new Set(rows().map(row=>row.dataset.moduleId)),new Set([a.id,b.id,c.id]));assert.equal(company(),nativeCompany);assert.equal(period(),nativePeriod);assert.equal(nativeCompany.value,'google');assert(reads.some(row=>row.url.includes('days=all')&&row.value.reports===2&&row.value.contributors===2));
  await field(nativeCompany,'meta');await wait(()=>rows().length===1);assert.equal(rows()[0].dataset.moduleId,a.id);assert.equal(period().value,'all');assert.equal(reads.length,3);assert.equal(nav.readLibraryRoute(location.hash).filters.view,'modules');assert.equal(calls.flushes,0);assert.equal(JSON.stringify(fixture.practice.saved),snapshot);
  console.log('PASS company library HTTP: actual App controls and shared sidebar consume anonymous JudgeHandler/InterviewService/SQLite summaries; company isolation, publication-period inclusion, honest counts and no draft/judge/social writes');
 }

}finally{if(root)await act(async()=>root.unmount());await server.close();rmSync(cache,{recursive:true,force:true});dom.window.close();for(const [key,descriptor]of originals){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
