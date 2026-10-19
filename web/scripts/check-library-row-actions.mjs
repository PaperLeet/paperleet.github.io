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
  './social': `import React,{useSyncExternalStore} from 'react';
    export {profileHref} from '/src/social.tsx';
    export const AccountProvider=({children})=>children;
    export const AccountNav=()=>null;export const socialApi=(...args)=>globalThis.__paperleetWorkspaceActionTest.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{constructor(message,status,code){super(message);this.status=status;this.code=code;}};export const SocialNotice=({error,children})=>React.createElement('p',null,error||children);export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const PersonLink=({user})=>React.createElement('a',{href:'#/u/'+user.username},user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const TestAccountBadge=()=>null;
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

const cache=mkdtempSync(path.join(os.tmpdir(),'paperleet-library-row-actions-'));
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

const api=[],sourcePending=[],bookmarks=new Map();let loseSave=true,bookmarkRevision=0;
fixture.account.capabilities={problem_bookmarks:true,study_lists:false};fixture.practice.owns=owner=>owner===fixture.practice.owner;
const originalOwner=fixture.account.user.id,originalSaved=fixture.practice.saved;
const bookmarkSnapshot=owner=>({owner_id:owner,revision:bookmarkRevision,items:[...(bookmarks.get(owner)||[])].map(problem_id=>({problem_id,created_at:'2026-10-18T00:00:00Z'}))});
fixture.request=(url,body,signal)=>{
 const owner=fixture.account.user?.id||'guest';api.push({url,body:body?structuredClone(body):undefined,signal,owner});
 if(url.startsWith('community/interviews/summary?')){assert.equal(body,undefined);const p=new URLSearchParams(url.split('?')[1]);return Promise.resolve(summary(p.get('company')||'google',p.get('days')||'365'));}
 if(url.startsWith('community/interviews?')){assert.equal(body,undefined);return new Promise((resolve,reject)=>sourcePending.push({url,signal,resolve,reject}));}
 assert.equal(url,'account/problem-bookmarks');if(!body)return Promise.resolve(bookmarkSnapshot(owner));
 assert.equal(body.expected_user_id,owner);if(!bookmarks.has(owner))bookmarks.set(owner,new Set());body.active?bookmarks.get(owner).add(body.problem_id):bookmarks.get(owner).delete(body.problem_id);bookmarkRevision++;
 if(loseSave){loseSave=false;return Promise.reject(Error('Response lost after bookmark commit'));}
 return Promise.resolve({...bookmarkSnapshot(owner),problem_id:body.problem_id,active:body.active});
};
const post=(n,concept=b.id,company='google')=>{const id=n.toString(16).padStart(32,'0');return {id,title:'Public implementation reflection '+n,body:'A clear public approach. ![no remote image](https://example.test/private.png) <img src=x onerror=bad()>',anonymous:true,category:'career',author:{id:'anonymous:'+id,username:'',display_name:'Anonymous',anonymous:true,avatar_color:'#48766b',affiliation:{kind:'institution',domain:'research.edu',name:'研究机构',name_en:'Research Institute'}},created_at:'2026-10-17T00:00:00Z',updated_at:'2026-10-17T00:00:00Z',counts:{likes:0,replies:0,reposts:0},viewer:{liked:false},interview:{company,role:'ml',stage:'technical',outcome:'unknown',interview_month:'',location:'Remote',questions:[concept]}};};
const feed=(items,total=items.length)=>({items,total,limit:3,offset:0,owner_id:null});
let root;
try{
 const {createRoot}=await import('react-dom/client'),h=React.createElement,{default:App}=await server.ssrLoadModule('/src/App.tsx');
 const nav=await server.ssrLoadModule('/src/libraryNavigation.ts'),interview=await server.ssrLoadModule('/src/interviewNavigation.ts'),source=await server.ssrLoadModule('/src/interviewSourcePreviewModel.ts'),account=await server.ssrLoadModule('/src/accountNavigation.ts'),{decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 const active=nav.libraryHref(undefined,{...nav.emptyLibraryFilters(),company:'google',company_days:'90',query:'注意力 & 100% _ 😀',sort:'title'}),sourceHref=interview.interviewHref({...interview.emptyInterviewFilters(),company:'google',days:'90',concept:b.id});
 assert.equal(source.libraryInterviewPreviewScope(active,sourceHref),active);
 for(const bad of [interview.interviewHref({...interview.emptyInterviewFilters(),company:'meta',days:'90',concept:b.id}),interview.interviewHref({...interview.emptyInterviewFilters(),company:'google',days:'30',concept:b.id}),interview.interviewHref({...interview.emptyInterviewFilters(),company:'google',days:'90',concept:b.id,role:'ml'}),interview.interviewHref({...interview.emptyInterviewFilters(),company:'google',days:'90',concept:b.id,saved:true})])assert.equal(source.libraryInterviewPreviewScope(active,bad),undefined);
 for(const href of ['https://evil.test/','#/22','#/interviews?company=google','#/?company=google&company_days=30','#/?company=google&company=meta','#/?company=google&code=private'])assert.equal(source.libraryInterviewPreviewScope(href,sourceHref),undefined);
 console.log('PASS library row evidence scope: exact closed company/publication cohort, canonical Unicode library views and rejection of mismatched, private, recursive and non-library previews');
 const wait=async fn=>{for(let i=0;i<240;i++){if(fn())return;await act(async()=>new Promise(r=>setTimeout(r,10)));}throw Error('Row actions timed out: '+document.body.textContent.slice(-1700));};
 const render=()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(h(React.Fragment,null,h('textarea',{id:'private-row-sentinel',defaultValue:'UNCHANGED PRIVATE ROW DRAFT 😀'}),h(App)));});
 const rows=()=>[...document.querySelectorAll('[data-module-id]')],row=()=>document.querySelector('[data-module-id="'+b.id+'"]')?.parentElement,tools=()=>row()?.querySelector('.library-company-row-tools'),toggle=()=>tools()?.querySelector('.library-company-row-actions>button'),panel=()=>tools()?.querySelector('section'),favorite=()=>tools()?.querySelector('[data-favorite-control]'),search=()=>document.querySelector('[aria-label="Search papers or modules"]');
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});},field=async(node,value)=>{assert(node);await act(async()=>{node.focus();Object.getOwnPropertyDescriptor(node.tagName==='SELECT'?window.HTMLSelectElement.prototype:window.HTMLInputElement.prototype,'value').set.call(node,value);node.dispatchEvent(new window.Event(node.tagName==='SELECT'?'change':'input',{bubbles:true}));});};
 const take=()=>{const value=sourcePending.shift();assert(value,'Expected a pending source read');return value;},answer=(request,value)=>act(async()=>request.resolve(value)),sourceReads=()=>api.filter(r=>r.url.startsWith('community/interviews?')),savedReads=()=>api.filter(r=>r.url==='account/problem-bookmarks'&&!r.body),savedWrites=()=>api.filter(r=>r.body);
 await render();await wait(()=>document.querySelectorAll('.library-company-row-tools').length===3&&favorite()&&!favorite().disabled);
 assert.equal(sourceReads().length,0);assert.equal(savedReads().length,1,'All row controls share one account snapshot');assert.equal(savedWrites().length,0);
 const code=document.querySelector('#private-row-sentinel');code.focus();code.setSelectionRange(2,14,'backward');code.scrollTop=51;code.scrollLeft=12;const link=row().querySelector('[data-module-id]'),rowToggle=toggle();assert.equal(link.tagName,'A');assert(!link.querySelector('button,a'));assert(tools().textContent.includes('3 contributors · 4 reports'));
 const start=location.href;await click(rowToggle);await wait(()=>sourcePending.length);const first=take(),p=new URLSearchParams(first.url.split('?')[1]);assert.equal(p.get('company'),'google');assert.equal(p.get('days'),'90');assert.equal(p.get('problem_id'),b.id);assert.equal(p.get('limit'),'3');assert(!p.has('q')&&!p.has('saved')&&!p.has('difficulty')&&!p.has('state'));code.focus();await answer(first,feed([post(1),post(2)],4));await wait(()=>panel().querySelectorAll('.interview-source-item').length===2);assert.equal(document.activeElement,code);assert.equal(location.href,start);assert(panel().textContent.includes('Research Institute'));assert.equal(panel().querySelectorAll('.interview-source-excerpt img,.anonymous-author a').length,0);assert([...panel().querySelectorAll('.interview-meta-tags a')].every(link=>link.target==='_blank'&&link.rel.includes('noopener')));const full=panel().querySelector('.interview-source-title');assert.equal(full.target,'_blank');assert(full.rel.includes('noopener')&&full.rel.includes('noreferrer'));assert.equal(interview.readInterviewRoute((await server.ssrLoadModule('/src/communityNavigation.ts')).communityPostReturn(full.getAttribute('href'),post(1).id)).company,'google');
 await field(search(),b.translations?.en?.title||b.title);assert.equal(toggle(),rowToggle);assert.equal(row().querySelector('[data-module-id]'),link);assert(!panel().hidden);assert.equal(sourceReads().length,1,'Local exercise search does not refetch the public report text');
 const localHref=location.href;await act(async()=>{localStorage.setItem('paperleet.locale','zh');window.dispatchEvent(new window.StorageEvent('storage',{key:'paperleet.locale'}));});assert.equal(toggle(),rowToggle);assert(!panel().hidden);assert.equal(sourceReads().length,1);assert.equal(location.href,localHref);await act(async()=>{localStorage.setItem('paperleet.locale','en');window.dispatchEvent(new window.StorageEvent('storage',{key:'paperleet.locale'}));});
 await click(panel().querySelector('.interview-source-panel-heading button'));const bad=take();await answer(bad,feed([post(3,b.id,'meta')]));await wait(()=>panel().querySelector('[role="alert"]'));assert.equal(panel().querySelectorAll('.interview-source-item').length,0);await click(panel().querySelector('.interview-source-panel-heading button'));await answer(take(),feed([]));await wait(()=>panel().textContent.includes('No experiences match this preview'));assert.equal(rows().length,1);assert.equal(savedWrites().length,0);
 panel().querySelector('.interview-source-panel-heading button').focus();await act(async()=>panel().dispatchEvent(new window.KeyboardEvent('keydown',{key:'Escape',ctrlKey:true,bubbles:true,cancelable:true})));assert(!panel().hidden);await act(async()=>panel().dispatchEvent(new window.KeyboardEvent('keydown',{key:'Escape',isComposing:true,bubbles:true,cancelable:true})));assert(!panel().hidden);await act(async()=>panel().dispatchEvent(new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true})));assert(panel().hidden);assert.equal(document.activeElement,rowToggle);assert.equal(location.href,localHref);
 console.log('PASS actual App row source reading: visible contributor/report counts, original anchor DOM, explicit bounded preview reads, anonymous affiliation and literal excerpts, new-tab source links, filter/locale/disclosure continuity, malformed/empty recovery, modifier-safe Escape and no private or judge writes');
 await click(favorite());await wait(()=>favorite()?.getAttribute('aria-label')==='Retry saving problem');assert.equal(favorite().getAttribute('aria-pressed'),'false');assert.equal(savedWrites().length,1);await click(favorite());await wait(()=>favorite()?.getAttribute('aria-pressed')==='true');assert.equal(savedWrites().length,2);assert(savedWrites().every(r=>r.url==='account/problem-bookmarks'&&r.body.active===true&&r.body.problem_id===b.id&&r.body.expected_user_id===originalOwner));
 await click(toggle());await wait(()=>sourcePending.length);const retired=take();await act(async()=>{fixture.account.loading=true;notify();});assert(retired.signal.aborted);fixture.account.user={id:'row-bob',username:'row_bob',display_name:'Bob'};fixture.practice.owner='user:row-bob';fixture.practice.saved={};await act(async()=>{fixture.account.loading=false;notify();});await wait(()=>favorite()&&!favorite().disabled);await answer(retired,feed([post(8)]));assert(panel().hidden);assert.equal(favorite().getAttribute('aria-pressed'),'false');assert.equal(savedReads().length,2);assert.equal(sourceReads().length,4);assert(!tools().textContent.includes('reflection 8'));assert.equal(JSON.stringify(originalSaved),snapshot);
 fixture.account.capabilities.problem_bookmarks=false;await act(async()=>notify());await wait(()=>!favorite());const reads=api.length;await act(async()=>window.dispatchEvent(new window.Event('focus')));assert.equal(api.length,reads);
 fixture.account.user=null;fixture.practice.owner='anonymous';fixture.account.capabilities.problem_bookmarks=true;await act(async()=>notify());await wait(()=>tools()?.querySelector('.problem-favorite-button[href]'));const login=tools().querySelector('.problem-favorite-button[href]').getAttribute('href');assert.equal(account.accountReturnHref(decodeProblemHash(login)),nav.libraryHref(nav.readLibraryRoute(location.hash).collectionId,nav.readLibraryRoute(location.hash).filters));assert.equal(savedReads().length,2);assert.equal(location.href,localHref);
 assert.deepEqual([code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],['UNCHANGED PRIVATE ROW DRAFT 😀',2,14,'backward',51,12]);assert.deepEqual(calls.runs,[]);assert.deepEqual(calls.submits,[]);assert.equal(calls.flushes,0);assert.deepEqual(failures,[]);
 console.log('PASS row account actions: coalesced private snapshot, explicit save with lost-response same-direction retry, truthful acknowledgement, account/capability retirement and late source abort, exact guest sign-in return, stable source/caret/scroll and unchanged practice grades');
 // Exercise real company pages with the actual App history and row enhancements.
 await act(async()=>root.unmount());root=undefined;
 const pagedIds=[...new Set([b.id,c.id,a.id,...fixture.catalog.problems.map(row=>row.id)])].slice(0,24),originalRequest=fixture.request;
 fixture.account.user={id:'row-paged',username:'row_paged',display_name:'Paged'};fixture.account.capabilities={problem_bookmarks:true,study_lists:false};fixture.practice.owner='user:row-paged';fixture.practice.saved={};
 fixture.request=(url,body,signal)=>{if(url.startsWith('community/interviews/summary?')){api.push({url,signal,owner:'row-paged'});return Promise.resolve({days:'90',basis:'published-community-reports',generated_at:'2026-10-18T00:00:00Z',reports:24,contributors:1,companies:[{company:'google',reports:24,contributors:1}],questions:pagedIds.map(problem_id=>({problem_id,reports:1,contributors:1,companies:[{company:'google',problem_id,reports:1,contributors:1,latest:'2026-10-18T00:00:00Z'}]}))});}return originalRequest(url,body,signal);};
 history.replaceState(null,'','#/?company=google&company_days=90');await render();await wait(()=>rows().length===20&&favorite()&&!favorite().disabled);
 const initialLink=row().querySelector('[data-module-id]'),initialTools=tools(),initialToggle=toggle(),pagingReads=api.length,pagingWrites=savedWrites().length;
 await click(initialToggle);await wait(()=>sourcePending.length);await answer(take(),feed([post(20)],1));await wait(()=>panel().querySelector('.interview-source-title'));
 const sourceCount=sourceReads().length,summaryCount=api.filter(row=>row.url.startsWith('community/interviews/summary?')).length,privateCount=savedReads().length;
 const beforePageHistory=history.length;
 await click(document.querySelector('.library-pagination:not(.library-pagination-compact) [aria-label="Next page"]'));await wait(()=>rows().length===4&&document.querySelectorAll('.library-company-row-tools').length===4);
 assert.equal(nav.readLibraryRoute(location.hash).filters.page,2);assert(history.length>beforePageHistory);assert.equal(initialLink.isConnected,false);assert(!document.querySelector('.library-load-more'));assert.equal(sourceReads().length,sourceCount);assert.equal(savedReads().length,privateCount);assert.equal(api.filter(row=>row.url.startsWith('community/interviews/summary?')).length,summaryCount);assert.equal(savedWrites().length,pagingWrites);assert.deepEqual(rows().map(node=>node.dataset.moduleId),pagedIds.slice(20));for(const node of rows())assert.equal(nav.readLibraryRoute(nav.libraryReturnHref(node.getAttribute('href'))).filters.page,2);
 await click(document.querySelector('.library-pagination:not(.library-pagination-compact) [aria-label="Page 1"]'));await wait(()=>rows().length===20&&toggle());assert.notEqual(row().querySelector('[data-module-id]'),initialLink);assert(panel().hidden);assert.equal(sourceReads().length,sourceCount);assert.equal(savedReads().length,privateCount);assert.equal(api.filter(row=>row.url.startsWith('community/interviews/summary?')).length,summaryCount);assert.equal(savedWrites().length,pagingWrites);
 await act(async()=>history.back());await wait(()=>rows().length===4&&nav.readLibraryRoute(location.hash).filters.page===2);await act(async()=>history.forward());await wait(()=>rows().length===20&&!nav.readLibraryRoute(location.hash).filters.page);assert(panel().hidden);assert.equal(sourceReads().length,sourceCount);assert.equal(savedWrites().length,pagingWrites);
 console.log('PASS company row pagination: real App twenty-row and final four-row pages, pushed browser back/forward and exact page-aware practice returns, retired off-page source DOM and safe closed return, complete company scope with no passive source, summary, private snapshot or write requests');
 if(process.env.PAPERLEET_LIBRARY_ROWS_HTTP){
  await act(async()=>root.unmount());root=undefined;
  const proof=JSON.parse(process.env.PAPERLEET_LIBRARY_ROWS_HTTP),transport=[],fetchHttp=originals.get('fetch').value;
  fixture.account.user=proof.user;fixture.account.capabilities={problem_bookmarks:true,study_lists:false};fixture.practice.owner='user:'+proof.user.id;fixture.practice.saved=originalSaved;
  fixture.request=async(url,body,signal)=>{const own=url==='account/problem-bookmarks';assert(!body||own);const response=await fetchHttp(proof.url+'/api/'+url,{method:body?'POST':'GET',signal,headers:{'Content-Type':'application/json',...(own?{Cookie:proof.cookie,Origin:proof.url}:{})},...(body?{body:JSON.stringify(body)}:{})});const value=await response.json();transport.push({url,body:body?structuredClone(body):undefined,status:response.status,value});if(!response.ok)throw Error(value.error||'HTTP fixture failed');return value;};
  history.replaceState(null,'','#/?company=google&company_days=90');await render();await wait(()=>document.querySelectorAll('.library-company-row-tools').length===2&&favorite()&&!favorite().disabled);assert(tools().textContent.includes('1 contributor · 1 report'));assert.equal(transport.filter(row=>row.url==='account/problem-bookmarks').length,1);assert(!transport.some(row=>row.url.startsWith('community/interviews?')));
  const initial=location.href;await click(toggle());await wait(()=>panel().querySelector('.interview-source-title')?.textContent.includes(proof.recentTitle));assert.equal(panel().querySelector('.interview-source-title').target,'_blank');assert.equal(location.href,initial);
  await click(favorite());await wait(()=>favorite()?.getAttribute('aria-pressed')==='true');assert.equal(location.href,initial);await click(favorite());await wait(()=>favorite()?.getAttribute('aria-pressed')==='false');assert.equal(location.href,initial);
  const nativePeriod=document.querySelector('[aria-label="Library report publication period"]');await field(nativePeriod,'all');await wait(()=>document.querySelectorAll('.library-company-row-tools').length===3);assert.equal(document.querySelector('[aria-label="Library report publication period"]'),nativePeriod);
  const oldRow=document.querySelector('[data-module-id="'+a.id+'"]').parentElement,oldTools=oldRow.querySelector('.library-company-row-tools');await click(oldTools.querySelector('.library-company-row-actions>button'));await wait(()=>oldTools.querySelector('.interview-source-title')?.textContent.includes(proof.oldTitle));assert(oldTools.textContent.includes('Anonymous'));assert(oldTools.textContent.includes('Test Institute'));assert.equal(oldTools.querySelectorAll('.anonymous-author a').length,0);
  const stats={summaries:transport.filter(row=>row.url.startsWith('community/interviews/summary?')).length,source_reads:transport.filter(row=>row.url.startsWith('community/interviews?')).length,bookmark_reads:transport.filter(row=>row.url==='account/problem-bookmarks'&&!row.body).length,bookmark_writes:transport.filter(row=>row.body).length};assert.deepEqual(stats,{summaries:2,source_reads:2,bookmark_reads:1,bookmark_writes:2});assert.deepEqual(transport.filter(row=>row.body).map(row=>row.body.active),[true,false]);assert(transport.filter(row=>row.body).every(row=>row.body.expected_user_id===proof.user.id&&row.body.problem_id===b.id));assert.equal(JSON.stringify(originalSaved),snapshot);assert.equal(calls.flushes,0);assert.deepEqual(calls.runs,[]);assert.deepEqual(calls.submits,[]);
  console.log('PASS library row HTTP: actual App source previews and period changes use anonymous JudgeHandler/InterviewService data, old anonymous affiliation is preserved, one private snapshot supports explicit save/remove acknowledgements, exact company sources and unchanged practice data');
  console.log('ROW_ACTIONS_HTTP '+JSON.stringify(stats));
 }

}finally{if(root)await act(async()=>root.unmount());await server.close();rmSync(cache,{recursive:true,force:true});dom.window.close();for(const [key,descriptor]of originals){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}
