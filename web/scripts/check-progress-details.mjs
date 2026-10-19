import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

// Render the actual App and split panels. Isolate account/runtime boundaries so
// these interactions never contact a server, run Python, or use real drafts.
const dom = new JSDOM('<!doctype html><div id="root"></div>', {url:'https://paperleet.test/#/1'});
const names = ['window','document','navigator','HTMLElement','CustomEvent','localStorage','location','requestAnimationFrame','cancelAnimationFrame','ResizeObserver','fetch','IS_REACT_ACT_ENVIRONMENT','__paperleetWorkspaceActionTest'];
const originals = new Map(names.map(name => [name,Object.getOwnPropertyDescriptor(globalThis,name)]));
for (const name of ['window','document','navigator','HTMLElement','CustomEvent','localStorage','location']) Object.defineProperty(globalThis,name,{value:dom.window[name],configurable:true,writable:true});
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
    run(entry,source){calls.runs.push({id:entry.id,source});this.active=true;this.problem=entry.id;this.stage='loading';this.progress={stage:'loading',step:'python'};delete this.syntaxErrors[entry.id];this.reasons[entry.id]='';this.diagnostics[entry.id]='';notify();return new Promise((resolve,reject)=>{runResolve=resolve;runReject=reject;});},
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
  './PaperViews': `export const PaperLibrary=()=>null;export const paperModules=()=>[];`,
  './ReportView': `import React from 'react';import Report from '/src/ReportView.tsx?fixture_actual=1';globalThis.__paperleetWorkspaceActionTest.calls.detailsLoads++;export default function FixtureReport(props){if(globalThis.__paperleetWorkspaceActionTest.failDetails)throw Error('Fixture detail failure');return React.createElement(Report,props);}`,
  './deferredPages': `import React,{useEffect} from 'react';
    import {deferredFeature} from '/src/deferredFeature.tsx';
    import {ProblemOverviewFallback} from '/src/ProblemOverviewFallback.tsx';
    import Overview from '/src/ProblemOverview.tsx';
    export const ProblemOverview=deferredFeature(()=>new Promise((resolve,reject)=>{const f=globalThis.__paperleetWorkspaceActionTest;f.resolveOverview=()=>resolve({default:Overview});f.rejectOverview=()=>reject(Error('Synthetic optional details failure'));}),['',''],{loading:p=>React.createElement(ProblemOverviewFallback,p),failure:(p,retry)=>React.createElement(ProblemOverviewFallback,{...p,retry})});
    export const PaperDetail=props=>React.createElement('section',{'data-test-paper-detail':true,'data-practice-ids':props.practiceProblems?.map(p=>p.id).join(','),'data-return-to':props.returnTo,'data-reload-blocked':String(!!props.recovery?.blocked)});
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
mocks['./social']=mocks['./social'].replace("export const AccountNav=()=>null;","export function AccountNav(){const value=useI18n();globalThis.__paperleetWorkspaceActionTest.setTestLocale=value.setLocale;return null;}").replace("import {useSyncExternalStore} from 'react';","import {useSyncExternalStore} from 'react';import {useI18n} from '/src/i18n.tsx';");
mocks['./progressFeature']=`import {deferredFeature as real} from '/src/deferredFeature.tsx';import Details from '/src/BrowserRunProgressDetails.tsx';export function deferredFeature(load,label,options){return real(()=>new Promise((resolve,reject)=>{const f=globalThis.__paperleetWorkspaceActionTest;(f.progressLoads??=[]).push({resolve:()=>resolve({default:Details}),reject:()=>reject(Error('Synthetic progress-detail bundle failure'))});}),label,options);}`;
mocks['/src/ReportView.tsx']=mocks['./ReportView'];
const server = await createServer({
  server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},
  define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_HOSTED_APP_AVAILABLE':JSON.stringify('1')},
  plugins:[{name:'workspace-action-fixtures',enforce:'pre',resolveId(source,importer){if(importer?.endsWith('/src/BrowserRunProgress.tsx')&&source==='./deferredFeature')return '\0workspace-actions:./progressFeature';if(importer?.endsWith('/src/ProblemOverview.tsx')&&source==='./deferredPages')return '\0overview-tags';if(importer?.endsWith('/src/WorkspaceTestResults.tsx')&&source==='./ReportView')return '\0workspace-actions:'+source;if(importer?.endsWith('/src/App.tsx')&&source in mocks)return '\0workspace-actions:'+source;},load(id){if(id==='\0overview-tags')return 'export const ProblemCompanyTags=()=>null;';if(id.startsWith('\0workspace-actions:'))return mocks[id.slice('\0workspace-actions:'.length)];}}],
});
let root;
try{
 const {createRoot}=await import('react-dom/client'),{default:App}=await server.ssrLoadModule('/src/App.tsx');
 const find=selector=>document.querySelector(selector),action=text=>[...document.querySelectorAll('.workspace-run-controls button')].find(button=>button.textContent.trim()===text);
 const click=async node=>{assert(node,'Missing progress control');await act(async()=>{node.focus();node.click();});};
 const change=async callback=>act(async()=>{callback();notify();});
 const settle=async predicate=>{for(let index=0;index<200&&!predicate();index++)await act(async()=>new Promise(resolve=>setTimeout(resolve,15)));assert(predicate(),'Progress fixture did not settle');};
 const stop=()=>find('.browser-progress-stop'),retry=()=>find('.browser-progress-retry'),fallback=()=>find('.browser-progress-fallback');
 root=createRoot(document.querySelector('#root'));await act(async()=>root.render(React.createElement(App)));await settle(()=>find('#problem-editor-panel textarea')&&find('.editor-font-size-select'));
 const editor=find('#problem-editor-panel textarea'),source='def attention(x):\n    # 私有实现 😀\n    return x\n';
 async function edit(value){await act(async()=>{Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set.call(editor,value);editor.dispatchEvent(new window.Event('input',{bubbles:true}));});}
 await edit(source);editor.setSelectionRange(4,18,'backward');editor.scrollTop=73;editor.scrollLeft=19;
 const href=location.href;
 const assertEditor=expected=>{assert(find('#problem-editor-panel textarea')===editor);assert.equal(calls.editorMounts,1);assert.equal(editor.value,expected);assert.equal(location.href,href);};
 const finish=async report=>act(async()=>{fixture.browser.active=false;fixture.browser.problem=undefined;fixture.browser.reports[problem.id]=report;notify();runResolve({type:'result',report});runResolve=undefined;runReject=undefined;});
 // Loading the optional body never gates stopping the actual run.
 await click(action('Run tests'));assert.equal(calls.runs.length,1);const immediateStop=stop();assert(immediateStop,'Stop is immediately usable while detailed progress code loads');await settle(()=>fixture.progressLoads?.length===1);assert.equal(stop(),immediateStop,'The stop control stays mounted when optional progress code arrives');assert(fallback());assert.equal(find('.browser-run-progress [role=status]').textContent,'Preparing the runtime…');assert.equal(find('.browser-run-progress progress').getAttribute('value'),null);assert(!find('.browser-run-progress ol'));
 const firstStop=stop(),firstLoad=fixture.progressLoads[0];assert(firstStop);await click(firstStop);await settle(()=>!action('Run tests').disabled);assert.equal(calls.stops,1);assert(!find('.browser-run-progress'));assert(find('#problem-results-panel').textContent.includes('Browser run stopped'));assert.equal(calls.submits.length,0);assert.equal(calls.appliedReports.length,0);
 editor.focus();await act(async()=>firstLoad.resolve());assert(!find('.browser-run-progress'));assert(document.activeElement===editor);assertEditor(source);
 // Counts/phases/locale keep updating under a pending or failed detail view.
 await click(action('Run tests'));const stableStop=stop();await act(async()=>stableStop.focus());assert.equal(fixture.progressLoads.length,2);
 await change(()=>{fixture.browser.stage='calibrating';fixture.browser.progress={stage:'calibrating'};});assert(find('.browser-run-progress [role=status]').textContent.includes('Validating judging…'));assert.equal(find('.browser-run-progress progress').getAttribute('value'),null);
 await change(()=>{fixture.browser.stage='calibrating';fixture.browser.progress={stage:'calibrating',step:'reference',completed:0,total:7};});assert(stop()===stableStop);assert(document.activeElement===stableStop);assert(find('.browser-run-progress [role=status]').textContent.includes('Checking reference tests · 0 / 7'));
 await change(()=>{fixture.browser.progress={stage:'calibrating',step:'mutants',completed:1,total:3};});assert(find('.browser-run-progress [role=status]').textContent.includes('Checking mutation cases · 1 / 3'));assert.equal(find('.browser-run-progress progress').getAttribute('value'),'1');assert.equal(find('.browser-run-progress progress').getAttribute('max'),'3');
 await act(async()=>fixture.setTestLocale('zh'));assert(stop()===stableStop);assert(document.activeElement===stableStop);assert.equal(stableStop.textContent,'停止运行');assert(find('.browser-run-progress [role=status]').textContent.includes('校验错误实现 · 1 / 3'));await act(async()=>fixture.setTestLocale('en'));
 await act(async()=>fixture.progressLoads[1].reject());assert(fallback());assert(find('.browser-progress-recovery[role=alert]').textContent.includes('Judging is still running'));assert(stop()===stableStop);assert(document.activeElement===stableStop);assert.equal(calls.runs.length,2);assert.equal(calls.stops,1);assert.equal(calls.submits.length,0);
 await change(()=>{fixture.browser.stage='running';fixture.browser.progress={stage:'running'};});assert.equal(find('.browser-run-progress [role=status]').textContent,'Running tests…');assert(!find('.browser-run-progress [role=status]').textContent.includes('completed'));assert.equal(find('.browser-run-progress progress').getAttribute('value'),null);
 await change(()=>{fixture.browser.stage='running';fixture.browser.progress={stage:'running',step:'tests',completed:2,total:7,activity:{index:2,test:'test_<img src=x>',layer:'T1',why:'Public <script> text stays literal.'}};});assert(find('.browser-run-progress [role=status]').textContent.includes('Tests completed · 2 / 7'));assert(!find('.browser-run-progress').textContent.includes('tests passed'));
 await click(retry());assert.equal(fixture.progressLoads.length,3);assert(stop()===stableStop);assert(document.activeElement===stableStop);assert(!retry());assert.equal(calls.runs.length,2,'Retrying presentation cannot restart judging');
 const edited=source+'# keep editing during the same run\n';await edit(edited);editor.focus();editor.setSelectionRange(8,22,'backward');editor.scrollTop=91;editor.scrollLeft=27;
 await act(async()=>fixture.progressLoads[2].resolve());assert(!fallback());assert(find('.browser-run-progress ol'));assert(stop()===stableStop);assert(document.activeElement===editor);assert(find('.browser-run-activity').textContent.includes('test_<img src=x>'));assert(!find('.browser-run-activity img,.browser-run-activity script'));assertEditor(edited);assert.deepEqual([editor.selectionStart,editor.selectionEnd,editor.selectionDirection,editor.scrollTop,editor.scrollLeft],[8,22,'backward',91,27]);assert.deepEqual(calls.runs.at(-1),{id:problem.id,source},'A presentation retry keeps the original source snapshot');
 await click(stableStop);await settle(()=>!action('Run tests').disabled);assert.equal(calls.stops,2);assert(!find('.browser-run-progress'));assertEditor(edited);
 // A presentation failure cannot reject a completed report or send a server fallback.
 await click(action('Run tests'));assert.equal(fixture.progressLoads.length,4);await act(async()=>fixture.progressLoads[3].reject());assert(retry());
 const report={problem:problem.id,source_sha256:createHash('sha256').update(edited).digest('hex'),runtime:'browser-numpy-wasm',summary:{T0:[1,1]},tests:[{id:'output',layer:'T0',passed:true}]};
 await finish(report);await settle(()=>find('.workspace-test-results > .result-head h3')?.textContent==='Tests passed');assert(!find('.browser-run-progress'));assert.equal(calls.runs.length,3);assert.equal(calls.submits.length,0);assert.equal(calls.stops,2);assertEditor(edited);
 // Late bodies belong to their original mounted run/account, including an owner cycle.
 await click(action('Run tests'));assert.equal(fixture.progressLoads.length,5);const oldOwnerLoad=fixture.progressLoads[4];
 await change(()=>{fixture.browser.active=false;fixture.browser.problem=undefined;fixture.browser.reports={};fixture.browser.syntaxErrors={};fixture.browser.calibrations={};fixture.browser.diagnostics={};fixture.practice.owner='user:other';fixture.practice.saved={};fixture.account.user={id:'other',username:'other',display_name:'Other'};runReject(new DOMException('Stopped','AbortError'));runResolve=undefined;runReject=undefined;});assert(!find('.browser-run-progress'));assert(!document.body.textContent.includes('私有实现'));
 await change(()=>{fixture.practice.owner='user:fixture-user';fixture.account.user={id:'fixture-user',username:'fixture-user',display_name:'Fixture'};});await settle(()=>!!find('#problem-editor-panel textarea')&&find('.editor-font-size-select')&&!action('Run tests').disabled);const returnedEditor=find('#problem-editor-panel textarea');
 await click(action('Run tests'));assert.equal(fixture.progressLoads.length,6);const currentStop=stop();returnedEditor.focus();await act(async()=>oldOwnerLoad.resolve());assert(fallback(),'The old owner/run body cannot resolve the current lazy attempt');assert(stop()===currentStop);assert(document.activeElement===returnedEditor);assert.equal(calls.runs.length,5);
 await click(currentStop);await settle(()=>!action('Run tests').disabled);await act(async()=>fixture.progressLoads[5].reject());assert(!find('.browser-run-progress'));assert.equal(calls.submits.length,0);
 console.log('PASS deferred progress details: actual App persistent stop control under loading/failure/retry/body arrival, live phase/reference/mutation/completion counters and locale, no invented pass/percent claims, stopping and late-loader retirement, original run/source and exact editor/caret/both-scroll/focus retention, escaped activity metadata, independent authoritative result and zero server fallback, run/owner-ABA isolation');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const[name,value]of originals){if(value)Object.defineProperty(globalThis,name,value);else delete globalThis[name];}}
