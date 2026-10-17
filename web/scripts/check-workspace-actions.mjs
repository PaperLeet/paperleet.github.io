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
mocks['/src/ReportView.tsx']=mocks['./ReportView'];
const server = await createServer({
  server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},
  define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_HOSTED_APP_AVAILABLE':JSON.stringify('1')},
  plugins:[{name:'workspace-action-fixtures',enforce:'pre',resolveId(source,importer){if(importer?.endsWith('/src/ProblemOverview.tsx')&&source==='./deferredPages')return '\0overview-tags';if(importer?.endsWith('/src/WorkspaceTestResults.tsx')&&source==='./ReportView')return '\0workspace-actions:'+source;if(importer?.endsWith('/src/App.tsx')&&source in mocks)return '\0workspace-actions:'+source;},load(id){if(id==='\0overview-tags')return 'export const ProblemCompanyTags=()=>null;';if(id.startsWith('\0workspace-actions:'))return mocks[id.slice('\0workspace-actions:'.length)];}}],
});
let root;
try {
  const {createRoot}=await import('react-dom/client');
  const {default:App}=await server.ssrLoadModule('/src/App.tsx');
  root=createRoot(document.querySelector('#root'));
  const find=selector=>document.querySelector(selector);
  const action=text=>[...document.querySelectorAll('.workspace-run-controls button')].find(button=>button.textContent.trim()===text);
  const click=async element=>{assert(element,'Expected an interactive control');await act(async()=>element.dispatchEvent(new window.MouseEvent('click',{bubbles:true,cancelable:true})));};
  const change=async mutate=>act(async()=>{mutate();notify();});
  const settle=async predicate=>{for(let i=0;i<150&&!predicate();i++)await act(async()=>new Promise(resolve=>setTimeout(resolve,20)));assert(predicate(),'Workspace did not settle; detail loads: '+calls.detailsLoads+'; '+document.querySelector('.workspace-test-details')?.textContent.slice(0,700));};
  const resultSelected=()=>find('#reading-tab-result')?.getAttribute('aria-selected')==='true';
  const assertInline=message=>{assert.equal(location.hash,'#/1',message);assert.equal(find('#problem-editor-panel textarea'),editor,'Result changes must preserve the mounted editor');assert.equal(calls.editorMounts,1);};
  const finishBrowser=async outcome=>act(async()=>{fixture.browser.active=false;fixture.browser.problem=undefined;if(outcome.report)fixture.browser.reports={[problem.id]:outcome.report};if(outcome.failure)fixture.browser.syntaxErrors={[problem.id]:outcome.failure};notify();runResolve(outcome);runResolve=undefined;runReject=undefined;});
  const finishSubmission=async report=>act(async()=>{fixture.judge.active=false;fixture.judge.pending=undefined;fixture.judge.job={status:'completed'};fixture.onServerReport(report,fixture.practice.owner);notify();});
  await act(async()=>root.render(React.createElement(App)));
  await settle(()=>find('#problem-editor-panel textarea'));
  const editor=find('#problem-editor-panel textarea');editor.getClientRects=()=>[new window.DOMRect(0,0,500,600)];
  find('#problem-results-panel').getClientRects=()=>find('#problem-results-panel').hidden||find('.app').dataset.focusMode||fixture.narrow&&find('.app').dataset.mobilePane==='code'?[]:[new window.DOMRect(0,0,500,600)];
  assert(find('.workspace-toolbar').contains(action('Run tests')));
  assert(find('.workspace-toolbar').contains(action('Submit')));
  assert.equal(find('#reading-tab-description').getAttribute('aria-selected'),'true');
  assert.equal(find('#reading-tab-result').textContent.trim(),'Result');
  assert.equal(find('#reading-tab-result').getAttribute('aria-controls'),'problem-results-panel');
  assert.equal(find('#problem-results-panel').getAttribute('aria-labelledby'),'reading-tab-result');
  assert(find('#problem-description-panel').contains(find('#problem-results-panel')),'Results belong to the left reading pane');
  assert(!find('#problem-editor-panel').contains(find('#problem-results-panel')),'The code pane has no result drawer');
  assert.equal(document.querySelectorAll('#problem-results-panel').length,1,'There is one result surface');
  assert.equal(find('.editor-bottom'),null,'The actions must not leave a second footer below the editor');
  assert.equal(document.querySelectorAll('.mobile-workspace-switch').length,1,'Mobile has one unambiguous pane switch');
  assert(find('#problem-results-panel').hidden);
  assert.equal(find('.editor-split'),null,'The editor no longer reserves a second result pane');
  assert.equal(find('.results-pane-toggle'),null);
  assert(!action('Run tests').disabled&&!action('Submit').disabled);

  // Switching reading tabs must not remount the editor or lose its
  // unsent selection/scroll. The real Python editor has separate editor tests.
  const source='def attention(x):\n    return x * 2 # private draft\n';
  await change(()=>{fixture.practice.saved={[problem.id]:{code:source}};});
  editor.focus();editor.setSelectionRange(4,13);editor.scrollTop=41;
  await click(find('.workspace-focus-toggle'));
  await settle(()=>find('.app').dataset.focusMode==='true');
  const layoutTrigger=find('.workspace-layout-trigger');await act(async()=>layoutTrigger.focus());await click(layoutTrigger);
  const readingPreset=find('.workspace-layout-options button');await act(async()=>readingPreset.focus());
  for(const options of [{isComposing:true},{keyCode:229},{repeat:true},{altKey:true},{ctrlKey:true},{metaKey:true},{shiftKey:true}]){
    const ignored=new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true,...options});await act(async()=>readingPreset.dispatchEvent(ignored));
    assert(!ignored.defaultPrevented&&find('.workspace-layout-popover'));assert.equal(find('.app').dataset.focusMode,'true','IME, repeated and modified keys retain both local layout and outer focus mode');
  }
  const layoutEscape=new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true});await act(async()=>readingPreset.dispatchEvent(layoutEscape));
  assert(layoutEscape.defaultPrevented&&!find('.workspace-layout-popover'));assert.equal(find('.app').dataset.focusMode,'true','Escape closes the layout panel while retaining the actual App focus mode');assert.equal(document.activeElement,layoutTrigger);
  await act(async()=>layoutTrigger.dispatchEvent(new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true})));
  assert.equal(find('.app').dataset.focusMode,undefined,'The next Escape retains the workspace exit-focus shortcut');assertInline('Local layout shortcuts preserve the problem route and mounted editor');
  assert.equal(editor.value,source);assert.equal(editor.selectionStart,4);assert.equal(editor.selectionEnd,13);assert.equal(editor.scrollTop,41);
  assert(find('#problem-reading-content article').textContent.includes('Return the attention output.'),'The real App renders its statement while optional overview loading is unresolved');
  assert(find('.problem-heading h1').textContent.includes('Attention'),'Problem identity remains visible while details load');
  await act(async()=>fixture.rejectOverview());
  await settle(()=>[...document.querySelectorAll('button')].some(b=>b.textContent==='Retry problem details'));
  assert(editor===find('#problem-editor-panel textarea')&&editor.value===source&&editor.selectionStart===4&&editor.scrollTop===41,'An optional details failure preserves the current editor and draft');
  assert(find('#problem-reading-content article').textContent.includes('Return the attention output.'),'Details failure leaves the statement available');
  await click([...document.querySelectorAll('button')].find(b=>b.textContent==='Retry problem details'));
  await act(async()=>fixture.resolveOverview());await settle(()=>find('.module-scope-card'));
  assert(editor===find('#problem-editor-panel textarea')&&editor.value===source&&editor.selectionStart===4&&editor.scrollTop===41,'Details retry does not remount or overwrite code');

  assert(calls.focusRequests.some(request=>!request.location&&typeof request.claimFocus==='function'),'The real App returns to code through a guarded selection-preserving request');
  const normalFrame=globalThis.requestAnimationFrame,queuedFrames=[];
  globalThis.requestAnimationFrame=callback=>{queuedFrames.push(callback);return queuedFrames.length;};
  try {
    await click(find('#reading-tab-explanation'));await click(find('#reading-tab-description'));
    const reader=find('#problem-reading-content');reader.scrollTop=331;find('#problem-results-panel').scrollTop=332;
    await act(async()=>document.dispatchEvent(new window.WheelEvent('wheel',{bubbles:true})));
    await act(async()=>{for(const callback of queuedFrames.splice(0))callback(0);});
    assert.equal(reader.scrollTop,331);assert.equal(find('#problem-results-panel').scrollTop,332,'A newer user scroll cancels queued reader resets');
    await click(find('#reading-tab-explanation'));await click(find('#reading-tab-description'));
    const latestReader=find('#problem-reading-content');let resets=0;latestReader.scrollTo=function(x,y){resets++;this.scrollLeft=x;this.scrollTop=y;};
    await act(async()=>{for(const callback of queuedFrames.splice(0))callback(0);});
    assert.equal(resets,1,'Only the latest tab request resets the current reader');assert(editor===find('#problem-editor-panel textarea')&&editor.value===source);
  } finally {globalThis.requestAnimationFrame=normalFrame;}

  await click(action('Run tests'));
  assert.deepEqual(calls.runs,[{id:problem.id,source}]);
  assert(resultSelected(),'Starting tests selects Result beside Description');
  assert(!find('#problem-results-panel').hidden&&find('#problem-reading-content').hidden);
  assert(find('#problem-results-panel .browser-run-progress'),'Browser progress appears inside the left Result tab');
  assertInline('Starting browser tests keeps the current problem URL');
  assert(action('Run tests').disabled&&action('Submit').disabled,'The moved controls still prevent overlapping runs');
  assert(find('.workspace-toolbar').contains(action('Stop')));
  await click(action('Stop'));
  assert.equal(calls.stops,1);
  await settle(()=>!action('Run tests').disabled);
  assert(resultSelected(),'Stopping tests keeps feedback in the selected Result tab');
  assert(find('#problem-results-panel').textContent.includes('Browser run stopped'),'The browser AbortError is shown inline');
  assert.equal(find('.workspace-result-heading > span').textContent,'Stopped','An intentional stop is distinct from a failed run');
  assert.equal(find('#problem-results-panel [role="alert"]'),null,'Stopping is a neutral status, not an error announcement');
  assert(find('.run-recovery button[data-retry-run]'),'The result panel offers a direct retry after stopping');
  assert.equal(calls.appliedReports.length,0,'Stopping cannot fabricate a completed report');
  assertInline('Stopping tests keeps the current problem URL');

  const report={problem:problem.id,source_sha256:createHash('sha256').update(source).digest('hex'),runtime:'browser-numpy-wasm',summary:{T0:[1,1]},tests:[{id:'output',layer:'T0',passed:true}]};
  await click(find('.run-recovery button[data-retry-run]'));
  assert.equal(calls.runs.length,2,'One retry starts exactly one browser run');
  assert.deepEqual(calls.runs.at(-1),{id:problem.id,source},'Retry executes the current draft in the same browser mode');
  assert.equal(find('.run-recovery'),null,'A new run clears the stopped recovery card');
  await finishBrowser({type:'result',report});
  await settle(()=>find('.workspace-test-results > .result-head h3')?.textContent==='Tests passed');
  assert(resultSelected()&&!find('#problem-results-panel').hidden,'Successful tests stay in the left Result tab');
  assertInline('Successful browser tests keep the current problem URL');
  assert.equal(find('.console'),null,'The obsolete second result summary is removed');
  const detail=find('.workspace-test-details');assert(detail&&!detail.open);
  assert.equal(find('#problem-results-panel .report-view'),null,'Full reports wait for explicit detail expansion');
  assert.equal(calls.detailsLoads,0,'Full diagnostic code is not loaded before it is requested');
  fixture.failDetails=true;
  const consoleError=console.error;console.error=()=>{};
  try{await act(async()=>{detail.open=true;detail.dispatchEvent(new window.Event('toggle'));});await settle(()=>find('.workspace-test-detail-error'));}finally{console.error=consoleError;}
  assert.equal(find('#problem-editor-panel textarea'),editor);assert.equal(editor.value,source);assert.equal(document.querySelectorAll('.workspace-test-results').length,1);
  fixture.failDetails=false;await click(find('.workspace-test-detail-error button'));
  await settle(()=>find('#problem-results-panel .report-view'));
  assert.equal(document.activeElement,detail.querySelector('summary'),'Retry keeps focus in the result disclosure');
  const detailedReport=find('#problem-results-panel .report-view');
  const passedFilter=find('.test-outcome-switch').querySelectorAll('button')[2];await click(passedFilter);
  assert.equal(passedFilter.getAttribute('aria-pressed'),'true');
  await click(find('#reading-tab-description'));
  assert(find('#problem-results-panel').hidden&&!find('#problem-reading-content').hidden);
  await click(find('#reading-tab-result'));
  assert(!find('#problem-results-panel').hidden);
  assert.equal(find('#problem-editor-panel textarea'),editor);
  assert.equal(editor.value,source);
  assert.equal(editor.selectionStart,4);assert.equal(editor.selectionEnd,13);assert.equal(editor.scrollTop,41);
  assert.equal(find('#problem-results-panel .report-view'),detailedReport,'Switching reading tabs preserves the existing report and selected test filter');
  assert.equal(passedFilter.getAttribute('aria-pressed'),'true');
  assertInline('Changing reading tabs never navigates away from the problem');

  const failed={...report,impl:'/tmp/learner.py',summary:{T0:[0,1]},tests:[{id:'output',layer:'T0',passed:false,message:'Traceback (most recent call last):\n  File "/tmp/learner.py", line 2, in attention\n    raise ValueError("fix this")\nValueError: fix this'}]};
  await click(action('Run tests'));
  await finishBrowser({type:'result',report:failed});
  await settle(()=>find('.report-workflow .source-location-button'));
  assert(find('#problem-results-panel .feedback-diagnostic').textContent.includes('fix this'));
  assert(resultSelected(),'Failed browser tests appear in the same Result tab');
  assertInline('Failed browser tests keep the current problem URL');
  await click(find('.report-workflow .source-location-button'));
  assert(!find('#problem-results-panel').hidden,'Locating an error keeps feedback visible beside the editor');
  assert.equal(calls.focusRequests.at(-1).location.line,2);
  assert.equal(calls.focusRequests.at(-1).source,source);
  assert.equal(find('#problem-editor-panel textarea'),editor);
  const editedSource=source.replace('* 2','* 3');await change(()=>{fixture.practice.saved={[problem.id]:{code:editedSource}};});
  await settle(()=>!find('.report-workflow .source-location-button'));
  assert.equal(editor.value,editedSource,'Returning from results never restores an older run over new code');
  await click(find('#reading-tab-description'));
  await change(()=>{fixture.practice.error='Draft synchronization failed';});
  assert(resultSelected()&&!find('#problem-results-panel').hidden,'A new save error is visible in the left Result tab');
  assert(find('#problem-results-panel').textContent.includes('Draft synchronization failed'));
  await change(()=>{fixture.practice.error='';fixture.browser.reports={};});
  await click(find('#reading-tab-description'));
  await click(action('Submit'));
  assert.deepEqual(calls.submits,[{id:problem.id,source:editedSource}],'Top Submit still sends the current code to server judging');
  assert.equal(calls.flushes,1,'Hosted drafts flush before server submission');
  assert.equal(calls.runs.length,3,'Submit must not silently become a browser-only run');
  assert(resultSelected()&&!find('#problem-results-panel').hidden,'Submit selects the same Result tab');
  assert(find('#problem-results-panel .judge-queue-notice').textContent.includes('Queued · 2 jobs ahead'));
  assertInline('Submitting server tests keeps the current problem URL');
  assert(action('Run tests').disabled&&action('Submit').disabled,'Queued server work prevents overlapping runs');
  await change(()=>{fixture.judge.job={status:'running'};});
  assert(find('#problem-results-panel .judge-queue-notice').textContent.includes('Running tests…'));
  const accepted={...report,runtime:'server',source_sha256:createHash('sha256').update(editedSource).digest('hex'),run_id:'a'.repeat(32)};
  await click(find('#reading-tab-description'));
  await act(async()=>new Promise(resolve=>setTimeout(resolve,0)));
  const chosenDescription=find('#problem-reading-content'),chosenArticle=chosenDescription.querySelector('article');chosenDescription.scrollTop=247;find('#reading-tab-description').focus();
  await finishSubmission(accepted);
  await settle(()=>find('.workspace-test-results > .result-head h3')?.textContent==='Accepted');
  assert(!resultSelected()&&find('#reading-tab-description').getAttribute('aria-selected')==='true','Server completion respects the reader tab selected after submission');
  assert(chosenDescription===find('#problem-reading-content')&&chosenArticle===find('#problem-reading-content article'));assert.equal(chosenDescription.scrollTop,247);assert(document.activeElement===find('#reading-tab-description'));
  const completeNotice=find('.submission-completion-notice');assert(completeNotice&&completeNotice.textContent.includes('Submission complete'));assert.equal(completeNotice.querySelector('a').getAttribute('href'),'#/submission/'+accepted.run_id);
  const runsBeforeNotice=calls.runs.length;const noticeChord=new window.KeyboardEvent('keydown',{key:'Enter',ctrlKey:true,bubbles:true,cancelable:true});await act(async()=>{completeNotice.querySelector('a').focus();completeNotice.querySelector('a').dispatchEvent(noticeChord);});assert(!noticeChord.defaultPrevented&&calls.runs.length===runsBeforeNotice,'The completion link keeps native modified activation rather than rerunning code');
  await click(completeNotice.querySelector('a'));await settle(()=>document.activeElement===find('#reading-tab-result'));assert(resultSelected(),'The explicit notification action opens the completed authoritative report');assert(!find('.submission-completion-notice'));assert.equal(location.hash,'#/1');
  assert.equal(calls.appliedReports.at(-1).report,accepted,'Server completion still saves the authoritative report');
  assertInline('Successful server submission must not redirect to a submission page');
  assert.equal(editor.value,editedSource);

  await click(find('#reading-tab-description'));
  await click(action('Submit'));
  const serverFailed={...accepted,run_id:'b'.repeat(32),summary:{T0:[0,1]},tests:[{id:'output',layer:'T0',passed:false,message:'Server output mismatch'}]};
  const pendingEdit=editedSource.replace('* 3','* 4');
  await change(()=>{fixture.practice.saved={[problem.id]:{...fixture.practice.saved[problem.id],code:pendingEdit}};});
  editor.setSelectionRange(6,15);editor.scrollTop=63;
  find('#problem-results-panel').scrollTop=390;
  await finishSubmission(serverFailed);
  assert.equal(find('#problem-results-panel').scrollTop,390,'Completion updates a visible report without resetting its reader scroll');
  await settle(()=>find('#problem-results-panel .feedback-diagnostic')?.textContent.includes('Server output mismatch'));
  assert(resultSelected(),'Failed server completion appears in the inline Result tab');
  assertInline('Failed server submission must not redirect to a submission page');
  assert.equal(editor.value,pendingEdit,'A queued result cannot overwrite edits made after submitting');
  assert.equal(editor.selectionStart,6);assert.equal(editor.selectionEnd,15);assert.equal(editor.scrollTop,63);
  assert(!find('.submission-completion-notice'),'Visible Result completion does not add a redundant notification');
  // Stored Result selection does not mean its panel is visible on a narrow code pane.
  fixture.narrow=true;await click(action('Submit'));await click(find('.mobile-workspace-switch button:last-child'));editor.focus();editor.setSelectionRange(5,14,'backward');editor.scrollTop=71;
  const mobileAccepted={...accepted,source_sha256:createHash('sha256').update(pendingEdit).digest('hex'),run_id:'c'.repeat(32)};
  await finishSubmission(mobileAccepted);await settle(()=>find('.submission-completion-notice'));
  assert.equal(find('.app').dataset.mobilePane,'code');assert(document.activeElement===editor,'Completion arrival never takes focus');assert(resultSelected(),'The stored Result selection is retained while the code pane stays active');
  const dismiss=find('.submission-completion-notice button');await act(async()=>dismiss.focus());await click(dismiss);await settle(()=>document.activeElement===editor);assert(!find('.submission-completion-notice'));assert.equal(editor.selectionStart,5);assert.equal(editor.selectionEnd,14);assert.equal(editor.scrollTop,71);
  await click(action('Submit'));await click(find('.mobile-workspace-switch button:last-child'));editor.focus();
  const mobileAgain={...mobileAccepted,run_id:'d'.repeat(32)};await finishSubmission(mobileAgain);await settle(()=>find('.submission-completion-notice'));
  await change(()=>{fixture.practice.saved={[problem.id]:{...fixture.practice.saved[problem.id],report:{...mobileAgain,run_id:'e'.repeat(32)}}};});
  const savedAction=find('.submission-completion-notice a');assert.equal(savedAction.textContent,'View saved result');let nativeSaved;
  document.addEventListener('click',event=>{if(event.target===savedAction){nativeSaved={prevented:event.defaultPrevented,href:savedAction.getAttribute('href')};event.preventDefault();}},{once:true});
  await click(savedAction);assert.deepEqual(nativeSaved,{prevented:false,href:'#/submission/'+mobileAgain.run_id},'A newer report keeps the old completion action bound to its immutable original record');assert.equal(find('.app').dataset.mobilePane,'code');assert.equal(location.hash,'#/1');
  await click(action('Submit'));await click(find('.mobile-workspace-switch button:last-child'));editor.focus();const newestCompletion={...mobileAccepted,run_id:'f'.repeat(32)};await finishSubmission(newestCompletion);await settle(()=>find('.submission-completion-notice'));
  await change(()=>{fixture.account.loading=true;});assert(!find('.submission-completion-notice'),'Account checking immediately hides the private result action');await change(()=>{fixture.account.loading=false;});await settle(()=>find('.submission-completion-notice'));
  const inlineAction=find('.submission-completion-notice a');await act(async()=>inlineAction.focus());await click(inlineAction);await settle(()=>document.activeElement===find('#reading-tab-result'));assert.equal(find('.app').dataset.mobilePane,'read');assert(!find('.submission-completion-notice'));assert(editor===find('#problem-editor-panel textarea')&&editor.value===pendingEdit);assert.equal(editor.selectionStart,5);assert.equal(editor.selectionEnd,14);assert.equal(location.hash,'#/1');fixture.narrow=false;


  await click(find('#reading-tab-description'));
  await click(action('Submit'));
  await change(()=>{fixture.judge.active=false;fixture.judge.pending=undefined;fixture.judge.job=undefined;fixture.judge.error='Server submission could not start. Your draft is preserved; try again.';});
  assert(resultSelected()&&find('#problem-results-panel [role="alert"]').textContent.includes('Server submission could not start'));
  assertInline('Server errors stay on the current problem');
  const freshBrowserReport={...report,source_sha256:createHash('sha256').update(pendingEdit).digest('hex')};
  await act(async()=>new Promise(resolve=>setTimeout(resolve,0)));
  find('#problem-results-panel').scrollTop=380;
  await click(action('Run tests'));
  await settle(()=>find('#problem-results-panel').scrollTop===0);
  assert(fixture.judge.error,'The old server failure remains in its hook for this regression case');
  assert(!find('#problem-results-panel').textContent.includes('Server submission could not start'),'Starting browser tests hides the old server error');
  await finishBrowser({type:'result',report:freshBrowserReport});
  await settle(()=>find('.workspace-test-results > .result-head h3')?.textContent==='Tests passed');
  assert(!find('#problem-results-panel').textContent.includes('Server submission could not start'),'A stale server failure cannot replace a fresh browser pass');
  assertInline('Browser success after a server error remains inline');

  await click(action('Submit'));
  await click(find('#problem-results-panel .judge-queue-notice button'));
  assert.equal(fixture.judge.job.status,'cancelled');
  assert(find('#problem-results-panel').textContent.includes('This submission was not executed'),'Cancelling the current server queue shows its outcome inline');
  await click(action('Run tests'));
  await finishBrowser({type:'result',report:freshBrowserReport});
  await settle(()=>find('.workspace-test-results > .result-head h3')?.textContent==='Tests passed');
  assert.equal(fixture.judge.job.status,'cancelled','The old cancellation remains in its hook for this regression case');
  assert(!find('#problem-results-panel').textContent.includes('This submission was not executed'),'A stale server cancellation cannot replace a fresh browser pass');
  assertInline('Browser success after cancelling a server submission remains inline');
  await change(()=>{fixture.judge.error='';fixture.judge.submitError='Fixture submit connection failed';});
  await click(find('#reading-tab-description'));
  await click(action('Submit'));
  assert(resultSelected()&&find('#problem-results-panel').textContent.includes('Fixture submit connection failed'),'Thrown submit failures are displayed inline');
  assertInline('Thrown submit errors stay on the current problem');
  await change(()=>{fixture.judge.submitError='';});

  const chord=new window.KeyboardEvent('keydown',{key:'Enter',ctrlKey:true,bubbles:true,cancelable:true});
  await act(async()=>editor.dispatchEvent(chord));
  assert(chord.defaultPrevented,'The editor run shortcut is still intercepted before inserting a newline');
  assert.equal(calls.runs.length,6);assert(resultSelected());
  const duplicate=new window.KeyboardEvent('keydown',{key:'Enter',metaKey:true,bubbles:true,cancelable:true});
  await act(async()=>editor.dispatchEvent(duplicate));
  assert.equal(calls.runs.length,6,'The shortcut cannot start another active run');
  const syntax={source:pendingEdit,diagnostic:{kind:'SyntaxError',message:'Expected a colon',line:1,column:17}};
  await finishBrowser({type:'syntax_error',failure:syntax});
  await settle(()=>!action('Run tests').disabled);
  assert(resultSelected()&&find('#problem-results-panel').textContent.includes('Expected a colon'),'Syntax errors use the same inline result surface');
  assertInline('Syntax errors keep the current problem URL');
  assert.equal(editor.value,pendingEdit);

  // Recovery is an action on the current editor, never a replay of an old source snapshot.
  await click(action('Run tests'));await click(action('Stop'));
  await settle(()=>find('.run-recovery button[data-retry-run]'));
  assert.equal(find('.workspace-test-results'),null,'A stopped attempt must not present an earlier report as its new verdict');
  assert.equal(find('.judge-runtime-details'),null,'A stopped attempt must not present calibration or timing from the previous result');
  assert.equal(find('#problem-results-panel [role="alert"]'),null);
  const recoverySource=pendingEdit+'# changed after stopping\n';
  await change(()=>{fixture.practice.saved={[problem.id]:{...fixture.practice.saved[problem.id],code:recoverySource}};});
  editor.setSelectionRange(7,12);editor.scrollTop=52;
  await click(find('.run-recovery .button.subtle'));
  assert.equal(document.activeElement,editor);assert.equal(editor.value,recoverySource);
  assert.equal(editor.selectionStart,7);assert.equal(editor.selectionEnd,12);assert.equal(editor.scrollTop,52);
  const retryButton=find('.run-recovery button[data-retry-run]'),beforeRetry=calls.runs.length;
  await click(retryButton);await click(retryButton);
  assert.equal(calls.runs.length,beforeRetry+1,'A repeated retry cannot start overlapping browser work');
  assert.equal(calls.runs.at(-1).source,recoverySource,'Retry runs edits made after stopping');
  await change(()=>{fixture.browser.reasons[problem.id]='runtime_failed';});
  await finishBrowser({type:'unsupported',reason:'runtime_failed'});
  await settle(()=>find('.run-recovery button[data-retry-run]'));
  assert.equal(find('.run-recovery button[data-retry-run]').textContent.trim(),'Run current code again','A browser failure keeps the browser retry mode');

  await change(()=>{fixture.judge.submitError='Temporary server connection failure';});
  await click(action('Submit'));
  await settle(()=>find('.run-recovery button[data-retry-run]')?.textContent.includes('Resubmit'));
  assert(find('#problem-results-panel [role="alert"]').textContent.includes('Temporary server connection failure'));
  await change(()=>{fixture.account.health.judge.ready=false;});
  assert(find('.run-recovery button[data-retry-run]').disabled,'An unavailable server cannot be retried just because browser execution is available');
  await change(()=>{fixture.account.health.judge.ready=true;fixture.judge.reloadRequired=true;});
  assert(find('.run-recovery button[data-retry-run]').disabled,'A required reload must not be bypassed through recovery');
  await change(()=>{fixture.judge.reloadRequired=false;fixture.judge.submitError='';});
  await change(()=>{fixture.judge.submitException=new DOMException('Server connection aborted','AbortError');});
  await click(find('.run-recovery button[data-retry-run]'));
  await settle(()=>find('#problem-results-panel [role="alert"]')?.textContent.includes('Server connection aborted'));
  assert.equal(find('.workspace-result-heading > span').textContent,'Run incomplete','A server AbortError is a connection failure, not an intentional browser stop');
  await change(()=>{fixture.judge.submitException=undefined;});
  const runsBeforeServerRetry=calls.runs.length,submitsBeforeRetry=calls.submits.length;
  await click(find('.run-recovery button[data-retry-run]'));
  assert.equal(calls.submits.length,submitsBeforeRetry+1);assert.equal(calls.runs.length,runsBeforeServerRetry,'A server retry must not silently become a browser result');
  assert.equal(calls.submits.at(-1).source,recoverySource);
  await finishSubmission({...accepted,source_sha256:createHash('sha256').update(recoverySource).digest('hex')});
  await settle(()=>!action('Run tests').disabled);
  await click(action('Run tests'));await click(action('Stop'));
  await settle(()=>find('.run-recovery'));

  await change(()=>{fixture.account.user=undefined;fixture.practice.owner='anonymous';});
  assert.equal(find('.run-recovery'),null,'A new workspace owner cannot inherit the previous owner’s stopped state or retry action');
  assert(action('Sign in to submit'),'Guests keep a sign-in entry in the same top action area');
  assert.equal(action('Submit'),undefined);
  assert(!action('Run tests').disabled,'Guest browser practice stays available');
  // Remount the real App at a known practice URL without a hashchange: the
  // workspace hook must validate the catalog entry and re-arm its persisted origin.
  await act(async()=>root.unmount());root=undefined;
  fixture.account.user={id:'fixture-user',username:'fixture-user',display_name:'Fixture'};fixture.practice.owner='user:fixture-user';
  const reading=await server.ssrLoadModule('/src/readingReturn.ts'),interviewNav=await server.ssrLoadModule('/src/interviewNavigation.ts'),libraryNav=await server.ssrLoadModule('/src/libraryNavigation.ts');
  const origin=interviewNav.interviewHref({...interviewNav.emptyInterviewFilters(),company:'google',saved:true,q:'mask & gradient',page:2});
  reading.rememberReadingReturn({owner:'account:fixture-user',href:origin,post:'a'.repeat(32),link:'practice',practice:problem.id,top:240,focus:true});
  const practiceHref=libraryNav.withLibraryContext(problem.number,origin);window.history.replaceState(null,'',practiceHref);assert.equal(reading.currentReadingReturn().phase,'opening');
  await act(async()=>{root=createRoot(document.querySelector('#root'));root.render(React.createElement(App));});
  await settle(()=>find('#problem-editor-panel textarea')&&reading.currentReadingReturn()?.phase==='reading');
  await change(()=>{fixture.judge.job={problem:problem.id,status:'cancelled'};});
  assert.equal(find('.run-recovery button[data-retry-run]').textContent.trim(),'Resubmit current code','A reconnected cancelled server job retains server retry mode without an in-memory latestRun');
  await change(()=>{fixture.judge.job=undefined;});
  assert.equal(location.hash,practiceHref);assert.equal(find('#problem-editor-panel textarea').value,fixture.practice.saved[problem.id].code,'Re-arming a reading return never changes learner code');reading.discardReadingReturn();
  await act(async()=>root.unmount());root=undefined;
  const savedOrigin='#/favorites?'+new URLSearchParams({q:'Attention',state:'todo'});
  reading.rememberReadingReturn({owner:'account:fixture-user',href:savedOrigin,post:problem.id,link:'practice',practice:problem.id,top:240,focus:true});
  window.history.replaceState(null,'',libraryNav.withLibraryContext(problem.number,savedOrigin));
  await act(async()=>{root=createRoot(document.querySelector('#root'));root.render(React.createElement(App));});
  await settle(()=>find('#problem-editor-panel textarea'));
  assert.equal(reading.currentReadingReturn()?.phase,'reading','The real workspace re-arms favorite practice returns on numeric URLs');
  assert.equal(find('.workspace-toolbar .back-link').getAttribute('href'),savedOrigin);
  assert.equal(find('[data-test-favorite]').getAttribute('data-test-favorite'),problem.id,'The practice favorite uses the canonical ID, even on numeric routes');
  assert.equal(find('#problem-editor-panel textarea').value,fixture.practice.saved[problem.id].code);
  await act(async()=>{location.hash=savedOrigin;window.dispatchEvent(new window.HashChangeEvent('hashchange'));});
  await settle(()=>find('[data-test-favorites]'));
  assert.equal(find('[data-test-favorites]').getAttribute('data-test-favorites'),savedOrigin.slice(2),'The real App routes favorites with their exact filters');
  await click(find('[data-test-favorite-filters]'));assert.equal(location.hash,'#/favorites?q=LoRA&state=todo');assert.equal(find('[data-test-favorites]').getAttribute('data-test-favorites'),'favorites?q=LoRA&state=todo','Favorite filter changes must update the App route as well as browser history');reading.discardReadingReturn();
  await act(async()=>root.unmount());root=undefined;
  const libraryPaper={id:'attention',number:1,title:'Attention Is All You Need',short_title:'Transformer',authors:'Ashish Vaswani',venue:'NeurIPS',year:2017,url:'https://arxiv.org/abs/1706.03762',arxiv:'1706.03762',tags:[],category:'architecture',summary:'',summary_en:'',module_ids:[problem.id]};
  // The real App labels public post origins and preserves the editor when only that origin changes.
  for(const locale of ['en','zh']){
    localStorage.setItem('paperleet.locale',locale);fixture.catalog={problems:[problem],papers:[libraryPaper],categories:[]};fixture.practice.ready=true;fixture.practice.loading=false;fixture.learning.ready=true;
    const sourcePost='#/community/'+'a'.repeat(32),otherPost='#/community/'+'b'.repeat(32);
    reading.rememberReadingReturn({owner:'account:fixture-user',href:origin,post:'a'.repeat(32),link:'title',top:240,focus:true});window.history.replaceState(null,'',libraryNav.withLibraryContext(problem.number,sourcePost));
    await act(async()=>{root=createRoot(document.querySelector('#root'));root.render(React.createElement(App));});await settle(()=>find('#problem-editor-panel textarea')&&reading.currentReadingReturn()?.phase==='reading');
    assert.equal(find('.workspace-toolbar .back-link').getAttribute('href'),sourcePost);assert.equal(find('.workspace-toolbar .back-link').textContent,locale==='en'?'Back to source post':'返回来源帖子');
    assert.equal(find('.topbar nav>a').getAttribute('href'),'#','Papers navigation must not be redirected to the source post');
    const retainedEditor=find('#problem-editor-panel textarea'),retainedCode=retainedEditor.value;retainedEditor.setSelectionRange(2,7,'backward');retainedEditor.scrollTop=23;retainedEditor.scrollLeft=9;
    await act(async()=>{location.hash=libraryNav.withLibraryContext(problem.number,otherPost);window.dispatchEvent(new window.HashChangeEvent('hashchange'));});await settle(()=>find('.workspace-toolbar .back-link')?.getAttribute('href')===otherPost);
    assert(find('#problem-editor-panel textarea')===retainedEditor);assert.equal(retainedEditor.value,retainedCode);assert.deepEqual([retainedEditor.selectionStart,retainedEditor.selectionEnd,retainedEditor.selectionDirection,retainedEditor.scrollTop,retainedEditor.scrollLeft],[2,7,'backward',23,9]);assert.equal(reading.currentReadingReturn(),undefined);
    await act(async()=>{location.hash=libraryNav.withLibraryContext(999,otherPost);window.dispatchEvent(new window.HashChangeEvent('hashchange'));});await settle(()=>find('.route-not-found a'));
    assert.equal(find('.route-not-found a').getAttribute('href'),otherPost);assert.equal(find('.route-not-found a').textContent,locale==='en'?'Back to source post':'返回来源帖子');
    await act(async()=>{location.hash=libraryNav.withLibraryContext('papers/attention',otherPost);window.dispatchEvent(new window.HashChangeEvent('hashchange'));});await settle(()=>find('[data-test-paper-detail]'));assert.equal(find('[data-test-paper-detail]').getAttribute('data-return-to'),otherPost);
    await act(async()=>root.unmount());root=undefined;reading.discardReadingReturn();
  }
  fixture.catalog=undefined;
  const acceptedLibraryReport={problem:problem.id,tests:[{id:'pass',layer:'T0',passed:true}],summary:{T0:[1,1]}};
  for(const locale of ['en','zh']) for(const scenario of ['empty','filtered-complete','all-complete','optional-only','loading']) {
    localStorage.setItem('paperleet.locale',locale);
    fixture.practice.ready=true;fixture.practice.loading=scenario==='loading';fixture.learning.ready=true;fixture.learning.records={};
    fixture.practice.saved={[problem.id]:{report:acceptedLibraryReport}};
    const module=scenario==='optional-only'?{...problem,module:{...problem.module,optional:true}}:problem;
    fixture.catalog={problems:[module],papers:[libraryPaper],categories:[]};
    window.history.replaceState(null,'',scenario==='empty'?'#/?q=missing-paper':scenario==='filtered-complete'?'#/?q=1706.03762':'#');
    await act(async()=>{root=createRoot(document.querySelector('#root'));root.render(React.createElement(App));});
    const expected=scenario==='empty'||scenario==='optional-only'?(locale==='en'?'No core modules in this view':'当前范围暂无核心模块'):
      scenario==='filtered-complete'?(locale==='en'?'Core modules complete in this view':'本范围核心模块已完成'):
      scenario==='all-complete'?(locale==='en'?'Core modules complete':'核心模块已完成'):undefined;
    await settle(()=>find('.topbar .continue-link')?.title===(expected||(locale==='en'?'Start practice':'开始练习')));
    assert.equal(find('.topbar .continue-link [aria-hidden="false"]').textContent,locale==='en'?'Workspace':'工作区','Navigation keeps a stable label while its title explains the current practice state');
    assert(find('.topbar .continue-link').disabled, 'An empty, complete or loading view cannot start a fabricated next module');
    if(scenario==='loading')assert(!find('.topbar .continue-link').title.includes('Core modules complete'),'Pending progress must not claim completion');
    await act(async()=>root.unmount());root=undefined;
  }
  fixture.practice.ready=true;fixture.practice.loading=false;fixture.practice.saved={};fixture.learning.ready=true;
  const position={...problem,id:'architecture/positional_encoding',number:29,title:'Sinusoidal positions',translations:{en:{...problem.translations.en,title:'Sinusoidal positions'}}};
  fixture.catalog={problems:[problem,position],papers:[{...libraryPaper,module_ids:[problem.id,position.id]}],categories:[]};
  const selectedView=libraryNav.libraryHref(undefined,{...libraryNav.emptyLibraryFilters(),query:'Sinusoidal'});
  window.history.replaceState(null,'',libraryNav.withLibraryContext('papers/attention',selectedView));
  await act(async()=>{root=createRoot(document.querySelector('#root'));root.render(React.createElement(App));});
  await settle(()=>find('[data-test-paper-detail]')?.getAttribute('data-practice-ids')===position.id);
  assert.equal(find('[data-test-paper-detail]').getAttribute('data-return-to'),selectedView,'The real App sends its current canonical library selection to paper details');
  assert.equal(find('[data-test-paper-detail]').getAttribute('data-reload-blocked'),'false');
  await act(async()=>{fixture.practice.pending=true;notify();});
  await settle(()=>find('[data-test-paper-detail]')?.getAttribute('data-reload-blocked')==='true');
  await act(async()=>{fixture.practice.pending=false;notify();});
  await settle(()=>find('[data-test-paper-detail]')?.getAttribute('data-reload-blocked')==='false');
  const emptyView=libraryNav.libraryHref(undefined,{...libraryNav.emptyLibraryFilters(),query:'missing-paper'});
  await act(async()=>{location.hash=libraryNav.withLibraryContext('papers/attention',emptyView);window.dispatchEvent(new window.HashChangeEvent('hashchange'));});
  await settle(()=>find('[data-test-paper-detail]')?.getAttribute('data-practice-ids')==='');
  assert.equal(find('[data-test-paper-detail]').getAttribute('data-return-to'),emptyView,'Changing a return view must retire the old recommendation scope');
  await act(async()=>root.unmount());root=undefined;

  // Profile-origin achievement URLs must carry their return behavior across a
  // fresh App mount, as a refresh or opening the link in another tab would.
  const mountAt=async href=>{
    if(root){await act(async()=>root.unmount());root=undefined;}
    window.history.replaceState(null,'',href);
    await act(async()=>{root=createRoot(document.querySelector('#root'));root.render(React.createElement(App));});
  };
  const navigate=async href=>act(async()=>{location.hash=href;window.dispatchEvent(new window.HashChangeEvent('hashchange'));});
  const achievementReturn=()=>find('[data-test-achievement-return]')?.getAttribute('href');
  const profileAchievements='#/achievements?from=profile';
  fixture.account.user={id:'alice',username:'alice',display_name:'Alice'};
  await mountAt('#/u/alice');
  await navigate(profileAchievements);
  assert.equal(achievementReturn(),'#/u/alice','Entering achievements from a profile returns to that account profile');
  await mountAt(profileAchievements);
  assert.equal(achievementReturn(),'#/u/alice','Reloading or opening the profile achievement link in a new tab must not lose its return target');
  await change(()=>{fixture.account.user={id:'bob',username:'bob',display_name:'Bob'};});
  assert.equal(achievementReturn(),'#/u/bob','An explicit profile origin always uses the current account, never a previous viewer');
  fixture.account.user=undefined;fixture.account.loading=true;
  await mountAt(profileAchievements);
  assert.equal(achievementReturn(),'#','No stale account profile is retained while a fresh session is unresolved');
  await change(()=>{fixture.account.user={id:'carol',username:'carol',display_name:'Carol'};fixture.account.loading=false;});
  assert.equal(achievementReturn(),'#/u/carol','Resolving the account after a refresh restores its profile return');
  const previousPage='#/community?sort=newest';
  await mountAt(previousPage);
  await navigate('#/achievements');
  assert.equal(achievementReturn(),previousPage,'Ordinary achievements entries preserve the existing in-memory return behavior');
  for(const href of ['#/achievements','#/achievements?from=https%3A%2F%2Foutside.example','#/achievements?from=someone_else']){
    await mountAt(href);
    assert.equal(achievementReturn(),'#','Only the fixed profile source marker can replace the default return target');
  }
  await act(async()=>root.unmount());root=undefined;
  console.log('PASS achievement App routing: profile return survives refresh/new-tab entry and late account resolution, follows account changes, and preserves ordinary entry behavior');
  console.log('PASS actual App workspace: source-safe inline judging, neutral stops, current-source/mode recovery with offline/reload/owner guards, retained editor/diagnostics, exact practice returns, live paper recommendation scope and bilingual empty/filtered-complete/all-complete/optional/loading states');
} finally {
  if(root)await act(async()=>root.unmount());
  await server.close();dom.window.close();
  for(const [name,descriptor] of originals)descriptor?Object.defineProperty(globalThis,name,descriptor):delete globalThis[name];
}
