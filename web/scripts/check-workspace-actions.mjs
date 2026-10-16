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
    clear(id){delete this.reports[id];delete this.syntaxErrors[id];notify();},
    run(entry,source){calls.runs.push({id:entry.id,source});this.active=true;this.problem=entry.id;this.stage='running';delete this.syntaxErrors[entry.id];this.reasons[entry.id]='';this.diagnostics[entry.id]='';notify();return new Promise((resolve,reject)=>{runResolve=resolve;runReject=reject;});},
    stop(){calls.stops++;fixture.browser.active=false;fixture.browser.problem=undefined;notify();runReject?.(new DOMException('Stopped','AbortError'));runResolve=undefined;runReject=undefined;}},
  judge:{active:false,async submit(id,source){calls.submits.push({id,source});if(this.submitError)throw Error(this.submitError);this.error='';this.active=true;this.pending={problem:id,source};this.job={problem:id,status:'queued',ahead:2};notify();},stopWaiting:noop,
    cancel(){this.active=false;this.pending=undefined;this.job={...this.job,status:'cancelled'};notify();}},
  learning:{ready:true,records:{},retry:noop,dismiss:noop,record:noop},
};
// App passes callbacks as event handlers; use bound fixtures just like the hooks.
fixture.practice.update=fixture.practice.update.bind(fixture.practice);
fixture.browser.run=fixture.browser.run.bind(fixture.browser);
globalThis.__paperleetWorkspaceActionTest = fixture;
const mocks = {
  './social': `import {useSyncExternalStore} from 'react';
    export const AccountProvider=({children})=>children;
    export const AccountNav=()=>null;
    export function useAccount(){const f=globalThis.__paperleetWorkspaceActionTest;useSyncExternalStore(f.subscribe,f.snapshot,f.snapshot);return f.account;}`,
  './PracticeWorkspace': `export const usePracticeWorkspace=()=>globalThis.__paperleetWorkspaceActionTest.practice;`,
  './useBrowserJudge': `export const useBrowserJudge=()=>globalThis.__paperleetWorkspaceActionTest.browser;`,
  './useQueuedJudge': `export function useQueuedJudge(owner,ready,onReport){const f=globalThis.__paperleetWorkspaceActionTest;f.onServerReport=onReport;return f.judge;}`,
  './useBrowserLearning': `export const useBrowserLearning=()=>globalThis.__paperleetWorkspaceActionTest.learning;`,
  './catalogIndex': `export const loadCatalogIndex=async()=>globalThis.__paperleetWorkspaceActionTest.catalog||({problems:[globalThis.__paperleetWorkspaceActionTest.problem],papers:[],categories:[]});`,
  './ProblemStats': `export const useProblemStats=()=>({stats:{},totalSubmissions:0,refresh:()=>{},loading:false});export const getProblemStats=()=>({});export const ProblemStats=()=>null;`,
  './PaperViews': `export const PaperDetail=()=>null;export const PaperLibrary=()=>null;export const paperModules=()=>[];`,
  './ReportView': `import React from 'react';import Report from '/src/ReportView.tsx?fixture_actual=1';globalThis.__paperleetWorkspaceActionTest.calls.detailsLoads++;export default function FixtureReport(props){if(globalThis.__paperleetWorkspaceActionTest.failDetails)throw Error('Fixture detail failure');return React.createElement(Report,props);}`,
  './deferredPages': `import React,{useEffect} from 'react';
    export const ProblemFavoriteButton=props=>React.createElement('button',{'data-test-favorite':props.problemId,'data-return-to':props.returnTo},'Save fixture problem');
    export const ProblemFavoritesPage=props=>React.createElement('section',{'data-test-favorites':props.route},React.createElement('button',{'data-test-favorite-filters':true,onClick:()=>props.onFilters('#/favorites?q=LoRA&state=todo')},'Change favorite filters'));
    export const InboxBell=()=>null,Messages=()=>null;
    export {default as ResizableSplit} from '/src/ResizableSplit.tsx';
    export {default as WorkspaceTestResults} from '/src/WorkspaceTestResults.tsx';
    export {default as WorkspaceLayoutMenu} from '/src/WorkspaceLayoutMenu.tsx';
    export function PythonEditor({value,onChange,label,readOnly,focusRequest,onFocusHandled}){useEffect(()=>{globalThis.__paperleetWorkspaceActionTest.calls.editorMounts++;},[]);useEffect(()=>{if(focusRequest){globalThis.__paperleetWorkspaceActionTest.calls.focusRequests.push(focusRequest);document.querySelector('#problem-editor-panel textarea')?.focus();onFocusHandled?.(focusRequest);}},[focusRequest]);return React.createElement('textarea',{'aria-label':label,value,readOnly,onChange:event=>onChange(event.target.value)});}
    export const Prose=({content})=>React.createElement('article',null,content);
    export const ReportView=({report})=>React.createElement('div',{'data-report':true},report ? 'Report available' : 'No report');
    ${['PreparationGuide','InterviewPracticeNextStep','Interviews','ProblemCompanyTags','Community','UserProfile','Account','SolutionHistory','SubmissionResultPage','Submissions','Visuals','WorkspaceBackup','QueueAdmissionNotice','ModuleNavigator','ResetCodeButton','BrowserDraftBackups','SettingsPage','Achievements','AchievementObserver','BrowserResultActions','JudgeRuntimeDetails','ReportWorkflow','OnlinePresence','ArtifactCharts'].map(name=>`export const ${name}=()=>null;`).join('\n')}`,
};
mocks['/src/ReportView.tsx']=mocks['./ReportView'];
const server = await createServer({
  server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},
  define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_HOSTED_APP_AVAILABLE':JSON.stringify('1')},
  plugins:[{name:'workspace-action-fixtures',enforce:'pre',resolveId(source,importer){if(importer?.endsWith('/src/WorkspaceTestResults.tsx')&&source==='./ReportView')return '\0workspace-actions:'+source;if(importer?.endsWith('/src/App.tsx')&&source in mocks)return '\0workspace-actions:'+source;},load(id){if(id.startsWith('\0workspace-actions:'))return mocks[id.slice('\0workspace-actions:'.length)];}}],
});
let root;
try {
  const {createRoot}=await import('react-dom/client');
  const {default:App}=await server.ssrLoadModule('/src/App.tsx');
  root=createRoot(document.querySelector('#root'));
  const find=selector=>document.querySelector(selector);
  const action=text=>[...document.querySelectorAll('.workspace-run-controls button')].find(button=>button.textContent.trim()===text);
  const click=async element=>{assert(element,'Expected an interactive control');await act(async()=>element.dispatchEvent(new window.MouseEvent('click',{bubbles:true})));};
  const change=async mutate=>act(async()=>{mutate();notify();});
  const settle=async predicate=>{for(let i=0;i<150&&!predicate();i++)await act(async()=>new Promise(resolve=>setTimeout(resolve,20)));assert(predicate(),'Workspace did not settle; detail loads: '+calls.detailsLoads+'; '+document.querySelector('.workspace-test-details')?.textContent.slice(0,700));};
  const resultSelected=()=>find('#reading-tab-result')?.getAttribute('aria-selected')==='true';
  const assertInline=message=>{assert.equal(location.hash,'#/1',message);assert.equal(find('#problem-editor-panel textarea'),editor,'Result changes must preserve the mounted editor');assert.equal(calls.editorMounts,1);};
  const finishBrowser=async outcome=>act(async()=>{fixture.browser.active=false;fixture.browser.problem=undefined;if(outcome.report)fixture.browser.reports={[problem.id]:outcome.report};if(outcome.failure)fixture.browser.syntaxErrors={[problem.id]:outcome.failure};notify();runResolve(outcome);runResolve=undefined;runReject=undefined;});
  const finishSubmission=async report=>act(async()=>{fixture.judge.active=false;fixture.judge.pending=undefined;fixture.judge.job={status:'completed'};fixture.onServerReport(report,fixture.practice.owner);notify();});
  await act(async()=>root.render(React.createElement(App)));
  await settle(()=>find('#problem-editor-panel textarea'));
  const editor=find('#problem-editor-panel textarea');
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
  assertInline('Stopping tests keeps the current problem URL');

  const report={problem:problem.id,source_sha256:createHash('sha256').update(source).digest('hex'),runtime:'browser-numpy-wasm',summary:{T0:[1,1]},tests:[{id:'output',layer:'T0',passed:true}]};
  await click(find('#reading-tab-description'));
  await click(action('Run tests'));
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
  await finishSubmission(accepted);
  await settle(()=>find('.workspace-test-results > .result-head h3')?.textContent==='Accepted');
  assert(resultSelected(),'Successful server completion selects the inline Result tab');
  assert.equal(calls.appliedReports.at(-1).report,accepted,'Server completion still saves the authoritative report');
  assertInline('Successful server submission must not redirect to a submission page');
  assert.equal(editor.value,editedSource);

  await click(find('#reading-tab-description'));
  await click(action('Submit'));
  const serverFailed={...accepted,run_id:'b'.repeat(32),summary:{T0:[0,1]},tests:[{id:'output',layer:'T0',passed:false,message:'Server output mismatch'}]};
  const pendingEdit=editedSource.replace('* 3','* 4');
  await change(()=>{fixture.practice.saved={[problem.id]:{...fixture.practice.saved[problem.id],code:pendingEdit}};});
  editor.setSelectionRange(6,15);editor.scrollTop=63;
  await finishSubmission(serverFailed);
  await settle(()=>find('#problem-results-panel .feedback-diagnostic')?.textContent.includes('Server output mismatch'));
  assert(resultSelected(),'Failed server completion appears in the inline Result tab');
  assertInline('Failed server submission must not redirect to a submission page');
  assert.equal(editor.value,pendingEdit,'A queued result cannot overwrite edits made after submitting');
  assert.equal(editor.selectionStart,6);assert.equal(editor.selectionEnd,15);assert.equal(editor.scrollTop,63);

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

  await change(()=>{fixture.account.user=undefined;fixture.practice.owner='anonymous';});
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
    if(expected) await settle(()=>find('.topbar .continue-link')?.textContent===expected);
    else await settle(()=>find('.topbar .continue-link')?.textContent.includes('Start practice'));
    assert(find('.topbar .continue-link').disabled, 'An empty, complete or loading view cannot start a fabricated next module');
    if(scenario==='loading')assert(!find('.topbar .continue-link').textContent.includes('Core modules complete'),'Pending progress must not claim completion');
    await act(async()=>root.unmount());root=undefined;
  }
  console.log('PASS actual App workspace: source-safe inline judging, retained editor and diagnostics, practice returns, and bilingual empty/filtered-complete/all-complete/optional/loading header states');
} finally {
  if(root)await act(async()=>root.unmount());
  await server.close();dom.window.close();
  for(const [name,descriptor] of originals)descriptor?Object.defineProperty(globalThis,name,descriptor):delete globalThis[name];
}
