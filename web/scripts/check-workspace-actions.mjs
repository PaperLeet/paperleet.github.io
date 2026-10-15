import assert from 'node:assert/strict';
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
window.HTMLElement.prototype.scrollTo = () => {};
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
let revision = 0, runResolve;
const listeners = new Set();
const notify = () => { revision++; for (const listener of listeners) listener(); };
const calls = {runs:[],submits:[],stops:0,flushes:0,editorMounts:0};
const fixture = {
  problem,calls,subscribe:listener => {listeners.add(listener);return()=>listeners.delete(listener);},snapshot:()=>revision,
  account:{user:{id:'fixture-user',username:'fixture-user',display_name:'Fixture'},health:{mode:'hosted',judge:{ready:true}},healthError:'',refreshHealth:noop,capabilities:{}},
  practice:{owner:'user:fixture-user',ready:true,loading:false,saved:{},conflicts:{},backups:[],owns:owner=>owner==='user:fixture-user',
    update(id,patch){this.saved={...this.saved,[id]:{...this.saved[id],...patch}};notify();},
    async flush(){calls.flushes++;},applyReport:noop,retry:noop,resolveConflict:noop},
  browser:{active:false,reports:{},syntaxErrors:{},reasons:{},calibrations:{},diagnostics:{},measurements:{},clear:noop,
    run(entry,source){calls.runs.push({id:entry.id,source});this.active=true;this.problem=entry.id;this.stage='running';notify();return new Promise(resolve=>{runResolve=resolve;});},
    stop(){calls.stops++;fixture.browser.active=false;notify();runResolve?.({type:'stopped'});runResolve=undefined;}},
  judge:{active:false,async submit(id,source){calls.submits.push({id,source});},stopWaiting:noop,cancel:noop},
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
  './useQueuedJudge': `export const useQueuedJudge=()=>globalThis.__paperleetWorkspaceActionTest.judge;`,
  './useBrowserLearning': `export const useBrowserLearning=()=>globalThis.__paperleetWorkspaceActionTest.learning;`,
  './catalogIndex': `export const loadCatalogIndex=async()=>({problems:[globalThis.__paperleetWorkspaceActionTest.problem],papers:[],categories:[]});`,
  './ProblemStats': `export const useProblemStats=()=>({stats:{},totalSubmissions:0,refresh:()=>{},loading:false});export const getProblemStats=()=>({});export const ProblemStats=()=>null;`,
  './PaperViews': `export const PaperDetail=()=>null;export const PaperLibrary=()=>null;export const paperModules=()=>[];`,
  './deferredPages': `import React,{useEffect} from 'react';
    export {default as ResizableSplit} from '/src/ResizableSplit.tsx';
    export {default as WorkspaceLayoutMenu} from '/src/WorkspaceLayoutMenu.tsx';
    export function PythonEditor({value,onChange,label,readOnly}){useEffect(()=>{globalThis.__paperleetWorkspaceActionTest.calls.editorMounts++;},[]);return React.createElement('textarea',{'aria-label':label,value,readOnly,onChange:event=>onChange(event.target.value)});}
    export const Prose=({content})=>React.createElement('article',null,content);
    export const ReportView=({report})=>React.createElement('div',{'data-report':true},report ? 'Report available' : 'No report');
    ${['PreparationGuide','InterviewPracticeNextStep','Interviews','ProblemCompanyTags','Community','UserProfile','Account','SolutionHistory','SubmissionResultPage','Submissions','Visuals','WorkspaceBackup','QueueAdmissionNotice','ModuleNavigator','ResetCodeButton','BrowserDraftBackups','SettingsPage','Achievements','AchievementObserver','BrowserResultActions','JudgeRuntimeDetails','ReportWorkflow','OnlinePresence'].map(name=>`export const ${name}=()=>null;`).join('\n')}`,
};
const server = await createServer({
  server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},
  define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_HOSTED_APP_AVAILABLE':JSON.stringify('1')},
  plugins:[{name:'workspace-action-fixtures',enforce:'pre',resolveId(source,importer){if(importer?.endsWith('/src/App.tsx')&&source in mocks)return '\0workspace-actions:'+source;},load(id){if(id.startsWith('\0workspace-actions:'))return mocks[id.slice('\0workspace-actions:'.length)];}}],
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
  const settle=async predicate=>{for(let i=0;i<40&&!predicate();i++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),'Workspace did not settle');};
  await act(async()=>root.render(React.createElement(App)));
  await settle(()=>find('#problem-editor-panel textarea'));
  const editor=find('#problem-editor-panel textarea');
  assert(find('.workspace-toolbar').contains(action('Run tests')));
  assert(find('.workspace-toolbar').contains(action('Submit')));
  assert.equal(find('.editor-bottom'),null,'The actions must not leave a second footer below the editor');
  assert.equal(document.querySelectorAll('.mobile-workspace-switch').length,1,'Mobile has one unambiguous pane switch');
  assert.equal(find('.editor-workbench').dataset.resultsVisible,'false','A new module has no empty result area');
  assert(find('#problem-results-panel').hidden);
  assert.equal(find('.editor-split').dataset.secondCollapsed,'true');
  assert(!action('Run tests').disabled&&!action('Submit').disabled);

  // Editing before expanding/collapsing must not remount the editor or lose its
  // unsent selection/scroll. The real Python editor has separate editor tests.
  const source='def attention(x):\n    return x * 2 # private draft\n';
  await change(()=>{fixture.practice.saved={[problem.id]:{code:source}};});
  editor.focus();editor.setSelectionRange(4,13);editor.scrollTop=41;
  await click(action('Run tests'));
  assert.deepEqual(calls.runs,[{id:problem.id,source}]);
  assert.equal(find('.editor-workbench').dataset.resultsVisible,'true');
  assert(!find('#problem-results-panel').hidden&&!find('#problem-results-content').hidden,'Starting a run reveals progress');
  assert(action('Run tests').disabled&&action('Submit').disabled,'The moved controls still prevent overlapping runs');
  assert(find('.workspace-toolbar').contains(action('Stop')));
  await click(action('Stop'));
  assert.equal(calls.stops,1);
  await settle(()=>!action('Run tests').disabled);
  assert.equal(find('.editor-workbench').dataset.resultsVisible,'false','A stopped run without results leaves no empty console');

  const report={problem:problem.id,runtime:'browser-numpy-wasm',summary:{T0:[1,1]},tests:[{id:'output',layer:'T0',passed:true}]};
  await click(action('Run tests'));
  await act(async()=>{fixture.browser.reports={[problem.id]:report};fixture.browser.active=false;notify();runResolve({type:'result',report});runResolve=undefined;});
  assert.equal(find('#reading-tab-description').getAttribute('aria-selected'),'true','Browser completion keeps the statement visible beside the editor');
  assert(!find('#problem-results-content').hidden,'Finished tests remain visible in the right result drawer');
  await click(find('.results-pane-toggle'));
  assert(find('#problem-results-content').hidden);
  assert.equal(find('.editor-split').dataset.secondCollapsed,'true');
  await click(find('.results-pane-toggle'));
  assert(!find('#problem-results-content').hidden);
  assert.equal(find('#problem-editor-panel textarea'),editor);
  assert.equal(editor.value,source);
  assert.equal(editor.selectionStart,4);assert.equal(editor.selectionEnd,13);assert.equal(editor.scrollTop,41);
  assert.equal(calls.editorMounts,1,'Result visibility changes preserve the existing editor instance');

  await click(find('.results-pane-toggle'));
  await change(()=>{fixture.practice.error='Draft synchronization failed';});
  assert(!find('#problem-results-content').hidden,'A save error cannot remain hidden in a collapsed drawer');
  assert(find('#problem-results-content').textContent.includes('Draft synchronization failed'));
  await change(()=>{fixture.practice.error='';fixture.browser.reports={};});
  await click(action('Submit'));
  assert.deepEqual(calls.submits,[{id:problem.id,source}],'Top Submit still sends the current code to server judging');
  assert.equal(calls.flushes,1,'Hosted drafts flush before server submission');
  assert.equal(calls.runs.length,2,'Submit must not silently become a browser-only run');

  const chord=new window.KeyboardEvent('keydown',{key:'Enter',ctrlKey:true,bubbles:true,cancelable:true});
  await act(async()=>editor.dispatchEvent(chord));
  assert(chord.defaultPrevented,'The editor run shortcut is still intercepted before inserting a newline');
  assert.equal(calls.runs.length,3);
  const duplicate=new window.KeyboardEvent('keydown',{key:'Enter',metaKey:true,bubbles:true,cancelable:true});
  await act(async()=>editor.dispatchEvent(duplicate));
  assert.equal(calls.runs.length,3,'The shortcut cannot start another active run');
  await click(action('Stop'));
  await settle(()=>!action('Run tests').disabled);

  await change(()=>{fixture.account.user=undefined;fixture.practice.owner='anonymous';});
  assert(action('Sign in to submit'),'Guests keep a sign-in entry in the same top action area');
  assert.equal(action('Submit'),undefined);
  assert(!action('Run tests').disabled,'Guest browser practice stays available');
  console.log('PASS actual App workspace: top Run/Submit/Stop, single mobile switch, hidden idle drawer, run and error expansion, preserved statement/editor/cursor, server submission, guest action, and keyboard duplicate-run guard');
} finally {
  if(root)await act(async()=>root.unmount());
  await server.close();dom.window.close();
  for(const [name,descriptor] of originals)descriptor?Object.defineProperty(globalThis,name,descriptor):delete globalThis[name];
}
