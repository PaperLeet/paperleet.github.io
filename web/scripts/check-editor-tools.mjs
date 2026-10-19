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
mocks['./social']=mocks['./social'].replace("export const AccountNav=()=>null;","export function AccountNav(){const value=useI18n();globalThis.__paperleetWorkspaceActionTest.setTestLocale=value.setLocale;return null;}").replace("import {useSyncExternalStore} from 'react';","import {useSyncExternalStore} from 'react';import {useI18n} from '/src/i18n.tsx';");
// Inject failure only into EditorTools; unrelated lazy features retain their own loaders.
mocks['./deferredFeature']=`import {deferredFeature as real} from '/src/deferredFeature.tsx';import Tools from '/src/EditorTools.tsx';export function deferredFeature(load,label,options){if(!String(load).includes('EditorTools'))return real(load,label,options);return real(()=>new Promise((resolve,reject)=>{const f=globalThis.__paperleetWorkspaceActionTest;f.toolAttempts=(f.toolAttempts||0)+1;f.resolveTools=()=>resolve({default:Tools});f.rejectTools=()=>reject(Error('Synthetic editor-tools bundle failure'));}),label,options);}`;
mocks['/src/ReportView.tsx']=mocks['./ReportView'];
const server = await createServer({
  server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},
  define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_HOSTED_APP_AVAILABLE':JSON.stringify('1')},
  plugins:[{name:'workspace-action-fixtures',enforce:'pre',resolveId(source,importer){if(importer?.endsWith('/src/ProblemOverview.tsx')&&source==='./deferredPages')return '\0overview-tags';if(importer?.endsWith('/src/WorkspaceTestResults.tsx')&&source==='./ReportView')return '\0workspace-actions:'+source;if(importer?.endsWith('/src/App.tsx')&&source in mocks)return '\0workspace-actions:'+source;},load(id){if(id==='\0overview-tags')return 'export const ProblemCompanyTags=()=>null;';if(id.startsWith('\0workspace-actions:'))return mocks[id.slice('\0workspace-actions:'.length)];}}],
});
const originalObjectURL=Object.getOwnPropertyDescriptor(URL,'createObjectURL'),originalRevokeURL=Object.getOwnPropertyDescriptor(URL,'revokeObjectURL');
const downloads=[],blobs=[];URL.createObjectURL=blob=>{blobs.push(blob);return 'blob:paperleet-test/'+blobs.length;};URL.revokeObjectURL=()=>{};
const nativeAnchorClick=window.HTMLAnchorElement.prototype.click;window.HTMLAnchorElement.prototype.click=function(){if(this.download){downloads.push({name:this.download,href:this.href});return;}nativeAnchorClick.call(this);};
let root;
try {
 const {createRoot}=await import('react-dom/client');const {default:App}=await server.ssrLoadModule('/src/App.tsx');
 let gets=0;const writes=[],pending=[],deadlines=new Map();let timerId=100000;
 const originalTimeout=window.setTimeout.bind(window),originalClear=window.clearTimeout.bind(window);
 window.setTimeout=(fn,delay,...args)=>{if(delay===8000){const id=timerId++;deadlines.set(id,()=>fn(...args));return id;}return originalTimeout(fn,delay,...args);};window.clearTimeout=id=>{if(deadlines.has(id))deadlines.delete(id);else originalClear(id);};
 Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:value=>{writes.push(value);return new Promise((resolve,reject)=>pending.push({resolve,reject,value}));}}});
 dom.window.HTMLDialogElement.prototype.showModal=function(){this.open=true;};dom.window.HTMLDialogElement.prototype.close=function(){this.open=false;};
 root=createRoot(document.querySelector('#root'));await act(async()=>root.render(React.createElement(App)));
 const ready=async condition=>{for(let n=0;n<150&&!condition();n++)await act(async()=>new Promise(r=>setTimeout(r,10)));assert(condition(),'Expected editor tool state');};
 const click=async node=>{assert(node,'Expected editor action');await act(async()=>{node.focus();node.click();});};
 const edit=async(value)=>act(async()=>{const node=editor();Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set.call(node,value);node.dispatchEvent(new window.Event('input',{bubbles:true}));});
 const editor=()=>document.querySelector('#problem-editor-panel .code-editor textarea'),copy=()=>document.querySelector('.editor-source-copy button'),manual=()=>document.querySelector('.editor-copy-manual textarea'),font=()=>document.querySelector('.editor-font-size-select');
 const begin=async()=>{const menu=document.querySelector('.editor-actions');if(!menu.open)await click(menu.querySelector('summary'));assert(menu.open);await click(copy());assert.equal(copy().getAttribute('aria-busy'),'true');return pending.shift();};
 const settle=async(item,fail=false)=>act(async()=>fail?item.reject(Error('Clipboard permission denied')):item.resolve());
 const key=async(node,options={})=>{const event=new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true,...options});await act(async()=>{node.focus();node.dispatchEvent(event);});return event;};
 await ready(()=>fixture.rejectTools&&editor());const originalEditor=editor(),mounts=calls.editorMounts;originalEditor.setSelectionRange(3,12,'backward');originalEditor.scrollTop=31;originalEditor.scrollLeft=14;const url=location.href;
 assert(document.querySelector('.editor-tools-recovery'));assert(!font());assert.equal(writes.length,0);assert.equal(calls.runs.length,0);assert.equal(fixture.toolAttempts,1,'Only the editor tools loader is controlled by this fixture');
 const unchanged=()=>{assert(editor()===originalEditor,'Tool presentation must not remount the editor');assert.equal(calls.editorMounts,mounts);assert.deepEqual([editor().selectionStart,editor().selectionEnd,editor().selectionDirection,editor().scrollTop,editor().scrollLeft],[3,12,'backward',31,14]);assert.equal(location.href,url);};
 await act(async()=>fixture.rejectTools());await ready(()=>!!document.querySelector('.editor-tools-recovery [role=alert]'));unchanged();const retry=document.querySelector('.editor-tools-recovery button'),download=document.querySelector('.editor-download-code');assert(!download.disabled);await click(download);assert.equal(downloads[0].name,'problem-1.py');assert.equal(await blobs[0].text(),problem.starter);await click(retry);assert(document.activeElement===download);assert.equal(fixture.toolAttempts,2);unchanged();originalEditor.focus();await act(async()=>fixture.resolveTools());await ready(()=>!!font()&&!!copy());assert(document.activeElement===originalEditor,'Arriving tools never steal newer editor focus');unchanged();
 const menu=document.querySelector('.editor-actions'),summary=menu.querySelector('summary');assert(document.querySelector('.editor-header').contains(menu),'File actions share the workspace tab row');assert(!menu.closest('[role=tablist]'),'Action menu does not enter the roving tab list');assert(!document.querySelector('.editor-code-panel .editor-tools'),'No second editor toolbar remains');assert(document.querySelector('#editor-tab-code .editor-tab-filename').textContent==='solution.py');assert(!menu.open);await click(summary);assert(menu.open);assert((await key(summary)).defaultPrevented);assert(!menu.open&&document.activeElement===summary);await click(summary);await act(async()=>document.querySelector('.code-editor').dispatchEvent(new window.MouseEvent('pointerdown',{bubbles:true})));assert(!menu.open,'Clicking outside closes the action disclosure');unchanged();
 const source='# 当前实现 😀\r\ndef answer(x):\r\n    return x + 1\r\n';fixture.practice.saved={...fixture.practice.saved,[problem.id]:{code:source}};await act(async()=>notify());editor().setSelectionRange(3,12,'backward');editor().scrollTop=31;editor().scrollLeft=14;unchanged();
 let first;await act(async()=>{copy().focus();copy().click();copy().click();first=pending.shift();});assert.equal(writes.length,1);assert.equal(first.value,source);assert.equal(copy().getAttribute('aria-disabled'),'true');assert.equal(deadlines.size,1);await key(copy(),{key:'Enter',ctrlKey:true});assert.equal(calls.runs.length,0,'Copy controls do not execute private code');await settle(first);assert(document.querySelector('.editor-source-copy [role=status]').textContent.includes('Code copied.'));unchanged();
 const denied=await begin();originalEditor.focus();await settle(denied,true);assert(document.activeElement===originalEditor);assert(manual()?.readOnly);assert.equal(manual().value,source.replace(/\r\n/g,'\n'));assert(manual().closest('.editor-copy-recovery'));manual().focus();assert.equal(manual().selectionEnd,manual().value.length);manual().setSelectionRange(2,20,'backward');manual().scrollTop=17;
 const selected=manual();await act(async()=>fixture.setTestLocale('zh'));assert(manual()===selected&&document.activeElement===selected);assert.deepEqual([selected.selectionStart,selected.selectionEnd,selected.selectionDirection,selected.scrollTop],[2,20,'backward',17]);await act(async()=>fixture.setTestLocale('en'));await act(async()=>window.dispatchEvent(new window.Event('resize')));assert(manual()===selected);await key(selected,{key:'Enter',metaKey:true});assert.equal(calls.runs.length,0);for(const extra of [{isComposing:true},{keyCode:229},{repeat:true},{ctrlKey:true},{altKey:true},{metaKey:true},{shiftKey:true}]){assert(!(await key(selected,extra)).defaultPrevented);assert(manual()===selected);}
 assert((await key(selected)).defaultPrevented);assert(!manual()&&document.activeElement===copy());unchanged();
 const timed=await begin();await act(async()=>[...deadlines.values()][0]());assert(manual());const newer=await begin();await settle(timed);assert.equal(copy().getAttribute('aria-busy'),'true');await settle(newer);assert(!manual());
 const originalCopy=copy(),fontNode=font();fontNode.focus();await act(async()=>{Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype,'value').set.call(fontNode,'18');fontNode.dispatchEvent(new window.Event('change',{bubbles:true}));});assert.equal(localStorage.getItem('paperleet.editor.fontSize'),'18');assert(font()===fontNode&&copy()===originalCopy);unchanged();
 const stale=await begin();await edit('return changed');fixture.practice.saved={...fixture.practice.saved,[problem.id]:{code:source}};await act(async()=>notify());await settle(stale,true);assert(!manual()&&!document.querySelector('.editor-source-copy [role=status]').textContent,'A-B-A edits retire older copy feedback');editor().setSelectionRange(3,12,'backward');editor().scrollTop=31;editor().scrollLeft=14;unchanged();
 const blocked=await begin();await act(async()=>{fixture.practice.ready=false;notify();});assert.equal(copy().getAttribute('aria-disabled'),'true');assert(document.querySelector('.editor-download-code').disabled);const count=writes.length;await click(copy());assert.equal(writes.length,count);const downloadCount=downloads.length;await click(document.querySelector('.editor-download-code'));assert.equal(downloads.length,downloadCount);await settle(blocked,true);assert(!manual());await act(async()=>{fixture.practice.ready=true;notify();});
 // Changing font or source keeps the reset review mounted; only the copy lifetime changes.
 fixture.practice.resetToStarter=(entry,expected)=>{gets++;assert.equal(expected.owner,fixture.practice.owner);assert.equal(expected.code,source);fixture.practice.saved={...fixture.practice.saved,[entry.id]:{...fixture.practice.saved[entry.id],code:entry.starter}};notify();return true;};
 await ready(()=>!!document.querySelector('[aria-label="Restore starter code"]'));const reset=document.querySelector('[aria-label="Restore starter code"]');await click(reset);await ready(()=>!!document.querySelector('.reset-code-dialog[open]'));const confirm=[...document.querySelectorAll('.reset-code-dialog button')].find(node=>node.textContent==='Restore starter');await click(confirm);assert.equal(gets,1);assert(document.querySelector('[aria-label="Restore starter code"]')===reset);assert.equal(editor().value,problem.starter);
 await edit('');const empty=await begin();assert.equal(empty.value,'');await settle(empty,true);assert(manual()&&manual().value==='');assert(document.querySelector('.editor-copy-manual').textContent.includes('0 bytes'));await click(document.querySelector('.editor-copy-manual button'));assert(!manual());
 const outsideCopy=await begin();await settle(outsideCopy,true);assert(manual());await act(async()=>document.querySelector('.code-editor').dispatchEvent(new window.MouseEvent('pointerdown',{bubbles:true})));assert(!menu.open);await click(document.querySelector('.editor-copy-manual button'));assert(!manual()&&document.activeElement===summary,'Closing manual copy returns to the visible disclosure when its action menu is closed');
 await edit(source);const routeCopy=await begin();window.history.replaceState(null,'','#/settings');await settle(routeCopy,true);assert(!manual());window.history.replaceState(null,'',url);
 // A reset preview belongs to the original account/problem and cannot reappear after an owner cycle.
 await edit('owner-private-preview');await click(document.querySelector('[aria-label="Restore starter code"]'));await ready(()=>!!document.querySelector('.reset-code-dialog[open]'));const ownerReset=document.querySelector('[aria-label="Restore starter code"]');await act(async()=>{fixture.practice.owner='user:another-owner';fixture.account.user={id:'another-owner',username:'another-owner'};fixture.practice.saved={};notify();});assert(!document.querySelector('.reset-code-dialog[open]'));assert(!document.body.textContent.includes('owner-private-preview'));assert(document.querySelector('[aria-label="Restore starter code"]')!==ownerReset);await act(async()=>{fixture.practice.owner='user:fixture-user';fixture.account.user={id:'fixture-user',username:'fixture-user'};notify();});assert(!document.querySelector('.reset-code-dialog[open]'));
 const leaving=await begin();await act(async()=>root.unmount());root=undefined;assert.equal(deadlines.size,0);await settle(leaving,true);assert.equal(document.querySelector('#root').textContent,'');
 console.log('PASS deferred editor tools: actual App loading/failure/retry with the same editor/source/caret/both-scroll and newer focus, independent lazy controls, exact full/empty clipboard payload and dedup, portal manual selection/close/IME/Run ownership, timeout/retry/source-ABA/readiness/route/unmount retirement, stable font and owner-bound reset review with original snapshot, no service submission or editor lifecycle changes');
}finally{if(originalObjectURL)Object.defineProperty(URL,'createObjectURL',originalObjectURL);else delete URL.createObjectURL;if(originalRevokeURL)Object.defineProperty(URL,'revokeObjectURL',originalRevokeURL);else delete URL.revokeObjectURL;if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const[name,value]of originals){if(value)Object.defineProperty(globalThis,name,value);else delete globalThis[name];}}
