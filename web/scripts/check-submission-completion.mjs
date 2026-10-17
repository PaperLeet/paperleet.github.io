import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<div id="root"></div>',{url:'https://paperleet.test/#/22?from=collections%2Fllm-basics',pretendToBeVisual:true});
const values={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,location:dom.window.location,localStorage:dom.window.localStorage,IS_REACT_ACT_ENVIRONMENT:true};
const old=new Map(Object.keys(values).map(k=>[k,Object.getOwnPropertyDescriptor(globalThis,k)]));for(const[k,v]of Object.entries(values))Object.defineProperty(globalThis,k,{value:v,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root;
try {
 const {createRoot}=await import('react-dom/client');
 const {default:Notice}=await server.ssrLoadModule('/src/SubmissionCompletionNotice.tsx');
 const {SubmissionCompletionFallback:Fallback}=await server.ssrLoadModule('/src/SubmissionCompletionFallback.tsx');
 const {submissionCompletion,resultPanelVisible}=await server.ssrLoadModule('/src/submissionCompletion.ts');
 const {deferredFeature}=await server.ssrLoadModule('/src/deferredFeature.tsx');
 let rejectPresentation,caught=0;const Deferred=deferredFeature(()=>new Promise((_resolve,reject)=>{rejectPresentation=reject;}),['',''],{loading:()=>null,failure:props=>React.createElement(Fallback,props)});
 const {useWorkspaceShortcuts}=await server.ssrLoadModule('/src/useWorkspaceShortcuts.ts');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const entry={id:'finetuning/lora',number:22},runId='a'.repeat(32),report={problem:entry.id,run_id:runId};
 const notice=submissionCompletion(report,'user:alice',entry);
 assert.deepEqual(notice,{owner:'user:alice',problem:entry.id,number:22,runId,href:'#/submission/'+runId});
 for(const [r,owner,e]of [[{...report,run_id:'../private'},'user:alice',entry],[{...report,problem:'another'},'user:alice',entry],[report,'anonymous',entry],[report,'local',entry],[report,'user:alice',{...entry,number:0}],[report,'user:alice',{...entry,number:NaN}],[report,'user:alice',{...entry,number:2.2}]])assert.equal(submissionCompletion(r,owner,e),undefined);
 const panel=document.createElement('div');document.body.append(panel);panel.getClientRects=()=>[new window.DOMRect(0,0,500,500)];assert(resultPanelVisible(panel));panel.hidden=true;assert(!resultPanelVisible(panel));panel.hidden=false;panel.getClientRects=()=>[];assert(!resultPanelVisible(panel),'A CSS-hidden narrow/focus-mode pane is not a visible result');panel.remove();assert(!resultPanelVisible(null));
 let props={notice,inline:true},useFallback=false,useDeferred=false,setLocale,runs=0,exits=0,action='inline',dismissals=[],activations=[];
 function Harness(){
  setLocale=useI18n().setLocale;useWorkspaceShortcuts({enabled:true,canRun:true,focused:true,onRun:()=>{runs++;},onExitFocus:()=>{exits++;}});
  return React.createElement(React.Fragment,null,React.createElement('textarea',{'aria-label':'Private code',defaultValue:'def mine(): return 42'}),React.createElement('button',{id:'outside'},'Outside'),React.createElement(useDeferred?Deferred:useFallback?Fallback:Notice,{...props,onView:(inline,modified)=>{activations.push({inline,modified});return action;},onDismiss:target=>dismissals.push(target)}));
 }
 root=createRoot(document.querySelector('#root'),{onCaughtError:()=>{caught++;}});const render=async()=>act(async()=>root.render(React.createElement(LocaleProvider,null,React.createElement(Harness))));
 const find=s=>document.querySelector(s),link=()=>find('.toast a'),close=()=>find('.toast button');
 const click=async(node,options={})=>{const e=new window.MouseEvent('click',{bubbles:true,cancelable:true,...options});await act(async()=>node.dispatchEvent(e));return e;};
 const key=async(node,options={})=>{const e=new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true,...options});await act(async()=>node.dispatchEvent(e));return e;};
 // Keep this local DOM on the practice route while observing native link intent.
 const native=[];document.addEventListener('click',event=>{if(event.target.closest?.('.toast a')&&!event.defaultPrevented){native.push(event.target.closest('a').getAttribute('href'));event.preventDefault();}});
 await render();const code=find('textarea');code.focus();code.setSelectionRange(2,9,'backward');code.scrollTop=41;code.scrollLeft=13;const href=location.href;
 assert(document.activeElement===code,'Notification arrival does not take focus');assert(find('.toast strong').textContent.includes('#22'));assert.equal(link().getAttribute('href'),notice.href);
 const chord=await key(link(),{key:'Enter',ctrlKey:true});assert(!chord.defaultPrevented);assert.equal(runs,0,'Native modified link activation does not run tests');
 await act(async()=>link().focus());await click(link());assert.deepEqual(activations.at(-1),{inline:true,modified:false});assert.equal(native.length,0);
 action='saved';await click(link(),{ctrlKey:true});assert.deepEqual(activations.at(-1),{inline:false,modified:true});assert.equal(native.at(-1),notice.href);
 props={...props,inline:false};await render();assert.equal(link().textContent,'View saved result');await click(link());assert.deepEqual(activations.at(-1),{inline:false,modified:false});assert.equal(native.at(-1),notice.href);
 action='ignore';const count=native.length;await click(link());assert.equal(native.length,count,'An obsolete owner/route action is inert');
 code.focus();await act(async()=>close().focus());await click(close());assert(dismissals.at(-1)===code,'Close remembers the latest outside editor without reading its content');
 await act(async()=>setLocale('zh'));assert.equal(link().textContent,'查看提交结果');assert(document.activeElement===close());assert.equal(code.selectionStart,2);assert.equal(code.selectionEnd,9);assert.equal(code.selectionDirection,'backward');assert.equal(code.scrollTop,41);assert.equal(code.scrollLeft,13);
 for(const options of [{isComposing:true},{keyCode:229},{repeat:true},{altKey:true},{ctrlKey:true},{metaKey:true},{shiftKey:true}]){const count=dismissals.length;assert(!(await key(close(),options)).defaultPrevented);assert.equal(dismissals.length,count);}
 const countDismiss=dismissals.length;const escaped=await key(close());assert(escaped.defaultPrevented&&dismissals.length===countDismiss+1);assert.equal(exits,0,'Local dismissal does not exit code focus mode');
 find('#outside').focus();const outside=await key(find('#outside'));assert(outside.defaultPrevented&&exits===1,'Outside Escape keeps the existing workspace shortcut');
 // The optional presentation can fail while the essential saved/inline action remains usable.
 useFallback=true;await render();assert(find('.submission-completion-fallback'));action='inline';props={...props,inline:true};await render();await click(link());assert.equal(activations.at(-1).inline,true);action='ignore';const prior=native.length;await click(link());assert.equal(native.length,prior);
 const fallbackChord=await key(link(),{key:'Enter',metaKey:true});assert(!fallbackChord.defaultPrevented&&runs===0);await click(close());assert.equal(dismissals.at(-1),undefined);
 code.focus();useDeferred=true;await render();assert(!find('.toast'));assert(document.activeElement===code);await act(async()=>rejectPresentation(Error('Synthetic optional presentation failure')));assert(find('.submission-completion-fallback'));assert.equal(caught,1);assert(document.activeElement===code);action='inline';await click(link());assert(activations.at(-1).inline);assert(document.querySelector('textarea')===code);
 assert.equal(location.href,href);assert.equal(code.value,'def mine(): return 42');assert.equal(code.selectionStart,2);assert.equal(code.selectionEnd,9);assert.equal(code.scrollTop,41);assert.equal(code.scrollLeft,13);
 console.log('PASS submission completion: canonical account/run/problem guards, visible-vs-hidden Result, neutral no-focus arrival, bilingual scoped actions and native modified links, inert stale activation, close origin and local Escape, shortcut ownership, presentation fallback and unchanged private code/selection/scroll');
} finally {if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const[k,d]of old){if(d)Object.defineProperty(globalThis,k,d);else delete globalThis[k];}}
