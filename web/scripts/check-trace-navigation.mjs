import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom=new JSDOM('<!doctype html><div id="root"></div><button id="other-task">Other task</button>',{url:'https://paperleet.test/#/22?from=collections%2Fllm-basics',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const nativeTimeout=window.setTimeout.bind(window),nativeClear=window.clearTimeout.bind(window),deadlines=new Map();let timerId=100000;
window.setTimeout=(callback,delay,...args)=>{if(delay===8000){const id=timerId++;deadlines.set(id,()=>callback(...args));return id;}return nativeTimeout(callback,delay,...args);};
window.clearTimeout=id=>{if(deadlines.has(id))deadlines.delete(id);else nativeClear(id);};
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});let root;
try{
  const {learnerTraceFrames,learnerTraceLocation,sourceOffset}=await server.ssrLoadModule('/src/sourceLocation.ts');
  const source='def forward(x):\n    return helper(x)\n\ndef helper(x):\n    raise ValueError("中文 😀")\n';
  const trace='tests/test_core.py:19: in test_reference\nsubmission.py:2: in forward\n    return helper(x)\n  File "/tmp/job/submission.py", line 5, in helper\n    raise ValueError("<script>literal</script>")\nreference.py:4: in expected\nValueError: wrong shape';
  const expected=[{row:1,location:{line:2}},{row:3,location:{line:5}}];
  assert.deepEqual(learnerTraceFrames(trace,'browser/submission.py',source,'current'),expected);assert.deepEqual(learnerTraceLocation(trace,'browser/submission.py',source,'current'),{line:5});
  for(const ending of ['\n','\r\n','\r'])assert.deepEqual(learnerTraceFrames(trace.replaceAll('\n',ending),'browser/submission.py',source.replaceAll('\n',ending),'current'),expected);
  const repeated=trace+'\nsubmission.py:2: in recursive\nsubmission.py:999: in outside';assert.deepEqual(learnerTraceFrames(repeated,'submission.py',source,'current'),[...expected,{row:7,location:{line:2}}]);assert.deepEqual(learnerTraceLocation(repeated,'submission.py',source,'current'),{line:2});
  for(const state of ['none','unknown','checking','changed'])assert.deepEqual(learnerTraceFrames(trace,'submission.py',source,state),[]);
  for(const impl of [undefined,'attention.py','x'.repeat(2049)])assert.deepEqual(learnerTraceFrames(trace,impl,source,'current'),[]);
  assert.deepEqual(learnerTraceFrames('workspace/attention.py:2: in forward\nC:\\work\\workspace\\attention.py:5: in helper','C:\\work\\workspace\\attention.py',source,'current'),[{row:0,location:{line:2}},{row:1,location:{line:5}}]);
  for(const message of ['reference.py:2: in forward','tests/test_core.py:2: in test_reference','torch/functional.py:2: in add','ValueError: submission.py:2: in f','    raise ValueError("submission.py:2: in f")','submission.py:0: in f','submission.py:9007199254740992: in f'])assert.deepEqual(learnerTraceFrames(message,'submission.py',source,'current'),[]);
  const {default:ReportView}=await server.ssrLoadModule('/src/ReportView.tsx'),{default:ReportWorkflow}=await server.ssrLoadModule('/src/ReportWorkflow.tsx');
  const {default:Diagnostic}=await server.ssrLoadModule('/src/FailureDiagnostic.tsx'),{LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
  const {createRoot}=await import('react-dom/client');
  const report={problem:'demo/core',impl:'browser/submission.py',summary:{T0:[0,1]},tests:[{id:'check_chain',layer:'T0',passed:false,message:trace}],created_at:'2026-10-16T00:00:00Z'};
  const original=JSON.stringify(report),problem={id:'demo/core',tests:{T0:1,T1:0,T2:0,T3:0}};
  let props={report,source,sourceState:'current',locatable:true},setLocale,direct=false;const located=[],runs=[];
  const locate=target=>{located.push(target);const editor=document.querySelector('[aria-label="Practice code"]');const offset=sourceOffset(props.source,target);assert.notEqual(offset,undefined);editor.focus({preventScroll:true});editor.setSelectionRange(offset,offset);};
  function Harness(){setLocale=useI18n().setLocale;const onLocate=props.locatable?locate:undefined;return React.createElement(React.Fragment,null,React.createElement('textarea',{'aria-label':'Practice code',readOnly:true,value:props.source}),direct?React.createElement(Diagnostic,{message:props.report.error,frames:learnerTraceFrames(props.report.error,props.report.impl,props.source,props.sourceState),onLocate}):React.createElement(React.Fragment,null,React.createElement(ReportWorkflow,{...props,onLocate,canRun:true,running:false,onEdit:()=>document.querySelector('[aria-label="Practice code"]').focus({preventScroll:true}),onRun:()=>runs.push(props.source)}),React.createElement(ReportView,{...props,onLocate,problem})));}
  root=createRoot(document.querySelector('#root'));
  const render=async patch=>{props={...props,...patch};await act(async()=>root.render(React.createElement(LocaleProvider,null,React.createElement(Harness))));};
  const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
  const waitFor=async(condition,message)=>{const until=Date.now()+2000;while(!condition()&&Date.now()<until)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(condition(),message);};
  const details=()=>document.querySelector('.failure-trace'),pre=()=>details().querySelector('.failure-message'),copy=()=>details().querySelector('.diagnostic-copy'),wrap=()=>details().querySelector('.diagnostic-wrap'),manual=()=>details().querySelector('.failure-diagnostic-manual textarea');
  const open=async()=>{await click(details().querySelector('summary'));await waitFor(()=>details().open&&details().querySelectorAll('.failure-frame-link').length,'The native disclosure exposes verified learner frames');};
  const pending=[],writes=[],clipboard={writeText(value){writes.push(value);return new Promise((resolve,reject)=>pending.push({resolve,reject,value}));}};
  Object.defineProperty(navigator,'clipboard',{value:clipboard,configurable:true});
  const begin=async()=>{await click(copy());assert.equal(copy().getAttribute('aria-busy'),'true');return pending.shift();};
  const settle=async(item,fail=false)=>act(async()=>fail?item.reject(Error('Denied')):item.resolve());
  await render();assert(!details().open);assert.equal(details().querySelectorAll('.failure-frame-link').length,0);assert.equal(pre().textContent,trace);assert.equal(writes.length,0);
  assert.equal(document.querySelector('.failure-insight .source-location-button').textContent.trim(),'↗Go to code line 5');
  const url=location.href,stored=JSON.stringify({...localStorage}),session=JSON.stringify({...sessionStorage});
  await open();const links=[...details().querySelectorAll('.failure-frame-link')];assert.equal(links.length,2);assert(links[0].textContent.includes('submission.py:2'));assert(links[1].textContent.includes('line 5'));assert.equal(pre().textContent,trace);assert(!document.querySelector('script'));
  const editor=document.querySelector('[aria-label="Practice code"]');editor.scrollTop=48;await click(links[0]);assert.equal(document.activeElement,editor);assert.equal(editor.selectionStart,sourceOffset(source,{line:2}));await click(links[1]);assert.equal(editor.selectionStart,sourceOffset(source,{line:5}));
  await click(wrap());assert(details().classList.contains('is-unwrapped'));const output=pre();output.scrollTop=91;output.scrollLeft=42;editor.setSelectionRange(3,12,'backward');
  const button=copy();let first;await act(async()=>{button.focus();button.click();button.click();first=pending.shift();});assert.equal(writes.length,1);assert.equal(first.value,trace);assert.equal(button.disabled,false);assert.equal(document.activeElement,button);assert.equal(deadlines.size,1);
  await act(async()=>setLocale('zh'));assert.equal(copy(),button);assert.equal(pre(),output);assert.equal(output.scrollTop,91);assert.equal(output.scrollLeft,42);assert.equal(wrap().getAttribute('aria-pressed'),'false');assert(details().open);await settle(first);assert(details().textContent.includes('完整诊断已复制'));
  await act(async()=>setLocale('en'));const denied=await begin();document.querySelector('#other-task').focus();await settle(denied,true);assert.equal(document.activeElement.id,'other-task');assert(manual().readOnly);assert.equal(manual().value,trace);manual().focus();assert.equal(manual().selectionStart,0);assert.equal(manual().selectionEnd,trace.length);
  const field=manual();field.setSelectionRange(5,19,'backward');field.scrollTop=31;await act(async()=>setLocale('zh'));assert.equal(manual(),field);assert.equal(field.selectionStart,5);assert.equal(field.selectionEnd,19);assert.equal(field.selectionDirection,'backward');assert.equal(field.scrollTop,31);await act(async()=>setLocale('en'));
  const timed=await begin();await act(async()=>[...deadlines.values()][0]());assert(manual());assert.equal(copy().getAttribute('aria-busy'),'false');const retry=await begin();await settle(timed);assert.equal(copy().getAttribute('aria-busy'),'true');await settle(retry);assert(!manual());
  assert.equal(document.querySelector('[aria-label="Practice code"]'),editor);assert.equal(editor.value,source);assert.equal(editor.selectionStart,3);assert.equal(editor.selectionEnd,12);assert.equal(editor.selectionDirection,'backward');assert.equal(editor.scrollTop,48);assert.equal(location.href,url);assert.equal(JSON.stringify({...localStorage}),stored);assert.equal(JSON.stringify({...sessionStorage}),session);
  await render({source:source+'# changed',sourceState:'changed'});assert.equal(details().querySelectorAll('.failure-frame-link').length,0);assert(!document.querySelector('.failure-insight .source-location-button'));assert(!details().querySelector('.failure-frame-hint'));assert.equal(pre().textContent,trace);await click(document.querySelector('.feedback-rerun'));assert.deepEqual(runs,[source+'# changed']);
  for(const sourceState of ['none','unknown','checking']){await render({sourceState});assert.equal(details().querySelectorAll('.failure-frame-link').length,0);}
  await render({source,sourceState:'current',locatable:false});assert.equal(details().querySelectorAll('.failure-frame-link').length,0);await render({locatable:true});assert.equal(details().querySelectorAll('.failure-frame-link').length,2);
  const old=await begin();const replacement={...report,tests:[{...report.tests[0],message:'RuntimeError: replacement'}]};await render({report:replacement});assert.equal(deadlines.size,0);await render({report});await settle(old,true);assert(!manual());assert(!details().open,'A replaced diagnostic starts collapsed and cannot revive old copy feedback');
  await open();Object.defineProperty(navigator,'clipboard',{value:undefined,configurable:true});await click(copy());assert(manual());assert.equal(deadlines.size,0);Object.defineProperty(navigator,'clipboard',{value:clipboard,configurable:true});
  // A run-level failure receives the same explicit frame navigation as a failed test.
  await render({report:{...report,error:trace,tests:[]}});assert(!details().open);await open();assert.equal(details().querySelectorAll('.failure-frame-link').length,2);await click(details().querySelector('.failure-frame-link'));assert.equal(editor.selectionStart,sourceOffset(source,{line:2}));
  const huge='log line\n'.repeat(20000)+'submission.py:5: in helper\nValueError: bad';direct=true;await render({report:{...report,error:huge,tests:[]}});assert.equal(pre().childNodes.length,1);await open();assert.equal(pre().textContent,huge);assert.equal(pre().querySelectorAll('button').length,1);assert(pre().childNodes.length<=3,'Long plain output remains coalesced text rather than one DOM node per log line');
  const unmounted=await begin();await act(async()=>root.unmount());root=undefined;assert.equal(deadlines.size,0);await settle(unmounted,true);assert.equal(document.querySelector('#root').textContent,'');assert.equal(JSON.stringify(report),original,'Reading and navigating never modifies reports, grades or source evidence');
  console.log('PASS traceback navigation: explicit current-source frame rows across Python/pytest/Windows/newlines/repeated calls, unchanged deepest target, inert library/stale/unknown frames, mounted test/run diagnostics and exact editor navigation, long-output text coalescing, wrap/locale/scroll continuity, bounded focused copy/manual fallback/retry/ABA/unmount, current-source rerun and immutable grades');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,descriptor]of previous)if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
