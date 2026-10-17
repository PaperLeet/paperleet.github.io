import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

// Real reference/code components and highlighter in an isolated DOM. Clipboard,
// geometry and focus are controlled here; no OS clipboard or browser is used.
const dom=new JSDOM('<!doctype html><div id="root"></div><button id="other">Other work</button>',{url:'https://paperleet.test/#/22?from=collections%2Fllm-basics',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true};
const originals=new Map(Object.keys(globals).map(k=>[k,Object.getOwnPropertyDescriptor(globalThis,k)]));
for(const[k,v]of Object.entries(globals))Object.defineProperty(globalThis,k,{value:v,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
window.matchMedia=()=>({matches:false});
window.scrollTo=()=>{};
window.HTMLElement.prototype.scrollIntoView=function(){};
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root;
try {
  const {createRoot}=await import('react-dom/client');
  const {default:Reference}=await server.ssrLoadModule('/src/ReferenceSolution.tsx');
  const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
  let solution='# Reference 中文 😀 <script> stays literal\n\ndef reference(x):\n\treturn "'+ 'long-token-'.repeat(30)+'"\n',owner='alice',setLocale,returns=0;
  const original=solution,writes=[],pending=[];
  Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText(value){writes.push(value);return new Promise((resolve,reject)=>pending.push({resolve,reject}));}}});
  function Harness(){setLocale=useI18n().setLocale;return React.createElement(React.Fragment,null,
    React.createElement('textarea',{'aria-label':'Own practice code',defaultValue:'def own_answer(x):\n    return x # private'}),
    React.createElement('div',{className:'description-scroll',style:{overflow:'auto',scrollPaddingTop:'16px',scrollPaddingBottom:'16px'}},
      React.createElement(Reference,{key:owner,solution,onCode:()=>{returns++;document.querySelector('[aria-label="Own practice code"]').focus({preventScroll:true});}})));}
  root=createRoot(document.querySelector('#root'));
  const render=async()=>act(async()=>root.render(React.createElement(LocaleProvider,null,React.createElement(Harness))));
  const find=s=>document.querySelector(s),details=()=>find('.reference-reader'),summary=()=>details().querySelector('summary'),block=()=>find('.community-code-block'),copy=()=>block()?.querySelector('[aria-busy]'),wrap=()=>block()?.querySelector('[aria-pressed]'),manual=()=>find('.community-code-manual textarea'),status=()=>block()?.querySelector('[role="status"]'),collapse=()=>find('.reference-reading-actions button'),back=()=>find('.reference-reading-actions button:last-child');
  const click=async node=>{assert(node,'Expected reference control');await act(async()=>{node.focus();node.click();});};
  const settle=async predicate=>{for(let n=0;n<150&&!predicate();n++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),'Expected reference reading state');};
  const reveal=async()=>{await click(summary());await settle(()=>details().open&&block());};
  const key=async(node,options={})=>{const e=new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true,...options});await act(async()=>{node.focus();node.dispatchEvent(e);});return e;};
  const begin=async()=>{await click(copy());assert.equal(copy().getAttribute('aria-busy'),'true');return pending.shift();};
  const finish=async(item,fail=false)=>act(async()=>fail?item.reject(Error('Clipboard unavailable')):item.resolve());
  await render();
  const code=find('[aria-label="Own practice code"]'),pane=find('.description-scroll'),href=location.href,stored=JSON.stringify({...sessionStorage});
  code.setSelectionRange(3,15,'backward');code.scrollTop=39;code.scrollLeft=17;
  const assertCode=()=>{assert(find('[aria-label="Own practice code"]')===code);assert.deepEqual([code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],['def own_answer(x):\n    return x # private',3,15,'backward',39,17]);assert.equal(JSON.stringify({...sessionStorage}),stored);};
  Object.defineProperties(pane,{clientHeight:{get:()=>500},clientTop:{get:()=>0}});
  pane.getBoundingClientRect=()=>({top:100,bottom:600,height:500});
  const moves=[];pane.scrollTo=options=>{moves.push(options);pane.scrollTop=options.top;};
  const measure=()=>{summary().getBoundingClientRect=()=>({top:130-pane.scrollTop,bottom:170-pane.scrollTop,height:40});};measure();
  assert(!details().open&&!block()&&!document.body.textContent.includes(solution));assert.equal(writes.length,0);
  assert.equal(summary().getAttribute('aria-expanded'),'false');
  await reveal();assert.equal(writes.length,0,'Revealing an answer never accesses the clipboard');assert.equal(summary().getAttribute('aria-expanded'),'true');
  await settle(()=>block().querySelector('pre.shiki'));
  assert(!block().querySelector('script'));assert.equal(block().querySelector('pre code').textContent,solution);
  await click(wrap());assert(block().classList.contains('is-wrapped'));assert.equal(wrap().getAttribute('aria-pressed'),'true');
  const first=await begin();await click(copy());assert.equal(writes.length,1);assert.equal(writes[0],solution);assert.equal(document.activeElement,copy());
  await act(async()=>setLocale('zh'));assert(block().classList.contains('is-wrapped'));assert(copy().textContent.includes('正在复制'));await finish(first);assert(status().textContent.includes('代码已复制'));
  await act(async()=>setLocale('en'));
  const rejected=await begin();find('#other').focus();await finish(rejected,true);assert(document.activeElement===find('#other'),'Copy failure cannot steal newer focus');assert.equal(manual().value,solution);manual().focus();assert.equal(manual().selectionEnd,solution.length);
  const field=manual();field.setSelectionRange(4,21,'backward');field.scrollTop=23;await act(async()=>setLocale('zh'));assert(manual()===field);assert.deepEqual([field.selectionStart,field.selectionEnd,field.selectionDirection,field.scrollTop],[4,21,'backward',23]);
  assert(!(await key(field)).defaultPrevented&&details().open,'Escape in manual selection belongs to the native field');await act(async()=>setLocale('en'));
  for(const options of [{isComposing:true},{keyCode:229},{repeat:true},{altKey:true},{ctrlKey:true},{metaKey:true},{shiftKey:true}])assert(!(await key(wrap(),options)).defaultPrevented&&details().open,'IME, repeat and modified shortcuts leave the reference open');
  const consumed=new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true});consumed.preventDefault();await act(async()=>wrap().dispatchEvent(consumed));assert(details().open);
  await click(back());assert.equal(returns,1);assert(document.activeElement===code);assertCode();assert.equal(location.href,href);
  pane.scrollTop=700;const lateClosed=await begin();await click(collapse());assert(!details().open&&!block());assert(document.activeElement===summary());assert.equal(pane.scrollTop,14,'Collapse reveals its summary in this reader using the shared visibility rule');
  await reveal();await finish(lateClosed,true);assert.equal(status().textContent,'');assert(!manual(),'Late copy failure cannot recreate a collapsed reader field');assert(!block().classList.contains('is-wrapped'));
  pane.scrollTop=640;const beforeMoves=moves.length;await act(async()=>{collapse().focus();collapse().click();code.focus();});assert(document.activeElement===code);assert.equal(pane.scrollTop,640);assert.equal(moves.length,beforeMoves,'Newer focus suppresses old collapse scrolling');
  await reveal();measure();const escaped=await key(wrap());assert(escaped.defaultPrevented&&!details().open);assert(document.activeElement===summary());
  await reveal();const routeSummary=summary();await act(async()=>{collapse().focus();collapse().click();window.history.replaceState(null,'','#/interviews');});assert(!details().open);assert(document.activeElement!==routeSummary,'A changed route owns focus');window.history.replaceState(null,'',href);
  await reveal();const old=await begin();solution='def updated(x):\r\n\treturn "新答案 😀"\r\n';await render();measure();assert(!details().open&&!block());assert(!document.body.textContent.includes(solution),'Updated reference requires fresh spoiler consent');await reveal();const updated=await begin();assert.equal(writes.at(-1),solution,'Copy preserves exact CRLF, tabs and Unicode source bytes');await finish(updated);solution=original;await render();measure();assert(!details().open);await reveal();await finish(old,true);assert.equal(status().textContent,'');assert(!manual(),'A → B → A source replacement cannot revive old copy state');
  const oldOwner=await begin();owner='bob';await render();measure();assert(!details().open&&!block());await reveal();await finish(oldOwner);assert.equal(status().textContent,'');
  solution=undefined;await render();assert(!details().open);await click(summary());await settle(()=>details().querySelector('.notice'));assert(details().open&&!block());assert(details().textContent.includes('No reference implementation is available yet.'));assert(back());await click(back());assertCode();
  const outsideEscape=await key(code);assert(!outsideEscape.defaultPrevented&&details().open,'Editor Escape never closes the reference');
  assert.equal(location.href,href);assertCode();assert(!writes.includes(code.value),'Reference copy never uses the learner’s private implementation');
  console.log('PASS reference solution reading: explicit spoiler consent, real safe highlighting and wrapping, exact raw/CRLF/Unicode copy, deduplicated async feedback and manual selection, localized continuity, guarded collapse/Escape/back-to-code, source/owner/ABA retirement and unchanged private code/caret/scroll/storage');
} finally {
  if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
  for(const[k,d]of originals){if(d)Object.defineProperty(globalThis,k,d);else delete globalThis[k];}
}
