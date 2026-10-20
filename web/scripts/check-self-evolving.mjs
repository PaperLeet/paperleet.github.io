import assert from 'node:assert/strict';
import React, {act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtemp, rm, readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
const dom = new JSDOM('<div id="root"></div>', {url:'https://paperleet.test/', pretendToBeVisual:true});
const previous = new Map();
for (const [key,value] of Object.entries({window:dom.window, document:dom.window.document, navigator:dom.window.navigator, localStorage:dom.window.localStorage, HTMLElement:dom.window.HTMLElement, IS_REACT_ACT_ENVIRONMENT:true})) {
  previous.set(key,Object.getOwnPropertyDescriptor(globalThis,key));Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
}
const timers=new Map(), listeners=new Set();let timerId=0, requests=0, hidden=false;
const media={matches:false,addEventListener:(_type,fn)=>listeners.add(fn),removeEventListener:(_type,fn)=>listeners.delete(fn)};
window.matchMedia=()=>media;window.setTimeout=fn=>{timers.set(++timerId,fn);return timerId;};window.clearTimeout=id=>timers.delete(id);
Object.defineProperty(document,'hidden',{get:()=>hidden});
const oldFetch=globalThis.fetch;globalThis.fetch=()=>{requests++;throw Error('Walkthrough cannot send requests');};
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-evolution-');
const server=await createServer({root:process.cwd(),configFile:false,esbuild:{jsx:'automatic'},cacheDir,server:{watch:null,middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root;
try {
  const h=React.createElement,{createRoot}=await import('react-dom/client');
  const {default:Flow}=await server.ssrLoadModule('/src/SelfEvolvingFlow.tsx');
  const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
  const {evolutionFlows,consensusExample,hintExample,renderExample}=await server.ssrLoadModule('/src/selfEvolving.ts');
  const nav=await server.ssrLoadModule('/src/libraryNavigation.ts'),account=await server.ssrLoadModule('/src/accountNavigation.ts');
  const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts'),{parseHandoff}=await server.ssrLoadModule('/src/draftHandoff.ts'),{discussionPracticeTarget}=await server.ssrLoadModule('/src/discussionPracticeReturn.ts');
  const catalog=JSON.parse(await readFile('public/data/index.json','utf8')).problems.filter(p=>p.id.startsWith('self_evolving/'));
  assert.equal(catalog.length,7);
  for(const problem of catalog){
    for(const origin of ['#/collections/self-evolving','#/interviews?role=llm&track=self-evolving']){
      const href=nav.withLibraryContext(problem.id,origin);assert.equal(nav.libraryReturnHref(href),origin);
      assert.equal(account.accountReturnHref(decodeProblemHash(account.signInHref(href))),href);
    }
    const post='a'.repeat(32),href=nav.withLibraryContext(problem.id,'#/community/'+post);
    assert.equal(discussionPracticeTarget(href,post),problem.id);
    const handoff={version:1,id:'test-handoff',problem:problem.id,code:'private draft',created:1000};
    assert.deepEqual(parseHandoff(JSON.stringify(handoff),1000),handoff);
  }
  let paper='r-zero',locale;
  function App(){locale=useI18n();return h(Flow,{paper});}
  const render=()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(h(React.StrictMode,null,h(LocaleProvider,null,h(App))));});
  const click=node=>act(async()=>node.click());
  const advance=()=>act(async()=>{const queued=[...timers.values()];timers.clear();queued.forEach(fn=>fn());});
  const score=()=>Number(document.querySelector('[data-score=reward]').textContent);
  const range=async(index,value)=>act(async()=>{const input=document.querySelectorAll('input[type=range]')[index];Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set.call(input,String(value));input.dispatchEvent(new window.Event('input',{bubbles:true}));});
  // Independent numeric boundaries, including median ties and invalid-format semantics.
  assert.deepEqual(consensusExample(.5,2,true),{uncertainty:1,reward:.8,keep:true});
  for(const p of [.25,.75])assert(consensusExample(p,10,true).keep);
  assert(!consensusExample(.8,1,true).keep);assert.equal(consensusExample(.5,1,false).reward,0);
  assert.equal(hintExample(.1,200,true).median,.1);assert(hintExample(.1,200,true).keep);
  assert(!hintExample(.6,200,true).keep);assert.equal(hintExample(.6,500,false).reward,-1.1);
  assert(Math.abs(hintExample(.6,300,true).reward-.47)<1e-12);
  assert.deepEqual(renderExample(0,.8,.6),{difficulty:.4,proposer:0,coder:2.2,failedCoder:-.1});
  assert(Math.abs(renderExample(3,.8,.6).proposer-.675)<1e-12);
  for(const id of Object.keys(evolutionFlows)){
    paper=id;await render();const flow=evolutionFlows[id];
    assert.equal(document.querySelector('.evolution').dataset.paper,id);
    assert.equal(document.querySelectorAll('.evolution-node').length,4);
    assert.equal(document.querySelector('a').href,flow.source);
    for(const input of document.querySelectorAll('input[type=range]'))assert.equal(input.labels[0].htmlFor,input.id,'Each slider has an explicit visible label despite the neighboring output control');
    assert.equal(timers.size,0,'No autoplay');
    for(let i=0;i<flow.steps.length;i++){
      await click(document.querySelectorAll('.evolution-steps button')[i]);
      const roles=[...document.querySelectorAll('.evolution-roles>span')];
      assert.equal(roles.filter(r=>r.dataset.updating==='true').length,flow.steps[i].update===undefined?0:1);
      if(flow.steps[i].update!==undefined)assert(roles[flow.steps[i].update].textContent.includes('Updating'));
      assert(document.querySelector(`.node-${flow.steps[i].node}`).dataset.active==='true');
      assert(document.querySelector('.evolution-detail').textContent.includes(flow.steps[i].detail[1]));
    }
    // Replay starts at zero, runs one bounded round and cleans its timer at the end.
    await click(document.querySelector('.evolution-play'));assert.equal(timers.size,1);
    for(let i=1;i<flow.steps.length;i++)await advance();
    assert.equal(document.querySelector('.evolution-play').getAttribute('aria-pressed'),'false');assert.equal(timers.size,0);
  }
  paper='r-zero';await render();assert.equal(score(),.6);await range(0,.9);assert.equal(score(),0);assert.equal(document.querySelector('.evolution-selection').dataset.kept,'false');
  await range(0,.5);await range(1,10);assert.equal(score(),0);assert.equal(document.querySelector('.evolution-selection').dataset.kept,'true');
  await click(document.querySelector('input[type=checkbox]'));assert.equal(document.querySelector('.evolution-selection').dataset.kept,'false');
  paper='visplay';await render();assert(document.body.textContent.includes('same-image'));assert.equal(score(),.6);
  paper='g-zero';await render();assert.equal(score(),0);await range(0,.6);assert.equal(score(),.5);assert.equal(document.querySelector('.evolution-selection').dataset.kept,'false');await range(1,300);assert.equal(score(),.47);
  paper='mm-zero';await render();assert.equal(score(),.675);await range(0,0);assert.equal(score(),0);assert.equal(document.querySelectorAll('[data-success=true]').length,0);await range(0,4);assert.equal(score(),.9);
  const input=document.querySelector('input');input.focus();await act(async()=>locale.setLocale('zh'));assert.equal(document.activeElement,input);assert.equal(document.querySelector('input'),input);assert(document.body.textContent.includes('成功渲染'));assert.equal(score(),.9);
  await click(document.querySelector('.evolution-play'));assert.equal(timers.size,1);hidden=true;await act(async()=>document.dispatchEvent(new window.Event('visibilitychange')));assert.equal(timers.size,0);hidden=false;
  await click(document.querySelector('.evolution-play'));media.matches=true;await act(async()=>listeners.forEach(fn=>fn()));assert.equal(timers.size,0);assert.equal(document.querySelector('.evolution').dataset.playing,'false');
  await click(document.querySelectorAll('.evolution-steps button')[3]);assert(document.body.textContent.includes('正在更新'));
  await click(document.querySelector('.evolution-play'));assert.equal(timers.size,1);await act(async()=>root.unmount());root=null;assert.equal(timers.size,0);assert.equal(listeners.size,0);assert.equal(requests,0);
  const lab=await readFile('src/Lab.tsx','utf8');assert(lab.includes('problem.id.startsWith("self_evolving/")'));assert(lab.includes('<SelfEvolvingFlow'));assert(lab.includes('<ToyExperiment'));
  console.log('PASS self-evolving: four paper-specific bilingual flows, alternating update/freeze ownership, source figures, numerical reward/filter boundaries, actual sliders, median ties, rendering failures, explicit bounded playback, visibility/reduced-motion pause, locale focus retention and cleanup; no network or training claims');
} finally {
  if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});globalThis.fetch=oldFetch;dom.window.close();for(const[k,v]of previous)v?Object.defineProperty(globalThis,k,v):delete globalThis[k];
}
