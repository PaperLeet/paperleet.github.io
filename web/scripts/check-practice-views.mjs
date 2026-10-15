import assert from 'node:assert/strict';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

const dom = new JSDOM('<div id="root"></div>', {url:'https://paperleet.test'});
const names = ['window','document','navigator','HTMLElement','CustomEvent','localStorage','IS_REACT_ACT_ENVIRONMENT'];
const originals = new Map(names.map(name => [name,Object.getOwnPropertyDescriptor(globalThis,name)]));
for (const name of names) Object.defineProperty(globalThis,name,{value:name === 'IS_REACT_ACT_ENVIRONMENT' ? true : dom.window[name],configurable:true,writable:true});
const { createRoot } = await import('react-dom/client');
const server = await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
const root = createRoot(document.getElementById('root'));
const originalNow = Date.now;
let now = originalNow(), visible = true;
Date.now = () => now;
Object.defineProperty(document,'visibilityState',{configurable:true,get:()=>visible ? 'visible' : 'hidden'});
try {
  const { parseReport } = await server.ssrLoadModule('/src/reports.ts');
  const { submissionDestination } = await server.ssrLoadModule('/src/submissionNavigation.ts');
  const id = 'a'.repeat(32);
  const report = parseReport({problem:'rl/dpo',run_id:id,tests:[{id:'loss',layer:'T0',passed:false}],summary:{T0:[0,1]}});
  assert.equal(report.run_id,id,'The queue report must retain the server run ID');
  assert.throws(()=>parseReport({...report,run_id:'../another-account'}));
  const entry = {id:'rl/dpo',number:3};
  for (const route of ['3','rl/dpo','3?from=collections/llm']) assert.deepEqual(submissionDestination(report,'user:alice',entry,route),{href:`#/submission/${id}`,redirect:true});
  for (const route of ['', '4', 'u/alice', `submission/${id}`]) assert.equal(submissionDestination(report,'user:alice',entry,route).redirect,false,'Completing a job cannot hijack another page');
  assert.equal(submissionDestination({...report,run_id:undefined},'user:alice',entry,'3'),undefined);
  assert.equal(submissionDestination(report,'anonymous',entry,'3'),undefined,'Browser-only tests have no server result URL');

  const { publishPracticeChange } = await server.ssrLoadModule('/src/practiceChanges.ts');
  const { usePracticeInvalidation } = await server.ssrLoadModule('/src/usePracticeInvalidation.ts');
  let refreshes = 0;
  function Probe({owner='user:alice'}) { usePracticeInvalidation(()=>refreshes++,owner,'rl/dpo'); return null; }
  await act(async()=>root.render(React.createElement(Probe)));
  const hint = change => act(async()=>publishPracticeChange({owner:'user:alice',problem:'rl/dpo',kind:'submission',...change}));
  await hint({kind:'draft'});
  await hint({owner:'user:bob'});
  await hint({problem:'rl/grpo'});
  assert.equal(refreshes,0,'Typing, another account, or another problem cannot refresh this history');
  await hint({});
  assert.equal(refreshes,1);
  visible = false;
  await hint({}); await hint({});
  assert.equal(refreshes,1,'Background hints wait until the view is visible');
  visible = true;
  await act(async()=>document.dispatchEvent(new window.Event('visibilitychange')));
  assert.equal(refreshes,2,'Coalesced background changes refresh once on return');
  await act(async()=>window.dispatchEvent(new window.Event('focus')));
  assert.equal(refreshes,2,'Focus and visibility events must not make duplicate requests');
  now += 5_000;
  await act(async()=>window.dispatchEvent(new window.Event('focus')));
  assert.equal(refreshes,3,'Focus recovers missed cross-window events');
  await act(async()=>root.render(React.createElement(Probe,{owner:'user:bob'})));
  await hint({});
  assert.equal(refreshes,3,'Changing accounts removes the old subscription');

  const { LocaleProvider, useI18n } = await server.ssrLoadModule('/src/i18n.tsx');
  function LocaleProbe() { return React.createElement('output',null,useI18n().locale); }
  await act(async()=>root.render(React.createElement(LocaleProvider,null,React.createElement(LocaleProbe))));
  localStorage.setItem('paperleet.locale','zh');
  await act(async()=>window.dispatchEvent(new window.StorageEvent('storage',{key:'paperleet.locale',newValue:'zh'})));
  assert.equal(document.querySelector('output').textContent,'zh');
  assert.equal(document.documentElement.lang,'zh-CN');
  localStorage.setItem('paperleet.locale','en');
  await act(async()=>window.dispatchEvent(new window.StorageEvent('storage',{key:'paperleet.locale',newValue:'en'})));
  assert.equal(document.querySelector('output').textContent,'en');
  assert.equal(refreshes,3,'Unmount removes server-view subscriptions');
  console.log('PASS submission routing, retained run ID, scoped/visible/focus view invalidation, cleanup, and cross-window language');
} finally {
  await act(async()=>root.unmount());
  Date.now = originalNow;
  await server.close(); dom.window.close();
  for (const [name,descriptor] of originals) descriptor ? Object.defineProperty(globalThis,name,descriptor) : delete globalThis[name];
}
