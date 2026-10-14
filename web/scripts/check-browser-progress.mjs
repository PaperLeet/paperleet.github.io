import assert from 'node:assert/strict';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
const originalStorage = globalThis.localStorage;
try {
  const { advanceBrowserProgress, browserProgressText } = await server.ssrLoadModule('/src/browserProgress.ts');
  const { BrowserRunProgress } = await server.ssrLoadModule('/src/BrowserRunProgress.tsx');
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const totals = { tests: 7, mutants: 3 };
  let current = { stage:'loading', step:'bundle' };
  const accept = message => { const next=advanceBrowserProgress(current,message,totals); assert(next); current=next; };
  const ignore = message => assert.equal(advanceBrowserProgress(current,message,totals),undefined);
  for (const step of ['python','syntax','packages','preparing']) accept({stage:'loading',step});
  ignore({stage:'loading',step:'python'});
  accept({stage:'calibrating'}); // Existing phase-only workers remain supported.
  accept({stage:'calibrating',step:'reference',completed:0,total:7});
  const activity = { index:0, test:'test_reference', layer:'T1', why:'The original reference contract.' };
  accept({stage:'calibrating',step:'reference',completed:0,total:7,activity});
  assert.deepEqual(current.activity,activity);
  ignore({stage:'calibrating',step:'reference',completed:0,total:7,activity});
  ignore({stage:'calibrating',step:'reference',completed:0,total:7,activity:{...activity,index:1}});
  for (const bad of [null, [], {stage:'unknown'}, {stage:'loading',step:'packages'},
    {stage:'calibrating',step:'tests',completed:0,total:7},
    ...[-1,8,NaN,Infinity,.5,'1'].map(completed=>({stage:'calibrating',step:'reference',completed,total:7})),
    {stage:'calibrating',step:'reference',completed:1,total:8}]) ignore(bad);
  accept({stage:'calibrating',step:'reference',completed:7,total:7});
  ignore({stage:'calibrating',step:'reference',completed:6,total:7});
  accept({stage:'calibrating',step:'mutants',completed:0,total:3});
  ignore({stage:'calibrating',step:'reference',completed:7,total:7});
  ignore({stage:'calibrating',step:'mutants',completed:0,total:3});
  accept({stage:'calibrating',step:'mutants',completed:0,total:3,activity:{...activity,mutant:'m1_wrong'}});
  accept({stage:'calibrating',step:'mutants',completed:0,total:3,activity:{...activity,index:1,mutant:'m1_wrong'}});
  ignore({stage:'calibrating',step:'mutants',completed:0,total:3,activity:{...activity,index:0,mutant:'m1_wrong'}});
  ignore({stage:'calibrating',step:'mutants',completed:0,total:3,activity:{...activity,index:2,mutant:'m2_other'}});
  accept({stage:'calibrating',step:'mutants',completed:1,total:3});
  assert.equal(current.activity,undefined,'Completed mutation progress clears its previous test');
  accept({stage:'calibrating',step:'mutants',completed:1,total:3,activity:{...activity,mutant:'m2_other'}});
  accept({stage:'calibrating',step:'mutants',completed:3,total:3});
  accept({stage:'running',step:'tests',completed:0,total:7});
  accept({stage:'running',step:'tests',completed:2,total:7});
  for (const bad of [null,[],{}, {...activity,index:-1}, {...activity,index:7}, {...activity,index:2.5}, {...activity,index:'2'},
    {...activity,index:2,test:'x'.repeat(201)}, {...activity,index:2,test:'\u0000'}, {...activity,index:2,why:' '},
    {...activity,index:2,why:'x'.repeat(321)}, {...activity,index:2,layer:'T4'}, {...activity,index:2,layer:{}}, {...activity,index:2,mutant:'wrong-stage'}]) {
    ignore({stage:'running',step:'tests',completed:2,total:7,activity:bad});
  }
  accept({stage:'running',step:'tests',completed:2,total:7,activity:{...activity,index:2,test:'<img src=x onerror=alert(1)>',secret:'ignored'}});
  assert.equal(current.activity.secret,undefined);
  assert.deepEqual(advanceBrowserProgress(current,{stage:'running',step:'tests',completed:3,total:7,activity:{...activity,index:3,why:'x'.repeat(321)}},totals),
    {stage:'running',step:'tests',completed:3,total:7},'Invalid optional details cannot block a valid completed-test counter');
  ignore({stage:'running'});
  ignore({stage:'running',step:'tests',completed:1,total:7});
  assert.equal(browserProgressText(current),'已完成 2 / 7 项测试');
  assert.equal(browserProgressText(current,true),'Tests completed 2 / 7');
  for (const en of [false,true]) {
    globalThis.localStorage = { getItem:()=>en?'en':'zh' };
    const markup = renderToString(React.createElement(LocaleProvider,null,React.createElement(BrowserRunProgress,{progress:current,onStop:()=>{}})));
    assert(markup.includes('aria-current="step"') && markup.includes('role="status"') && markup.includes('aria-live="polite"'));
    assert(markup.includes('value="2"') && markup.includes('max="7"'));
    assert(markup.includes(en?'Stop run':'停止运行'));
    assert(!markup.includes(en?'2 / 7 tests passed':'2 / 7 项通过'), 'Completion counts must not claim acceptance');
    assert(markup.includes(en?'Current test':'当前测试'));
    assert(markup.includes('&lt;img')&&!markup.includes('<img'),'Public test descriptions render as escaped text');
    assert(markup.includes('The original reference contract.'));
    const mutation = renderToString(React.createElement(LocaleProvider,null,React.createElement(BrowserRunProgress,{progress:{stage:'calibrating',step:'mutants',completed:0,total:3,activity:{...activity,mutant:'m1_wrong'}},onStop:()=>{}})));
    assert(mutation.includes('m1_wrong')&&mutation.includes(en?'Mutation case':'错误实现'));
    const loading = renderToString(React.createElement(LocaleProvider,null,React.createElement(BrowserRunProgress,{progress:{stage:'loading',step:'packages'},onStop:()=>{}})));
    assert(!loading.includes('value='), 'Loading uses an indeterminate bar, never an invented percentage');
    assert(loading.includes(en?'first run':'首次运行'));
  }
  accept({stage:'running',step:'tests',completed:7,total:7});
  assert.equal(current.activity,undefined);
  ignore({stage:'running',step:'tests',completed:7,total:7,activity:{...activity,index:6}});
  ignore({stage:'calibrating'});
  console.log('PASS browser progress: actual test/mutation activity, bounded escaped metadata, monotonic counts, stale/duplicate rejection, legacy phases, bilingual accessible steps and no fabricated completion claims');
} finally {
  if (originalStorage === undefined) delete globalThis.localStorage;
  else globalThis.localStorage = originalStorage;
  await server.close();
}
