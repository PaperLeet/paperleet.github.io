import assert from 'node:assert/strict';
import React, {act} from 'react';
import {JSDOM} from 'jsdom';

export async function checkFailurePicker(server, Workflow, props) {
  const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/22'});
  const names=['window','document','HTMLElement','localStorage','location','IS_REACT_ACT_ENVIRONMENT'];
  const previous=new Map(names.map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  for(const name of names)Object.defineProperty(globalThis,name,{value:name==='IS_REACT_ACT_ENVIRONMENT'?true:dom.window[name],configurable:true,writable:true});
  localStorage.setItem('paperleet.locale','en');
  const {createRoot}=await import('react-dom/client');
  const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
  let locale,root;
  let edits=0,runs=0,located=[];
  const report={...props.report,error:'RuntimeError: interrupted',tests:Array.from({length:15},(_,index)=>({id:'check_'+index,layer:'T0',passed:index===7,message:`submission.py:2: in f\nE   ValueError: failure ${index}`}))};
  const before=structuredClone(report);
  let current={...props,report,onEdit(){edits++;},onRun(){runs++;},onLocate(value){located.push(value);}};
  function Harness(){locale=useI18n();return React.createElement(React.Fragment,null,React.createElement('textarea',{'aria-label':'Code',defaultValue:props.source}),React.createElement(Workflow,current));}
  const render=async()=>act(async()=>root.render(React.createElement(LocaleProvider,null,React.createElement(Harness))));
  const picker=()=>document.querySelector('.feedback-failure-picker');
  const choose=async value=>act(async()=>{picker().focus();picker().value=value;picker().dispatchEvent(new dom.window.Event('change',{bubbles:true}));});
  const click=async node=>act(async()=>{node.focus();node.click();});
  try {
    root=createRoot(document.getElementById('root'));await render();
    assert(picker(),'Many failures need a direct native selection control');
    assert.equal(picker().getAttribute('aria-label'),'Choose a failure');
    assert.equal(picker().options.length,15,'The interrupted run and fourteen failed tests are selectable; passing tests are excluded');
    assert(picker().options[0].textContent.includes('Run interrupted'));
    const draft=document.querySelector('textarea');draft.value+='\n# unsent changes';draft.setSelectionRange(2,8);draft.scrollTop=27;
    const source=location.href,stored={...localStorage};
    await choose('test-14');
    assert.equal(document.activeElement,picker());assert.equal(document.querySelector('.feedback-diagnostic').textContent,'ValueError: failure 14');
    assert.equal(document.querySelector('.feedback-error-nav>span').textContent,'15 / 15');
    assert(document.querySelector('button[aria-label="Next failure"]').disabled);
    await click(document.querySelector('.source-location-button'));assert.deepEqual(located,[{line:2}]);
    await choose('test-0');await click(document.querySelector('button[aria-label="Previous failure"]'));assert.equal(picker().value,'run-error');
    await click(document.querySelector('button[aria-label="Next failure"]'));assert.equal(picker().value,'test-0');
    await choose('test-14');await act(async()=>locale.setLocale('zh'));
    assert.equal(picker().getAttribute('aria-label'),'选择失败项');assert.equal(picker().value,'test-14');
    current={...current,sourceState:'changed',source:props.source+'# modified'};await render();
    assert(!document.querySelector('.source-location-button'));assert(!document.querySelector('.feedback-code-context'));
    assert(document.querySelector('.feedback-diagnostic').textContent.includes('上次运行：'));
    current={...current,running:true};await render();
    assert(picker().disabled);assert([...document.querySelectorAll('.feedback-error-nav button')].every(node=>node.disabled));
    assert(!document.querySelector('.feedback-diagnostic'),'Old diagnostics stay hidden during a new run');
    current={...current,running:false,source:props.source,sourceState:'current',report:{...report,error:undefined,tests:report.tests.slice(0,2)}};await render();
    assert.equal(picker().value,'test-0','A new report resets to its first failure');
    current={...current,report:{...current.report,tests:[report.tests[0]]}};await render();assert(!picker(),'A single failure does not need a picker');
    assert.equal(document.querySelector('textarea'),draft);assert(draft.value.endsWith('# unsent changes'));assert.equal(draft.selectionStart,2);assert.equal(draft.selectionEnd,8);assert.equal(draft.scrollTop,27);
    assert.equal(location.href,source);assert.deepEqual({...localStorage},{...stored,'paperleet.locale':'zh'});assert.equal(edits,0);assert.equal(runs,0);
    assert.deepEqual(report,before,'Selecting failures cannot mutate the report or grades');
    await click(document.querySelector('.feedback-actions>.button:not(.source-location-button)'));assert.equal(edits,1);
    await click(document.querySelector('.feedback-rerun'));assert.equal(runs,1,'An explicit run action still reaches the workspace');
    console.log('PASS direct failure picker: full failed-only list, run interruption, distant selection, arrows/count, verified line location, locale continuity, changed/running guards, new-report reset and retained editor/caret/scroll/source');
  } finally {
    // This fixture has only synchronous effect cleanup. Flush the unmount
    // before closing JSDOM instead of waiting on an extra async act scope.
    if(root){
      act(()=>root.unmount());
      assert.equal(dom.window.document.getElementById('root').childElementCount,0,'Unmount must clear the fixture before its DOM is closed');
    }
    dom.window.close();
    for(const [name,descriptor] of previous)if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];
  }
}
