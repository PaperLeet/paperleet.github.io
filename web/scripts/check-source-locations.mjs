import assert from 'node:assert/strict';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', optimizeDeps:{noDiscovery:true,include:[]} });
const storage = globalThis.localStorage;
try {
  const { editorSource, sourceOffset, learnerTraceLocation } = await server.ssrLoadModule('/src/sourceLocation.ts');
  const { SourceLocationButton } = await server.ssrLoadModule('/src/SourceLocationButton.tsx');
  const { BrowserSyntaxNotice } = await server.ssrLoadModule('/src/BrowserSyntaxNotice.tsx');
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const source = '# 🧪\r\n变量 = "😀" + (\r\n\treturn 3\r';
  const normalized = editorSource(source);
  assert.equal(normalized, '# 🧪\n变量 = "😀" + (\n\treturn 3\n');
  assert.equal(sourceOffset(normalized, { line: 2, column: 12 }), normalized.indexOf('('));
  assert.equal(sourceOffset(normalized, { line: 3, column: 2 }), normalized.indexOf('return'));
  assert.equal(sourceOffset(normalized, { line: 2, column: 500 }), normalized.indexOf('\n', 6));
  assert.equal(sourceOffset(normalized, { line: 5 }), normalized.length);
  assert.equal(sourceOffset('', { line: 1 }), 0);
  assert.equal(sourceOffset('a\n', { line: 2 }), 2);
  for (const location of [{ line: 0 }, { line: -1 }, { line: Infinity }, { line: 1.5 }, { line: 10 }, { line: 1, column: 0 }, { line: 1, column: NaN }]) assert.equal(sourceOffset(normalized, location), undefined);
  const trace = `tests/test_reference.py:32: in test_reference\n    output = impl.forward(x)\nsubmission.py:2: in forward\n    return inner(x)\nsubmission.py:3: in inner\n    raise ValueError('bad')\nE   ValueError: bad\nreference.py:1: in reference\n`;
  assert.deepEqual(learnerTraceLocation(trace, 'browser/submission.py', normalized, 'current'), { line: 3 });
  assert.deepEqual(learnerTraceLocation('/tmp/paperleet-job-abc/submission.py:2: in forward', 'submission.py', normalized, 'current'), { line: 2 });
  assert.deepEqual(learnerTraceLocation('workspace/attention.py:2: in forward', '/Users/student/paperleet/workspace/attention.py', normalized, 'current'), { line: 2 });
  assert.deepEqual(learnerTraceLocation('C:\\papers\\workspace\\attention.py:2: in forward', 'C:\\papers\\workspace\\attention.py', normalized, 'current'), { line: 2 });
  assert.deepEqual(learnerTraceLocation('  File "submission.py", line 2, in forward\n  File "submission.py", line 3, in inner\n  File "reference.py", line 1, in helper','browser/submission.py',normalized,'current'),{line:3},'Standard Python tracebacks select the deepest explicit learner frame');
  assert.equal(learnerTraceLocation('  File "reference.py", line 2, in forward','browser/submission.py',normalized,'current'),undefined);
  assert.equal(learnerTraceLocation('  File "submission.py", line 2, in forward','browser/submission.py',normalized,'changed'),undefined);
  assert.deepEqual(learnerTraceLocation('/home/a:b/attention.py:2: in forward', '/home/a:b/attention.py', normalized, 'current'), { line: 2 });
  for (const state of ['none', 'unknown', 'checking', 'changed']) assert.equal(learnerTraceLocation(trace, 'browser/submission.py', normalized, state), undefined);
  for (const message of ['tests/test_reference.py:2: in test_reference', 'reference.py:2: in forward', 'solution.py:2: in forward', 'paperleet/browser/torch.py:2: in add', 'submission.py:50: in forward', 'submission.py:0: in forward', 'submission.py:1.5: in forward', 'ValueError: submission.py:2: wrong size', '    raise ValueError("submission.py:2: in forward")', 'File "submission.py", line 2', 'submission.py:2: ValueError']) assert.equal(learnerTraceLocation(message, 'browser/submission.py', normalized, 'current'), undefined, message);
  assert.equal(learnerTraceLocation('attention.py:2: in forward', '/work/workspace/attention.py', normalized, 'current'), undefined, 'Ambiguous basename alone must not match a local implementation');
  assert.equal(learnerTraceLocation('solution.py:2: in forward', 'solution.py', normalized, 'current'), undefined);
  assert.equal(learnerTraceLocation(trace, undefined, normalized, 'current'), undefined);
  for (const en of [false, true]) {
    globalThis.localStorage = { getItem: () => en ? 'en' : 'zh' };
    const render = (Component, props) => renderToString(React.createElement(LocaleProvider, null, React.createElement(Component, props)));
    const button = render(SourceLocationButton, { location: { line: 3 }, onLocate() {} });
    assert(button.includes('type="button"') && button.includes(en ? 'Go to code line 3' : '定位代码第 3 行'));
    assert.equal(render(SourceLocationButton, { location: { line: 3 } }), '');
    const failure = { source: normalized, diagnostic: { kind: 'SyntaxError', message: 'unexpected <tag>', line: 2, column: 12 } };
    const props = { failure, source: normalized, onLocate() {} };
    assert(render(BrowserSyntaxNotice, props).includes(en ? 'Go to code line 2' : '定位代码第 2 行'));
    assert(!render(BrowserSyntaxNotice, { ...props, source: normalized + '#' }).includes('<button'), 'A previous syntax failure cannot navigate modified code');
    assert(!render(BrowserSyntaxNotice, { ...props, onLocate: undefined }).includes('<button'), 'An unresolved workspace cannot navigate an editor');
  }
  console.log('PASS source navigation: Unicode/line endings/EOF, explicit learner frames, deepest frame, native/browser paths, version gating and bilingual actions');
} finally {
  if (storage === undefined) delete globalThis.localStorage; else globalThis.localStorage = storage;
  await server.close();
}
