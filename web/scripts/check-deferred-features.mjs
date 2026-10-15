import assert from 'node:assert/strict';
import React from 'react';
import { localizedRenderer } from "./render-locale.mjs";
import { createServer } from 'vite';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
try {
  const renderToString = await localizedRenderer(server, "zh");
  const { EditorFailure, EditorLoading } = await server.ssrLoadModule('/src/EditorFallback.tsx');
  const { FeatureLoading, FeatureFailure, deferredFeature } = await server.ssrLoadModule('/src/deferredFeature.tsx');
  const { Prose } = await server.ssrLoadModule('/src/RichText.tsx');
  const code = '# 私有草稿 <script>\ndef forward(x):\n    return x + 1\n';
  const props = { value: code, readOnly: false, theme: 'dark', font: 16, label: 'Python code editor', onChange() {} };
  const editor = renderToString(React.createElement(EditorFailure, { ...props, retry() {} }));
  assert(editor.includes('<textarea') && editor.includes('aria-label="Python code editor"') && !/\sreadonly=/i.test(editor));
  assert(editor.includes('&lt;script&gt;') && !editor.includes('<script>'), 'Fallback code must remain escaped text');
  assert(editor.includes('普通文本模式') && editor.includes('重试加载') && editor.includes('刷新页面'));
  const recoveryProps = {...props, retry(){}, recovery:{blocked:false,reason:'',reload(){}}};
  const recovery = renderToString(React.createElement(EditorFailure,recoveryProps));
  assert(recovery.includes('重新加载编辑器') && recovery.includes('账号中已保存的解答会保留') && recovery.includes('游客代码会丢失'));
  assert(!recovery.includes('重试加载') && !recovery.includes('刷新页面</button>'),'The workspace editor must use its guarded reload rather than ineffective repeated imports or an unguarded refresh');
  for(const reason of ['saving draft','storage failed','draft conflict','running tests','saving learning progress','owner unresolved']){
    const waiting = renderToString(React.createElement(EditorFailure,{...recoveryProps,recovery:{...recoveryProps.recovery,blocked:true,reason}}));
    assert(waiting.includes('disabled=""') && waiting.includes(reason) && waiting.includes('role="status"'));
    assert(!/\sreadonly=/i.test(waiting),'A blocked recovery must still allow editing in plain-text mode');
  }
  assert(/\sreadonly=/i.test(renderToString(React.createElement(EditorFailure, { ...props, readOnly:true, retry() {} }))));
  const loading = renderToString(React.createElement(EditorLoading, props));
  assert(loading.includes('aria-busy="true"') && loading.includes('私有草稿') && !loading.includes('<textarea'));
  assert(renderToString(React.createElement(FeatureLoading,{label:['实验室','lab']})).includes('role="status"'));
  assert(renderToString(React.createElement(FeatureFailure,{label:['实验室','lab'],retry(){}})).includes('role="alert"'));
  const Pending = deferredFeature(() => new Promise(()=>{}), ['实验室','lab']);
  const shell = renderToString(React.createElement(Pending));
  assert(shell.includes('正在加载实验室'), 'An unresolved optional import must render its local loading state');
  const prose = renderToString(React.createElement(Prose,{content:'$x^2$\n\n[Next](../layer_norm/README.md)\n\n<viz id="attention" />\n\n```python\nprint("<script>")\n```',problemId:'architecture/attention',onViz(){}}));
  assert(prose.includes('katex') && prose.includes('href="#/architecture/layer_norm"'));
  assert(prose.includes('打开交互实验') && prose.includes('&lt;script&gt;') && !prose.includes('<script>'));
  console.log('PASS deferred features: local loading/recovery, safe readable fallback code, private-workspace read-only gate, native text editing controls, math, lab links and source highlighting');
} finally { await server.close(); }
