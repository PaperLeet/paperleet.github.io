import assert from 'node:assert/strict';
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import React from 'react';
import {renderToString} from 'react-dom/server';
import {createServer} from 'vite';
import {syntaxCheck} from '../public/browser-judge/syntax.mjs';

const invalid=['def broken(:\n    return 1\n','def f():\nreturn 1\n','def f():\n\treturn 1\n        return 2\n','变量 = (\n','x = \0','def f():\r    x = 1\r    bad = (\r'];
const noExecution=["raise RuntimeError('must not execute')",'while True: pass','import deliberately_missing_module'];
const catalog=JSON.parse(fs.readFileSync('public/data/catalog.json','utf8'));
const maintained=[];
for(const problem of catalog.problems.filter(p=>p.browser_judge)){
  maintained.push(problem.solution,problem.starter);
  const bundle=JSON.parse(fs.readFileSync(`public/data/${problem.browser_judge.bundle_url}`,'utf8'));
  maintained.push(...Object.values(bundle.mutants));
}
const sources=[...invalid,...noExecution,...maintained];
const python=String.raw`import json,sys
payload=json.load(sys.stdin)
results=[]
for source in payload['sources']:
    scope={'learner_source':source}
    exec(payload['script'],scope)
    results.append(scope['_paperleet_syntax_result'])
print(json.dumps(results))`;
const pythonBinary=process.env.PYTHON||(fs.existsSync('../.venv/bin/python')?'../.venv/bin/python':'python3');
const diagnostics=JSON.parse(execFileSync(pythonBinary,['-c',python],{input:JSON.stringify({script:syntaxCheck,sources}),encoding:'utf8',timeout:5000}));
assert.equal(diagnostics.length,sources.length);
assert.deepEqual(diagnostics.slice(0,3).map(x=>x.kind),['SyntaxError','IndentationError','TabError']);
assert.deepEqual(diagnostics.slice(0,3).map(x=>x.line),[1,2,3]);
assert(diagnostics.slice(invalid.length).every(x=>x===null),'Every maintained source compiles without executing imports, training, raises or loops');

const server=await createServer({server:{middlewareMode:true},appType:'custom'});
const originalStorage=globalThis.localStorage;
try{
  const {parseSyntaxDiagnostic}=await server.ssrLoadModule('/src/browserSyntax.ts');
  const {BrowserSyntaxNotice}=await server.ssrLoadModule('/src/BrowserSyntaxNotice.tsx');
  const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
  for(let i=0;i<invalid.length;i++)assert(parseSyntaxDiagnostic(diagnostics[i],invalid[i]));
  const source='def broken(:\n',valid=diagnostics[0];
  for(const value of [null,[],{}, {...valid,kind:{}},{...valid,kind:'RuntimeError'}, {...valid,message:'x'.repeat(501)}, {...valid,line:0},{...valid,line:50},{...valid,column:0},{...valid,column:1e8},{...valid,line:NaN}])assert.equal(parseSyntaxDiagnostic(value,source),undefined);
  assert.deepEqual(parseSyntaxDiagnostic({...valid,passed:true,tests:[{passed:true}]},source),valid);
  for(const en of [false,true]){
    globalThis.localStorage={getItem:()=>en?'en':'zh'};
    const failure={source,diagnostic:{...valid,message:'expected <script>alert(1)</script>'}};
    const render=current=>renderToString(React.createElement(LocaleProvider,null,React.createElement(BrowserSyntaxNotice,{failure,source:current})));
    const current=render(source),changed=render(source+'# edited\n');
    assert(current.includes('role="alert"')&&current.includes(en?'Only syntax was checked':'仅完成语法检查'));
    assert(current.includes('&lt;script&gt;')&&!current.includes('<script>'));
    assert(changed.includes('role="status"')&&changed.includes(en?'code changed':'代码已修改'));
    assert(!current.includes('tests passed')&&!current.includes('项通过'),'Syntax feedback is not a test report');
  }
  console.log(`PASS Python syntax preflight: ${maintained.length} maintained sources, compile-only safety, error locations, bounded protocol, source versions and bilingual feedback`);
}finally{
  if(originalStorage===undefined)delete globalThis.localStorage;else globalThis.localStorage=originalStorage;
  await server.close();
}
