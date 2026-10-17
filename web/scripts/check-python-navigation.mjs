import assert from 'node:assert/strict';
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import React, {act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom = new JSDOM('<div id="root"></div>', {url:'http://paperleet.test/#/10', pretendToBeVisual:true});
const globals = {window:dom.window, document:dom.window.document, navigator:dom.window.navigator,
  HTMLElement:dom.window.HTMLElement, Element:dom.window.Element, Node:dom.window.Node, Text:dom.window.Text,
  Window:dom.window.Window, MutationObserver:dom.window.MutationObserver,
  getComputedStyle:dom.window.getComputedStyle.bind(dom.window), localStorage:dom.window.localStorage,
  requestAnimationFrame:dom.window.requestAnimationFrame.bind(dom.window),
  cancelAnimationFrame:dom.window.cancelAnimationFrame.bind(dom.window), IS_REACT_ACT_ENVIRONMENT:true};
const originals = new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for (const [key,value] of Object.entries(globals)) Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
dom.window.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
dom.window.HTMLElement.prototype.scrollIntoView=function(){};
dom.window.Range.prototype.getClientRects=()=>[];
dom.window.Range.prototype.getBoundingClientRect=()=>new dom.window.DOMRect(0,0,8,20);
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root;
try {
  const {pythonDefinitions}=await server.ssrLoadModule('/src/pythonDefinitions.ts');
  const {editorSource}=await server.ssrLoadModule('/src/sourceLocation.ts');
  const fixtures=[
    '# def comment(): pass\ntext="""class StringOnly:\n  def fake(): pass\n"""\n@decorator(1)\nasync def real(x):\n    def nested():\n        return x\n    return nested()\n',
    '# 🧪\r\nclass 模型:\r\n    @staticmethod\r\n    async def 前向(x):\r\n        return x\r\n',
    'class Outer:\n    class Inner:\n        def forward(self):pass\n\ndef outer():\n    class Local:\n        def method(self):pass\n    return Local\n',
    'def duplicate():pass\ndef duplicate():pass\n',
    'def Kelvin():pass\n',
  ];
  const catalog=JSON.parse(fs.readFileSync('public/data/catalog.json','utf8'));
  const maintained=catalog.problems.flatMap(problem=>[problem.starter,problem.solution]);
  const sources=[...fixtures,...maintained];
  const oracle=String.raw`import ast,json,sys
class Definitions(ast.NodeVisitor):
    def __init__(self):self.scope=[];self.rows=[]
    def declaration(self,node,kind):
        self.rows.append({'qualifiedName':'.'.join([*self.scope,node.name]),'kind':kind,'line':node.lineno})
        self.scope.append(node.name);self.generic_visit(node);self.scope.pop()
    def visit_FunctionDef(self,node):self.declaration(node,'function')
    def visit_AsyncFunctionDef(self,node):self.declaration(node,'function')
    def visit_ClassDef(self,node):self.declaration(node,'class')
results=[]
for source in json.load(sys.stdin):
    visitor=Definitions();visitor.visit(ast.parse(source));results.append(visitor.rows)
print(json.dumps(results))`;
  const python=process.env.PYTHON||'python3';
  const expected=JSON.parse(execFileSync(python,['-c',oracle],{input:JSON.stringify(sources),encoding:'utf8',timeout:10000,maxBuffer:4*1024*1024}));
  for (let i=0;i<sources.length;i++) {
    const source=editorSource(sources[i]),definitions=pythonDefinitions(sources[i]);
    assert.deepEqual(definitions.map(({qualifiedName,kind,line})=>({qualifiedName:qualifiedName.normalize('NFKC'),kind,line})),expected[i],`Python AST parity for source ${i}`);
    assert.equal(new Set(definitions.map(x=>x.from)).size,definitions.length);
    for (const definition of definitions) assert.equal(source.slice(definition.from,definition.to),definition.name,'Offsets select the actual identifier in normalized UTF-16 editor text');
  }
  for (const source of ['', '# def absent(): pass', 'text="def fake(): pass"', 'def unfinished(', 'class Incomplete:\n    def method(', '\0\n@bad\ndef f(x):\n return (']) {
    assert.doesNotThrow(()=>pythonDefinitions(source));
  }
  assert.deepEqual(pythonDefinitions('text="def fake(): pass"'),[]);
  assert.deepEqual(pythonDefinitions('def unfinished(').map(x=>x.name),['unfinished'],'A partially typed signature remains navigable');
  const large=Array.from({length:1500},(_,i)=>`def fn_${i}(): return ${i}\n`).join('');
  assert.equal(pythonDefinitions(large).at(-1).name,'fn_1499','Navigation includes definitions beyond the initial editor viewport');

  const {createRoot}=await import('react-dom/client');
  const {EditorView}=await import('@uiw/react-codemirror');
  const {default:PythonEditor}=await server.ssrLoadModule('/src/PythonEditor.tsx');
  const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
  const raw='# 🧪\r\nclass 模型:\r\n    def 前向(self, x):\r\n        return x\r\n\ndef helper():\r\n    return 1\r\n';
  let changes=[],setLocale;
  let props={value:raw,readOnly:false,theme:'light',font:16,label:'Fixture Python editor',onChange:value=>changes.push(value)};
  function Harness(){setLocale=useI18n().setLocale;return React.createElement(PythonEditor,props);}
  root=createRoot(document.querySelector('#root'));
  const render=async patch=>{props={...props,...patch};await act(async()=>root.render(React.createElement(LocaleProvider,null,React.createElement(Harness))));};
  const settle=async predicate=>{for(let i=0;i<100&&!predicate();i++)await act(async()=>new Promise(done=>setTimeout(done,10)));assert(predicate(),'Editor did not settle');};
  const select=()=>document.querySelector('.python-definition-nav select');
  const go=()=>document.querySelector('.python-definition-nav button');
  const view=()=>EditorView.findFromDOM(document.querySelector('.cm-editor'));
  const choose=async name=>{
    const option=[...select().options].find(option=>option.textContent.includes(name));assert(option,`Missing definition ${name}`);
    await act(async()=>{select().focus();select().value=option.value;select().dispatchEvent(new dom.window.Event('change',{bubbles:true}));});
    return Number(option.value);
  };
  const activate=async()=>act(async()=>{go().focus();go().click();});
  const jump=async name=>{const position=await choose(name);await activate();return position;};
  await render();await settle(()=>select()&&!select().disabled);
  const editor=view(),content=editor.contentDOM;
  assert.equal(changes.length,0,'Loading navigation must not normalize or save the untouched CRLF source');
  assert.equal(select().getAttribute('aria-label'),'Choose a function or class');assert(go().disabled);
  const position=await choose('模型.前向');
  assert.equal(document.activeElement,select(),'Selecting an option retains focus for keyboard arrows and type-ahead');
  assert.equal(editor.state.selection.main.head,0,'Choosing a target does not navigate before explicit activation');
  await activate();
  assert.equal(editor.state.selection.main.head,position);assert(editor.hasFocus);
  assert.equal(editor.state.doc.toString().slice(position,position+2),'前向');
  assert.equal(changes.length,0,'A definition jump must not emit a source edit');assert.equal(props.value,raw);
  assert.equal(select().value,String(position),'The chosen target remains available for repeat activation');
  await act(async()=>editor.dispatch({selection:{anchor:0}}));await activate();assert.equal(editor.state.selection.main.head,position);
  assert.equal(view(),editor);assert.equal(editor.contentDOM,content);
  await act(async()=>setLocale('zh'));
  assert.equal(select().getAttribute('aria-label'),'选择要定位的函数或类');assert.equal(view(),editor);assert.equal(editor.state.selection.main.head,position);
  await render({font:18,theme:'dark'});assert.equal(view(),editor);assert.equal(changes.length,0);
  const before=editor.state.selection.main.head;
  await render({readOnly:true});assert(select().disabled&&go().disabled);await jump('helper');assert.equal(editor.state.selection.main.head,before,'An unresolved/read-only workspace cannot receive navigation');
  await render({value:'def replacement():\n    return 2\n',readOnly:false});
  await settle(()=>!select().disabled&&select().textContent.includes('replacement'));
  assert(!select().textContent.includes('helper'),'The picker follows the current source rather than the previous draft');
  assert(go().disabled,'Editing invalidates the previous target until it is selected again');
  changes=[];await jump('replacement');assert.equal(changes.length,0);
  const newer='def fresh_draft():\n    return 3\n';
  await act(async()=>editor.dispatch({changes:{from:0,to:editor.state.doc.length,insert:newer}}));
  const newerPosition=editor.state.selection.main.head;
  await jump('replacement');
  assert.equal(editor.state.doc.toString(),newer);assert.equal(editor.state.selection.main.head,newerPosition,'An old picker must not navigate an editor that already received newer source');
  assert.equal(changes.length,1,'Only the deliberate source edit emits a change');
  await render({value:newer});await settle(()=>select().textContent.includes('fresh_draft'));
  const request={source:props.value,location:{line:2}};let handled=0;
  await render({focusRequest:request,onFocusHandled:()=>handled++});assert.equal(handled,1);
  await jump('fresh_draft');const manual=editor.state.selection.main.head;
  await render({font:16,onFocusHandled:()=>handled++});assert.equal(editor.state.selection.main.head,manual,'A handled old diagnostic cannot take focus back from a newer definition choice');
  assert.equal(handled,1);
  await render({focusRequest:{...request}});assert.equal(handled,2,'A newly requested diagnostic still works after manual navigation');
  assert.equal(editor.state.doc.lineAt(editor.state.selection.main.head).number,2);
  const selection=editor.state.selection,sourceBefore=editor.state.doc.toString(),changedBefore=changes.length;
  const acknowledgements=[];let claims=0;
  await render({focusRequest:{source:props.value,claimFocus:()=>{claims++;return true;}},onFocusHandled:(_request,focused)=>acknowledgements.push(focused)});
  assert.equal(claims,1);assert.equal(acknowledgements.at(-1),true);assert(document.activeElement===editor.contentDOM);assert(editor.state.selection===selection,'Returning without a diagnostic preserves the exact CodeMirror selection object');
  select().focus();await render({focusRequest:{source:props.value,location:{line:1},claimFocus:()=>false}});
  assert(document.activeElement===select(),'A retired diagnostic cannot steal newer focus');assert(editor.state.selection===selection);assert.equal(acknowledgements.at(-1),false);assert.equal(editor.state.doc.toString(),sourceBefore);assert.equal(changes.length,changedBefore);
  await render({value:'# no definitions\n',focusRequest:undefined});await settle(()=>select().disabled);assert.equal(view(),editor);
  console.log(`PASS Python navigation: AST parity for ${maintained.length} maintained sources, strings/comments, Unicode/CRLF, decorators/async/nesting, duplicates, partial syntax and full-document coverage; actual CodeMirror jumps preserve source/view/focus with live locale, stale-source/read-only and diagnostic ownership guards`);
} finally {
  if(root)await act(async()=>root.unmount());
  await server.close();dom.window.close();
  for(const [key,descriptor] of originals)if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];
}
