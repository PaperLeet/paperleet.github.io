import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {createServer} from 'vite';

// No sockets or dependency scan are needed for these server-rendered fixtures.
const server=await createServer({server:{middlewareMode:true,ws:false},optimizeDeps:{noDiscovery:true,include:[]},appType:'custom'});
const originalStorage=globalThis.localStorage;
const unsafeClass=/\bclass="[^"]*\bprototype-math\b/;
const externalResource=/(?:src|href)="https:\/\/math-probe\.invalid/;
const htmlCommand=String.raw`\htmlClass{prototype-math}{x}`;
const imageCommand=String.raw`\includegraphics{https://math-probe.invalid/pixel.png}`;

function withPrototype(fields,run) {
  const previous=new Map(Object.keys(fields).map(key=>[key,Object.getOwnPropertyDescriptor(Object.prototype,key)]));
  try {
    for(const [key,value] of Object.entries(fields))Object.defineProperty(Object.prototype,key,{value,writable:true,configurable:true});
    return run();
  } finally {
    for(const [key,descriptor] of previous) {
      if(descriptor)Object.defineProperty(Object.prototype,key,descriptor);
      else delete Object.prototype[key];
    }
  }
}

try {
  const {default:CommunityMarkdown}=await server.ssrLoadModule('/src/CommunityMarkdown.tsx');
  const {Prose}=await server.ssrLoadModule('/src/RichText.tsx');
  const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
  globalThis.localStorage={getItem:()=> 'en'};
  const renderers={
    community:body=>React.createElement(CommunityMarkdown,{body}),
    reading:content=>React.createElement(Prose,{content}),
  };
  const render=(view,body)=>renderToStaticMarkup(React.createElement(LocaleProvider,null,view(body)));
  const commands=`$${htmlCommand}$\n\n$${imageCommand}$`;
  for(const [name,view] of Object.entries(renderers)) {
    const math=render(view,String.raw`Inline $\alpha/r$.

$$
\begin{aligned}
\Delta W &= \frac{\alpha}{r}BA \\
A &= \begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}
\end{aligned}
$$`);
    assert(math.includes('<math')&&math.includes('<mfrac>')&&/<mtable(?:\s|>)/.test(math)&&math.includes('katex-display'),`${name}: real inline/display fractions, alignment and matrices remain readable`);
    assert(!math.includes('katex-error'),`${name}: maintained formulas must render rather than fall back to errors`);
    for(const [scenario,fields] of [
      ['trust',{trust:true}],
      ['default',{default:true}],
      ['processor',{trust:false,processor:value=>value===false?true:value}],
    ]) {
      const html=withPrototype(fields,()=>render(view,commands));
      assert(!unsafeClass.test(html)&&!externalResource.test(html),`${name}: inherited ${scenario} must not enable trusted math HTML or resource loads`);
    }
    const inherited=withPrototype({'\\prototypeMath':String.raw`\frac{91}{73}`},()=>render(view,String.raw`$\prototypeMath$`));
    assert(!inherited.includes('<mn>91</mn>')&&!inherited.includes('<mn>73</mn>'),`${name}: inherited namespace entries must not become macros`);
  }

  // Exercise the renderer resolved by each consumer, catching nested vulnerable
  // copies even when the direct dependency and its stylesheet are patched.
  const require=createRequire(import.meta.url);
  const consumers=[['direct',require],...['rehype-katex','micromark-extension-math'].map(name=>[name,createRequire(require.resolve(name))])];
  for(const [name,resolve] of consumers) {
    const katex=resolve('katex');
    const options=Object.assign(Object.create({trust:true}),{strict:'ignore'});
    assert(!unsafeClass.test(katex.renderToString(htmlCommand,options)),`${name}: renderer options must ignore inherited trust`);
    for(const fields of [{trust:true},{default:true},{processor:()=>true}]) {
      const html=withPrototype(fields,()=>katex.renderToString(htmlCommand,{trust:false,strict:'ignore'}));
      assert(!unsafeClass.test(html),`${name}: polluted settings metadata must not override an explicit trust:false`);
    }
    assert(unsafeClass.test(katex.renderToString(htmlCommand,{trust:true,strict:'ignore'})),`${name}: explicit own-property options still work`);
  }
  console.log('PASS math security: both application renderers and every KaTeX consumer reject inherited settings/macros; inline/display fractions, aligned equations and matrices remain supported');
} finally {
  if(originalStorage===undefined)delete globalThis.localStorage;else globalThis.localStorage=originalStorage;
  await server.close();
}
