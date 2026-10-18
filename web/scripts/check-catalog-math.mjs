import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {createServer} from 'vite';
import {unified} from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import {localizedRenderer} from './render-locale.mjs';

const root=fileURLToPath(new URL('..',import.meta.url));
const server=await createServer({root,server:{middlewareMode:true,hmr:false,ws:false},optimizeDeps:{noDiscovery:true,include:[]},appType:'custom'});
const parser=unified().use(remarkParse).use(remarkGfm).use(remarkMath);
try{
 const {Prose}=await server.ssrLoadModule('/src/RichText.tsx');
 const catalog=JSON.parse(await readFile(new URL('../public/data/catalog.json',import.meta.url),'utf8'));
 const errors=[];let documents=0,formulas=0;
 for(const locale of ['zh','en']){
  const render=await localizedRenderer(server,locale,renderToStaticMarkup);
  for(const problem of catalog.problems)for(const field of ['readme','notes']){
   const content=(locale==='en'?problem.translations?.en?.[field]:problem[field])||'';
   if(!content)continue;
   const label=`${problem.id} ${locale} ${field}`,tree=parser.parse(content),lines=content.split('\n'),stack=[tree];
   while(stack.length){
    const node=stack.pop();
    if(node.type==='math'||node.type==='inlineMath')formulas++;
    if(node.type==='math'){
     // remark-math treats text beside an opening block fence as metadata and
     // only recognizes a closing fence on its own line. Both can silently eat text.
     if(node.meta||!/^\s*\${2,}\s*$/.test(lines[node.position.start.line-1])||!/^\s*\${2,}\s*$/.test(lines[node.position.end.line-1]))errors.push(`${label}:${node.position.start.line}: invalid display-math fences`);
    }
    stack.push(...node.children||[]);
   }
   const html=render(React.createElement(Prose,{content,problemId:problem.id}));
   if(/class="[^"]*\bkatex-error\b/.test(html))errors.push(`${label}: KaTeX parse error`);
   if(problem.id==='architecture/block_attention_residual'&&field==='readme'){
    if(!html.includes('<table>')||!html.includes(locale==='en'?'Normalize keys only.':'只有 key 经过归一化'))errors.push(`${label}: explanation/table swallowed by math`);
   }
   documents++;
  }
 }
 assert.deepEqual(errors,[],errors.join('\n'));
 console.log(`PASS catalog math: ${documents} Chinese/English statements and notes, ${formulas} formulas, valid block fences, no KaTeX errors, and the Kimi contract table remains outside math`);
}finally{await server.close();}
