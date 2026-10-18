import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtempSync,rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const cache=mkdtempSync(path.join(os.tmpdir(),'paperleet-admin-progress-'));
const server=await createServer({cacheDir:cache,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
try{
 const {default:Progress}=await server.ssrLoadModule('/src/AdminPaperProgress.tsx');
 const row={id:'a'.repeat(32),arxiv_id:'2601.12345',version:2,maintainer_review:{approved:false}};
 for(const [status,current] of [['pending_review',0],['suitability',1],['generation',1],['validation',1],['pool',2],['ready',2],['published',3],['changes',undefined],['rejected',undefined],['unknown',undefined]]){
  const dom=new JSDOM(renderToStaticMarkup(React.createElement(Progress,{paper:{...row,status}})));
  try{
   const section=dom.window.document.querySelector('section'),steps=section.querySelectorAll('ol li');
   assert.equal(steps.length,4);
   assert.equal(section.querySelectorAll('[aria-current=step]').length,current===undefined?0:1);
   if(current!==undefined)assert.equal(steps[current].getAttribute('aria-current'),'step');
   assert.equal(section.querySelector('h3').id,section.getAttribute('aria-labelledby'));
   assert.equal(section.querySelectorAll('button,a,input').length,0,'Progress cannot approve or publish a submission');
   if(['pool','ready'].includes(status))assert(section.textContent.includes('Approval applies only to this version'));
   if(['changes','rejected'].includes(status))assert(section.textContent.includes('does not publish an earlier candidate'));
   if(current===1)assert(section.textContent.includes('generation cannot be approved again'));
   if(status==='unknown')assert(section.textContent.includes('stage is unavailable'));
  }finally{dom.window.close();}
 }
 const approved=renderToStaticMarkup(React.createElement(Progress,{paper:{...row,status:'ready',maintainer_review:{approved:true}}}));
 assert(approved.includes('Candidate v2 has administrator approval'));
 assert(approved.includes('community review and the publication process still apply'));
 const malformed=renderToStaticMarkup(React.createElement(Progress,{paper:{...row,status:'pool',maintainer_review:{approved:'true'}}}));
 assert(!malformed.includes('has administrator approval'));
 const stale=renderToStaticMarkup(React.createElement(Progress,{paper:{...row,status:'generation',maintainer_review:{approved:true}}}));
 assert(!stale.includes('has administrator approval'));
 console.log('PASS administrator paper progress: ten states, one accessible current stage, exact candidate approval, processing and restart guidance, no publication controls');
}finally{await server.close();rmSync(cache,{recursive:true,force:true});}
