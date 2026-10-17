import assert from 'node:assert/strict';
import fs from 'node:fs';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

// Render public metadata locally. Account/company evidence is tested separately;
// this check never connects to an account or uses a learner's stored workspace.
const storage=Object.getOwnPropertyDescriptor(globalThis,'localStorage');
let locale='en',companyProps;
Object.defineProperty(globalThis,'localStorage',{configurable:true,value:{getItem:key=>key==='paperleet.locale'?locale:null}});
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
  name:'overview-company-fixture',enforce:'pre',
  resolveId(source,importer){if(importer?.endsWith('/src/ProblemOverview.tsx')&&source==='./deferredPages')return '\0overview-company';},
  load(id){if(id==='\0overview-company')return `export function ProblemCompanyTags(props){globalThis.__overviewCompanyProps=props;return null;}`;},
}]});
const original=Object.getOwnPropertyDescriptor(globalThis,'__overviewCompanyProps');
try {
  const {default:Overview}=await server.ssrLoadModule('/src/ProblemOverview.tsx');
  const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
  const {libraryHref,emptyLibraryFilters,withLibraryContext}=await server.ssrLoadModule('/src/libraryNavigation.ts');
  const {ProblemOverviewFallback}=await server.ssrLoadModule('/src/ProblemOverviewFallback.tsx');
  const catalog=JSON.parse(fs.readFileSync('public/data/catalog.json','utf8'));
  const problem=structuredClone(catalog.problems.find(p=>p.parent));
  assert(problem?.module&&problem?.paper);
  const parent=catalog.problems.find(p=>p.id===problem.parent);
  assert(parent);
  const returnTo=libraryHref('llm-basics',{...emptyLibraryFilters(),query:'attention',difficulty:'M'});
  const stats={attempted_users:1200,solved_users:40,total_submissions:100,accepted_submissions:20,failed_submissions:80,acceptance_rate:20};
  const onCode=()=>{};
  const props={problem,prerequisite:{number:parent.number},returnTo,course:false,notesUnlocked:true,stats,statsLoading:false,onCode};
  const render=(component,properties)=>new JSDOM(renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(component,properties)))).window.document;
  for(locale of ['zh','en']) {
    const d=render(Overview,props);companyProps=globalThis.__overviewCompanyProps;
    assert.equal(d.querySelectorAll('h1').length,1);
    assert.equal(d.querySelector('h1').textContent,`${problem.number}. ${problem.title}`);
    assert(d.querySelector('.accepted').textContent.includes(locale==='zh'?'已通过':'Solved'));
    assert.equal(d.querySelector('.heading-tags a').getAttribute('href'),withLibraryContext(parent.number,returnTo));
    assert.equal(d.querySelector('.paper-card').getAttribute('href'),problem.paper.url||`https://arxiv.org/abs/${problem.paper.arxiv}`);
    assert.equal(d.querySelector('.paper-card').getAttribute('target'),'_blank');
    assert.equal(d.querySelector('.paper-card').getAttribute('rel'),'noreferrer');
    assert.equal(d.querySelector('.paper-card strong').textContent,problem.paper.title);
    assert.deepEqual([...d.querySelectorAll('.module-functions:not(.provided-functions) code')].map(n=>n.textContent),problem.module.required_functions);
    assert.deepEqual([...d.querySelectorAll('.provided-functions code')].map(n=>n.textContent),problem.module.provided_functions);
    assert.equal(d.querySelector('[data-stat="attempted_users"] dd').textContent,'1,200');
    assert.equal(d.querySelector('[data-stat="acceptance_rate"] dd').textContent,'20.0%');
    assert.equal(d.querySelector('.problem-stats').getAttribute('aria-busy'),'false');
    assert.equal(companyProps.problemId,problem.id);assert.equal(companyProps.number,problem.number);assert(companyProps.onCode===onCode);
    const unavailable=render(Overview,{...props,stats:undefined,statsLoading:true,notesUnlocked:false,course:true,prerequisite:undefined});
    assert(!unavailable.querySelector('.accepted'));
    assert.equal(unavailable.querySelector('.heading-tags a').getAttribute('href'),`#/${problem.parent}`);
    assert(unavailable.querySelector('.paper-card small').textContent.includes(locale==='zh'?'参考资料':'REFERENCES'));
    assert.equal(unavailable.querySelector('.problem-stats').getAttribute('aria-busy'),'true');
    assert(unavailable.querySelector('.stats-availability')&&!unavailable.querySelector('.stats-availability').hasAttribute('open'));
    const fallback=render(ProblemOverviewFallback,{problem});
    assert.equal(fallback.querySelector('h1').textContent,`${problem.number}. ${problem.title}`);
    assert(fallback.querySelector('[role="status"][aria-busy="true"]'));
    const failed=render(ProblemOverviewFallback,{problem,retry:()=>{}});
    assert.equal(failed.querySelectorAll('button').length,1,'Optional details recovery offers retry without a page reload');
    assert(!failed.querySelector('[aria-busy="true"]'));
  }
  console.log('PASS deferred problem overview: bilingual public metadata, scoped prerequisites, references, module goals, live statistics, availability and identifiable retry fallbacks');
} finally {
  await server.close();
  if(storage)Object.defineProperty(globalThis,'localStorage',storage);else delete globalThis.localStorage;
  if(original)Object.defineProperty(globalThis,'__overviewCompanyProps',original);else delete globalThis.__overviewCompanyProps;
}
