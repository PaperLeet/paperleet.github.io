import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';

const dom=new JSDOM('<html data-routing="path"><body></body></html>',{url:'https://paperleet.test/',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,location:dom.window.location,history:dom.window.history,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,Element:dom.window.Element,HTMLElement:dom.window.HTMLElement,HashChangeEvent:dom.window.HashChangeEvent};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-network-biology-');
const server=await createServer({root:process.cwd(),configFile:false,esbuild:{jsx:'automatic'},cacheDir,server:{watch:null,middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
try{
 const ids=['degree_distribution','hubs_centrality','connected_components','shortest_paths','feed_forward_loops','community_modules'].map(id=>'network_biology/'+id);
 const catalog=JSON.parse(await readFile('public/data/catalog.json','utf8'));
 const problems=catalog.problems.filter(problem=>problem.id.startsWith('network_biology/'));
 assert.deepEqual(problems.map(problem=>problem.id).sort(),[...ids].sort());
 const papers=catalog.papers.filter(paper=>paper.category==='network-biology');
 assert.equal(papers.length,5);assert.deepEqual(papers.flatMap(paper=>paper.module_ids).sort(),[...ids].sort());
 assert(catalog.categories.some(category=>category.id==='network-biology'));
 const nav=await server.ssrLoadModule('/src/libraryNavigation.ts'),interview=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const account=await server.ssrLoadModule('/src/accountNavigation.ts'),{decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 const {parseHandoff}=await server.ssrLoadModule('/src/draftHandoff.ts'),reading=await server.ssrLoadModule('/src/readingReturn.ts'),discussion=await server.ssrLoadModule('/src/discussionPracticeReturn.ts');
 const {selectLibraryPractice}=await server.ssrLoadModule('/src/libraryPractice.ts');
 const filters={...nav.emptyLibraryFilters(),category:'network-biology'},from=nav.libraryHref(undefined,filters);
 const scope=selectLibraryPractice(catalog.papers,catalog.problems,{},nav.readLibraryRoute(from));
 assert.deepEqual(scope.problems.map(problem=>problem.id).sort(),[...ids].sort());
 assert.deepEqual(scope.papers.map(paper=>paper.id).sort(),papers.map(paper=>paper.id).sort());
 const pathFilters={...interview.emptyInterviewFilters(),role:'research',track:'network-biology',saved:true};
 const pathFrom=interview.interviewHref(pathFilters),favoriteFrom=nav.favoriteHref({query:'bridge',state:'all'}),post='a'.repeat(32),postFrom='#/community/'+post;
 assert.deepEqual(interview.readInterviewRoute(pathFrom),pathFilters);
 for(const problem of problems){
  for(const origin of [from,pathFrom,favoriteFrom,postFrom]){
   for(const destination of [problem.id,problem.number]){
    const href=nav.withLibraryContext(destination,origin);
    assert.equal(nav.libraryReturnHref(href),origin,problem.id+' exact source return');
    assert.equal(account.accountReturnHref(decodeProblemHash(account.signInHref(href))),href,problem.id+' sign-in return');
   }
  }
  assert.equal(account.accountReturnHref(decodeProblemHash(account.signInHref('#/'+problem.id))),'#/'+problem.id,'Canonical deep links survive sign-in without a source query');
  const handoff={version:1,id:'biology-handoff',problem:problem.id,code:'private implementation',created:1000};
  assert.deepEqual(parseHandoff(JSON.stringify(handoff),1000),handoff);
  const favoritePractice=nav.withLibraryContext(problem.id,favoriteFrom),postPractice=nav.withLibraryContext(problem.id,postFrom);
  assert.equal(reading.readingPracticeTarget(favoritePractice,favoriteFrom),problem.id);
  reading.rememberReadingReturn({owner:'account:test',href:favoriteFrom,post:problem.id,practice:problem.id,link:'practice',top:21,focus:true});
  assert.equal(reading.currentReadingReturn()?.practice,problem.id,'Favorites retain the canonical module and reading position');reading.discardReadingReturn();
  const questions=interview.interviewHref({...pathFilters,view:'questions'});
  assert.equal(reading.readingSourceConcept(questions,problem.id),problem.id);
  assert.equal(discussion.discussionPracticeTarget(postPractice,post),problem.id);
  discussion.rememberDiscussionReturn({owner:'guest',post,practice:problem.id,view:{order:'oldest',authorOnly:false,query:'',limit:20},expanded:[],anchor:{href:postPractice,module:problem.id,place:'practice',index:0,top:21,keyboard:true}});
  assert.equal(discussion.currentDiscussionReturn()?.practice,problem.id,'Discussion returns retain the canonical module');discussion.discardDiscussionReturn();
 }
 for(const invalid of ['network_biology/../degree_distribution','network_biology/degree_distribution/extra','unknown/degree_distribution']){
  assert.throws(()=>nav.withLibraryContext(invalid,from));
  assert.equal(parseHandoff(JSON.stringify({version:1,id:'biology-handoff',problem:invalid,code:'x',created:1000}),1000),undefined);
 }
 console.log('PASS network biology routes: all six modules preserve library, study, favorites and discussion sources, sign-in, handoff and reading returns; malformed and unknown namespaces stay rejected');

 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx'),{PaperLibrary}=await server.ssrLoadModule('/src/PaperViews.tsx'),{default:PaperDetail}=await server.ssrLoadModule('/src/PaperDetail.tsx'),{default:Lab}=await server.ssrLoadModule('/src/Lab.tsx');
 const render=(Component,props)=>JSDOM.fragment(renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(Component,props))));
 const library=render(PaperLibrary,{...catalog,saved:{},route:from});
 assert.deepEqual([...library.querySelectorAll('[data-paper-id]')].map(row=>row.dataset.paperId).sort(),papers.map(paper=>paper.id).sort(),'The category discovers every new paper');
 for(const paper of papers){
  const detail=render(PaperDetail,{paper,problems:catalog.problems,practiceProblems:scope.problems,saved:{},returnTo:from});
  assert.equal(detail.querySelector('.paper-library-link').getAttribute('href'),from);
  assert.deepEqual([...detail.querySelectorAll('.paper-module-card')].map(link=>decodeProblemHash(link.getAttribute('href')).split('?')[0]).sort(),paper.module_ids.map(id=>String(problems.find(problem=>problem.id===id).number)).sort());
  assert([...detail.querySelectorAll('.paper-module-card')].every(link=>nav.libraryReturnHref(link.getAttribute('href'))===from));
 }
 for(const problem of problems){
  assert(problem.browser_judge?.bundle_url,problem.id+' browser exercise');
  const reference=render(Lab,{problem,initialView:'concept'});
  assert(reference.querySelector('.toy-experiment .chart,.toy-experiment .bars'),problem.id+' reference experiment renders with the existing chart component');
  const report={problem:problem.id,summary:{T3:[1,1]},tests:[{id:'network-experiment',layer:'T3',passed:true,artifacts:{charts:[{kind:'bar',title:'Current network result',x:['A'],series:[{name:'Degree',values:[1]}]}],metrics:{nodes:1}}}]};
  const run=render(Lab,{problem,report,initialView:'run',sourceState:'current'});
  assert(run.textContent.includes('Current network result')&&run.querySelector('.chart,.bars'),problem.id+' T3 run renders');
 }
 console.log('PASS network biology discovery and practice: five category papers expose six runnable modules with exact library returns; reference and T3 results use the existing toy renderer');

 const {startPathNavigation}=await server.ssrLoadModule('/src/pathNavigation.ts'),{publicHref,registerCatalogRoutes,appLocation}=await server.ssrLoadModule('/src/appLocation.ts');
 startPathNavigation();
 for(const problem of problems){
  assert.equal(publicHref('#/'+problem.id),'/problems/'+problem.id+'/','Canonical public links work before catalog registration');
 }
 registerCatalogRoutes(catalog.problems);
 for(const problem of problems){
  const href=nav.withLibraryContext(problem.number,from);appLocation.hash=href;
  assert.equal(location.pathname,'/problems/'+problem.id+'/');assert.equal(location.hash,'');
  assert.equal(appLocation.hash,href);assert.equal(nav.libraryReturnHref(appLocation.hash),from);
 }
 appLocation.hash=from;assert.equal(appLocation.hash,from);assert.equal(location.pathname,'/');
 console.log('PASS network biology public paths: canonical and numbered routes reach the same module and return to the category filter without fragments');
}finally{
 await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();
 for(const[key,value]of previous)value?Object.defineProperty(globalThis,key,value):delete globalThis[key];
}
