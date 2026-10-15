import assert from 'node:assert/strict';
import fs from 'node:fs';
import React from 'react';
import { renderToString } from 'react-dom/server';

export async function checkPracticeScope(server) {
  const { selectLibraryPractice, matchingPaperModules, paperProgress } = await server.ssrLoadModule('/src/libraryPractice.ts');
  const { recommendPractice, nextPractice } = await server.ssrLoadModule('/src/practice-recommendation.ts');
  const { readLibraryRoute, libraryHref, withLibraryContext, libraryReturnHref } = await server.ssrLoadModule('/src/libraryNavigation.ts');
  const { paperCollections } = await server.ssrLoadModule('/src/paperCollections.ts');
  const { PaperLibrary } = await server.ssrLoadModule('/src/PaperViews.tsx');
  const { PracticeNextStep } = await server.ssrLoadModule('/src/PracticeJourney.tsx');
  const catalog = JSON.parse(fs.readFileSync('public/data/catalog.json', 'utf8'));
  const { papers, problems, categories } = catalog;
  const byId = Object.fromEntries(problems.map(p => [p.id, p]));
  const bpr = byId['retrieval/bpr'], sasrec = byId['retrieval/sasrec'], attention = byId['architecture/attention'];
  const retrieval = paperCollections.find(c => c.id === 'retrieval');
  const view = readLibraryRoute('collections/retrieval');
  const accepted = p => ({ problem: p.id, summary: Object.fromEntries(Object.entries(p.tests).map(([layer, n]) => [layer, [n,n]])), tests: Object.entries(p.tests).flatMap(([layer,n]) => Array.from({length:n},(_,i) => ({id:`${layer}-${i}`,layer,passed:true}))) });
  const saved = { [bpr.id]: { acceptedReport: accepted(bpr) }, [attention.id]: {code:'# newer outside draft',updated:'2099-01-01'}, [sasrec.id]:{code:'# inside draft',updated:'2026-10-14'} };
  const before = structuredClone({ papers, problems, saved, view });
  const scoped = selectLibraryPractice(papers, problems, saved, view);
  assert(scoped.problems.length && scoped.problems.every(p => retrieval.paperIds.includes(p.paper_id)));
  assert.equal(nextPractice(bpr,[bpr.id],scoped.problems,saved)?.id,sasrec.id,'A newer draft outside the collection must not hijack the next step');
  assert.equal(nextPractice(bpr,[bpr.id],problems,saved)?.id,attention.id,'The old unscoped call demonstrates the real collection escape');
  assert.deepEqual({papers,problems,saved,view},before,'Scope selection must not reorder the source catalog or mutate private drafts');
  const completed = Object.fromEntries(scoped.problems.map(p => [p.id,{report:accepted(p)}]));
  assert.equal(nextPractice(bpr,[bpr.id],selectLibraryPractice(papers,problems,completed,view).problems,completed),undefined,'A finished collection must not fall back to the full catalog');
  assert(recommendPractice(problems,completed),'Other collections still have unfinished work');

  const queryView = readLibraryRoute('collections/retrieval?q=BPR&difficulty=M&runtime=browser');
  assert.deepEqual(selectLibraryPractice(papers,problems,{},queryView).problems.map(p=>p.id),[bpr.id,'retrieval/lightgcn']);
  const empty = readLibraryRoute('collections/retrieval?q=definitely-no-matching-paper');
  assert.deepEqual(selectLibraryPractice(papers,problems,{},empty).problems,[]);
  assert.equal(recommendPractice(selectLibraryPractice(papers,problems,{},empty).problems,{}),undefined,'No matches means no unrelated first-practice offer');
  const moduleView = readLibraryRoute('?q=LayerNorm+and+Add+%26+Norm&difficulty=E');
  const moduleScope = selectLibraryPractice(papers,problems,{},moduleView);
  assert.deepEqual(moduleScope.problems.map(p=>p.id),['architecture/layer_norm']);
  const transformer = papers.find(p=>p.id==='transformer');
  assert.equal(nextPractice(attention,transformer.module_ids,moduleScope.problems,{})?.id,'architecture/layer_norm','Continue the paper using only modules matching the originating filters');
  const mixed = problems.map(p=>p.id==='finetuning/lora'?{...p,browser_judge:undefined}:p);
  for (const runtime of ['server','browser']) assert.deepEqual(
    selectLibraryPractice(papers,mixed,{},readLibraryRoute(`?runtime=${runtime}`)).problems.map(p=>p.id),
    selectLibraryPractice(papers,mixed,{},readLibraryRoute('')).problems.map(p=>p.id),
    'Retired runtime links cannot leave a hidden scope restriction');
  const paper = papers.find(p=>p.id==='bpr');
  const tagged = { ...queryView, filters:{...queryView.filters,category:paper.category,tag:paper.tags[0]} };
  assert.deepEqual(selectLibraryPractice(papers,problems,{},tagged).problems.map(p=>p.id),[bpr.id]);
  const started = {[attention.id]:{report:accepted(attention)}};
  assert.equal(paperProgress(transformer,{problems,saved:started}).state,'started');
  const startedScope = selectLibraryPractice(papers,problems,started,readLibraryRoute('?state=started'));
  assert(startedScope.problems.length && startedScope.problems.every(p=>p.paper_id===transformer.id),'Status filtering follows whole-paper progress, matching the library');
  const todo = selectLibraryPractice(papers,problems,started,readLibraryRoute('?state=todo'));
  assert(!todo.problems.some(p=>p.paper_id===transformer.id));
  const completedView = readLibraryRoute('collections/retrieval?state=todo');
  assert.deepEqual(selectLibraryPractice(papers,problems,completed,completedView).problems,[],'Completing the last todo paper must not introduce an out-of-scope recommendation');
  const noContext = selectLibraryPractice(papers,problems,{},undefined);
  assert.equal(noContext.problems,problems,'Direct module URLs keep the established global fallback');
  assert.equal(selectLibraryPractice(papers,problems,{},readLibraryRoute('lists/retrieval')).problems.length,scoped.problems.length);
  const href = libraryHref(queryView.collectionId,queryView.filters);
  assert.equal(libraryReturnHref(withLibraryContext(bpr.number,href)),href);
  const html = renderToString(React.createElement(PaperLibrary,{papers,problems,categories,saved:{},route:'collections/retrieval?q=BPR',collectionId:'retrieval',progressReady:true,onGuide:()=>{}}));
  const intro = html.slice(html.indexOf('class="practice-start"'),html.indexOf('class="featured-collections"'));
  assert(intro.includes(withLibraryContext(bpr.number,libraryHref('retrieval',readLibraryRoute('collections/retrieval?q=BPR').filters)).replaceAll('&','&amp;')) && !intro.includes(`href="#/${attention.number}"`),'The visible first-practice card uses the filtered results too');
  const noResults = renderToString(React.createElement(PaperLibrary,{papers,problems,categories,saved:{},route:'collections/retrieval?q=definitely-no-matching-paper',collectionId:'retrieval',progressReady:true,onGuide:()=>{}}));
  const emptyIntro = noResults.slice(noResults.indexOf('class="practice-start"'),noResults.indexOf('class="featured-collections"'));
  assert(emptyIntro.includes('当前范围暂无核心模块') && !emptyIntro.includes('本范围核心模块已完成') && !emptyIntro.includes('开始这道题'),'An empty search must not be presented as completed work');
  const finished = renderToString(React.createElement(PracticeNextStep,{samePaper:false,onNotes:()=>{},onLab:()=>{},returnTo:href}));
  assert(finished.includes('本范围核心模块已完成') && !finished.includes('核心题库已完成'),'Scoped completion must not claim the whole catalog is finished');
  assert(finished.includes(href.replaceAll('&','&amp;')),'The completed-scope link preserves all public filters');
  // Every collection uses exactly the same search/difficulty predicate as its rows.
  for (const collection of paperCollections) for (const query of ['', 'Attention', '检索']) for (const difficulty of ['all','E','M','H']) {
    const filters = {...view.filters,query,difficulty};
    const selected=selectLibraryPractice(papers,problems,{}, {collectionId:collection.id,filters});
    const expected=papers.filter(p=>collection.paperIds.includes(p.id)).flatMap(p=>matchingPaperModules(p,problems,query,difficulty));
    assert.deepEqual([...selected.problems.map(p=>p.id)].sort(),[...expected.map(p=>p.id)].sort());
  }
  console.log('PASS practice scope: collection boundaries, filtered start/resume/next, outside drafts, paper-level status, empty/completed views, direct URLs, legacy lists and preserved return filters');
}
