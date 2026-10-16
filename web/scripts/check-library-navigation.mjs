import assert from 'node:assert/strict';
import fs from 'node:fs';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { createServer } from 'vite';
import { JSDOM } from 'jsdom';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
const storage = globalThis.sessionStorage, localeStorage = globalThis.localStorage;
try {
  const { emptyLibraryFilters, libraryHref, libraryTags, readLibraryRoute, libraryReturnHref, libraryContinuation, normalizeLibraryFilters, sortLibraryPapers, withLibraryContext, routePath } = await server.ssrLoadModule('/src/libraryNavigation.ts');
  const { signInHref, accountReturnHref } = await server.ssrLoadModule('/src/accountNavigation.ts');
  const { decodeProblemHash } = await server.ssrLoadModule('/src/reports.ts');
  const { collectionIdFromRoute, paperCollections, papersInCollection } = await server.ssrLoadModule('/src/paperCollections.ts');
  const { PaperLibrary } = await server.ssrLoadModule('/src/PaperViews.tsx');
  const { default: PaperDetail } = await server.ssrLoadModule('/src/PaperDetail.tsx');
  const { matchingPaperModules, selectLibraryPractice } = await server.ssrLoadModule('/src/libraryPractice.ts');
  const { PracticeNextStep } = await server.ssrLoadModule('/src/PracticeJourney.tsx');
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const catalog = JSON.parse(fs.readFileSync('public/data/catalog.json', 'utf8'));
  for (const collectionId of [undefined, 'llm-basics', 'diffusion', 'retrieval']) for (const query of ['', 'attention', ' 张量 & mask=1? 100%_% # + / 😀 ']) {
    const filters = { query, category: 'architecture', tag: 'attention & mask', difficulty: 'M', runtime: 'browser', state: 'started' };
    const href = libraryHref(collectionId, filters);
    assert.deepEqual(readLibraryRoute(href), { collectionId, filters });
    assert.deepEqual(readLibraryRoute(decodeProblemHash(href)), { collectionId, filters });
    assert.equal(accountReturnHref(decodeProblemHash(signInHref(href))), href);
    assert.equal(accountReturnHref(decodeProblemHash(signInHref(decodeProblemHash(href)))), href);
    for (const path of ['6', 'foundations/next_token_loss', 'foundations/top_p_sampling', 'diffusion/schedules', 'papers/ddpm', 'retrieval/semantic_ids', 'retrieval/generative_retrieval']) {
      const target = withLibraryContext(path, href);
      assert.equal(libraryReturnHref(target), href);
      assert.equal(libraryReturnHref(decodeProblemHash(target)), href);
      assert.equal(routePath(decodeProblemHash(target)), path);
      assert.equal(accountReturnHref(decodeProblemHash(signInHref(target))), target);
      assert.equal(accountReturnHref(decodeProblemHash(signInHref(decodeProblemHash(target)))), target);
    }
  }
  assert.equal(libraryHref(undefined, emptyLibraryFilters()), '#');
  for (const sort of ['newest', 'oldest', 'title']) {
    const filters = { ...emptyLibraryFilters(), query: 'attention', sort };
    const view = libraryHref('llm-basics', filters);
    assert.deepEqual(readLibraryRoute(view).filters, filters);
    assert.equal(libraryReturnHref(withLibraryContext(22, view)), view, 'Sorting must survive the round trip through practice');
    assert.equal(accountReturnHref(decodeProblemHash(signInHref(withLibraryContext(22, view)))), withLibraryContext(22, view));
  }
  assert.equal(libraryHref(undefined, { ...emptyLibraryFilters(), sort: 'default' }), '#');
  assert.equal(readLibraryRoute('#/?sort=unknown'), undefined);
  assert.equal(readLibraryRoute('#/?sort=title&sort=newest'), undefined);
  const samples = [
    { id: 'b', number: 3, short_title: 'Beta', year: 2025 },
    { id: 'missing', number: 4, short_title: 'Gamma' },
    { id: 'a', number: 1, short_title: 'Alpha', year: 2025 },
    { id: 'old', number: 2, short_title: 'Model 2', year: 2017 },
  ];
  assert.equal(sortLibraryPapers(samples), samples, 'Discovery order must remain stable');
  assert.deepEqual(sortLibraryPapers(samples, 'newest').map(p => p.id), ['a', 'b', 'old', 'missing']);
  assert.deepEqual(sortLibraryPapers(samples, 'oldest').map(p => p.id), ['old', 'a', 'b', 'missing']);
  assert.deepEqual(sortLibraryPapers(samples, 'title').map(p => p.id), ['a', 'b', 'missing', 'old']);
  assert.equal(samples[0].id, 'b', 'Sorting must not mutate catalog order');
  assert.equal(libraryHref(undefined, { ...emptyLibraryFilters(), owner: 'private', code: 'private source' }), '#');
  assert.equal(withLibraryContext(6, '#'), '#/6');
  assert.equal(withLibraryContext('papers/ddpm'), '#/papers/ddpm');
  assert.equal(collectionIdFromRoute('collections/llm-basics?q=attention'), 'llm-basics');
  assert.equal(collectionIdFromRoute('lists/llm-basics?q=attention'), 'llm-basics');
  assert.equal(libraryContinuation('lists/llm-basics?q=attention'), 'collections/llm-basics?q=attention');
  for (const bad of ['https://evil.test/', '//evil.test/', 'javascript:alert(1)', 'collections/missing', 'collections/../diffusion', 'collections/diffusion/extra', '?from=community', '?unknown=1', '?q=a&q=b', '?difficulty=X', '?runtime=gpu', '?state=accepted', '?category=../private', '?q=' + 'x'.repeat(201), '?tag=' + 'x'.repeat(81), '?q=%00', '#/%E0%A4%A']) {
    assert.equal(readLibraryRoute(bad), undefined, bad);
    assert.equal(withLibraryContext(6, bad), '#/6', bad);
  }
  for (const from of ['https://evil.test/', '//evil.test/', 'community', 'account/continue/6', 'papers/ddpm?from=collections/diffusion', 'collections/diffusion?from=collections/rl']) {
    const href = '#/6?' + new URLSearchParams({ from });
    assert.equal(libraryReturnHref(href), undefined);
    assert.equal(signInHref(href), '#/account');
  }
  assert.equal(libraryReturnHref('#/6?from=collections/diffusion&from=collections/rl'), undefined);
  assert.equal(libraryReturnHref('#/community/a?from=collections/diffusion'), undefined);
  assert.equal(libraryReturnHref('#/6?from=collections/diffusion&code=private'), undefined);
  for (const problem of catalog.problems) assert(withLibraryContext(problem.number, '#/collections/llm-basics').startsWith('#/' + problem.number + '?from='));
  const normalized = normalizeLibraryFilters({ ...emptyLibraryFilters(), category: 'retired-category', tag: 'retired-tag', query: 'keep query' }, catalog.papers);
  assert.deepEqual(normalized, { ...emptyLibraryFilters(), query: 'keep query' });
  const originalMetadata = JSON.stringify(catalog.papers);
  const tags = libraryTags(catalog.papers);
  assert.equal(tags.filter(tag => tag.toLowerCase() === 'top-k').length, 1, 'Case aliases share one native topic option');
  assert(tags.indexOf('Attention') < tags.indexOf('Diffusion') && tags.indexOf('Diffusion') < tags.indexOf('Flow Matching'), 'Topic names have a predictable alphabetical order');
  for (const alias of ['Top-K', 'Top-k', 'tOp-k']) {
    const filters = { ...emptyLibraryFilters(), tag: alias, sort: 'title' };
    assert.equal(normalizeLibraryFilters(filters, catalog.papers).tag, alias, 'Existing link spelling must not be rewritten');
    const selection = selectLibraryPractice(catalog.papers, catalog.problems, {}, { filters });
    assert.deepEqual(selection.papers.map(paper => paper.id).sort(), ['llm-training-generation', 'sdpo'], 'Either spelling finds both maintained tagged papers');
    assert(selection.problems.some(problem => problem.id === 'foundations/top_k_sampling') && selection.problems.some(problem => problem.id === 'rl/sdpo_topk'), 'Practice follows the same inclusive topic results');
    const view = libraryHref(undefined, filters);
    assert.equal(libraryReturnHref(withLibraryContext(selection.problems[0].number, view)), view, 'The exact original topic and sort survive a practice return');
    const foundations = selectLibraryPractice(catalog.papers, catalog.problems, {}, { filters: { ...filters, category: 'foundations' } });
    assert.deepEqual(foundations.papers.map(paper => paper.id), ['llm-training-generation'], 'Case matching cannot bypass the chosen category');
    const done = selectLibraryPractice(catalog.papers, catalog.problems, {}, { filters: { ...filters, state: 'done' } });
    assert.equal(done.papers.length, 0, 'Topic aliases cannot bypass the progress filter');
  }
  assert(!libraryTags(catalog.papers, 'architecture').some(tag => tag.toLowerCase() === 'top-k'), 'The topic picker stays within its category');
  assert.equal(JSON.stringify(catalog.papers), originalMetadata, 'Topic matching must not rewrite source metadata');
  for (const runtime of ['browser','server']) assert.equal(normalizeLibraryFilters({...emptyLibraryFilters(),runtime},catalog.papers).runtime,'all','Retired runtime links must never create a hidden filter');
  const returnTo = libraryHref('diffusion', { ...emptyLibraryFilters(), query: 'DDPM', runtime: 'browser', difficulty: 'E' });
  const currentReturn = libraryHref('diffusion', {...readLibraryRoute(returnTo).filters,runtime:'all'});
  const ddpm = catalog.papers.find(p => p.id === 'ddpm');
  const transformer = catalog.papers.find(p => p.id === 'transformer');
  const bibliographyQueries = ['1706.03762', ' arXiv:1706.03762v3 ', 'https://arxiv.org/pdf/1706.03762v2.pdf?download=1#page=2',
    'HTTPS://EXPORT.ARXIV.ORG/ABS/1706.03762v2', 'Ashish Vaswani', 'NeurIPS', '2017', 'Vaswani 2017', '2017 Vaswani'];
  for (const query of bibliographyQueries) {
    assert.deepEqual(matchingPaperModules(transformer, catalog.problems, query).map(p => p.id), transformer.module_ids,
      `A paper metadata match retains all its modules: ${query}`);
    const view = { filters: { ...emptyLibraryFilters(), query } };
    assert(selectLibraryPractice(catalog.papers, catalog.problems, {}, view).papers.some(p => p.id === transformer.id),
      'The list and practice entry use the same bibliography results');
  }
  const legacy = { ...transformer, authors: ['Ada Lovelace', 'Grace Hopper'], arxiv: 'hep-th/9901001', url: 'https://arxiv.org/abs/hep-th/9901001' };
  for (const query of ['Grace Hopper', 'arXiv:hep-th/9901001v2', 'https://arxiv.org/pdf/hep-th/9901001v2.pdf']) {
    assert.equal(matchingPaperModules(legacy, catalog.problems, query).length, transformer.module_ids.length);
  }
  assert.equal(matchingPaperModules({ ...transformer, arxiv: undefined }, catalog.problems, 'arXiv:1706.03762v2').length, transformer.module_ids.length, 'The source URL supports catalog entries without a separate identifier');
  for (const query of ['https://evil.test/abs/1706.03762', 'https://arxiv.org.evil.test/abs/1706.03762', '1706.037620', 'arXiv:1706.03762v0']) {
    assert.equal(matchingPaperModules(transformer, catalog.problems, query).length, 0, `Unknown or invalid reference stays unmatched: ${query}`);
  }
  const queryFilters = { ...emptyLibraryFilters(), query: '1706.03762' };
  assert.equal(selectLibraryPractice(catalog.papers, catalog.problems, {}, { collectionId: 'retrieval', filters: queryFilters }).problems.length, 0, 'Metadata search stays within the selected collection');
  assert.equal(selectLibraryPractice(catalog.papers, catalog.problems, {}, { filters: { ...queryFilters, state: 'done' } }).problems.length, 0, 'Metadata matches cannot bypass progress filters');
  const attention = catalog.problems.find(p => p.id === 'architecture/attention');
  assert.deepEqual(matchingPaperModules(transformer, catalog.problems, '1706.03762', attention.difficulty).map(p => p.id),
    catalog.problems.filter(p => transformer.module_ids.includes(p.id) && p.difficulty === attention.difficulty).map(p => p.id));
  assert.equal(matchingPaperModules(transformer, catalog.problems, '1706.03762', 'all', 'server').length, 0, 'Metadata matches preserve runtime boundaries');
  assert.equal(matchingPaperModules(transformer, catalog.problems, 'Scaled Dot-Product Attention').length, 1, 'A module title match stays specific to that module');
  assert.deepEqual(matchingPaperModules(transformer, catalog.problems, 'Vaswani Scaled Dot-Product').map(p => p.id), [attention.id], 'Author and implementation terms narrow to the matching module');
  assert.equal(matchingPaperModules(transformer, catalog.problems, 'Vaswani missing-paper').length, 0, 'Every query term must match the paper or the same module');
  for (const en of [false, true]) {
    globalThis.localStorage = { getItem: () => en ? 'en' : 'zh' };
    const render = (Component, props) => renderToString(React.createElement(LocaleProvider, null, React.createElement(Component, props)));
    for (const query of bibliographyQueries) {
      const results = render(PaperLibrary, { ...catalog, saved: {}, route: libraryHref(undefined, { ...emptyLibraryFilters(), query }) });
      assert(results.includes('data-paper-id="transformer"'), 'Bibliography queries render the linked paper in both languages');
      assert(!results.includes(en ? 'No matching papers' : '没有符合条件的论文'));
    }
    const library = render(PaperLibrary, { ...catalog, saved: {}, collectionId: 'diffusion', route: decodeProblemHash(returnTo) });
    assert(library.includes('value="DDPM"') && library.includes('value="E" selected=""'));
    assert(library.includes('collection-page-header') && !library.includes('featured-collections') && !library.includes('practice-start'), 'A collection opens its own page, without homepage sections');
    assert(!library.includes('Filter by judge location') && !library.includes('判题位置筛选') && !library.includes('judge-capability-badge'));
    assert(library.includes(withLibraryContext('papers/ddpm', currentReturn).replaceAll('&', '&amp;')));
    for (const collection of paperCollections) {
      const page = render(PaperLibrary, { ...catalog, saved: {}, collectionId: collection.id, route: `collections/${collection.id}` });
      assert.equal((page.match(/<h1>/g) || []).length, 1, 'Collection pages have one dedicated page title');
      assert(page.includes(`<h1>${en ? collection.name_en.replaceAll('&', '&amp;') : collection.name.replaceAll('&', '&amp;')}</h1>`));
      assert.equal((page.match(/data-paper-id=/g) || []).length, papersInCollection(catalog.papers, collection).length, 'The collection page lists its own papers');
      assert(page.includes('class="back-link" href="#"') && page.includes('class="button collection-practice"'), 'Each collection has home and practice navigation');
    }
    const legacyServer = render(PaperLibrary,{...catalog,saved:{},route:'?runtime=server'});
    assert.equal((legacyServer.match(/data-paper-id=/g)||[]).length,catalog.papers.length,'A saved server filter must not empty an all-browser catalog');
    const waiting = render(PaperLibrary, { ...catalog, saved: {}, progressReady: false, route: libraryHref(undefined, { ...emptyLibraryFilters(), state: 'done' }) });
    assert(waiting.includes(en ? 'Checking your learning progress' : '正在核对学习进度'));
    assert(!waiting.includes(en ? 'No matching papers' : '没有符合条件的论文') && !waiting.includes('data-paper-id='), 'Pending account progress must not fabricate empty or unstarted status-filter results');
    const detail = render(PaperDetail, { paper: ddpm, problems: catalog.problems, saved: {}, returnTo });
    assert(!detail.includes('judge-capability-badge'),'Browser execution is standard, so module cards need no capability tag');
    assert(detail.includes(en ? 'Back to library results' : '返回题库结果') && detail.includes(returnTo.replaceAll('&', '&amp;')));
    assert(detail.includes(withLibraryContext(6, returnTo).replaceAll('&', '&amp;')));
    const position = catalog.problems.find(p => p.id === 'architecture/positional_encoding');
    const positionView = libraryHref(undefined, { ...emptyLibraryFilters(), query: 'Vaswani Sinusoidal' });
    const positionScope = selectLibraryPractice(catalog.papers, catalog.problems, {}, readLibraryRoute(positionView)).problems;
    const detailPage = props => JSDOM.fragment(render(PaperDetail, {
      paper: transformer, problems: catalog.problems, saved: {},
      returnTo: positionView, practiceProblems: positionScope, ...props,
    }));
    const primary = page => page.querySelector('.paper-primary-action a');
    const positionHref = withLibraryContext(position.number, positionView);
    const scoped = detailPage({});
    assert.equal(primary(scoped).getAttribute('href'), positionHref, 'A paper primary action must honor the module search, rather than restart at Attention');
    assert.equal(scoped.querySelector('.paper-completion-card a.button').getAttribute('href'), positionHref, 'Both paper practice entries use the same scope');
    assert.equal(scoped.querySelectorAll('.paper-module-card').length, transformer.module_ids.length, 'The complete paper curriculum stays available for explicit exploration');
    assert.equal(scoped.querySelector('[data-next] h3').textContent, `${position.number}. ${en ? position.translations.en.title : position.title}`);
    const drafts = { [attention.id]: { code: attention.starter + '\n# recent outside-scope draft', updated: '2026-10-16T10:00:00Z' },
      [position.id]: { code: position.starter + '\n# scoped draft', updated: '2026-10-15T10:00:00Z' } };
    const resumed = detailPage({saved:drafts});
    assert.equal(primary(resumed).getAttribute('href'), positionHref, 'A newer draft outside the filter cannot replace the scoped recommendation');
    assert(primary(resumed).textContent.includes(en ? 'Continue practice' : '继续实现'));
    const acceptedFor = problem => ({ problem: problem.id,
      tests: Object.entries(problem.tests).flatMap(([layer,count])=>Array.from({length:count},(_,i)=>({id:`${layer}_${i}`,layer,passed:true}))),
      summary: Object.fromEntries(Object.entries(problem.tests).map(([layer,count])=>[layer,[count,count]])) });
    const reviewed = detailPage({saved:{[position.id]:{report:acceptedFor(position)}}});
    assert(primary(reviewed).textContent.includes(en ? 'Review implementation' : '回顾实现'), 'Review follows the selected module, even when the rest of its paper is unfinished');
    assert.equal(reviewed.querySelector('[data-next]'), null, 'A solved filtered module must not be marked as unfinished Up next');
    const mediumView = libraryHref('llm-basics', {...emptyLibraryFilters(),query:'Vaswani',difficulty:'M'});
    const mediumScope = selectLibraryPractice(catalog.papers,catalog.problems,{},readLibraryRoute(mediumView)).problems;
    const medium = detailPage({returnTo:mediumView,practiceProblems:mediumScope});
    assert.equal(primary(medium).getAttribute('href'), withLibraryContext(27,mediumView), 'Collection and difficulty bounds also govern the detail recommendation');
    const empty = detailPage({practiceProblems:[]});
    assert.equal(primary(empty),null);
    assert.equal(empty.querySelector('.paper-completion-card a.button'),null);
    assert(empty.textContent.includes(en ? 'No modules match this library view' : '当前题库筛选中没有匹配模块'));
    assert.equal(empty.querySelector('.paper-library-link').getAttribute('href'),positionView);
    const pending = detailPage({progressReady:false});
    assert.equal(primary(pending),null);
    assert(pending.textContent.includes(en ? 'Loading progress' : '正在加载学习进度'));
    assert(!pending.textContent.includes(en ? 'No modules match this library view' : '当前题库筛选中没有匹配模块'));
    const course = catalog.papers.find(p=>p.kind==='course');
    const courseModules = catalog.problems.filter(p=>course.module_ids.includes(p.id));
    const completed = Object.fromEntries(courseModules.map(p=>[p.id,{report:acceptedFor(p)}]));
    const scopedCourse = JSDOM.fragment(render(PaperDetail,{paper:course,problems:catalog.problems,practiceProblems:courseModules,saved:completed,returnTo:'#/collections/llm-basics'}));
    assert(primary(scopedCourse).textContent.includes(en ? 'Continue learning' : '继续学习'), 'A completed course retains its explicitly labeled cross-paper continuation');
    assert(scopedCourse.querySelector('.paper-action-target').textContent.includes(en ? 'Another paper' : '其他论文'));
    assert.equal(primary(scopedCourse).getAttribute('href'),`#/${attention.number}`,'The explicit cross-paper continuation starts a fresh scope');
    const next = render(PracticeNextStep, { next: catalog.problems.find(p => p.number === 6), samePaper: true, returnTo, onNotes() {}, onLab() {} });
    assert(next.includes(withLibraryContext(6, returnTo).replaceAll('&', '&amp;')));
    const escaped = render(PaperLibrary, { ...catalog, saved: {}, route: libraryHref(undefined, { ...emptyLibraryFilters(), query: '<script>alert(1)</script>' }) });
    assert(escaped.includes('&lt;script&gt;') && !escaped.includes('<script>'));
  }
  const cache = new Map([['paperleet.library.positions.v1', JSON.stringify([
    ['#/collections/diffusion', { y: 800, anchor: { id: 'ddpm', top: 150 } }],
    ['javascript:alert(1)', { y: 2 }], ['#/?q=bad', { y: -5 }],
  ])]]);
  globalThis.sessionStorage = { getItem: key => cache.get(key) ?? null, setItem: (key, value) => cache.set(key, value) };
  const { rememberLibraryPosition, readLibraryPosition } = await server.ssrLoadModule('/src/libraryPosition.ts');
  assert.deepEqual(readLibraryPosition('#/collections/diffusion'), { y: 800, anchor: { id: 'ddpm', top: 150 } });
  assert.equal(readLibraryPosition('javascript:alert(1)'), undefined);
  assert.equal(readLibraryPosition('#/?q=bad'), undefined);
  rememberLibraryPosition('#/?q=second', { y: 200 });
  assert.equal(JSON.parse(cache.get('paperleet.library.positions.v1')).length, 2, 'Saving after reload must retain other hydrated positions');
  for (let i = 0; i < 20; i++) rememberLibraryPosition('#/?q=' + i, { y: i });
  assert.equal(JSON.parse(cache.get('paperleet.library.positions.v1')).length, 8);
  assert.equal(readLibraryPosition('#/?q=0'), undefined);
  assert.deepEqual(readLibraryPosition('#/?q=19'), { y: 19 });
  rememberLibraryPosition('#/?q=19', { y: Infinity });
  assert.deepEqual(readLibraryPosition('#/?q=19'), { y: 19 });
  globalThis.sessionStorage = { getItem() { throw Error('blocked'); }, setItem() { throw Error('blocked'); } };
  rememberLibraryPosition('#/?q=volatile', { y: 123 });
  assert.deepEqual(readLibraryPosition('#/?q=volatile'), { y: 123 });
  console.log('PASS library navigation: scoped paper recommendations, matching draft/review/empty/loading/course actions, complete paper exploration, bibliography search, sorted case-alias topics with unchanged metadata and exact legacy returns, preserved collection/difficulty/progress boundaries and bilingual results');
} finally {
  if (storage === undefined) delete globalThis.sessionStorage; else globalThis.sessionStorage = storage;
  if (localeStorage === undefined) delete globalThis.localStorage; else globalThis.localStorage = localeStorage;
  await server.close();
}
