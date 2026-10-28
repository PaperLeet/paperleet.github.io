import assert from 'node:assert/strict';
import fs from 'node:fs';
import {mkdtempSync, rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import React, {act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {localizedRenderer} from './render-locale.mjs';

// Reading metadata remains separate from executable modules. Verify all 62
// curated papers now resolve to practice, plus safe behavior for a missing paper.
const data = JSON.parse(fs.readFileSync('src/data/collection-reading-lists.json', 'utf8'));
const catalog = JSON.parse(fs.readFileSync('public/data/catalog.json', 'utf8'));
const expectedCounts = {multimodal: 16, robotics: 16, 'ai-infra': 16, 'diffusion-rl': 14};
const expectedAvailable = {...expectedCounts};
const expectedStages = {multimodal: 4, robotics: 5, 'ai-infra': 4, 'diffusion-rl': 5};
const flat = collection => collection.stages.flatMap(stage => stage.papers);
const normalizedTitle = value => value.normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
const arxivId = value => /arxiv\.org\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5})/i.exec(value || '')?.[1];
const sourceKey = value => arxivId(value) || value.replace(/\/$/, '');
const textField = (item, key) => assert(typeof item[key] === 'string' && item[key].trim(), `${item.id}: missing ${key}`);
const bilingual = (item, zh, en) => {
  textField(item, zh); textField(item, en);
  assert(/[\u3400-\u9fff]/.test(item[zh]), `${item.id}: ${zh} must include Chinese`);
  assert(/[A-Za-z]/.test(item[en]) && item[zh] !== item[en], `${item.id}: ${en} needs its own translation`);
};
assert.equal(data.schema_version, 1);
assert.match(data.reviewed_on, /^\d{4}-\d{2}-\d{2}$/);
assert(!Number.isNaN(Date.parse(data.reviewed_on)));
assert.deepEqual(data.collections.map(item => item.id).sort(), Object.keys(expectedCounts).sort());
const ids = new Set(), sources = new Set(), availableIds = new Set();
for (const collection of data.collections) {
  bilingual(collection, 'description', 'description_en');
  assert.equal(collection.stages.length, expectedStages[collection.id]);
  assert.equal(new Set(collection.stages.map(stage => stage.id)).size, collection.stages.length);
  assert.equal(flat(collection).length, expectedCounts[collection.id]);
  assert.equal(flat(collection).filter(item => item.paper_id).length, expectedAvailable[collection.id]);
  assert.equal(flat(collection).filter(item => item.priority).length, 8);
  for (const stage of collection.stages) {
    assert.match(stage.id, /^[a-z0-9]+(?:-[a-z0-9]+)*$/);
    bilingual(stage, 'title', 'title_en');
    assert(stage.papers.length, `${stage.id}: empty stage`);
    for (const entry of stage.papers) {
      assert.match(entry.id, /^[a-z0-9]+(?:-[a-z0-9]+)*$/);
      assert(!ids.has(entry.id), `Duplicate reading id: ${entry.id}`); ids.add(entry.id);
      for (const field of ['title', 'short_title', 'url']) textField(entry, field);
      bilingual(entry, 'summary', 'summary_en');
      assert(Number.isInteger(entry.year) && entry.year >= 2000 && entry.year <= Number(data.reviewed_on.slice(0, 4)), `${entry.id}: invalid year`);
      assert.equal(typeof entry.priority, 'boolean');
      const url = new URL(entry.url);
      assert.equal(url.protocol, 'https:');
      assert(!url.username && !url.password && !url.hash && url.pathname !== '/', `${entry.id}: direct paper source required`);
      const source = sourceKey(entry.url);
      assert(!sources.has(source), `Duplicate paper source: ${source}`); sources.add(source);
      if (entry.paper_id) {
        assert(!availableIds.has(entry.paper_id), `Duplicate practice identity: ${entry.paper_id}`); availableIds.add(entry.paper_id);
        const paper = catalog.papers.find(item => item.id === entry.paper_id);
        assert(paper?.module_ids.length, `${entry.id}: practice mapping must resolve to an executable catalog paper`);
        assert.equal(normalizedTitle(entry.title), normalizedTitle(paper.title), `${entry.id}: practice mapping points to a different paper`);
        // Official anthology/project URLs may differ; compare arXiv identities
        // when both sources expose them, in addition to the full title above.
        if (arxivId(entry.url) && arxivId(paper.url)) assert.equal(arxivId(entry.url), arxivId(paper.url));
        for (const id of paper.module_ids) assert(catalog.problems.some(problem => problem.id === id && problem.paper_id === paper.id));
      } else {
        assert(!catalog.papers.some(paper => paper.id === entry.id || normalizedTitle(paper.title) === normalizedTitle(entry.title)), `${entry.id}: reading-only record leaked into executable catalog`);
        assert(!catalog.problems.some(problem => problem.paper_id === entry.id), `${entry.id}: missing practice mapping for an executable paper`);
      }
    }
  }
}
assert.equal(ids.size, 62);
assert.equal(availableIds.size, 62);
assert.equal(ids.size - availableIds.size, 0);

const dom = new JSDOM('<!doctype html><div id="root"></div>', {url: 'https://paperleet.test/#/collections/diffusion-rl', pretendToBeVisual: true});
const globals = {
  window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
  HTMLElement: dom.window.HTMLElement, Element: dom.window.Element,
  location: dom.window.location, history: dom.window.history,
  localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage,
  innerHeight: 900, scrollY: 0, scrollTo() {},
  ResizeObserver: class {observe() {} disconnect() {}},
  requestAnimationFrame: callback => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout,
  IS_REACT_ACT_ENVIRONMENT: false,
};
const originals = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, {value, writable: true, configurable: true});
window.matchMedia = () => ({matches: false, addEventListener() {}, removeEventListener() {}});
window.scrollTo = () => {};
HTMLElement.prototype.scrollIntoView = function() {};
localStorage.setItem('paperleet.locale', 'en');
const cacheDir = mkdtempSync(path.join(os.tmpdir(), 'paperleet-reading-check-'));
const server = await createServer({cacheDir, server: {middlewareMode: true, hmr: false, ws: false}, appType: 'custom', optimizeDeps: {noDiscovery: true, include: []}});
let root;
try {
  const {paperCollections, papersInCollection} = await server.ssrLoadModule('/src/paperCollections.ts');
  const {default: ReadingList} = await server.ssrLoadModule('/src/CollectionReadingList.tsx');
  const {default: LibraryCollections} = await server.ssrLoadModule('/src/LibraryCollections.tsx');
  const {PaperLibrary} = await server.ssrLoadModule('/src/PaperViews.tsx');
  const {selectLibraryPractice, paperProgress} = await server.ssrLoadModule('/src/libraryPractice.ts');
  const {eligibleUnattemptedProblems, pickRandomUnattemptedProblem} = await server.ssrLoadModule('/src/randomProblem.ts');
  const navigation = await server.ssrLoadModule('/src/libraryNavigation.ts');
  const {LocaleProvider, useI18n} = await server.ssrLoadModule('/src/i18n.tsx');
  const props = {...catalog, saved: {}, progressReady: true};
  const parsed = html => {const element = document.createElement('div'); element.innerHTML = html; return element;};
  const sorted = values => [...values].sort();
  const accepted = problem => ({acceptedReport: {
    problem: problem.id,
    summary: Object.fromEntries(Object.entries(problem.tests).map(([layer, count]) => [layer, [count, count]])),
    tests: Object.entries(problem.tests).flatMap(([layer, count]) => Array.from({length: count}, (_, i) => ({id: `${layer}-${i}`, layer, passed: true}))),
  }});
  for (const reading of data.collections) {
    const collection = paperCollections.find(item => item.id === reading.id);
    assert(collection, reading.id);
    assert.equal(collection.readingCount, expectedCounts[reading.id]);
    assert.deepEqual(sorted(collection.paperIds), sorted(flat(reading).flatMap(entry => entry.paper_id ? [entry.paper_id] : [])));
    const members = papersInCollection(catalog.papers, collection);
    const moduleIds = new Set(members.flatMap(paper => paper.module_ids));
    const route = `#/collections/${reading.id}`, modulesRoute = `${route}?view=modules`;
    const practice = selectLibraryPractice(catalog.papers, catalog.problems, {}, navigation.readLibraryRoute(modulesRoute));
    assert.deepEqual(sorted(practice.problems.map(problem => problem.id)), sorted(moduleIds));
    assert.deepEqual(sorted(eligibleUnattemptedProblems(practice.problems, {}).map(problem => problem.id)), sorted(moduleIds));
    for (const rng of [() => 0, () => 0.999999]) assert(moduleIds.has(pickRandomUnattemptedProblem(practice.problems, {}, rng)?.id));
    const completed = Object.fromEntries(practice.problems.map(problem => [problem.id, accepted(problem)]));
    assert.equal(pickRandomUnattemptedProblem(practice.problems, completed, () => 0), undefined);
    for (const paper of members) {
      const expected = paper.module_ids.filter(id => !catalog.problems.find(problem => problem.id === id).module.optional).length;
      assert.equal(paperProgress(paper, {problems: catalog.problems, saved: {}}).total, expected);
      assert.deepEqual(paperProgress(paper, {problems: catalog.problems, saved: completed}), {completed: expected, total: expected, state: 'done'});
    }
    for (const locale of ['zh', 'en']) {
      const render = await localizedRenderer(server, locale);
      const page = parsed(render(React.createElement(ReadingList, {...props, collectionId: reading.id, route})));
      const list = page.querySelector(`.collection-reading-list[data-collection-id="${reading.id}"]`);
      assert(list, `${reading.id}: missing reading view in ${locale}`);
      assert.equal(list.querySelectorAll('.collection-reading-paper').length, expectedCounts[reading.id]);
      assert.equal(list.querySelectorAll('.reading-practice-link').length, expectedAvailable[reading.id]);
      assert.equal(list.querySelectorAll('.reading-priority').length, 8);
      assert.equal(list.querySelector('input[type=search]').getAttribute('aria-label'), locale === 'zh' ? '搜索合集论文' : 'Search collection papers');
      assert(page.textContent.includes(locale === 'zh' ? reading.description : reading.description_en));
      assert.equal(list.querySelector('.collection-reading-views [aria-current="page"]').getAttribute('href'), route);
      assert(list.querySelector(`.collection-reading-views a[href="${modulesRoute}"]`));
      for (const stage of reading.stages) {
        const section = list.querySelector(`[data-stage-id="${stage.id}"]`);
        assert(section?.querySelector('h2').textContent.includes(locale === 'zh' ? stage.title : stage.title_en));
        assert.equal(section.querySelectorAll('.collection-reading-paper').length, stage.papers.length);
        for (const entry of stage.papers) {
          const card = section.querySelector(`[data-reading-id="${entry.id}"]`);
          assert(card, `${entry.id}: missing reading card`);
          assert.equal(card.querySelector('.reading-paper-title').textContent, entry.title);
          assert.equal(card.querySelector('.reading-paper-summary').textContent, locale === 'zh' ? entry.summary : entry.summary_en);
          assert.equal(card.querySelector('.reading-source-link').getAttribute('href'), entry.url);
          if (entry.paper_id) {
            assert.equal(card.dataset.practiceState, 'available');
            const link = card.querySelector('.reading-practice-link');
            assert.equal(link.dataset.paperId, entry.paper_id);
            assert.equal(navigation.libraryReturnHref(link.getAttribute('href')), route);
            assert(link.getAttribute('href').startsWith(`#/papers/${entry.paper_id}?`));
            assert(!card.querySelector('.reading-no-exercise'));
          } else {
            assert.equal(card.dataset.practiceState, 'missing');
            assert.equal(card.querySelector('.reading-no-exercise').textContent, locale === 'zh' ? '暂无练习' : 'No exercise yet');
            assert(!card.querySelector('.reading-practice-link, button, a[href^="#"]'), `${entry.id}: reading-only paper must not offer fake practice`);
            assert.equal(card.querySelectorAll('a').length, 1);
            assert.equal(card.dataset.progress, undefined);
          }
        }
      }
      // Preserve the no-exercise fallback with a deliberate incomplete catalog.
      // A paper_id alone must never manufacture practice or completion progress.
      const unavailable = flat(reading)[0];
      const missingPaper = catalog.papers.find(paper => paper.id === unavailable.paper_id);
      const incompletePapers = catalog.papers.filter(paper => paper.id !== missingPaper.id);
      const incomplete = parsed(render(React.createElement(ReadingList, {...props, papers: incompletePapers, collectionId: reading.id, route})));
      const missingCard = incomplete.querySelector(`[data-reading-id="${unavailable.id}"]`);
      assert.equal(missingCard.dataset.practiceState, 'missing');
      assert.equal(missingCard.dataset.progress, undefined);
      assert.equal(missingCard.querySelector('.reading-no-exercise').textContent, locale === 'zh' ? '暂无练习' : 'No exercise yet');
      assert(!missingCard.querySelector('.reading-practice-link, button, a[href^="#"]'));
      assert.equal(missingCard.querySelectorAll('a').length, 1);
      assert.equal(missingCard.querySelector('.reading-source-link').getAttribute('href'), unavailable.url);
      assert.equal(incomplete.querySelectorAll('.reading-practice-link').length, expectedAvailable[reading.id] - 1);
      const incompletePractice = selectLibraryPractice(incompletePapers, catalog.problems, {}, navigation.readLibraryRoute(modulesRoute));
      assert.deepEqual(sorted(incompletePractice.problems.map(problem => problem.id)), sorted([...moduleIds].filter(id => !missingPaper.module_ids.includes(id))));
      assert(eligibleUnattemptedProblems(incompletePractice.problems, {}).every(problem => !missingPaper.module_ids.includes(problem.id)));
      const incompleteModules = parsed(render(React.createElement(PaperLibrary, {...props, papers: incompletePapers, collectionId: reading.id, route: modulesRoute})));
      assert.equal(incompleteModules.querySelector('.library-progress b').textContent, `0 / ${incompletePractice.problems.length}`);
      const cards = parsed(render(React.createElement(LibraryCollections, props)));
      const collectionCard = cards.querySelector(`.collection-card.collection-${reading.id}`);
      assert.equal(collectionCard.getAttribute('href'), route);
      assert(collectionCard.getAttribute('aria-label').includes(`${expectedCounts[reading.id]} ${locale === 'zh' ? '篇论文' : 'papers'}`));
      assert(collectionCard.querySelector('.collection-learning-progress').textContent.includes(`0/${moduleIds.size}`));
      const modules = parsed(render(React.createElement(PaperLibrary, {...props, collectionId: reading.id, route: modulesRoute})));
      assert(!modules.querySelector('.collection-reading-list'));
      assert.deepEqual(sorted([...modules.querySelectorAll('[data-module-id]')].map(node => node.dataset.moduleId)), sorted(moduleIds));
      assert.equal(modules.querySelector('.library-progress b').textContent, `0 / ${moduleIds.size}`);
      const solved = parsed(render(React.createElement(PaperLibrary, {...props, collectionId: reading.id, route: modulesRoute, saved: completed})));
      assert.equal(solved.querySelector('.library-progress b').textContent, `${moduleIds.size} / ${moduleIds.size}`);
      assert(solved.querySelector('.random-problem').disabled);
    }
  }

  // Exercise the lazy default route, real controlled input, clear button, and
  // actual navigation anchors. The harness supplies only the app's route owner.
  // Static renders above do not own client subscriptions; enable act tracking
  // only for the mounted interaction journey below.
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const {createRoot} = await import('react-dom/client');
  let route = '#/collections/diffusion-rl', setLocale;
  const changes = [];
  const tree = () => React.createElement(LocaleProvider, null, React.createElement(Frame));
  function navigate(href) {route = href; history.replaceState(null, '', href); root.render(tree());}
  function Frame() {
    setLocale = useI18n().setLocale;
    return React.createElement(PaperLibrary, {...props, route, collectionId: navigation.readLibraryRoute(route)?.collectionId, onFilters: href => {changes.push(href); navigate(href);}});
  }
  const wait = async (predicate, message) => {
    for (let i = 0; i < 150 && !predicate(); i++) await act(async () => {await new Promise(resolve => setTimeout(resolve, 10));});
    assert(predicate(), message);
  };
  const click = async node => {
    assert(node, 'Expected a clickable control');
    await act(async () => {node.focus(); node.click(); await new Promise(resolve => setTimeout(resolve, 20));});
  };
  const onNavigation = event => {
    const link = event.target.closest('a');
    if (!link || !link.getAttribute('href')?.startsWith('#/collections/')) return;
    event.preventDefault(); navigate(link.getAttribute('href'));
  };
  document.getElementById('root').addEventListener('click', onNavigation);
  await act(async () => {root = createRoot(document.getElementById('root')); root.render(tree());});
  await wait(() => document.querySelectorAll('.collection-reading-paper').length === 14, 'Default collection route must load the full reading list');
  const type = async value => {
    const input = document.querySelector('.collection-reading-list input[type=search]');
    assert(input);
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(input, value);
      input.dispatchEvent(new window.Event('input', {bubbles: true}));
    });
    await wait(() => navigation.readLibraryRoute(route)?.filters.query === value, 'Search input must update the q route');
    assert.equal(new URL(location.href).hash, route);
  };
  await type('DDPO');
  assert.deepEqual([...document.querySelectorAll('.collection-reading-paper')].map(card => card.dataset.readingId), ['ddpo']);
  assert.equal(document.querySelector('[data-reading-id="ddpo"] .reading-practice-link').dataset.paperId, 'ddpo');
  await click(document.querySelector('.search-clear'));
  assert.equal(document.querySelectorAll('.collection-reading-paper').length, 14);
  await act(async () => {setLocale('zh');});
  await type('前向');
  assert.deepEqual([...document.querySelectorAll('.collection-reading-paper')].map(card => card.dataset.readingId), ['diffusion-nft']);
  assert(document.querySelector('.reading-paper-summary').textContent.includes('前向'));
  // A reading summary need not appear verbatim in exercise metadata. Use the
  // shared paper title when checking filtered practice results across tabs.
  await type('DiffusionNFT');
  const filteredOrigin = route;
  assert.equal(navigation.libraryReturnHref(document.querySelector('.reading-practice-link').getAttribute('href')), filteredOrigin);
  await click(document.querySelector('.collection-reading-views a:not([aria-current])'));
  await wait(() => document.querySelector('[data-module-id]'), 'Practice tab must load executable modules');
  await wait(() => document.querySelector('.library-result-actions') && document.querySelector('.library-module-source .library-search-match'), 'Filtered practice controls and search highlighting must finish loading');
  assert.equal(navigation.readLibraryRoute(route).filters.view, 'modules');
  assert.equal(navigation.readLibraryRoute(route).filters.query, 'DiffusionNFT');
  assert.equal(document.querySelector('.collection-reading-list'), null);
  assert.equal(document.querySelector('.library-progress b').textContent, '0 / 14');
  await click(document.querySelector('.collection-reading-views a:not([aria-current])'));
  await wait(() => document.querySelector('[data-reading-id="diffusion-nft"]'), 'Reading tab must restore the filtered reading list');
  assert.equal(route, filteredOrigin);
  await type('no-such-paper-92817');
  assert.equal(document.querySelectorAll('.collection-reading-paper').length, 0);
  assert(document.querySelector('.collection-reading-list .no-results'));
  await click(document.querySelector('.collection-reading-list .no-results button'));
  assert.equal(document.querySelectorAll('.collection-reading-paper').length, 14);
  assert(changes.some(href => href.includes('q=DDPO')) && changes.some(href => href.includes('q=%E5%89%8D%E5%90%91')));
  document.getElementById('root').removeEventListener('click', onNavigation);
  console.log('PASS collection reading lists: 62 verified records / 62 practice papers; missing-catalog fallback without fake practice; bilingual stages and sources; scoped practice/random/progress; live search and reading/practice navigation');
} finally {
  if (root) await act(async () => {
    // Let optional chunks resolve while their subscriptions still belong to act.
    await new Promise(resolve => setTimeout(resolve, 80));
    root.unmount();
  });
  // Client subscriptions are gone; remaining lazy SSR work is teardown only.
  globalThis.IS_REACT_ACT_ENVIRONMENT = false;
  await server.close();
  rmSync(cacheDir, {recursive: true, force: true});
  dom.window.close();
  for (const [key, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
}
