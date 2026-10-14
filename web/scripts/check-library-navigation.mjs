import assert from 'node:assert/strict';
import fs from 'node:fs';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
const storage = globalThis.sessionStorage, localeStorage = globalThis.localStorage;
try {
  const { emptyLibraryFilters, libraryHref, readLibraryRoute, libraryReturnHref, libraryContinuation, normalizeLibraryFilters, withLibraryContext, routePath } = await server.ssrLoadModule('/src/libraryNavigation.ts');
  const { signInHref, accountReturnHref } = await server.ssrLoadModule('/src/accountNavigation.ts');
  const { decodeProblemHash } = await server.ssrLoadModule('/src/reports.ts');
  const { collectionIdFromRoute } = await server.ssrLoadModule('/src/paperCollections.ts');
  const { PaperLibrary, PaperDetail } = await server.ssrLoadModule('/src/PaperViews.tsx');
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
    for (const path of ['6', 'diffusion/schedules', 'papers/ddpm', 'retrieval/semantic_ids', 'retrieval/generative_retrieval']) {
      const target = withLibraryContext(path, href);
      assert.equal(libraryReturnHref(target), href);
      assert.equal(libraryReturnHref(decodeProblemHash(target)), href);
      assert.equal(routePath(decodeProblemHash(target)), path);
      assert.equal(accountReturnHref(decodeProblemHash(signInHref(target))), target);
      assert.equal(accountReturnHref(decodeProblemHash(signInHref(decodeProblemHash(target)))), target);
    }
  }
  assert.equal(libraryHref(undefined, emptyLibraryFilters()), '#');
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
  const returnTo = libraryHref('diffusion', { ...emptyLibraryFilters(), query: 'DDPM', runtime: 'browser', difficulty: 'E' });
  const ddpm = catalog.papers.find(p => p.id === 'ddpm');
  for (const en of [false, true]) {
    globalThis.localStorage = { getItem: () => en ? 'en' : 'zh' };
    const render = (Component, props) => renderToString(React.createElement(LocaleProvider, null, React.createElement(Component, props)));
    const library = render(PaperLibrary, { ...catalog, saved: {}, collectionId: 'diffusion', route: decodeProblemHash(returnTo), onGuide() {} });
    assert(library.includes('value="DDPM"') && library.includes('value="browser" selected=""') && library.includes('value="E" selected=""'));
    assert(library.includes(withLibraryContext('papers/ddpm', returnTo).replaceAll('&', '&amp;')));
    const waiting = render(PaperLibrary, { ...catalog, saved: {}, progressReady: false, route: libraryHref(undefined, { ...emptyLibraryFilters(), state: 'done' }), onGuide() {} });
    assert(waiting.includes(en ? 'Checking your learning progress' : '正在核对学习进度'));
    assert(!waiting.includes(en ? 'No matching papers' : '没有符合条件的论文') && !waiting.includes('data-paper-id='), 'Pending account progress must not fabricate empty or unstarted status-filter results');
    const detail = render(PaperDetail, { paper: ddpm, problems: catalog.problems, saved: {}, returnTo, onGuide() {} });
    assert(detail.includes(en ? 'Back to library results' : '返回题库结果') && detail.includes(returnTo.replaceAll('&', '&amp;')));
    assert(detail.includes(withLibraryContext(6, returnTo).replaceAll('&', '&amp;')));
    const next = render(PracticeNextStep, { next: catalog.problems.find(p => p.number === 6), samePaper: true, returnTo, onNotes() {}, onLab() {} });
    assert(next.includes(withLibraryContext(6, returnTo).replaceAll('&', '&amp;')));
    const escaped = render(PaperLibrary, { ...catalog, saved: {}, route: libraryHref(undefined, { ...emptyLibraryFilters(), query: '<script>alert(1)</script>' }), onGuide() {} });
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
  console.log('PASS library navigation: complete filter URLs, reserved Unicode, paper/module/auth round trips, legacy collections, safe public-only context, source-independent routing, bounded scroll persistence and bilingual output');
} finally {
  if (storage === undefined) delete globalThis.sessionStorage; else globalThis.sessionStorage = storage;
  if (localeStorage === undefined) delete globalThis.localStorage; else globalThis.localStorage = localeStorage;
  await server.close();
}
