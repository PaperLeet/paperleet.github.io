import assert from 'node:assert/strict';
import fs from 'node:fs';
import React from 'react';
import {createServer} from 'vite';
import {localizedRenderer} from './render-locale.mjs';

const server = await createServer({
  server: {middlewareMode: true, hmr: false, ws: false}, appType: 'custom',
  optimizeDeps: {noDiscovery: true, include: []},
});
try {
  const catalog = JSON.parse(fs.readFileSync('public/data/catalog.json', 'utf8'));
  const {paperCollections, papersInCollection} = await server.ssrLoadModule('/src/paperCollections.ts');
  const {PaperLibrary} = await server.ssrLoadModule('/src/PaperViews.tsx');
  const {default: LibraryCollections} = await server.ssrLoadModule('/src/LibraryCollections.tsx');
  const navigation = await server.ssrLoadModule('/src/libraryNavigation.ts');
  const {selectLibraryPractice} = await server.ssrLoadModule('/src/libraryPractice.ts');
  const {accountReturnHref, signInHref} = await server.ssrLoadModule('/src/accountNavigation.ts');
  const {decodeProblemHash} = await server.ssrLoadModule('/src/reports.ts');
  const {parseHandoff} = await server.ssrLoadModule('/src/draftHandoff.ts');
  const {discussionPracticeTarget} = await server.ssrLoadModule('/src/discussionPracticeReturn.ts');
  const sections = {
    multimodal: ['multimodal/siglip'],
    robotics: ['robotics/diffusion_policy', 'robotics/supermimic'],
    'ai-infra': ['ai_infra/paged_attention'],
    'diffusion-rl': ['diffusion_rl/diffusion_dpo', 'diffusion_rl/flow_grpo'],
  };
  const props = {...catalog, saved: {}, progressReady: true};
  for (const [section, newIds] of Object.entries(sections)) {
    const collection = paperCollections.find(item => item.id === section);
    assert(collection, section);
    const category = catalog.categories.find(item => item.id === section);
    assert(category?.name && category.name_en, `${section}: category must be exported in both languages`);
    const members = papersInCollection(catalog.papers, collection);
    assert.equal(members.length, collection.paperIds.length);
    const ids = new Set(members.flatMap(paper => paper.module_ids));
    const view = navigation.readLibraryRoute(`#/collections/${section}`);
    assert(view);
    const selected = selectLibraryPractice(catalog.papers, catalog.problems, {}, view);
    assert.deepEqual([...selected.problems.map(problem => problem.id)].sort(), [...ids].sort());
    for (const locale of ['zh', 'en']) {
      const render = await localizedRenderer(server, locale);
      const cards = render(React.createElement(LibraryCollections, props));
      assert(cards.includes(`collection-${section}`) && cards.includes(`href="#/collections/${section}"`));
      const html = render(React.createElement(PaperLibrary, {...props, collectionId: section}));
      assert(html.includes(locale === 'zh' ? collection.name : collection.name_en));
      assert(html.includes('collection-practice'), `${section}: section has a usable practice entry`);
      for (const paper of members) assert(html.includes(`#/papers/${paper.id}`));
    }
    for (const id of newIds) {
      assert(ids.has(id), `${id}: new module must be discoverable in its section`);
      const problem = catalog.problems.find(item => item.id === id);
      assert(problem?.browser_judge?.bundle_url, `${id}: browser grading bundle required`);
      for (const key of ['starter', 'solution', 'readme', 'notes']) assert(problem[key]?.trim(), `${id}: missing ${key}`);
      for (const key of ['readme', 'notes']) assert(problem.translations.en[key]?.trim(), `${id}: missing English ${key}`);
      assert(problem.vizData?.metrics && problem.vizData?.charts?.length, `${id}: experiment must be exported`);
      assert(catalog.papers.find(paper => paper.id === problem.paper_id)?.category === section);
      const origin = `#/collections/${section}`;
      for (const address of [id, problem.number]) {
        const href = navigation.withLibraryContext(address, origin);
        assert.equal(navigation.libraryReturnHref(href), origin);
        assert.equal(accountReturnHref(decodeProblemHash(signInHref(href))), href);
      }
      const handoff = {version: 1, id: 'research-section', problem: id, code: 'private draft', created: 1000};
      assert.deepEqual(parseHandoff(JSON.stringify(handoff), 1000), handoff);
      const post = 'a'.repeat(32);
      assert.equal(discussionPracticeTarget(navigation.withLibraryContext(id, '#/community/' + post), post), id);
    }
  }
  for (const bad of ['robotics/../private', 'ai_infra/module/extra', 'diffusion_rl//flow_grpo']) {
    const handoff = {version: 1, id: 'invalid', problem: bad, code: 'draft', created: 1000};
    assert.equal(parseHandoff(JSON.stringify(handoff), 1000), undefined);
  }
  console.log('PASS research sections: bilingual discovery, executable content, section-scoped practice, numbered/canonical sign-in returns, private draft handoff and discussion returns');
} finally {
  await server.close();
}
