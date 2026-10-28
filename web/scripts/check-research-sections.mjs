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
    "multimodal": [
      "multimodal/siglip",
      "multimodal/align",
      "multimodal/siglip_2",
      "multimodal/blip",
      "multimodal/blip_2",
      "multimodal/flamingo",
      "multimodal/llava",
      "multimodal/instructblip",
      "multimodal/imagebind",
      "multimodal/chameleon",
      "multimodal/qwen2_vl",
      "multimodal/qwen2_5_vl",
      "multimodal/internvl3",
      "multimodal/qwen3_vl",
      "multimodal/qwen2_5_omni"
    ],
    "robotics": [
      "robotics/diffusion_policy",
      "robotics/supermimic",
      "robotics/dagger",
      "robotics/act",
      "robotics/rdt_1b",
      "robotics/rt_1",
      "robotics/rt_2",
      "robotics/octo",
      "robotics/openvla",
      "robotics/pi0",
      "robotics/pi0_5",
      "robotics/dreamerv3",
      "robotics/daydreamer",
      "robotics/td_mpc2",
      "robotics/deepmimic",
      "robotics/diffmimic"
    ],
    "ai-infra": [
      "ai_infra/paged_attention",
      "ai_infra/gpipe",
      "ai_infra/megatron_lm",
      "ai_infra/zero",
      "ai_infra/flash_attention_2",
      "ai_infra/orca",
      "ai_infra/sglang",
      "ai_infra/sarathi_serve",
      "ai_infra/distserve",
      "ai_infra/gptq",
      "ai_infra/smoothquant",
      "ai_infra/awq"
    ],
    "diffusion-rl": [
      "diffusion_rl/diffusion_dpo",
      "diffusion_rl/flow_grpo",
      "diffusion_rl/imagereward",
      "diffusion_rl/diffusion_kto",
      "diffusion_rl/d3po",
      "diffusion_rl/pc_flow",
      "diffusion_rl/ddpo",
      "diffusion_rl/dpok",
      "diffusion_rl/alignprop",
      "diffusion_rl/draft",
      "diffusion_rl/dancegrpo",
      "diffusion_rl/mixgrpo",
      "diffusion_rl/flow_dppo"
    ]
  };
  assert.equal(Object.values(sections).flat().length, 56, "All six initial and fifty expansion modules must stay covered");
  const props = {...catalog, saved: {}, progressReady: true};
  for (const [section, newIds] of Object.entries(sections)) {
    const collection = paperCollections.find(item => item.id === section);
    assert(collection, section);
    const category = catalog.categories.find(item => item.id === section);
    assert(category?.name && category.name_en, `${section}: category must be exported in both languages`);
    const members = papersInCollection(catalog.papers, collection);
    assert.equal(members.length, collection.paperIds.length);
    const ids = new Set(members.flatMap(paper => paper.module_ids));
    const practiceRoute = `#/collections/${section}?view=modules`;
    const view = navigation.readLibraryRoute(practiceRoute);
    assert(view);
    const selected = selectLibraryPractice(catalog.papers, catalog.problems, {}, view);
    assert.deepEqual([...selected.problems.map(problem => problem.id)].sort(), [...ids].sort());
    for (const locale of ['zh', 'en']) {
      const render = await localizedRenderer(server, locale);
      const cards = render(React.createElement(LibraryCollections, props));
      assert(cards.includes(`collection-${section}`) && cards.includes(`href="#/collections/${section}"`));
      const html = render(React.createElement(PaperLibrary, {...props, collectionId: section, route: practiceRoute}));
      assert(html.includes(locale === 'zh' ? collection.name : collection.name_en));
      assert(html.includes('collection-practice'), `${section}: section has a usable practice entry`);
      for (const id of ids) assert(html.includes(`data-module-id="${id}"`), `${section}: practice tab must list ${id}`);
    }
    for (const id of newIds) {
      assert(ids.has(id), `${id}: new module must be discoverable in its section`);
      const problem = catalog.problems.find(item => item.id === id);
      assert(problem?.browser_judge?.bundle_url, `${id}: browser grading bundle required`);
      for (const key of ['starter', 'solution', 'readme', 'notes']) assert(problem[key]?.trim(), `${id}: missing ${key}`);
      for (const key of ['readme', 'notes']) assert(problem.translations.en[key]?.trim(), `${id}: missing English ${key}`);
      assert(problem.vizData?.metrics && problem.vizData?.charts?.length, `${id}: experiment must be exported`);
      assert(catalog.papers.find(paper => paper.id === problem.paper_id)?.category === section);
      for (const origin of [`#/collections/${section}`, practiceRoute]) {
        for (const address of [id, problem.number]) {
          const href = navigation.withLibraryContext(address, origin);
          assert.equal(navigation.libraryReturnHref(href), origin);
          assert.equal(accountReturnHref(decodeProblemHash(signInHref(href))), href);
        }
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
  console.log('PASS research sections: all 56 new modules, bilingual discovery, executable content, section-scoped practice, numbered/canonical sign-in returns, private draft handoff and discussion returns');
} finally {
  await server.close();
}
