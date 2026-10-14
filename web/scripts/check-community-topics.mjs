import assert from 'node:assert/strict';
import { createServer } from 'vite';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
try {
  const { topicHref, practiceHref, topicIdFromRoute, topicFromProblem, topicTitle } = await server.ssrLoadModule('/src/communityTopics.ts');
  const { decodeProblemHash } = await server.ssrLoadModule('/src/reports.ts');
  for (const id of ['architecture/attention', 'finetuning/lora', 'diffusion/schedules']) {
    assert.equal(topicIdFromRoute(decodeProblemHash(topicHref(id))), id);
    assert.equal(decodeProblemHash(practiceHref(id)), id);
  }
  for (const route of ['community', 'community/a12b34', 'architecture/attention', 'community/problem/22', 'community/problem/../attention', 'community/problem/architecture/attention/extra']) {
    assert.equal(topicIdFromRoute(route), undefined, `${route} must not become a module-scoped feed`);
  }
  const module = { id: 'architecture/attention', number: 22, title: '缩放点积注意力', translations: { en: { title: 'Scaled Dot-Product Attention' } } };
  const topic = topicFromProblem(module);
  assert.equal(topicTitle(topic, false), '#22 · 缩放点积注意力');
  assert.equal(topicTitle(topic, true), '#22 · Scaled Dot-Product Attention');
  assert.equal(topicTitle({ id: 'retired/module', title: 'retired/module', retired: true }, true), 'retired/module');
  assert.equal('starter' in topic || 'solution' in topic, false);
  console.log('PASS module discussion routes, practice return paths, bilingual metadata, retired topics and minimal public context');
} finally {
  await server.close();
}
