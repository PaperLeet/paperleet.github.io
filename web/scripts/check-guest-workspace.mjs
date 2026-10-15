import assert from 'node:assert/strict';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', optimizeDeps: { noDiscovery: true, entries: [] } });
try {
  const { createGuestWorkspace } = await server.ssrLoadModule('/src/guestWorkspace.ts');
  const { useGuestWorkspace } = await server.ssrLoadModule('/src/useGuestWorkspace.ts');
  const { useBrowserLearning, persistsBrowserLearning } = await server.ssrLoadModule('/src/useBrowserLearning.ts');
  const names = ['localStorage', 'sessionStorage', 'indexedDB'];
  const original = Object.fromEntries(names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
  let touches = 0;
  for (const name of names) Object.defineProperty(globalThis, name, {
    configurable: true,
    get() { touches++; throw Error(`Guest practice must not access ${name}`); },
  });
  try {
    const problem = { id: 'architecture/demo', starter: '# starter', tests: { T0: 1 } };
    const passed = { problem: problem.id, tests: [{ id: 'test', layer: 'T0', passed: true }], summary: { T0: [1, 1] } };
    const failed = { ...passed, tests: [{ ...passed.tests[0], passed: false }], summary: { T0: [0, 1] } };
    for (const owner of ['anonymous', 'local']) {
      let current = true;
      const workspace = createGuestWorkspace(owner, () => current);
      assert(workspace.getSnapshot().ready);
      assert.deepEqual(workspace.getSnapshot().saved, {}, 'A guest visit starts without importing old source');
      assert(workspace.update(problem.id, { code: '' }), 'An empty file is a valid guest edit');
      const before = workspace.getSnapshot().saved[problem.id];
      assert(workspace.update(problem.id, { code: '# edited during run' }));
      workspace.applyReport(problem, passed, owner);
      workspace.applyReport(problem, failed, owner);
      const saved = workspace.getSnapshot().saved[problem.id];
      assert.equal(saved.code, '# edited during run', 'Late reports do not replace current source');
      assert.equal(saved.acceptedReport, passed, 'Current-page acceptance remains usable after a later failed run');
      assert.equal(saved.report, failed);
      assert.equal(workspace.replaceCode(problem, '# stale', { owner, ...before }), false, 'A stale preview cannot overwrite a newer edit');
      assert.equal(workspace.resetToStarter(problem, { owner, code: saved.code, updated: saved.updated }), true);
      assert.equal(workspace.getSnapshot().saved[problem.id].code, problem.starter);
      assert.deepEqual(createGuestWorkspace(owner).getSnapshot().saved, {}, 'Reloading creates an empty workspace');
      current = false;
      const snapshot = workspace.getSnapshot();
      assert.equal(workspace.update(problem.id, { code: '# stale owner' }), false);
      workspace.applyReport(problem, passed, owner);
      assert.equal(workspace.getSnapshot(), snapshot, 'Callbacks retained across account changes cannot mutate prior guest state');
      assert.equal(persistsBrowserLearning(owner), false);
      let hookView, learningView;
      function Probe() { hookView = useGuestWorkspace(owner); learningView = useBrowserLearning(owner); return null; }
      renderToString(React.createElement(Probe));
      assert(hookView.ready && learningView.ready, 'Guests can edit and run without waiting for durable storage');
      assert.deepEqual(learningView.records, {});
      assert.deepEqual(hookView.saved, {});
    }
    assert.equal(createGuestWorkspace().update(problem.id, { code: '# unresolved' }), false);
    assert.equal(persistsBrowserLearning('unresolved'), false);
    assert.equal(persistsBrowserLearning('user:'), false);
    assert.equal(persistsBrowserLearning('user:alice'), true, 'Authenticated learning behavior is retained');
    assert.equal(touches, 0, 'Guest source/results must neither read nor write old browser caches');
  } finally {
    for (const name of names) {
      if (original[name]) Object.defineProperty(globalThis, name, original[name]);
      else delete globalThis[name];
    }
  }
  console.log('PASS guest practice: memory-only source/results, refresh reset, stale owner guards, current-source preservation, immediately ready, and zero browser storage access');
} finally {
  await server.close();
}
