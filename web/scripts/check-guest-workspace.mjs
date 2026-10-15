import assert from 'node:assert/strict';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { createServer } from 'vite';
import 'fake-indexeddb/auto';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', optimizeDeps: { noDiscovery: true, entries: [] } });
const originalFetch = globalThis.fetch;
let requests = 0;
globalThis.fetch = async () => { requests++; throw Error('Guest caching must not call a server'); };
try {
  const { useGuestWorkspace } = await server.ssrLoadModule('/src/useGuestWorkspace.ts');
  const { useBrowserLearning, persistsBrowserLearning } = await server.ssrLoadModule('/src/useBrowserLearning.ts');
  const { loadBrowserWorkspace, writeBrowserDraft, writeBrowserReport } = await server.ssrLoadModule('/src/browserWorkspaceStore.ts');
  const { draftSnapshotMatches, withPracticeReport } = await server.ssrLoadModule('/src/practiceRecords.ts');
  const { status } = await server.ssrLoadModule('/src/types.ts');
  const problem = { id: 'architecture/guest-cache-test', starter: '# starter', tests: { T0: 1 } };
  const passed = { problem: problem.id, runtime: 'browser-numpy-wasm', tests: [{ id: 'test', layer: 'T0', passed: true }], summary: { T0: [1, 1] } };
  const failed = { ...passed, tests: [{ ...passed.tests[0], passed: false }], summary: { T0: [0, 1] } };
  const code = '# 你好 🧪\r\ndef solve():\r\n\treturn 42\r\n';
  for (const owner of ['anonymous', 'local']) {
    const writer = `${owner}-tab`;
    const first = await writeBrowserDraft(owner, problem.id, writer, '', code, 'first');
    let fresh = await loadBrowserWorkspace(owner);
    assert.equal(fresh.records[problem.id].saved.code, code, 'A fresh page must recover exact Unicode source and CRLF');
    const empty = await writeBrowserDraft(owner, problem.id, writer, first.record.version, '', 'empty');
    fresh = await loadBrowserWorkspace(owner);
    assert.equal(fresh.records[problem.id].saved.code, '', 'An intentionally empty file must not revert to the starter');
    const during = await writeBrowserDraft(owner, problem.id, writer, empty.record.version, '# edited during run', 'newer');
    await writeBrowserReport(owner, problem, passed);
    await writeBrowserReport(owner, problem, failed);
    fresh = await loadBrowserWorkspace(owner);
    const saved = fresh.records[problem.id].saved;
    assert.equal(saved.code, '# edited during run', 'A late run cannot restore its submitted source over a newer draft');
    assert.equal(saved.report.tests[0].passed, false);
    assert.equal(status(problem, saved), 'done', 'An earlier complete pass survives later failures and reload');
    assert(!draftSnapshotMatches(saved, { owner, code, updated: 'first' }), 'A stale source review cannot authorize replacing a newer draft');
    const staleTab = await writeBrowserDraft(owner, problem.id, `${owner}-other-tab`, empty.record.version, '# concurrent edit', 'other');
    assert(staleTab.conflict, 'Concurrent tabs must not silently overwrite the newest source');
    fresh = await loadBrowserWorkspace(owner);
    assert.equal(fresh.records[problem.id].saved.code, '# edited during run');
    assert(fresh.backups.some(row => row.code === '# concurrent edit'), 'The conflicting tab remains recoverable');
    const prior = fresh.records[problem.id];
    const put = IDBObjectStore.prototype.put;
    let memory = { ...saved, code: '# keep editing while storage is full' };
    try {
      IDBObjectStore.prototype.put = function(...args) { if (this.name === 'records') throw new DOMException('Full', 'QuotaExceededError'); return put.apply(this, args); };
      await assert.rejects(writeBrowserDraft(owner, problem.id, writer, during.record.version, memory.code, 'failed-save'), { name: 'QuotaExceededError' });
      memory = withPracticeReport(memory, problem, failed);
      assert.equal(memory.code, '# keep editing while storage is full');
    } finally { IDBObjectStore.prototype.put = put; }
    fresh = await loadBrowserWorkspace(owner);
    assert.equal(fresh.records[problem.id].saved.code, prior.saved.code, 'Storage failure must retain the last durable draft');
    await writeBrowserDraft(owner, problem.id, writer, prior.version, memory.code, 'retry');
    assert.equal((await loadBrowserWorkspace(owner)).records[problem.id].saved.code, memory.code, 'Retry saves the in-memory source');
    assert.equal(persistsBrowserLearning(owner), false, 'Guest results must have one cache, not a second stale learning overlay');
    let hookView, learningView;
    function Probe() { hookView = useGuestWorkspace(owner); learningView = useBrowserLearning(owner); return null; }
    renderToString(React.createElement(Probe));
    assert(!hookView.ready && hookView.loading, 'The editor waits for browser-cache hydration before showing a writable starter');
    assert.deepEqual(hookView.saved, {});
    assert(learningView.ready && !Object.keys(learningView.records).length);
  }
  await writeBrowserDraft('anonymous', 'guest/only', 'guest', '', '# guest-only', 'stamp');
  assert(!(await loadBrowserWorkspace('local')).records['guest/only'], 'Browser owners remain separate');
  assert.equal(persistsBrowserLearning('unresolved'), false);
  assert.equal(persistsBrowserLearning('user:'), false);
  assert.equal(persistsBrowserLearning('user:alice'), true, 'Signed-in feedback retains its existing separate cache');
  let inactive;
  function Inactive() { inactive = useGuestWorkspace(undefined); return null; }
  renderToString(React.createElement(Inactive));
  assert.deepEqual(inactive.saved, {}, 'Resolving or signed-in owners must not expose the guest snapshot');
  assert.equal(inactive.update(problem.id, { code: '# not a guest' }), false);
  assert.equal(requests, 0, 'Guest drafts and browser results never contact account or practice APIs');
  console.log('PASS guest local cache: fresh-page recovery, empty/Unicode drafts, late-run source safety, durable passes, cross-tab conflicts, quota retry, owner separation, hydration gating and zero server uploads');
} finally {
  globalThis.fetch = originalFetch;
  await server.close();
}
