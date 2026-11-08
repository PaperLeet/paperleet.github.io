import assert from 'node:assert/strict';
import { createServer } from 'vite';

const server = await createServer({ configFile: false, server: { middlewareMode: true, watch: null, hmr: false, ws: false }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
const stores = [];
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const failure = (status, code) => Object.assign(Error(code), { status, code });
try {
  const { createAssemblyDraft, readAssemblyDocument, assemblyScope, assemblyErrorText } = await server.ssrLoadModule('/src/rebuttalAssemblyDraft.ts');
  const memory = () => { const values = new Map(); return { values, getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) }; };
  const document = (source = null, revision = 0, extra = {}) => ({ user_id: 'account_a', paper_id: 'paper_a', scope: 'all', source, revision, updated_at: revision ? '2030-01-01T00:00:00Z' : null, ...extra });
  const create = options => { const store = createAssemblyDraft({ owner: 'account_a', paperId: 'paper_a', scope: 'all', cloud: true, storage: memory(), debounceMs: 60000, ...options }); stores.push(store); return store; };
  assert.equal(assemblyScope('reviewer_a'), 'reviewer-reviewer_a');
  const unavailableLocal = create({ cloud: false, storage: undefined }); await unavailableLocal.load(); assert.equal(unavailableLocal.getSnapshot().ready, true); assert.equal(unavailableLocal.getSnapshot().recoveryError, true, 'No-storage memory fallback retains its explicit warning');
  const unreadableLocal = create({ cloud: false, storage: { getItem() { throw Error('Storage access denied'); }, setItem() { throw Error('Must not write unreadable storage'); }, removeItem() {} } });
  await unreadableLocal.load(); assert.equal(unreadableLocal.getSnapshot().ready, false); assert.equal(unreadableLocal.getSnapshot().blocked, true);
  const cloudWithoutRecovery = create({ storage: undefined, request: async () => document('Known server source', 1) }); await cloudWithoutRecovery.load();
  assert.equal(cloudWithoutRecovery.getSnapshot().ready, true); assert.equal(cloudWithoutRecovery.getSnapshot().source, 'Known server source');
  assert.throws(() => readAssemblyDocument(document(null, 0, { user_id: 'account_b' }), 'account_a', 'paper_a', 'all'));
  assert.throws(() => readAssemblyDocument(document(null, 0, { scope: 'reviewer-all' }), 'account_a', 'paper_a', 'all'));

  const localStorage = memory();
  const local = create({ cloud: false, storage: localStorage }); await local.load();
  assert.equal(local.getSnapshot().source, null); assert.equal(localStorage.values.size, 0, 'Generated responses are not frozen into personal drafts');
  local.update('Personal revision'); assert.equal(local.getSnapshot().status, 'saved');
  const localReload = create({ cloud: false, storage: localStorage }); await localReload.load(); assert.equal(localReload.getSnapshot().source, 'Personal revision');
  const otherOwner = create({ cloud: false, storage: localStorage, owner: 'account_b' }); await otherOwner.load(); assert.equal(otherOwner.getSnapshot().source, null);
  const otherPaper = create({ cloud: false, storage: localStorage, paperId: 'paper_b' }); await otherPaper.load(); assert.equal(otherPaper.getSnapshot().source, null);
  const otherScope = create({ cloud: false, storage: localStorage, scope: 'reviewer-all' }); await otherScope.load(); assert.equal(otherScope.getSnapshot().source, null);
  localReload.update(''); const blank = create({ cloud: false, storage: localStorage }); await blank.load(); assert.equal(blank.getSnapshot().source, '', 'An intentional blank is not replaced by assembly on reload');
  blank.update(null); const reset = create({ cloud: false, storage: localStorage }); await reset.load(); assert.equal(reset.getSnapshot().source, null);
  const corruptStorage = memory(); corruptStorage.setItem([...localStorage.values.keys()][0], '{bad json');
  const corrupt = create({ cloud: false, storage: corruptStorage }); await corrupt.load(); assert.equal(corrupt.getSnapshot().blocked, true); assert.equal(corrupt.getSnapshot().ready, false); assert.equal(corrupt.update('Do not replace the unreadable draft'), false); assert.equal([...corruptStorage.values.values()][0], '{bad json');

  const first = deferred(), calls = [];
  const serial = create({ request: async (_path, body) => { if (!body) return document(); calls.push(structuredClone(body)); return calls.length === 1 ? first.promise : document(body.source, body.expected_revision + 1); } });
  await serial.load(); assert.equal(calls.length, 0);
  serial.update('First'); const saving = serial.flush(); serial.update('Latest');
  assert.equal(calls.length, 1); first.resolve(document('First', 1)); assert.equal(await saving, true);
  assert.equal(calls.length, 2); assert.equal(calls[1].source, 'Latest'); assert.equal(calls[1].expected_revision, 1);
  assert.notEqual(calls[0].request_id, calls[1].request_id); assert.equal(calls[1].expected_user_id, 'account_a');
  assert.equal(serial.getSnapshot().source, 'Latest'); assert.equal(serial.getSnapshot().status, 'saved');
  serial.update(''); await serial.flush(); assert.equal(calls.at(-1).source, '');
  serial.update(null); await serial.flush(); assert.equal(calls.at(-1).source, null); assert.equal(serial.getSnapshot().revision, 4);

  // Frozen commands survive a lost response and account/paper/scope-keyed recovery.
  const recovery = memory(), attempts = [];
  const lost = create({ storage: recovery, request: async (_path, body) => { if (!body) return document(); attempts.push(structuredClone(body)); throw Error('Lost response'); } });
  await lost.load(); lost.update('Already committed'); assert.equal(await lost.flush(), false); lost.dispose();
  const recovered = create({ storage: recovery, request: async (_path, body) => { if (!body) return document('Already committed', 1); attempts.push(structuredClone(body)); return document(body.source, 1); } });
  await recovered.load(); assert.equal(recovered.getSnapshot().source, 'Already committed'); assert.equal(recovered.getSnapshot().status, 'error');
  assert.equal(await recovered.flush(), true); assert.deepEqual(attempts[0], attempts[1]); assert.equal(recovery.values.size, 0);

  // Idempotent receipts describe the original commit, not a newer revision already observed.
  const staleRecovery = memory();
  const oldReceipt = create({ storage: staleRecovery, request: async (_path, body) => { if (!body) return document(); throw Error('Lost acknowledgement'); } });
  await oldReceipt.load(); oldReceipt.update('Original commit'); await oldReceipt.flush(); oldReceipt.dispose();
  const laterRevision = create({ storage: staleRecovery, request: async (_path, body) => body ? document('Original commit', 1) : document('Newer window', 2) });
  await laterRevision.load(); assert.equal(await laterRevision.flush(), false);
  assert.equal(laterRevision.getSnapshot().revision, 2); assert.equal(laterRevision.getSnapshot().status, 'conflict'); assert.equal(laterRevision.getSnapshot().source, 'Original commit');

  let remote = document(), conflictPosts = 0;
  const conflicting = create({ request: async (_path, body) => { if (!body) return remote; conflictPosts++; throw failure(409, 'draft_revision_conflict'); } });
  await conflicting.load(); conflicting.update('Keep this personal rewrite'); await conflicting.flush();
  assert.equal(conflicting.getSnapshot().status, 'conflict'); assert.equal(await conflicting.flush(), false); assert.equal(conflictPosts, 1);
  remote = document('Other window', 2); await conflicting.load(); assert.equal(conflicting.getSnapshot().source, 'Keep this personal rewrite');
  await conflicting.load(true); assert.equal(conflicting.getSnapshot().source, 'Other window'); assert.equal(conflicting.hasPending(), false);

  // Confirmed discard locks editing and mutation retry until the fresh GET settles.
  const delayedReload = deferred(); let reloadReads = 0, reloadPosts = 0;
  const reloading = create({ request: async (_path, body) => { if (body) { reloadPosts++; throw Error('Lost write'); } return ++reloadReads === 1 ? document() : delayedReload.promise; } });
  await reloading.load(); reloading.update('Preserved until explicit discard'); await reloading.flush();
  const loadingSaved = reloading.load(true);
  assert.equal(reloading.update('Must not be accepted during reload'), false); assert.equal(await reloading.flush(), false); assert.equal(reloadPosts, 1);
  delayedReload.resolve(document('Explicitly loaded version', 3)); await loadingSaved;
  assert.equal(reloading.getSnapshot().source, 'Explicitly loaded version'); assert.equal(reloading.getSnapshot().revision, 3);
  for (const error of ['Lost response', 'Your unsaved personal draft was recovered', 'Browser storage is unavailable', 'Personal draft exceeds 200 KiB', 'Invalid account']) {
    assert(/[\p{Script=Han}]/u.test(assemblyErrorText({ ...reloading.getSnapshot(), status: 'error', error }, 'zh')), 'Chinese error UI never exposes raw English transport errors');
  }

  const oversizedCalls = [];
  const oversized = create({ request: async (_path, body) => { if (!body) return document(); oversizedCalls.push(body); return document(body.source, body.expected_revision + 1); } });
  await oversized.load(); oversized.update('中'.repeat(70000)); assert.equal(await oversized.flush(), false); assert.equal(oversizedCalls.length, 0); assert.equal(oversized.getSnapshot().source.length, 70000);
  oversized.update('Short enough'); assert.equal(await oversized.flush(), true);

  let currentOwner = 'account_a'; const oldWrite = deferred(), ownerPosts = [];
  const retired = create({ canAct: () => currentOwner === 'account_a', request: async (_path, body) => { if (!body) return document(); ownerPosts.push(body); return oldWrite.promise; } });
  await retired.load(); retired.update('Account A'); const retiring = retired.flush(); retired.update('Queued A');
  currentOwner = 'account_b'; oldWrite.resolve(document('Account A', 1)); assert.equal(await retiring, false); assert.equal(ownerPosts.length, 1); assert.equal(retired.getSnapshot().revision, 0);
  const stale = deferred(); let reads = 0;
  const strict = create({ request: async () => ++reads === 1 ? stale.promise : document('Current lifecycle', 2) });
  const staleLoad = strict.load(); strict.dispose(); strict.activate(); await strict.load(); stale.resolve(document('Obsolete lifecycle', 1)); await staleLoad;
  assert.equal(strict.getSnapshot().source, 'Current lifecycle'); assert.equal(strict.getSnapshot().status, 'saved');
  console.log('PASS personal assembly drafts: auto/null versus edited/blank source, owner/paper/reviewer isolation, local persistence and corruption protection, serial revisions, reset tombstones, exact retry recovery, conflicts, UTF-8 limits, account retirement and lifecycle');
} finally { stores.forEach(store => store.dispose()); await server.close(); }
await import('./check-rebuttal-assembly-ui.mjs');
