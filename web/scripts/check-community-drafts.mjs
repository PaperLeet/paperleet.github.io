import assert from 'node:assert/strict';
import { createServer } from 'vite';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
try {
  const { draftKey, hasDraft, readPostDraft, writePostDraft, clearPublishedDraft } = await server.ssrLoadModule('/src/communityDrafts.ts');
  const values = new Map();
  const storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
  const original = { title: 'Masked queries', body: 'What happens to the gradient?\n\n```python\nweights = scores.softmax(-1)\n```', category: 'questions' };
  assert.equal(writePostDraft(storage, 'alice', 'architecture/attention', original), true);
  assert.deepEqual(readPostDraft(storage, 'alice', 'architecture/attention'), original, 'Refresh/close must recover exact whitespace and code');
  assert.equal(readPostDraft(storage, 'bob', 'architecture/attention'), undefined, 'Never show another account’s draft');
  assert.equal(readPostDraft(storage, 'alice', 'finetuning/lora'), undefined, 'Never attach a draft to the wrong module');
  assert.equal(readPostDraft(storage, 'alice', 'general'), undefined);
  assert.notEqual(draftKey('a:b', 'c'), draftKey('a', 'b:c'), 'Escaped keys must not collide');
  storage.setItem(draftKey('bob', 'architecture/attention'), storage.getItem(draftKey('alice', 'architecture/attention')));
  assert.equal(readPostDraft(storage, 'bob', 'architecture/attention'), undefined, 'Envelope owner must match even if copied into another key');
  for (const raw of ['{bad json', 'null', JSON.stringify({ version: 2, owner: 'alice', scope: 'general', draft: original }), JSON.stringify({ version: 1, owner: 'alice', scope: 'general', draft: {...original, category: 'unknown'} }), JSON.stringify({ version: 1, owner: 'alice', scope: 'general', draft: {...original, body: 'a'.repeat(20001)} }), 'a'.repeat(100001)]) {
    storage.setItem(draftKey('alice', 'general'), raw);
    assert.equal(readPostDraft(storage, 'alice', 'general'), undefined);
  }
  const newer = {...original, body: original.body + '\nNew work while the request was in flight.'};
  writePostDraft(storage, 'alice', 'architecture/attention', newer);
  assert.equal(clearPublishedDraft(storage, 'alice', 'architecture/attention', original), false);
  assert.deepEqual(readPostDraft(storage, 'alice', 'architecture/attention'), newer, 'Late publish must preserve newer edits');
  assert.equal(clearPublishedDraft(storage, 'alice', 'architecture/attention', newer), true);
  assert.equal(readPostDraft(storage, 'alice', 'architecture/attention'), undefined);
  writePostDraft(storage, 'alice', 'general', original);
  assert.equal(writePostDraft(storage, 'alice', 'general', {...original, title: '', body: ''}), true);
  assert.equal(storage.getItem(draftKey('alice', 'general')), null, 'Discard and empty drafts must not be restored');
  assert.equal(hasDraft({...original, title: ' ', body: '\n'}), false);
  const blocked = { getItem() { throw Error('blocked'); }, setItem() { throw Error('quota'); }, removeItem() { throw Error('blocked'); } };
  assert.equal(readPostDraft(blocked, 'alice', 'general'), undefined);
  assert.equal(writePostDraft(blocked, 'alice', 'general', original), false);
  assert.equal(writePostDraft(undefined, 'alice', 'general', original), false);
  assert.equal(writePostDraft(storage, '', 'general', original), false);
  console.log('PASS community draft refresh recovery, account/module isolation, validation, publish/discard cleanup, late-result preservation and blocked storage');
} finally {
  await server.close();
}
