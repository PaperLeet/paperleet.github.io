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
  const { replyDraftKey, hasReplyDraft, readReplyDraft, writeReplyDraft, clearPublishedReplyDraft } = await server.ssrLoadModule('/src/replyDrafts.ts');
  const post = 'a'.repeat(32), otherPost = 'b'.repeat(32), parent = 'c'.repeat(32);
  const reply = { body: '  About the mask: 中文 🧪\n\n```python\nweights.sum(-1)\n```\n', parentId: parent };
  assert(writeReplyDraft(storage,'alice',post,reply));
  assert.deepEqual(readReplyDraft(storage,'alice',post),reply,'Refresh must preserve reply text, whitespace, Markdown and parent identity');
  assert.equal(readReplyDraft(storage,'bob',post),undefined,'Reply drafts belong to the signed-in account');
  assert.equal(readReplyDraft(storage,'alice',otherPost),undefined,'Reply drafts must not move to a different post');
  assert.notEqual(replyDraftKey('a:b','c'),replyDraftKey('a','b:c'));
  assert.notEqual(replyDraftKey('alice',post),draftKey('alice',post),'Post and reply drafts use distinct namespaces');
  storage.setItem(replyDraftKey('bob',post),storage.getItem(replyDraftKey('alice',post)));
  assert.equal(readReplyDraft(storage,'bob',post),undefined,'Envelope owner must match the reply-draft key');
  storage.setItem(replyDraftKey('alice',otherPost),storage.getItem(replyDraftKey('alice',post)));
  assert.equal(readReplyDraft(storage,'alice',otherPost),undefined,'Envelope post must match the reply-draft key');
  for (const bad of ['{bad json','null','a'.repeat(100001),JSON.stringify({version:2,owner:'alice',post,draft:reply}),
    ...[{body:4},{body:'a'.repeat(10001)},{...reply,parentId:null},{...reply,parentId:'not-a-reply'},[],null].map(draft=>JSON.stringify({version:1,owner:'alice',post,draft}))]) {
    storage.setItem(replyDraftKey('alice',post),bad);
    assert.equal(readReplyDraft(storage,'alice',post),undefined,'Malformed saved replies must not break the thread');
  }
  assert(writeReplyDraft(storage,'alice',post,reply));
  const newerReply={...reply,body:reply.body+'New work after returning to the thread.'};
  writeReplyDraft(storage,'alice',post,newerReply);
  assert.equal(clearPublishedReplyDraft(storage,'alice',post,reply),false);
  assert.deepEqual(readReplyDraft(storage,'alice',post),newerReply,'A late publish response must not erase a newer reply draft');
  const retargeted={...newerReply,parentId:'d'.repeat(32)};
  writeReplyDraft(storage,'alice',post,retargeted);
  assert.equal(clearPublishedReplyDraft(storage,'alice',post,newerReply),false,'Changing the target is also new work');
  assert(clearPublishedReplyDraft(storage,'alice',post,retargeted));
  assert.equal(readReplyDraft(storage,'alice',post),undefined,'A successful publish removes only its exact saved draft');
  writeReplyDraft(storage,'alice',post,reply);
  assert(writeReplyDraft(storage,'alice',post,{body:''}));
  assert.equal(readReplyDraft(storage,'alice',post),undefined,'Discarding clears the text and reply target');
  assert.equal(hasReplyDraft({body:' \n'}),false);
  assert.equal(hasReplyDraft({body:'',parentId:parent}),true,'A selected reply target can survive refresh before typing');
  assert.equal(readReplyDraft(blocked,'alice',post),undefined);
  assert.equal(writeReplyDraft(blocked,'alice',post,reply),false);
  assert.equal(writeReplyDraft(undefined,'alice',post,reply),false);
  assert.equal(writeReplyDraft(storage,'',post,reply),false);
  assert.equal(writeReplyDraft(storage,'alice',post,{body:'a'.repeat(10001)}),false);
  console.log('PASS community post/reply draft refresh recovery, account/module/post isolation, parent identity, validation, publish/discard cleanup, late-result preservation and blocked storage');
} finally {
  await server.close();
}
