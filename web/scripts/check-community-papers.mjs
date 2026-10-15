import assert from 'node:assert/strict';
import { createServer } from 'vite';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
try {
  const { parseArxivUrl, canonicalArxivUrl, readPaperPreview } = await server.ssrLoadModule('/src/communityPapers.ts');
  const url = 'https://arxiv.org/abs/1706.03762';
  const secondUrl = 'https://arxiv.org/abs/2210.02747v2';
  for (const input of [url, 'http://arxiv.org/abs/1706.03762', 'https://www.arxiv.org/pdf/1706.03762.pdf', ' HTTP://EXPORT.ARXIV.ORG/pdf/1706.03762 ']) {
    assert.equal(canonicalArxivUrl(input), url, 'Abstract and PDF URLs identify the same paper');
  }
  assert.deepEqual(parseArxivUrl('https://export.arxiv.org/pdf/hep-th/9901001v2.pdf'), {
    arxivId: 'hep-th/9901001v2', url: 'https://arxiv.org/abs/hep-th/9901001v2', pdfUrl: 'https://arxiv.org/pdf/hep-th/9901001v2',
  });
  assert.equal(canonicalArxivUrl('https://arxiv.org/abs/0704.0001v9999'), 'https://arxiv.org/abs/0704.0001v9999');
  for (const input of [null, {}, '', '1706.03762', 'https://arxiv.org.evil.example/abs/1706.03762',
    'https://arxiv.org@evil.example/abs/1706.03762', 'https://someone@arxiv.org/abs/1706.03762',
    'https://arxiv.org:443/abs/1706.03762', 'https://arxiv.org:80/abs/1706.03762',
    'javascript:alert(1)', '//arxiv.org/abs/1706.03762', 'file:///abs/1706.03762',
    `${url}?`, `${url}#`, `${url}?download=1`, `${url}#page=1`, `${url}/`, `${url}.pdf`,
    'https://arxiv.org/pdf/1706.03762.pdf.pdf', 'https://arxiv.org/abs/1700.03762',
    'https://arxiv.org/abs/1713.03762', 'https://arxiv.org/abs/1706.037',
    'https://arxiv.org/abs/1706.03762v0', 'https://arxiv.org/abs/1706.03762v01',
    'https://arxiv.org/abs/1706.03762v10000', 'https://arxiv.org/abs/hep-th/9913001',
    'https://arxiv.org/abs/%31%37%30%36.03762', 'https://arxiv.org\\@evil.example/abs/1706.03762',
    'https://arxiv.org/abs/1706.0\n3762', `${url}\n`, `https://arxiv.org/abs/${'1'.repeat(600)}`]) {
    assert.equal(canonicalArxivUrl(input), undefined, `Reject unsafe or malformed paper URL: ${JSON.stringify(input)}`);
  }

  const paper = { arxiv_id: '1706.03762', url, pdf_url: 'https://arxiv.org/pdf/1706.03762', status: 'ready',
    title: 'Attention Is All You Need', authors: ['A. Author'], abstract: 'A paper abstract.', published: '2017-06-12T00:00:00Z',
    thumbnail_url: `/api/community/papers/thumbnails/${'a'.repeat(64)}.png` };
  assert.deepEqual(readPaperPreview(paper, url), paper);
  assert.equal(readPaperPreview(paper, secondUrl), undefined, 'A preview must belong to the attached paper');
  assert.equal(readPaperPreview(paper, null), undefined, 'A post without an attachment cannot render an unrelated preview');
  for (const change of [{ url: 'https://evil.example' }, { pdf_url: 'https://evil.example/paper.pdf' },
    { pdf_url: 'https://arxiv.org/pdf/2210.02747' }, { url: 'http://arxiv.org/abs/1706.03762' },
    { arxiv_id: '2210.02747' }, { status: 'loading' }, { authors: ['Author', 7] }, { title: 5 }, { abstract: 'x'.repeat(30001) }]) {
    assert.equal(readPaperPreview({ ...paper, ...change }, url), undefined, 'Reject malformed or mismatched API metadata');
  }
  for (const thumbnail_url of ['https://evil.example/x.png', '//evil.example/x.png', '/api/community/papers/thumbnails/../../private.png',
    `/api/community/papers/thumbnails/${'a'.repeat(64)}.png?token=x`, '/api/community/papers/thumbnails/not-a-hash.png']) {
    const safe = readPaperPreview({ ...paper, thumbnail_url }, url);
    assert.equal(safe.url, url);
    assert.equal(safe.thumbnail_url, undefined, 'Unsafe thumbnail must never become an image request');
  }
  assert.deepEqual(readPaperPreview({ arxiv_id: paper.arxiv_id, url, pdf_url: paper.pdf_url, status: 'unavailable', title: null, authors: null }),
    { arxiv_id: paper.arxiv_id, url, pdf_url: paper.pdf_url, status: 'unavailable' });

  const drafts = await server.ssrLoadModule('/src/communityDrafts.ts');
  const edits = await server.ssrLoadModule('/src/communityEdits.ts');
  const requests = await server.ssrLoadModule('/src/publicationRequests.ts');
  const values = new Map();
  const storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
  const draft = { title: '', body: '', category: 'papers', paperUrl: ' http://export.arxiv.org/pdf/1706.03762.pdf ' };
  assert(drafts.hasDraft(draft), 'A link entered before title/body is still a draft');
  assert(drafts.writePostDraft(storage, 'alice', 'general', draft));
  assert.deepEqual(drafts.readPostDraft(storage, 'alice', 'general'), draft, 'Reload keeps the exact raw link while typing');
  assert.equal(drafts.readPostDraft(storage, 'bob', 'general'), undefined);
  const newer = { ...draft, paperUrl: secondUrl };
  drafts.writePostDraft(storage, 'alice', 'general', newer);
  assert.equal(drafts.clearPublishedDraft(storage, 'alice', 'general', draft), false, 'A delayed publish cannot erase a changed link');
  assert.deepEqual(drafts.readPostDraft(storage, 'alice', 'general'), newer);
  assert(drafts.clearPublishedDraft(storage, 'alice', 'general', newer));
  const malformedDraft = { ...draft, paperUrl: 'https://arxiv.org/abs/still-typing' };
  assert(drafts.writePostDraft(storage, 'alice', 'general', malformedDraft));
  assert.deepEqual(drafts.readPostDraft(storage, 'alice', 'general'), malformedDraft, 'Incomplete URL input survives refresh');
  assert(!drafts.writePostDraft(storage, 'alice', 'general', { ...draft, paperUrl: 'x'.repeat(2049) }));
  values.set(drafts.draftKey('alice', 'general'), JSON.stringify({ version: 1, owner: 'alice', scope: 'general', draft: { ...draft, paperUrl: 42 } }));
  assert.equal(drafts.readPostDraft(storage, 'alice', 'general'), undefined);

  const post = { title: 'Paper question', body: 'What changes in this result?', category: 'papers', edit_version: 'a'.repeat(64) };
  assert(!Object.hasOwn(edits.newPostEdit(post), 'paperUrl'), 'Legacy edits retain their original shape');
  const edit = edits.newPostEdit({ ...post, paper_url: url });
  assert.equal(edit.paperUrl, url);
  assert(edits.writePostEdit(storage, 'alice', 'post', edit));
  assert.deepEqual(edits.readPostEdit(storage, 'alice', 'post'), edit);
  const newEdit = { ...edit, paperUrl: secondUrl };
  assert(!edits.samePostEdit(edit, newEdit));
  edits.writePostEdit(storage, 'alice', 'post', newEdit);
  assert(!edits.clearSavedPostEdit(storage, 'alice', 'post', edit));
  assert.deepEqual(edits.readPostEdit(storage, 'alice', 'post'), newEdit);
  assert.equal(edits.readPostEdit(storage, 'bob', 'post'), undefined);
  assert(edits.clearSavedPostEdit(storage, 'alice', 'post', newEdit));

  const complete = { ...draft, title: 'Paper question', body: 'What changes in this result?' };
  const payload = requests.postPublication(complete);
  assert.equal(payload.paper_url, url);
  assert(!Object.hasOwn(requests.postPublication({ ...complete, category: 'questions' }), 'paper_url'), 'Hidden links are not published outside Papers');
  assert(!Object.hasOwn(requests.postPublication({ ...complete, paperUrl: '' }), 'paper_url'), 'No attachment preserves the legacy receipt shape');
  const first = requests.preparePublication(storage, 'paper-alice', 'post', 'general', payload);
  assert.equal(requests.preparePublication(storage, 'paper-alice', 'post', 'general', requests.postPublication({ ...complete, paperUrl: url })).receipt.requestId,
    first.receipt.requestId, 'Equivalent abstract/PDF links share the original idempotency key');
  const reload = await server.ssrLoadModule('/src/publicationRequests.ts?paper-reload');
  assert.equal(reload.readPublication(storage, 'paper-alice', 'post', 'general').receipt.requestId, first.receipt.requestId);
  assert.equal(reload.readPublication(storage, 'paper-bob', 'post', 'general'), undefined);
  const second = requests.preparePublication(storage, 'paper-alice', 'post', 'general', requests.postPublication({ ...complete, paperUrl: secondUrl }));
  assert.notEqual(second.receipt.requestId, first.receipt.requestId, 'A changed attachment is new publication content');
  assert(!requests.clearPublication(storage, first.receipt), 'A stale acknowledgement cannot clear a newer paper publication');
  assert.equal(requests.readPublication(storage, 'paper-alice', 'post', 'general').receipt.requestId, second.receipt.requestId);
  const legacyPayload = requests.postPublication({ ...complete, paperUrl: undefined });
  const legacyReceipt = { ...first.receipt, owner: 'legacy-paper', payload: legacyPayload };
  values.set(requests.publicationKey('legacy-paper', 'post', 'general'), JSON.stringify(legacyReceipt));
  assert.equal(reload.readPublication(storage, 'legacy-paper', 'post', 'general').receipt.requestId, legacyReceipt.requestId, 'Old pending paper posts remain retryable');
  const nullReceipt = { ...legacyReceipt, owner: 'null-paper', payload: { ...legacyPayload, paper_url: null } };
  values.set(requests.publicationKey('null-paper', 'post', 'general'), JSON.stringify(nullReceipt));
  assert.equal(reload.readPublication(storage, 'null-paper', 'post', 'general').receipt.requestId, nullReceipt.requestId);
  for (const paperUrl of ['https://evil.example/paper.pdf', `${url}#`, `${url}\n`, 'http://arxiv.org:80/abs/1706.03762']) {
    const invalid = requests.postPublication({ ...complete, paperUrl });
    assert.throws(() => requests.preparePublication(storage, 'invalid-paper', 'post', 'general', invalid), /Invalid publication request/, 'Rendering an invalid draft is safe; sending it is rejected');
  }
  const invalidReceipt = { ...legacyReceipt, owner: 'malformed-paper', payload: { ...payload, paper_url: 'https://evil.example' } };
  values.set(requests.publicationKey('malformed-paper', 'post', 'general'), JSON.stringify(invalidReceipt));
  assert.equal(reload.readPublication(storage, 'malformed-paper', 'post', 'general'), undefined);
  console.log('PASS arXiv URL and API preview safety; paper draft/edit reload, account isolation, legacy receipts, retry identity and stale acknowledgement protection');
} finally {
  await server.close();
}
