import assert from 'node:assert/strict';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { localizedRenderer } from './render-locale.mjs';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
try {
  const { default: Card } = await server.ssrLoadModule('/src/PaperPreviewCard.tsx');
  const { default: Field } = await server.ssrLoadModule('/src/PaperLinkField.tsx');
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const { apiUrl } = await server.ssrLoadModule('/src/apiClient.ts');
  const render = await localizedRenderer(server, 'en');
  const first = 'https://arxiv.org/abs/1706.03762';
  const second = 'https://arxiv.org/abs/2210.02747';
  const paper = {
    arxiv_id: '1706.03762', url: first, pdf_url: 'https://arxiv.org/pdf/1706.03762', status: 'ready',
    title: '<script>alert(1)</script>', authors: ['<img src=x onerror=alert(1)>'],
    abstract: '<iframe src="javascript:alert(1)">', published: 'bogus',
    thumbnail_url: 'https://evil.example/beacon.png',
  };
  const html = render(React.createElement(Card, { url: first, paper }));
  assert(html.includes('&lt;script&gt;') && html.includes('&lt;img') && html.includes('&lt;iframe'));
  assert(!html.includes('<script>') && !html.includes('<iframe') && !html.includes('evil.example') && !html.includes('Invalid Date'),
    'Metadata is text, invalid dates are omitted, and an untrusted thumbnail never becomes a request');
  assert.equal((html.match(/href="https:\/\/arxiv.org\//g) || []).length, 4);
  assert(!html.includes('<img'), 'Without a valid API thumbnail or a local thumbnail the cover is the PDF placeholder');
  // Layout: by default the cover sits beside the copy, which ends with the links; linksUnderCover (roadmap cards) stacks the cover
  // and the links in .paper-preview-side and leaves the copy everything else.
  const card = markup => new JSDOM(markup).window.document.querySelector('aside.paper-preview-card');
  const shape = element => [...element.children].map(child => child.localName + '.' + child.className);
  const underHtml = render(React.createElement(Card, { url: first, paper, linksUnderCover: true })), plain = card(html), under = card(underHtml);
  assert.deepEqual(shape(plain), ['a.paper-preview-cover', 'div.paper-preview-copy'], 'Without linksUnderCover the cover sits beside the copy');
  assert.equal(shape(plain.querySelector('.paper-preview-copy')).at(-1), 'div.paper-preview-links', 'Without linksUnderCover the links end the copy');
  assert.deepEqual(shape(under), ['div.paper-preview-side', 'div.paper-preview-copy'], 'With linksUnderCover the side column comes before the copy');
  assert.deepEqual(shape(under.querySelector('.paper-preview-side')), ['a.paper-preview-cover', 'div.paper-preview-links'], 'The links follow the cover');
  assert.deepEqual(shape(under.querySelector('.paper-preview-copy')), shape(plain.querySelector('.paper-preview-copy')).slice(0, -1), 'The copy keeps everything but the links');
  assert.deepEqual([...under.querySelectorAll('.paper-preview-links a')].map(link => [link.textContent, link.getAttribute('href')]), [['Paper page', first], ['Open PDF', 'https://arxiv.org/pdf/1706.03762']]);
  assert.equal((underHtml.match(/href="https:\/\/arxiv.org\//g) || []).length, 4, 'Both layouts keep the four arXiv links');
  // A local thumbnail (roadmap covers) is used only when the validated preview has no API thumbnail.
  const apiThumbnail = `/api/community/papers/thumbnails/${'a'.repeat(64)}.png`, local = './roadmap-thumbnails/1706.03762.webp';
  const sources = markup => [...markup.matchAll(/<img [^>]*src="([^"]*)"/g)].map(match => match[1]);
  assert.deepEqual(sources(render(React.createElement(Card, { url: first, paper: { ...paper, thumbnail_url: apiThumbnail }, thumbnail: local }))), [apiUrl(apiThumbnail)], 'The API thumbnail wins over the local one');
  assert.deepEqual(sources(render(React.createElement(Card, { url: first, paper, thumbnail: local }))), [local], 'A local thumbnail alone renders, also beside a rejected API thumbnail');
  assert.deepEqual(sources(render(React.createElement(Card, { url: first, paper: { ...paper, thumbnail_url: apiThumbnail } }))), [apiUrl(apiThumbnail)], 'Without the prop the API thumbnail renders as before');
  assert.equal(render(React.createElement(Card, { url: 'javascript:alert(1)', paper })), '');
  const fallback = render(React.createElement(Card, { url: first, paper: { ...paper, url: 'https://evil.example' } }));
  assert(fallback.includes('Paper details are unavailable') && !fallback.includes('evil.example'),
    'Malformed metadata falls back to the canonical attached paper');
  const unavailable = render(React.createElement(Field, { value: first, onChange() {}, owner: 'alice', available: false }));
  assert(unavailable.includes('disabled=""') && unavailable.includes('Paper previews are unavailable'));

  const dom = new JSDOM('<div id="app"></div>', { url: 'https://paperleet.test' });
  const originals = new Map();
  const pending = [];
  const globals = {
    window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
    localStorage: dom.window.localStorage, IS_REACT_ACT_ENVIRONMENT: true,
    fetch: (path, options) => new Promise(resolve => pending.push({ path, options, resolve })),
  };
  let root;
  try {
    for (const [key, value] of Object.entries(globals)) {
      originals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
      Object.defineProperty(globalThis, key, { configurable: true, value, writable: true });
    }
    root = createRoot(document.getElementById('app'));
    const draw = value => React.createElement(LocaleProvider, null,
      React.createElement(Field, { value, onChange() {}, owner: 'alice', available: true }));
    const waitForRequest = async count => {
      // Keep the production debounce intact; poll only the observable request.
      for (let attempts = 0; pending.length < count && attempts < 100; attempts += 1) {
        await act(async () => new Promise(resolve => setTimeout(resolve, 20)));
      }
      assert.equal(pending.length, count, 'A valid link issues one debounced preview request');
    };
    await act(async () => root.render(draw(first)));
    await waitForRequest(1);
    assert.equal(JSON.parse(pending[0].options.body).expected_user_id, 'alice');
    await act(async () => root.render(draw(second)));
    assert(pending[0].options.signal.aborted, 'Changing papers cancels the previous request');
    await waitForRequest(2);
    const response = (url, title) => ({ ok: true, json: async () => ({ paper: {
      arxiv_id: url.split('/').at(-1), url, pdf_url: url.replace('/abs/', '/pdf/'), status: 'ready', title,
    } }) });
    await act(async () => pending[1].resolve(response(second, 'Current second paper')));
    assert(document.body.textContent.includes('Current second paper'));
    // Deliberately resolve a cancelled fetch: some transports ignore AbortSignal.
    await act(async () => pending[0].resolve(response(first, 'Stale first paper')));
    assert(document.body.textContent.includes('Current second paper') && !document.body.textContent.includes('Stale first paper'),
      'A late old response cannot replace the current paper');
    await act(async () => root.render(draw('https://evil.example/paper.pdf')));
    assert(!document.body.textContent.includes('Current second paper'), 'Invalid input immediately hides the previous paper');
    assert.equal(pending.length, 2, 'Invalid input does not send a preview request');
  } finally {
    if (root) await act(async () => root.unmount());
    for (const [key, descriptor] of originals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
    dom.window.close();
  }
  console.log('PASS paper UI: escaped metadata, safe link fallback, API thumbnail before a local one, links in the copy or under the cover, capability fallback, debounced requests, abort and stale-response protection');
} finally {
  await server.close();
}
