import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { createServer } from 'vite';

const html = fs.readFileSync('index.html', 'utf8');
const source = html.match(/<script id="app-startup">([\s\S]*?)<\/script>/)?.[1];
assert(source, 'Recovery must run inline, before downloaded app modules');
assert(html.includes('id="startup-title"') && html.includes('<noscript>'), 'The document must be readable even when the app cannot load');

function harness({ english = false, blockedStorage = false, parsed = true } = {}) {
  let connected = parsed, cleared = false, timer;
  const nodes = Object.fromEntries(['startup-title', 'startup-message', 'startup-retry'].map(id => [id, { textContent: '', hidden: true, href: '' }]));
  const events = () => {
    const listeners = new Map();
    return {
      addEventListener(name, fn) { listeners.set(name, fn); },
      removeEventListener(name) { listeners.delete(name); },
      dispatch(name, extra = {}) { listeners.get(name)?.({ type: name, ...extra }); },
      listeners,
    };
  };
  const window = events(), document = { ...events(), getElementById: id => connected ? nodes[id] : null };
  vm.runInNewContext(source, {
    window, document, URL, Date,
    location: { href: 'https://paperleet.com/?campaign=demo#/account/continue/22' },
    localStorage: { getItem() { if (blockedStorage) throw Error('Storage blocked'); return english ? 'en' : null; } },
    setTimeout(fn, ms) { assert.equal(ms, 10000); timer = fn; return 1; },
    clearTimeout(id) { assert.equal(id, 1); cleared = true; },
  });
  return { nodes, window, document, parse() { connected = true; document.dispatch('DOMContentLoaded'); }, timeout() { timer(); }, cleared: () => cleared };
}

const failed = harness({ english: true, parsed: false });
failed.window.dispatch('error', { target: { tagName: 'SCRIPT' }, message: 'private exception detail' });
failed.parse();
assert.equal(failed.nodes['startup-title'].textContent, 'The page could not load');
assert.equal(failed.nodes['startup-retry'].hidden, false);
const retry = new URL(failed.nodes['startup-retry'].href);
assert.equal(retry.origin, 'https://paperleet.com');
assert.equal(retry.hash, '#/account/continue/22');
assert.equal(retry.searchParams.get('campaign'), 'demo');
assert(retry.searchParams.has('_paperleet_reload'), 'Recovery must bypass a stale cached HTML document');
assert(!JSON.stringify(failed.nodes).includes('private exception detail'));

const slow = harness({ blockedStorage: true });
slow.parse();
assert.equal(slow.nodes['startup-retry'].hidden, true);
slow.timeout();
assert.equal(slow.nodes['startup-retry'].hidden, false);
assert(slow.nodes['startup-message'].textContent.includes('加载时间较长'));

const success = harness();
success.parse();
success.window.dispatch('error', { target: { tagName: 'IMG' } });
assert.equal(success.nodes['startup-retry'].hidden, true, 'An unrelated image failure is not an app failure');
success.document.dispatch('paperleet:ready');
assert(success.cleared());
assert.equal(success.window.listeners.size, 0, 'After mount, startup recovery must not interrupt active work');
assert.equal(success.document.listeners.size, 0);
const before = structuredClone(success.nodes);
success.window.dispatch('unhandledrejection');
assert.deepEqual(success.nodes, before);

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
try {
  const { recoveryUrl } = await server.ssrLoadModule('/src/appRecovery.ts');
  const url = new URL(recoveryUrl('https://paperleet.com/?lang=en&_paperleet_reload=old#/22', 123));
  assert.equal(url.hash, '#/22');
  assert.equal(url.searchParams.get('lang'), 'en');
  assert.deepEqual(url.searchParams.getAll('_paperleet_reload'), ['123']);
} finally { await server.close(); }
console.log('PASS startup recovery: missing entry, delayed load, blocked storage, fresh-document retry, route preservation, and cleanup without automatic reload');
