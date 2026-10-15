import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: 'https://paperleet.test/' });
const names = ['window', 'document', 'navigator', 'HTMLElement', 'fetch', 'IS_REACT_ACT_ENVIRONMENT'];
const originals = new Map(names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
for (const name of ['window', 'document', 'navigator', 'HTMLElement']) {
  Object.defineProperty(globalThis, name, { value: dom.window[name], configurable: true, writable: true });
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const { createRoot } = await import('react-dom/client');
const project = fileURLToPath(new URL('../', import.meta.url));
const hour = 3_600_000;
const statistics = (accepted, failed) => ({
  attempted_users: 2, solved_users: accepted ? 1 : 0,
  total_submissions: accepted + failed, accepted_submissions: accepted, failed_submissions: failed,
  acceptance_rate: accepted + failed ? accepted / (accepted + failed) * 100 : null,
});
const fixture = { 'architecture/attention': statistics(3, 2), 'optimization/adamw': statistics(1, 6) };
const success = stats => ({ available: true, stats });

// Fake only the browser timers used by the hook, leaving React/Vite scheduling
// alone. Both the 12-second request deadline and hourly poll remain observable.
function harness(useProblemStats) {
  let now = 0, sequence = 0, visible = true, online = true, root, value;
  const requests = [], timers = new Map();
  const originalNow = Date.now;
  const originalSetTimeout = window.setTimeout, originalClearTimeout = window.clearTimeout;
  Date.now = () => now;
  window.setTimeout = (callback, delay = 0, ...args) => {
    const id = ++sequence;
    timers.set(id, { at: now + Math.max(0, Number(delay) || 0), callback: () => callback(...args) });
    return id;
  };
  window.clearTimeout = id => timers.delete(id);
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => visible ? 'visible' : 'hidden' });
  Object.defineProperty(navigator, 'onLine', { configurable: true, get: () => online });
  globalThis.fetch = (url, options = {}) => new Promise((resolve, reject) => {
    const request = { url, options, resolve, reject };
    requests.push(request);
    options.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
  });
  function Probe({ refreshKey }) {
    value = useProblemStats(refreshKey);
    return React.createElement('output', null, value.totalSubmissions === undefined ? 'unavailable' : String(value.totalSubmissions));
  }
  async function render(refreshKey = 'library') {
    if (!root) root = createRoot(document.querySelector('#root'));
    await act(async () => root.render(React.createElement(Probe, { refreshKey })));
  }
  async function reply(index, data, status = 200) {
    assert(requests[index], `Missing request ${index}`);
    await act(async () => requests[index].resolve(Response.json(data, { status })));
  }
  async function refresh() {
    await act(async () => { void value.refresh(); });
  }
  async function tick(delay) {
    const end = now + delay;
    for (let callbacks = 0; ; callbacks += 1) {
      assert(callbacks < 100, 'Timer callbacks must make progress instead of polling in a loop');
      const next = [...timers.entries()].sort((a, b) => a[1].at - b[1].at)[0];
      if (!next || next[1].at > end) break;
      now = next[1].at;
      timers.delete(next[0]);
      await act(async () => next[1].callback());
    }
    now = end;
  }
  async function visibility(next) {
    visible = next;
    await act(async () => document.dispatchEvent(new window.Event('visibilitychange')));
  }
  async function connectivity(next) {
    online = next;
    await act(async () => window.dispatchEvent(new window.Event(next ? 'online' : 'offline')));
  }
  async function unmount() {
    if (root) await act(async () => root.unmount());
    root = undefined;
  }
  async function dispose() {
    await unmount();
    Date.now = originalNow;
    window.setTimeout = originalSetTimeout;
    window.clearTimeout = originalClearTimeout;
  }
  return { render, reply, refresh, tick, visibility, connectivity, unmount, dispose, requests, timers,
    get value() { return value; }, get text() { return document.querySelector('output')?.textContent; } };
}

let server, current;
try {
  server = await createServer({
    root: project, configFile: false, server: { middlewareMode: true, hmr: false }, appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
    define: {
      'import.meta.env.VITE_PUBLIC_SITE': '"0"',
      'import.meta.env.VITE_APP_EDITION': '"hosted"',
      'import.meta.env.VITE_API_ORIGIN': '""',
    },
  });
  const { useProblemStats, COMMUNITY_STATS_REFRESH_MS } = await server.ssrLoadModule('/src/ProblemStats.tsx');
  assert.equal(COMMUNITY_STATS_REFRESH_MS, hour);
  current = harness(useProblemStats);
  await current.render();
  assert.equal(current.requests.length, 1, 'Mount fetches the real community aggregate once');
  assert.equal(current.value.totalSubmissions, undefined, 'Loading must never invent zero');
  assert.equal(current.text, 'unavailable');
  assert.equal(current.requests[0].url, '/api/practice/stats');
  assert.equal(current.requests[0].options.credentials, 'omit', 'The aggregate does not send account cookies');
  assert.equal(current.requests[0].options.cache, 'no-store');
  await current.reply(0, success(fixture));
  assert.equal(current.value.totalSubmissions, 12, 'Count accepted and failed submissions across all problems, not users or problem rows');
  assert.equal(current.text, '12');
  assert.equal(current.value.available, true);
  assert.equal(current.value.loading, false);
  await current.render();
  assert.equal(current.requests.length, 1, 'An unchanged route does not refetch');
  await current.render('problem/attention');
  assert.equal(current.requests.length, 2, 'Changing refreshKey still refreshes immediately');
  await current.reply(1, success({}));
  assert.equal(current.value.totalSubmissions, 0, 'A successfully loaded empty aggregate is a real zero');
  assert.equal(current.text, '0');
  await current.refresh();
  await current.reply(2, { available: false });
  assert.equal(current.value.totalSubmissions, undefined);
  assert.equal(current.value.available, false);
  assert.equal(current.value.error, null);
  assert.equal(current.text, 'unavailable', 'An unavailable service must clear a previously valid zero');
  await current.refresh();
  await current.reply(3, success(fixture));
  assert.equal(current.value.totalSubmissions, 12);
  await current.refresh();
  await current.reply(4, { error: 'Unavailable' }, 503);
  assert.equal(current.value.totalSubmissions, undefined, 'A failed refresh clears a previously valid count');
  assert.equal(current.value.available, null);
  assert(current.value.error);
  await current.refresh();
  await current.reply(5, success({ broken: { ...statistics(0, 0), total_submissions: -1 } }));
  assert.equal(current.value.totalSubmissions, undefined, 'Malformed statistics must not reach the aggregate');
  assert(current.value.error);
  await current.refresh();
  await current.reply(6, success({ first: statistics(Number.MAX_SAFE_INTEGER, 0), second: statistics(1, 0) }));
  assert.equal(current.value.available, true, 'Each problem can be valid even when their combined count overflows');
  assert.equal(current.value.totalSubmissions, undefined, 'Never display a rounded unsafe integer sum');
  await current.dispose();
  current = harness(useProblemStats);
  await current.render();
  await current.reply(0, success(fixture));
  await current.tick(hour - 1);
  assert.equal(current.requests.length, 1);
  await current.tick(1);
  assert.equal(current.requests.length, 2, 'Visible, online tabs refresh at the one-hour boundary');
  await current.reply(1, success(fixture));
  await current.tick(5 * 60_000);
  await current.visibility(false);
  assert.equal(current.timers.size, 0, 'Hiding cancels the scheduled poll');
  await current.tick(5 * 60_000);
  await current.visibility(true);
  assert.equal(current.requests.length, 2, 'Returning before the deadline does not fetch');
  await current.tick(50 * 60_000 - 1);
  assert.equal(current.requests.length, 2);
  await current.tick(1);
  assert.equal(current.requests.length, 3, 'Early visibility recovery preserves the original deadline');
  await current.reply(2, success(fixture));
  await current.visibility(false);
  await current.tick(2 * hour);
  assert.equal(current.requests.length, 3, 'Hidden tabs do not poll even across multiple deadlines');
  await current.visibility(true);
  await current.visibility(true);
  await current.connectivity(true);
  assert.equal(current.requests.length, 4, 'An overdue resume fetches exactly once, without catch-up bursts');
  await current.reply(3, success(fixture));
  await current.tick(hour - 1);
  assert.equal(current.requests.length, 4);
  await current.tick(1);
  assert.equal(current.requests.length, 5, 'Overdue recovery starts a fresh one-hour interval');
  await current.reply(4, success(fixture));
  await current.connectivity(false);
  assert.equal(current.timers.size, 0, 'Going offline cancels the scheduled poll');
  await current.tick(hour + 1000);
  assert.equal(current.requests.length, 5);
  await current.connectivity(true);
  assert.equal(current.requests.length, 6, 'An overdue connection recovery fetches once');
  await current.reply(5, success(fixture));
  await current.connectivity(false);
  await current.tick(1000);
  await current.connectivity(true);
  assert.equal(current.requests.length, 6, 'A brief offline period does not force a new request');
  await current.tick(hour - 1001);
  assert.equal(current.requests.length, 6);
  await current.tick(1);
  assert.equal(current.requests.length, 7, 'Online recovery preserves the next deadline');
  await current.unmount();
  assert.equal(current.requests[6].options.signal.aborted, true, 'Unmount aborts an in-flight refresh');
  assert.equal(current.timers.size, 0, 'Unmount clears both poll and request timeout');
  await current.visibility(true);
  await current.connectivity(true);
  await current.tick(2 * hour);
  assert.equal(current.requests.length, 7, 'Unmount removes lifecycle listeners and cannot resume polling');
  await current.dispose();
  current = undefined;
  await server.close();

  server = await createServer({
    root: project, configFile: false, server: { middlewareMode: true, hmr: false }, appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
    define: {
      'import.meta.env.VITE_PUBLIC_SITE': '"1"',
      'import.meta.env.VITE_APP_EDITION': '"local"',
      'import.meta.env.VITE_HOSTED_APP_AVAILABLE': '"1"',
      'import.meta.env.VITE_HOSTED_APP_URL': '"https://app.paperleet.com"',
      'import.meta.env.VITE_API_ORIGIN': '""',
    },
  });
  const publicModule = await server.ssrLoadModule('/src/ProblemStats.tsx');
  current = harness(publicModule.useProblemStats);
  await current.render();
  await current.render('problem/attention');
  await current.refresh();
  await current.visibility(false);
  await current.tick(2 * hour);
  await current.visibility(true);
  await current.connectivity(false);
  await current.connectivity(true);
  assert.equal(current.requests.length, 0, 'Public-site builds never call the statistics API, including route/manual/lifecycle refresh');
  assert.equal(current.timers.size, 0, 'Public-site builds never schedule polling');
  assert.equal(current.value.totalSubmissions, undefined);
  assert.equal(current.value.available, false);
  assert.equal(current.value.loading, false);
  console.log('PASS community submissions: validated aggregate, zero versus unavailable, overflow, route refresh, hourly polling, hidden/offline deadlines, public-site suppression, and unmount cleanup');
} finally {
  if (current) await current.dispose();
  if (server) await server.close();
  dom.window.close();
  for (const name of names) {
    const descriptor = originals.get(name);
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  }
}
