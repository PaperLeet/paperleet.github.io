import assert from "node:assert/strict";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom", optimizeDeps: { noDiscovery: true, entries: [] } });
const flush = async () => { for (let i = 0; i < 8; i += 1) await Promise.resolve(); };
try {
  const { createPresenceHeartbeat, presenceVisitorId, presenceStorageKey } =
    await server.ssrLoadModule("/src/presence.ts");
  const id = "91a6d752-1d29-46f5-97a6-d84e98cfb125";
  const secondId = "76051123-4220-4e5a-8a21-2d338e20497a";
  const values = new Map();
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value) };
  assert.match(presenceVisitorId(storage), /^[\da-f]{8}-[\da-f]{4}-4[\da-f]{3}-[89ab][\da-f]{3}-[\da-f]{12}$/);
  assert.equal(presenceVisitorId(storage), values.get(presenceStorageKey));
  values.set(presenceStorageKey, id);
  assert.equal(presenceVisitorId(storage), id, "Adopt the browser identity written by another tab");
  values.set(presenceStorageKey, "invalid");
  assert.equal(presenceVisitorId(storage), id, "Repair corrupt storage without changing this tab's identity");
  const blocked = { getItem: () => { throw Error("blocked"); }, setItem: () => { throw Error("blocked"); } };
  assert.equal(presenceVisitorId(blocked), id, "Blocked storage keeps a usable in-memory visitor ID");

  function harness() {
    let now = 0, nextTimer = 0, visitor = id;
    const timers = new Map(), calls = [], updates = [];
    const heartbeat = createPresenceHeartbeat({
      visitorId: () => visitor,
      now: () => now,
      onChange: (count) => updates.push(count),
      schedule: (fn, delay) => { const timer = ++nextTimer; timers.set(timer, { at: now + delay, fn }); return timer; },
      cancel: (timer) => timers.delete(timer),
      fetch: (url, options) => new Promise((resolve, reject) => calls.push({ url, options, resolve, reject })),
    });
    async function tick(delay) {
      const end = now + delay;
      for (;;) {
        const next = [...timers.entries()].sort((a, b) => a[1].at - b[1].at)[0];
        if (!next || next[1].at > end) break;
        now = next[1].at;
        timers.delete(next[0]);
        next[1].fn();
        await flush();
      }
      now = end;
      await flush();
    }
    const respond = async (index, data = { online: 4, window_seconds: 3600, heartbeat_seconds: 3600 }, ok = true) => {
      calls[index].resolve({ ok, json: async () => data });
      await flush();
    };
    return { heartbeat, calls, updates, timers, tick, respond, setVisitor: (value) => { visitor = value; } };
  }

  const live = harness();
  live.heartbeat.resume();
  live.heartbeat.resume();
  assert.equal(live.calls.length, 1, "Immediate heartbeat, with no overlapping repeated resume");
  assert.equal(live.calls[0].url, "/api/presence/heartbeat");
  assert.equal(live.calls[0].options.credentials, undefined, "The heartbeat must preserve the configured API transport's cookie policy");
  assert.equal(live.calls[0].options.cache, "no-store");
  assert.deepEqual(JSON.parse(live.calls[0].options.body), { visitor_id: id }, "Send only the opaque browser ID; authentication comes from the session cookie");
  await live.respond(0);
  assert.deepEqual(live.updates, [4]);
  await live.tick(3_599_999);
  assert.equal(live.calls.length, 1);
  await live.tick(1);
  assert.equal(live.calls.length, 2, "Visible viewers refresh once per hour");
  await live.respond(1, {}, false);
  assert.equal(live.updates.at(-1), null, "Failures clear the count instead of preserving stale data or inventing zero");
  live.heartbeat.pause();
  await live.tick(100000);
  assert.equal(live.calls.length, 2, "Hidden/offline viewers do not send requests");
  live.setVisitor(secondId);
  live.heartbeat.resume();
  assert.equal(JSON.parse(live.calls[2].options.body).visitor_id, secondId, "Recovery uses the latest shared browser identity");
  await live.respond(2, { online: 0, window_seconds: 3600, heartbeat_seconds: 3600 });
  assert.equal(live.updates.at(-1), 0, "A real zero from the service remains distinguishable from an unavailable service");
  live.heartbeat.dispose();
  await live.tick(100000);
  assert.equal(live.calls.length, 3);
  assert.equal(live.timers.size, 0);

  const visible = harness();
  visible.heartbeat.resume();
  await visible.respond(0);
  await visible.tick(30_000);
  visible.heartbeat.pause();
  assert.deepEqual(visible.updates, [4], "Hiding a tab preserves the current hourly count");
  await visible.tick(30_000);
  visible.heartbeat.resume();
  assert.equal(visible.calls.length, 1, "Returning to a tab before the hour does not force an extra request");
  await visible.tick(3_539_999);
  assert.equal(visible.calls.length, 1);
  await visible.tick(1);
  assert.equal(visible.calls.length, 2, "Visibility changes preserve the original hourly refresh deadline");
  await visible.respond(1, { online: 7, window_seconds: 3600, heartbeat_seconds: 3600 });
  assert.equal(visible.updates.at(-1), 7);
  visible.heartbeat.pause();
  visible.setVisitor(secondId);
  visible.heartbeat.resume(true);
  assert.equal(visible.calls.length, 3, "A changed browser identity is recorded immediately");
  await visible.respond(2);
  visible.heartbeat.dispose();
  assert.equal(visible.timers.size, 0);

  const pending = harness();
  pending.heartbeat.resume();
  pending.heartbeat.pause();
  assert.equal(pending.calls[0].options.signal.aborted, true);
  pending.heartbeat.resume();
  assert.equal(pending.calls.length, 1, "Visibility recovery waits for the aborted request to settle");
  await pending.respond(0, { online: 99, window_seconds: 3600, heartbeat_seconds: 3600 });
  assert(!pending.updates.includes(99), "An aborted request cannot restore a stale count");
  assert.equal(pending.calls.length, 2, "Recovery starts immediately once the old request settles");
  pending.heartbeat.dispose();
  assert.equal(pending.calls[1].options.signal.aborted, true);
  const before = [...pending.updates];
  await pending.respond(1);
  assert.deepEqual(pending.updates, before, "Unmount prevents late state updates");
  assert.equal(pending.timers.size, 0);

  const stalled = harness();
  stalled.heartbeat.resume();
  await stalled.tick(8000);
  assert.equal(stalled.calls[0].options.signal.aborted, true, "Network requests have an eight-second deadline");
  assert.equal(stalled.updates.at(-1), null);
  await stalled.respond(0);
  assert(!stalled.updates.includes(4), "A response after timeout cannot resurrect a stale count");
  await stalled.tick(60_000);
  assert.equal(stalled.calls.length, 2, "A timed-out connection retries after one minute instead of leaving the display unavailable for an hour");
  stalled.heartbeat.dispose();
  stalled.calls[1].reject(Error("aborted"));
  await flush();

  for (const bad of [
    { online: -1, window_seconds: 3600, heartbeat_seconds: 3600 },
    { online: "4", window_seconds: 3600, heartbeat_seconds: 3600 },
    { online: 1.5, window_seconds: 3600, heartbeat_seconds: 3600 },
    { online: 4, window_seconds: 0, heartbeat_seconds: 3600 },
    { online: 4, window_seconds: 3600, heartbeat_seconds: 0 },
    null,
  ]) {
    const malformed = harness();
    malformed.heartbeat.resume();
    await malformed.respond(0, bad);
    assert.equal(malformed.updates.at(-1), null, "Malformed responses must not display a number or alter the poll rate");
    malformed.heartbeat.dispose();
  }
  console.log("PASS hourly presence: rolling-hour protocol, hourly display refresh, visibility-stable deadline, shared visitor identity, cookie credentials, visible lifecycle, single-flight requests, timeouts, recovery, unavailable states, and cleanup");
} finally {
  await server.close();
}

const originalFetch = globalThis.fetch;
for (const deployment of [
  { publicSite: false, hosted: false, available: false, api: '' },
  { publicSite: false, hosted: true, available: true, api: '' },
  { publicSite: false, hosted: true, available: true, api: 'https://app.paperleet.com' },
  { publicSite: true, hosted: false, available: true, api: 'https://other-api.example' },
  { publicSite: true, hosted: false, available: false, api: '' },
]) {
  const modeServer = await createServer({
    server: { middlewareMode: true }, appType: 'custom', optimizeDeps: { noDiscovery: true, entries: [] },
    define: {
      'import.meta.env.VITE_PUBLIC_SITE': JSON.stringify(deployment.publicSite ? '1' : '0'),
      'import.meta.env.VITE_APP_EDITION': JSON.stringify(deployment.hosted ? 'hosted' : 'local'),
      'import.meta.env.VITE_HOSTED_APP_AVAILABLE': JSON.stringify(deployment.available ? '1' : '0'),
      'import.meta.env.VITE_HOSTED_APP_URL': JSON.stringify('https://app.paperleet.com'),
      'import.meta.env.VITE_API_ORIGIN': JSON.stringify(deployment.api),
    },
  });
  try {
    const requests = [];
    globalThis.fetch = async (url, options) => {
      requests.push({ url, options });
      return Response.json({ online: 3, window_seconds: 3600, heartbeat_seconds: 3600 });
    };
    const { presenceRequest, createPresenceRequest, presenceHeartbeatPath } = await modeServer.ssrLoadModule('/src/presenceApi.ts');
    const { createPresenceHeartbeat } = await modeServer.ssrLoadModule('/src/presence.ts');
    if (deployment.publicSite && !deployment.available) {
      assert.equal(presenceRequest, undefined, 'An unpublished account app never receives guest heartbeats');
      assert.equal(requests.length, 0);
      continue;
    }
    assert(presenceRequest);
    const updates = [];
    const heartbeat = createPresenceHeartbeat({
      visitorId: () => '91a6d752-1d29-46f5-97a6-d84e98cfb125',
      onChange: value => updates.push(value), fetch: presenceRequest,
    });
    heartbeat.resume();
    await flush();
    heartbeat.dispose();
    assert.equal(requests.length, 1);
    const origin = deployment.publicSite ? 'https://app.paperleet.com' : deployment.api;
    assert.equal(requests[0].url, origin + presenceHeartbeatPath);
    assert.equal(requests[0].options.credentials, origin ? 'include' : 'same-origin');
    assert.equal(requests[0].options.method, 'POST');
    assert.deepEqual(updates, [3]);
    for (const path of ['/api/account/me', '/api/practice/workspace', '/api/presence/heartbeat?next=other', '//evil.example/api/presence/heartbeat']) {
      assert.throws(() => presenceRequest(path), /presence heartbeat endpoint/, 'The public heartbeat transport cannot become a general cross-origin API proxy');
    }
    for (const hostedAppUrl of ['http://app.paperleet.com', 'https://user:password@app.paperleet.com', 'https://app.paperleet.com/api', 'https://app.paperleet.com/?next=other', 'https://app.paperleet.com/#/']) {
      assert.throws(() => createPresenceRequest({ publicSite: true, hostedAppAvailable: true, hostedAppUrl }));
    }
  } finally {
    globalThis.fetch = originalFetch;
    await modeServer.close();
  }
}
console.log('PASS presence routing: hosted same/split origin cookies, validated public-only heartbeat destination, unavailable app suppression, and no account API expansion');
