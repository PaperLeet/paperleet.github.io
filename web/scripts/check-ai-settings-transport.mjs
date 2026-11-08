import assert from 'node:assert/strict';
import { createServer as createHttpServer } from 'node:http';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { createServer } from 'vite';

// Exercise the real fetch redirect behavior with synthetic data on loopback only.
const secret = 'synthetic-ai-transport-key';
const originalFetch = globalThis.fetch;
const received = [];
const receiver = createHttpServer(async (request, response) => {
  let body = ''; for await (const chunk of request) body += chunk;
  received.push({ url: request.url, body });
  response.writeHead(200, { 'Content-Type': 'application/json' }); response.end('{"ok":true}');
});
const listen = server => new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const close = server => new Promise(resolve => server.close(resolve));
const trustedRequests = [];
let redirectCode = 307;
let trusted, vite;
const cacheDir = await mkdtemp(`${tmpdir()}/paperleet-ai-transport-`);
try {
  await listen(receiver);
  const target = `http://127.0.0.1:${receiver.address().port}/capture`;
  trusted = createHttpServer(async (request, response) => {
    let body = ''; for await (const chunk of request) body += chunk;
    trustedRequests.push({ url: request.url, body, headers: request.headers });
    response.writeHead(redirectCode, { Location: target }); response.end();
  });
  await listen(trusted);
  const base = `http://127.0.0.1:${trusted.address().port}`;
  const fetchOptions = [];
  globalThis.fetch = (url, options) => {
    const destination = new URL(url, base);
    assert(destination.pathname.startsWith('/api/'), 'Credential requests use the deployment API namespace');
    fetchOptions.push(options);
    // Keep this regression on loopback even when CI sets a production API origin.
    return originalFetch(new URL(destination.pathname + destination.search, base), options);
  };
  vite = await createServer({ configFile: false, cacheDir, esbuild: { jsx: 'automatic' }, server: { middlewareMode: true, watch: null, hmr: false, ws: false }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  const { socialApi } = await vite.ssrLoadModule('/src/social.tsx');
  for (redirectCode of [301, 302, 303, 307, 308]) {
    for (const path of ['account/ai-settings', 'account/ai-settings/models', 'account/ai-settings/test']) {
      await assert.rejects(socialApi(path, { provider: 'gemini', api_key: secret, expected_user_id: 'owner-a' }));
      assert.equal(fetchOptions.at(-1).cache, 'no-store');
      assert.equal(trustedRequests.at(-1).body.includes(secret), true);
      assert.equal(trustedRequests.at(-1).url.includes(secret), false);
    }
    await assert.rejects(socialApi('account/ai-settings/status'));
    assert.equal(trustedRequests.at(-1).body, '');
  }
  assert.equal(received.length, 0, 'No redirect destination receives a credential request or account metadata probe');
  assert(trustedRequests.every(request => !JSON.stringify(request.headers).includes(secret)), 'Keys are only sent in the JSON body to the selected API origin');
  // Unrelated account/community behavior is unchanged by the scoped policy.
  await socialApi('community/posts');
  assert.equal(received.length, 1); assert.equal(received[0].body, '');
  console.log('PASS personal AI transport: real fetch refuses 301/302/303/307/308 before redirect destinations receive keys; request URLs/headers remain secret-free and no-store is retained');
} finally {
  globalThis.fetch = originalFetch;
  if (vite) await vite.close();
  if (trusted) await close(trusted);
  await close(receiver);
  await rm(cacheDir, { recursive: true, force: true });
}
