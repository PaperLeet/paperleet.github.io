import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import {createHash} from 'node:crypto';
import {pinnedFile} from './pinned-runtime-file.mjs';

const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'paperleet-runtime-download-'));
const bytes = Buffer.from('reviewed runtime bytes');
const digest = createHash('sha256').update(bytes).digest('hex');
const target = name => path.join(directory, name);
const url = 'https://runtime.invalid/pinned.whl';
let server;
try {
  await fs.writeFile(target('cached'), bytes);
  await pinnedFile(url, target('cached'), digest, {fetcher() { throw Error('Valid cache must not access the network'); }});

  let attempts = 0;
  const delays = [];
  await pinnedFile(url, target('network'), digest, {
    fetcher: async () => { if (++attempts < 3) throw new TypeError('fetch failed: connection timeout'); return new Response(bytes); },
    wait: async delay => { delays.push(delay); },
  });
  assert.equal(attempts, 3); assert.deepEqual(delays, [1000, 2000]);
  assert.deepEqual(await fs.readFile(target('network')), bytes);

  await fs.writeFile(target('body'), 'older runtime');
  attempts = 0;
  await pinnedFile(url, target('body'), digest, {
    fetcher: async () => {
      assert.equal(await fs.readFile(target('body'), 'utf8'), 'older runtime', 'Partial downloads cannot replace the existing cache');
      if (++attempts === 1) return new Response(new ReadableStream({start(controller) { controller.enqueue(new Uint8Array([1])); controller.error(new TypeError('body interrupted')); }}));
      return new Response(bytes);
    }, wait: async () => {},
  });
  assert.equal(attempts, 2); assert.deepEqual(await fs.readFile(target('body')), bytes);

  for (const status of [400, 401, 403, 404]) {
    attempts = 0;
    await assert.rejects(pinnedFile(url, target('missing-' + status), digest, {
      fetcher: async () => { attempts++; return new Response('', {status}); },
      wait: async () => { throw Error('Permanent refusals cannot retry'); },
    }), new RegExp(`\\(${status}\\)`));
    assert.equal(attempts, 1);
  }
  attempts = 0;
  await fs.writeFile(target('integrity'), 'previous cache');
  await assert.rejects(pinnedFile(url, target('integrity'), digest, {
    fetcher: async () => { attempts++; return new Response('wrong bytes'); },
    wait: async () => { throw Error('Integrity mismatch cannot retry'); },
  }), /integrity mismatch/);
  assert.equal(attempts, 1); assert.equal(await fs.readFile(target('integrity'), 'utf8'), 'previous cache');

  attempts = 0;
  const rateDelays = [];
  await pinnedFile(url, target('rate'), digest, {
    fetcher: async () => ++attempts === 1 ? new Response('', {status:429, headers:{'Retry-After':'3'}}) : new Response(bytes),
    wait: async delay => { rateDelays.push(delay); },
  });
  assert.deepEqual(rateDelays, [3000]);
  attempts = 0;
  await assert.rejects(pinnedFile(url, target('long-refusal'), digest, {
    fetcher: async () => { attempts++; return new Response('', {status:503, headers:{'Retry-After':'120'}}); },
    wait: async () => { throw Error('Long Retry-After cannot be shortened'); },
  }), /503/);
  assert.equal(attempts, 1);

  const originalNow = Date.now;
  const dateDelays = [];
  attempts = 0;
  try {
    Date.now = () => Date.UTC(2026, 9, 15, 12);
    await pinnedFile(url, target('date'), digest, {
      fetcher: async () => ++attempts === 1 ? new Response('', {status:503, headers:{'Retry-After':'Thu, 15 Oct 2026 12:00:04 GMT'}}) : new Response(bytes),
      wait: async delay => { dateDelays.push(delay); },
    });
  } finally { Date.now = originalNow; }
  assert.deepEqual(dateDelays, [4000]);
  attempts = 0;
  await assert.rejects(pinnedFile(url, target('missing-directory/file'), digest, {
    fetcher: async () => { attempts++; return new Response(bytes); },
    wait: async () => { throw Error('Filesystem errors cannot restart network downloads'); },
  }), /ENOENT/);
  assert.equal(attempts, 1);

  let httpAttempts = 0, timeoutAttempts = 0;
  server = http.createServer((request, response) => {
    if (request.url === '/stall') return;
    if (++httpAttempts === 1) { response.writeHead(503, {'Retry-After':'0'}); response.end('temporarily unavailable'); }
    else response.end(bytes);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  await pinnedFile(origin + '/asset', target('http'), digest, {wait: async () => {}});
  assert.equal(httpAttempts, 2); assert.deepEqual(await fs.readFile(target('http')), bytes);
  await assert.rejects(pinnedFile(origin + '/stall', target('timeout'), digest, {
    timeoutMs: 80, fetcher: (...args) => { timeoutAttempts++; return fetch(...args); }, wait: async () => {},
  }), /failed after 3 attempts/);
  assert.equal(timeoutAttempts, 3);
  await assert.rejects(fs.access(target('timeout')));
  assert(!(await fs.readdir(directory)).some(name => name.endsWith('.tmp')), 'No partial cache files remain');
  console.log('PASS runtime downloads: valid cache reuse, bounded connection/body/HTTP retries, real HTTP recovery and timeout, Retry-After, terminal refusals/integrity failures, atomic cache and cleanup');
} finally {
  if (server) { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
  await fs.rm(directory, {recursive:true, force:true});
}
