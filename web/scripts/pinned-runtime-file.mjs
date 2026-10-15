import fs from 'node:fs/promises';
import {createHash, randomUUID} from 'node:crypto';
import {setTimeout as sleep} from 'node:timers/promises';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
class RuntimeResponseError extends Error {
  constructor(url, status, retryDelay) {
    super(`Runtime asset unavailable: ${url} (${status})`);
    this.retryDelay = retryDelay;
  }
}

function responseDelay(response, attempt) {
  if (![408, 429].includes(response.status) && (response.status < 500 || response.status > 599)) return;
  const header = response.headers.get('Retry-After');
  let minimum = 0;
  if (header) {
    const value = /^\d+$/.test(header) ? Number(header) * 1000 : Date.parse(header) - Date.now();
    if (!Number.isNaN(value)) minimum = Math.max(0, value);
  }
  // A longer refusal ends this build rather than retrying before the server permits it.
  if (minimum > 30_000) return;
  return Math.max(1000 * 2 ** attempt, minimum);
}

/** Build-time downloads only; validated bytes are committed atomically to the cache. */
export async function pinnedFile(url, destination, digest, {fetcher = fetch, wait = sleep, timeoutMs = 60_000} = {}) {
  try { if (hash(await fs.readFile(destination)) === digest) return; } catch { /* Missing or stale cache. */ }
  let bytes;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetcher(url, {signal: AbortSignal.timeout(timeoutMs)});
      if (!response.ok) {
        const delay = responseDelay(response, attempt);
        await response.body?.cancel().catch(() => {});
        throw new RuntimeResponseError(url, response.status, delay);
      }
      bytes = Buffer.from(await response.arrayBuffer());
      break;
    } catch (error) {
      if (error instanceof RuntimeResponseError && error.retryDelay === undefined) throw error;
      if (attempt === 2) throw new Error(`Runtime download failed after 3 attempts: ${url}`, {cause: error});
      await wait(error instanceof RuntimeResponseError ? error.retryDelay : 1000 * 2 ** attempt);
    }
  }
  // Integrity and filesystem errors are terminal; retries cannot waive the pinned digest.
  if (hash(bytes) !== digest) throw new Error(`Runtime integrity mismatch: ${url}`);
  const temporary = `${destination}.${randomUUID()}.tmp`;
  try {
    await fs.writeFile(temporary, bytes, {flag: 'wx'});
    await fs.rename(temporary, destination);
  } finally {
    await fs.rm(temporary, {force: true}).catch(() => {});
  }
}
