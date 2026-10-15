import type { Problem } from './types';

type RecordValue = Record<string, unknown>;
type RuntimeBundle = { version: 1; backend: string; adapter: string; harness: string; files: Record<string, string> };
const sharedFiles = ['paperleet/__init__.py', 'paperleet/common/__init__.py', 'paperleet/common/asserts.py', 'paperleet/common/comparisons.py'];
const object = (value: unknown): value is RecordValue => !!value && typeof value === 'object' && !Array.isArray(value);
const stringFiles = (value: unknown): value is Record<string, string> => object(value) && Object.entries(value).every(
  ([name, source]) => /^(?:[a-zA-Z0-9_]+\/)*(?:[a-zA-Z0-9_]+\.py|problem\.yaml)$/.test(name) && typeof source === 'string');
const stopped = () => new DOMException('Stopped', 'AbortError');

/** Cancelling one run must not cancel another run's shared, public download. */
function untilAborted<T>(pending: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(stopped());
  return new Promise((resolve, reject) => {
    const abort = () => { signal.removeEventListener('abort', abort); reject(stopped()); };
    signal.addEventListener('abort', abort, { once: true });
    pending.then(value => { if (signal.aborted) reject(stopped()); else resolve(value); }, reject)
      .finally(() => signal.removeEventListener('abort', abort));
  });
}

/** Cache only immutable public code. Every run still gets its own learner worker. */
export function createBrowserBundleLoader(fetcher?: typeof fetch) {
  const runtimes = new Map<string, Promise<RuntimeBundle>>();
  async function download(url: URL, signal: AbortSignal): Promise<unknown> {
    const response = await (fetcher || globalThis.fetch)(url, { signal, cache: 'force-cache', credentials: 'omit' });
    if (!response.ok) throw Error('Browser bundle download failed');
    return response.json();
  }
  function runtime(url: URL) {
    const key = url.href;
    let pending = runtimes.get(key);
    if (pending) runtimes.delete(key); // Keep the small cache in recent-use order.
    else {
      pending = download(url, AbortSignal.timeout(30_000)).then(value => {
        if (!object(value) || value.version !== 1 || typeof value.backend !== 'string' || !['numpy-wasm', 'numpy-autograd-wasm'].includes(value.backend)
          || typeof value.adapter !== 'string' || typeof value.harness !== 'string' || !stringFiles(value.files)
          || Object.keys(value.files).length !== sharedFiles.length || !sharedFiles.every(name => Object.hasOwn(value.files as object, name)))
          throw Error('Invalid public browser runtime');
        return value as RuntimeBundle;
      });
      const current = pending;
      void pending.catch(() => { if (runtimes.get(key) === current) runtimes.delete(key); });
    }
    runtimes.set(key, pending);
    if (runtimes.size > 4) runtimes.delete(runtimes.keys().next().value!);
    return pending;
  }
  return async function load(problem: Problem, base: string, signal: AbortSignal): Promise<RecordValue> {
    if (signal.aborted) throw stopped();
    const metadata = problem.browser_judge;
    if (!metadata || !/^browser-judge\/\d+-[a-f0-9]{20}\.json$/.test(metadata.bundle_url)) throw Error('Invalid browser bundle path');
    const bundle = await download(new URL(`data/${metadata.bundle_url}`, base), AbortSignal.any([signal, AbortSignal.timeout(30_000)]));
    if (signal.aborted) throw stopped();
    if (!object(bundle) || bundle.problem !== problem.id) throw Error('Browser bundle identity mismatch');
    if (bundle.version === 1) return bundle; // Existing content-addressed exports remain usable.
    if (bundle.version !== 2 || typeof bundle.runtime_bundle_url !== 'string'
      || !/^browser-judge\/runtime-[a-f0-9]{20}\.json$/.test(bundle.runtime_bundle_url)
      || !stringFiles(bundle.files) || sharedFiles.some(name => Object.hasOwn(bundle.files as object, name))
      || typeof bundle.reference !== 'string' || !object(bundle.mutants) || !Object.values(bundle.mutants).every(value => typeof value === 'string')
      || !object(bundle.tests) || typeof bundle.test_file !== 'string' || !Object.hasOwn(bundle.files, bundle.test_file))
      throw Error('Invalid browser problem bundle');
    const shared = await untilAborted(runtime(new URL(`data/${bundle.runtime_bundle_url}`, base)), signal);
    if (signal.aborted) throw stopped();
    if (shared.backend !== metadata.backend) throw Error('Browser runtime backend mismatch');
    return { ...bundle, adapter: shared.adapter, harness: shared.harness, files: { ...shared.files, ...bundle.files } };
  };
}

export const loadBrowserBundle = createBrowserBundleLoader();
