import type { Problem } from './types';
import { parseBrowserBudgets } from './browserBudgets';

type RecordValue = Record<string, unknown>;
type RuntimeBundle = { version: 1; backend: string; adapter: string; harness: string; files: Record<string, string> };
const sharedFiles = ['paperleet/__init__.py', 'paperleet/common/__init__.py', 'paperleet/common/asserts.py', 'paperleet/common/comparisons.py'];
const libraryPath = (name: string) => /^(?:torchwasm|stridetorch|paperleet_torch)\//.test(name);
const libraryNotice = (name: string) => /^(?:torchwasm|stridetorch|paperleet_torch)\/(?:LICENSE|NOTICE)$/.test(name);
const libraryFile = (name: string) => /^(?:torchwasm|stridetorch|paperleet_torch)\/[a-zA-Z0-9_]+\.py$/.test(name) || libraryNotice(name);
const object = (value: unknown): value is RecordValue => !!value && typeof value === 'object' && !Array.isArray(value);
const stringFiles = (value: unknown): value is Record<string, string> => object(value) && Object.entries(value).every(
  ([name, source]) => (/^(?:[a-zA-Z0-9_]+\/)*(?:[a-zA-Z0-9_]+\.py|problem\.yaml)$/.test(name) || libraryNotice(name)) && typeof source === 'string');
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
  // Even immutable URLs can hold cached error bodies. Replace only failed
  // resources on a later run, then restore their normal cache path.
  const failedDownloads = new Set<string>();
  function failed(url: URL) {
    failedDownloads.delete(url.href);
    failedDownloads.add(url.href);
    if (failedDownloads.size > 128) failedDownloads.delete(failedDownloads.values().next().value!);
  }
  async function download(url: URL, signal: AbortSignal): Promise<unknown> {
    try {
      const response = await (fetcher || globalThis.fetch)(url, {
        signal, cache: failedDownloads.has(url.href) ? 'reload' : 'force-cache', credentials: 'omit',
      });
      if (!response.ok) throw Error('Browser bundle download failed');
      return await response.json();
    } catch (error) {
      if (!signal.aborted) failed(url);
      throw error;
    }
  }
  function runtime(url: URL) {
    const key = url.href;
    let pending = runtimes.get(key);
    if (pending) runtimes.delete(key); // Keep the small cache in recent-use order.
    else {
      pending = download(url, AbortSignal.timeout(30_000)).then(value => {
        if (!object(value) || value.version !== 1 || typeof value.backend !== 'string' || !['numpy-wasm', 'numpy-autograd-wasm'].includes(value.backend)
          || typeof value.adapter !== 'string' || typeof value.harness !== 'string' || !stringFiles(value.files)
          || !sharedFiles.every(name => Object.hasOwn(value.files as object, name))
          || Object.keys(value.files).some(name => !sharedFiles.includes(name) && !libraryFile(name)))
          throw Error('Invalid public browser runtime');
        failedDownloads.delete(key);
        return value as RuntimeBundle;
      });
      const current = pending;
      void pending.catch(() => {
        failed(url);
        if (runtimes.get(key) === current) runtimes.delete(key);
      });
    }
    runtimes.set(key, pending);
    if (runtimes.size > 4) runtimes.delete(runtimes.keys().next().value!);
    return pending;
  }
  return async function load(problem: Problem, base: string, signal: AbortSignal): Promise<RecordValue> {
    if (signal.aborted) throw stopped();
    const metadata = problem.browser_judge;
    if (!metadata || !/^browser-judge\/\d+-[a-f0-9]{20}\.json$/.test(metadata.bundle_url)) throw Error('Invalid browser bundle path');
    const bundleUrl = new URL(`data/${metadata.bundle_url}`, base);
    function invalid(message: string): never { failed(bundleUrl); throw Error(message); }
    const bundle = await download(bundleUrl, AbortSignal.any([signal, AbortSignal.timeout(30_000)]));
    if (signal.aborted) throw stopped();
    if (!object(bundle) || bundle.problem !== problem.id) invalid('Browser bundle identity mismatch');
    if (bundle.version === 1) { // Existing content-addressed exports remain usable.
      failedDownloads.delete(bundleUrl.href);
      return bundle;
    }
    if ((bundle.version !== 2 && bundle.version !== 3) || typeof bundle.runtime_bundle_url !== 'string'
      || !/^browser-judge\/runtime-[a-f0-9]{20}\.json$/.test(bundle.runtime_bundle_url)
      || !stringFiles(bundle.files) || Object.keys(bundle.files).some(name => sharedFiles.includes(name) || libraryPath(name))
      || typeof bundle.reference !== 'string' || !object(bundle.mutants) || !Object.values(bundle.mutants).every(value => typeof value === 'string')
      || !object(bundle.tests) || typeof bundle.test_file !== 'string' || !Object.hasOwn(bundle.files, bundle.test_file))
      invalid('Invalid browser problem bundle');
    if (bundle.version === 3) {
      if (!Array.isArray(bundle.test_files) || !bundle.test_files.length
          || new Set(bundle.test_files).size !== bundle.test_files.length
          || bundle.test_files[0] !== bundle.test_file
          || !bundle.test_files.every(name => typeof name === 'string' && /\/tests\/test_[a-zA-Z0-9_]+\.py$/.test(name) && Object.hasOwn(bundle.files as object, name))
          || Object.keys(bundle.files).some(name => /\/tests\/test_[a-zA-Z0-9_]+\.py$/.test(name) && !(bundle.test_files as unknown[]).includes(name))
          || !Array.isArray(bundle.packages) || new Set(bundle.packages).size !== bundle.packages.length
          || !bundle.packages.every(name => name === 'pyyaml') || !parseBrowserBudgets(bundle.budgets)
          || Object.keys(bundle.files).some(name => name.endsWith('/problem.yaml')) && !bundle.packages.includes('pyyaml'))
        invalid('Invalid browser execution configuration');
      const budgets = parseBrowserBudgets(bundle.budgets)!;
      if (metadata.budgets && JSON.stringify(parseBrowserBudgets(metadata.budgets)) !== JSON.stringify(budgets))
        invalid('Browser budget metadata mismatch');
    }
    failedDownloads.delete(bundleUrl.href);
    const runtimeUrl = new URL(`data/${bundle.runtime_bundle_url}`, base);
    const shared = await untilAborted(runtime(runtimeUrl), signal);
    if (signal.aborted) throw stopped();
    if (shared.backend !== metadata.backend) {
      failed(runtimeUrl);
      runtimes.delete(runtimeUrl.href);
      throw Error('Browser runtime backend mismatch');
    }
    return { ...bundle, adapter: shared.adapter, harness: shared.harness, files: { ...shared.files, ...bundle.files } };
  };
}

export const loadBrowserBundle = createBrowserBundleLoader();
