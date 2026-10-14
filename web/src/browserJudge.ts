import { parseReport } from './reports';
import type { Problem, Report } from './types';

export type JudgeMode = 'auto' | 'browser' | 'server';
export type BrowserStage = 'loading' | 'calibrating' | 'running';
export type BrowserOutcome =
  | { type: 'result'; report: Report; calibration: { reference_passed: number; mutants_caught: Record<string, string[]> } }
  | { type: 'unsupported'; reason: string; detail?: string };

export function browserReason(reason: string, en = false) {
  const messages: Record<string, [string, string]> = {
    python_api_unsupported: ['当前实现使用了浏览器兼容层未支持的 Python / PyTorch API', 'This implementation uses Python / PyTorch APIs outside the browser adapter'],
    calibration_failed: ['浏览器运行时未通过参考实现及错误实现校验', 'The browser runtime failed reference and mutation calibration'],
    collection_failed: ['浏览器未能收集完整的原始测试', 'The browser could not collect the complete maintained suite'],
    runtime_failed: ['浏览器 Python 运行时加载或执行失败', 'The browser Python runtime could not load or execute'],
    timeout: ['浏览器运行超时', 'The browser run timed out'],
    unavailable: ['此环境不支持隔离的浏览器判题', 'Isolated browser judging is unavailable in this environment'],
  };
  return (messages[reason] || messages.unavailable)[en ? 1 : 0];
}

/** Client feedback, never authority for hosted progress/rankings. */
export async function runBrowserJudge(problem: Problem, source: string, signal: AbortSignal, progress: (stage: BrowserStage) => void): Promise<BrowserOutcome> {
  if (signal.aborted) throw new DOMException('Stopped', 'AbortError');
  if (!problem.browser_judge || !globalThis.Worker || !globalThis.WebAssembly)
    return { type: 'unsupported', reason: 'unavailable' };
  const base = new URL(import.meta.env.BASE_URL, location.href).href;
  if (!/^browser-judge\/\d+-[a-f0-9]{20}\.json$/.test(problem.browser_judge.bundle_url)) return { type: 'unsupported', reason: 'runtime_failed' };
  const response = await fetch(new URL(`data/${problem.browser_judge.bundle_url}`, base), { signal: AbortSignal.any([signal, AbortSignal.timeout(30_000)]), cache: 'force-cache', credentials: 'omit' });
  if (!response.ok) return { type: 'unsupported', reason: 'runtime_failed' };
  const bundle = await response.json();
  if (bundle.version !== 1 || bundle.problem !== problem.id) return { type: 'unsupported', reason: 'runtime_failed' };
  return new Promise((resolve, reject) => {
    const frame = document.createElement('iframe');
    frame.setAttribute('sandbox', 'allow-scripts');
    frame.setAttribute('aria-hidden', 'true');
    frame.title = 'Isolated browser judge';
    frame.hidden = true;
    const nonce = crypto.randomUUID();
    let settled = false;
    let phase = -1;
    const hardDeadline = setTimeout(() => finish({ type: 'unsupported', reason: 'timeout' }), 120_000);
    let timer: ReturnType<typeof setTimeout>;
    const finish = (outcome?: BrowserOutcome, error?: unknown) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      clearTimeout(hardDeadline);
      window.removeEventListener('message', receive);
      signal.removeEventListener('abort', abort);
      frame.contentWindow?.postMessage({ nonce, stop: true }, '*');
      frame.remove();
      if (error) reject(error); else resolve(outcome!);
    };
    const abort = () => finish(undefined, new DOMException('Stopped', 'AbortError'));
    const arm = (ms: number) => {
      clearTimeout(timer);
      timer = setTimeout(() => finish({ type: 'unsupported', reason: 'timeout' }), ms);
    };
    const receive = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.data?.nonce !== nonce || settled) return;
      const message = event.data.payload;
      if (message?.type === 'ready') {
        frame.contentWindow!.postMessage({ nonce, input: { base, bundle, source } }, '*');
      } else if (message?.type === 'progress') {
        const nextPhase = ['loading', 'calibrating', 'running'].indexOf(message.stage);
        if (nextPhase > phase) {
          phase = nextPhase;
          progress(message.stage);
          arm(message.stage === 'running' ? 30_000 : 90_000);
        }
      } else if (message?.type === 'unsupported') {
        finish({ type: 'unsupported', reason: String(message.reason), detail: message.detail });
      } else if (message?.type === 'result') {
        try {
          const report = parseReport(message.report);
          if (report.problem !== problem.id || report.runtime !== 'browser-numpy-wasm') throw Error('Invalid browser report');
          finish({ type: 'result', report, calibration: message.calibration });
        } catch { finish({ type: 'unsupported', reason: 'collection_failed' }); }
      }
    };
    window.addEventListener('message', receive);
    signal.addEventListener('abort', abort, { once: true });
    // The opaque origin cannot read site storage. Blob workers inherit this
    // CSP: only public runtime assets, never /api or third-party destinations.
    const asset = new URL('browser-runtime/', base).href;
    const scripts = new URL('browser-judge/', base).href;
    const csp = `default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' blob: ${asset} ${scripts}; worker-src blob:; connect-src ${asset};`;
    const workerImport = `self.onmessage = async event => { try { await import('${new URL('browser-judge/worker.mjs', base).href}'); self.onmessage(event); } catch (error) { self.postMessage({type:'unsupported', reason:'runtime_failed', detail:String(error)}); } };`;
    const bootstrap = `
      const nonce = ${JSON.stringify(nonce)};
      const relay = payload => parent.postMessage({nonce, payload}, '*');
      let worker;
      addEventListener('message', event => {
        if (event.source !== parent || event.data?.nonce !== nonce) return;
        if (event.data.stop) { worker?.terminate(); return; }
        if (!event.data.input || worker) return;
        try {
          const url = URL.createObjectURL(new Blob([${JSON.stringify(workerImport)}], {type:'text/javascript'}));
          worker = new Worker(url);
          URL.revokeObjectURL(url);
          worker.onmessage = event => relay(event.data);
          worker.onerror = event => relay({type:'unsupported', reason:'runtime_failed', detail:String(event.message || 'Worker error') + ' ' + (event.filename || '')});
          worker.postMessage(event.data.input);
        } catch { relay({type:'unsupported', reason:'unavailable'}); }
      });
      relay({type:'ready'});`;
    frame.srcdoc = `<meta http-equiv="Content-Security-Policy" content="${csp.replaceAll('"', '&quot;')}"><script>${bootstrap.replaceAll('</script', '<\\/script')}<\/script>`;
    arm(90_000);
    document.body.append(frame);
    if (signal.aborted) abort();
  });
}
