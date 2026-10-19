import { parseReport } from './reports';
import { loadBrowserBundle } from './browserBundles';
import { browserBudget } from './browserBudgets';
import { parseSyntaxDiagnostic, type BrowserSyntaxFailure } from './browserSyntax';
import { advanceBrowserProgress, type BrowserProgress, type BrowserStage } from './browserProgress';
export type { BrowserStage, BrowserProgress } from './browserProgress';
import type { Problem, Report } from './types';
import { browserWorkerAsset, browserWorkerDigest } from './browserWorkerAsset.generated';
import { calibrationProof, createCalibrationCache } from './browserCalibration';
import { browserMeasurements, browserRunDetails } from './browserRunDetails';
import {readCustomResult,type CustomCase,type CustomCaseResult} from './customCaseModel';
import type { BrowserMeasurements, CalibrationSummary, CalibrationProof } from './browserCalibration';
const compatibility = createCalibrationCache();

export type JudgeMode = 'auto' | 'browser' | 'server';
export type BrowserOutcome =
  | { type: 'result'; report: Report; calibration: CalibrationSummary & {reused?:boolean}; measurements?:BrowserMeasurements }
  | { type: 'syntax_error'; failure: BrowserSyntaxFailure }
  | { type: 'unsupported'; reason: string; detail?: string };

export { browserReason } from './browserMessages';

/** Client feedback, never authority for hosted progress/rankings. */
export async function runBrowserJudge(problem: Problem, source: string, signal: AbortSignal, progress: (stage: BrowserStage, detail?: BrowserProgress) => void, requestedAt = performance.now()): Promise<BrowserOutcome> {
  const outcome=await runBrowserTask(problem,source,signal,progress,requestedAt);
  return outcome.type==='custom_result'?{type:'unsupported',reason:'collection_failed'}:outcome;
}
export async function runBrowserCustomCase(problem:Problem,source:string,customCase:CustomCase,signal:AbortSignal,progress:(stage:BrowserStage)=>void) {
  return runBrowserTask(problem,source,signal,progress,performance.now(),customCase);
}
type TaskOutcome=BrowserOutcome|{type:'custom_result';result:CustomCaseResult};
async function runBrowserTask(problem: Problem, source: string, signal: AbortSignal, progress: (stage: BrowserStage, detail?: BrowserProgress) => void, requestedAt:number,customCase?:CustomCase): Promise<TaskOutcome> {
  if (signal.aborted) throw new DOMException('Stopped', 'AbortError');
  if (!problem.browser_judge || !globalThis.Worker || !globalThis.WebAssembly)
    return { type: 'unsupported', reason: 'unavailable' };
  const base = new URL(import.meta.env.BASE_URL, location.href).href;
  let lastProgress: BrowserProgress = { stage: 'loading', step: 'bundle' };
  progress(lastProgress.stage, lastProgress);
  let bundle: Record<string, unknown>;
  try { bundle = await loadBrowserBundle(problem, base, signal); }
  catch (error) {
    if (signal.aborted) throw new DOMException('Stopped', 'AbortError');
    return { type: 'unsupported', reason: 'runtime_failed', detail: `Stage: Public test bundles\n${String(error).slice(-3000)}` };
  }
  let calibrationKey = '';
  let reuseAllowed = true;
  try {
    const digest = await crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify(bundle)));
    calibrationKey = browserWorkerDigest + ':' + [...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('');
  } catch {
    reuseAllowed = false;
    calibrationKey = 'uncached:' + crypto.randomUUID();
  }
  if(signal.aborted) throw new DOMException('Stopped','AbortError');
  const totals = { tests: Object.values(problem.tests || {}).reduce((sum, count) => sum + count, 0),
    mutants: bundle.mutants && typeof bundle.mutants === 'object' ? Object.keys(bundle.mutants).length : 0 };
  const budget = browserBudget(problem, bundle.budgets);
  return new Promise((resolve, reject) => {
    const frame = document.createElement('iframe');
    frame.setAttribute('sandbox', 'allow-scripts');
    frame.setAttribute('aria-hidden', 'true');
    frame.title = 'Isolated browser judge';
    frame.hidden = true;
    const nonce = crypto.randomUUID();
    let settled = false;
    let phase = -1;
    let calibrationSeen = false;
    let reused = false;
    let readySeen = false;
    let granted: CalibrationProof | undefined;
    let checkedCalibration: CalibrationSummary | undefined;
    const hardDeadline = setTimeout(() => finish({ type: 'unsupported', reason: 'timeout' }), budget.total);
    let timer: ReturnType<typeof setTimeout>;
    const finish = (outcome?: TaskOutcome, error?: unknown) => {
      if (settled) return;
      settled = true;
      if(error || outcome?.type === 'unsupported' || outcome?.type === 'syntax_error') compatibility.remove(calibrationKey);
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
      if (message?.type === 'ready' && !readySeen) {
        readySeen = true;
        granted = reuseAllowed ? compatibility.get(calibrationKey,bundle) : undefined;
        frame.contentWindow!.postMessage({ nonce, input: { base, bundle, source, calibrationKey, cachedCalibration:granted,customCase } }, '*');
      } else if(message?.type==='custom_result'&&customCase&&readySeen) {
        const result=readCustomResult(message.result);
        finish(result?{type:'custom_result',result}:{type:'unsupported',reason:'collection_failed'});
      } else if (message?.type === 'progress') {
        const next = advanceBrowserProgress(lastProgress, message, totals);
        if (!next) return;
        lastProgress = next;
        const nextPhase = ['loading', 'calibrating', 'running'].indexOf(next.stage);
        if (nextPhase > phase) {
          phase = nextPhase;
          arm(next.stage === 'running' ? customCase?10_000:budget.run : next.stage === 'calibrating' ? budget.calibration : 90_000);
        }
        progress(next.stage, next);
      } else if (message?.type === 'calibrated' && readySeen && !calibrationSeen && phase === 1) {
        const proof = calibrationProof(message.proof,calibrationKey,bundle);
        if(proof && typeof message.reused === 'boolean'
            && (!message.reused || granted && JSON.stringify(proof) === JSON.stringify(granted))) {
          calibrationSeen = true; reused = message.reused;
          checkedCalibration = proof.summary;
          // Reuse cannot renew the lifetime of an existing compatibility proof.
          if(!reused && reuseAllowed) compatibility.put(calibrationKey,bundle,proof);
        }
      } else if (message?.type === 'syntax_error') {
        const diagnostic = parseSyntaxDiagnostic(message.diagnostic,source);
        finish(diagnostic ? {type:'syntax_error',failure:{source,diagnostic}} : {type:'unsupported',reason:'collection_failed'});
      } else if (message?.type === 'unsupported') {
        finish({ type: 'unsupported', reason: String(message.reason), detail: message.detail });
      } else if (message?.type === 'result'&&!customCase) {
        try {
          const report = parseReport(message.report);
          if (report.problem !== problem.id || report.runtime !== 'browser-numpy-wasm') throw Error('Invalid browser report');
          if(!checkedCalibration || phase !== 2) throw Error('Missing pre-execution compatibility evidence');
          const measured = browserMeasurements(message.measurements);
          const overhead = measured ? Math.max(0,performance.now()-requestedAt-measured.total_ms) : 0;
          const measurements = calibrationSeen && measured?.reused === reused
            ? {...measured,loading_ms:measured.loading_ms+overhead,total_ms:measured.total_ms+overhead} : undefined;
          // Persist only the coordinator-checked summary with this exact report.
          // Historical display data must never populate the page-local proof cache.
          const recorded = report.source_sha256 ? browserRunDetails({version:1,
            calibration:{reference_passed:checkedCalibration.reference_passed,mutants_caught:Object.keys(checkedCalibration.mutants_caught).length,reused},measurements}) : undefined;
          if(recorded) report.browser_run = recorded;
          else delete report.browser_run;
          finish({ type: 'result', report, calibration: {...checkedCalibration,reused}, measurements });
        } catch { finish({ type: 'unsupported', reason: 'collection_failed' }); }
      }
    };
    window.addEventListener('message', receive);
    signal.addEventListener('abort', abort, { once: true });
    // The opaque origin cannot read site storage. Blob workers inherit this
    // CSP: only public runtime assets, never /api or third-party destinations.
    const asset = new URL('browser-runtime/', base).href;
    const worker = new URL(browserWorkerAsset,base).href;
    const csp = `default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' blob: ${asset} ${worker}; worker-src blob:; connect-src ${asset};`;
    const workerImport = `self.onmessage = async event => { try { await import('${worker}'); self.onmessage(event); } catch (error) { self.postMessage({type:'unsupported', reason:'runtime_failed', detail:'Stage: Browser worker import\\n'+String(error)}); } };`;
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
