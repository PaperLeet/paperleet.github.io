import { useEffect, useRef, useState } from 'react';
import { runBrowserJudge, type BrowserOutcome, type BrowserStage } from './browserJudge';
import type { Problem, Report } from './types';

type View = { owner: string; problem?: string; stage?: BrowserStage; reports: Record<string, Report>; reasons: Record<string, string>; diagnostics: Record<string, string>; calibrations: Record<string, { reference_passed: number; mutants_caught: Record<string, string[]> }> };
const empty = (owner: string): View => ({ owner, reports: {}, reasons: {}, diagnostics: {}, calibrations: {} });
export function useBrowserJudge(owner: string) {
  const [view, setView] = useState<View>(() => empty(owner));
  const controller = useRef<AbortController | undefined>(undefined);
  const currentOwner = useRef(owner);
  currentOwner.current = owner;
  const visible = view.owner === owner ? view : empty(owner);
  useEffect(() => {
    controller.current?.abort();
    setView(empty(owner));
    return () => controller.current?.abort();
  }, [owner]);

  async function run(problem: Problem, source: string): Promise<BrowserOutcome> {
    controller.current?.abort();
    const next = new AbortController();
    controller.current = next;
    setView((previous) => ({ ...(previous.owner === owner ? previous : empty(owner)), problem: problem.id, stage: 'loading', reasons: { ...previous.reasons, [problem.id]: '' }, diagnostics: { ...previous.diagnostics, [problem.id]: '' } }));
    try {
      const outcome = await runBrowserJudge(problem, source, next.signal, (stage) => {
        if (!next.signal.aborted && currentOwner.current === owner) setView((previous) => ({ ...previous, stage }));
      });
      if (next.signal.aborted || currentOwner.current !== owner) throw new DOMException('Stopped', 'AbortError');
      setView((previous) => ({ ...previous, problem: undefined, stage: undefined,
        calibrations: outcome.type === 'result' ? { ...previous.calibrations, [problem.id]: outcome.calibration } : previous.calibrations,
        reports: outcome.type === 'result' ? { ...previous.reports, [problem.id]: outcome.report } : previous.reports,
        diagnostics: { ...previous.diagnostics, [problem.id]: outcome.type === 'unsupported' ? outcome.detail || '' : '' },
        reasons: { ...previous.reasons, [problem.id]: outcome.type === 'unsupported' ? outcome.reason : '' } }));
      return outcome;
    } catch (error) {
      if (currentOwner.current === owner) setView((previous) => ({ ...previous, problem: undefined, stage: undefined }));
      if (next.signal.aborted || currentOwner.current !== owner) throw new DOMException('Stopped', 'AbortError');
      // Failed asset downloads also take the automatic server fallback path.
      const outcome = { type: 'unsupported' as const, reason: 'runtime_failed', detail: String(error).slice(0, 500) };
      setView((previous) => ({ ...previous, reasons: { ...previous.reasons, [problem.id]: outcome.reason }, diagnostics: { ...previous.diagnostics, [problem.id]: outcome.detail } }));
      return outcome;
    }
  }
  function clear(problem: string) {
    setView((previous) => {
      const reports = { ...previous.reports };
      delete reports[problem];
      return { ...previous, reports };
    });
  }
  return { ...visible, active: !!visible.problem, run, clear, stop: () => controller.current?.abort() };
}
