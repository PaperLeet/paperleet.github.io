import { startTransition, useEffect, useRef, useState } from 'react';
import { runBrowserJudge, type BrowserOutcome, type BrowserStage, type BrowserProgress } from './browserJudge';
import type { Problem, Report } from './types';
import type { BrowserSyntaxFailure } from './browserSyntax';

type View = { owner: string; problem?: string; stage?: BrowserStage; progress?: BrowserProgress; reports: Record<string, Report>; syntaxErrors: Record<string,BrowserSyntaxFailure>; reasons: Record<string, string>; diagnostics: Record<string, string>; calibrations: Record<string, { reference_passed: number; mutants_caught: Record<string, string[]> }> };
const empty = (owner: string): View => ({ owner, reports: {}, syntaxErrors:{}, reasons: {}, diagnostics: {}, calibrations: {} });
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
    setView((previous) => {
      const current = previous.owner === owner ? previous : empty(owner);
      const syntaxErrors={...current.syntaxErrors};delete syntaxErrors[problem.id];
      return { ...current, syntaxErrors, problem: problem.id, stage: 'loading', progress: { stage: 'loading', step: 'bundle' },
        reasons: { ...current.reasons, [problem.id]: '' }, diagnostics: { ...current.diagnostics, [problem.id]: '' } };
    });
    try {
      const outcome = await runBrowserJudge(problem, source, next.signal, (stage, progress) => {
        if (!next.signal.aborted && currentOwner.current === owner && controller.current === next)
          startTransition(() => setView((previous) => previous.owner === owner && previous.problem === problem.id && controller.current === next ? { ...previous, stage, progress } : previous));
      });
      if (next.signal.aborted || currentOwner.current !== owner || controller.current !== next) throw new DOMException('Stopped', 'AbortError');
      setView((previous) => {
        if(previous.owner !== owner || controller.current !== next)return previous;
        const calibrations={...previous.calibrations},syntaxErrors={...previous.syntaxErrors};
        if(outcome.type==='syntax_error'){delete calibrations[problem.id];syntaxErrors[problem.id]=outcome.failure;}
        if(outcome.type==='result')calibrations[problem.id]=outcome.calibration;
        return { ...previous, problem: undefined, stage: undefined, progress: undefined, calibrations, syntaxErrors,
        reports: outcome.type === 'result' ? { ...previous.reports, [problem.id]: outcome.report } : previous.reports,
        diagnostics: { ...previous.diagnostics, [problem.id]: outcome.type === 'unsupported' ? outcome.detail || '' : '' },
        reasons: { ...previous.reasons, [problem.id]: outcome.type === 'unsupported' ? outcome.reason : '' } };
      });
      return outcome;
    } catch (error) {
      if (currentOwner.current === owner && controller.current === next)
        setView((previous) => previous.owner === owner && controller.current === next ? { ...previous, problem: undefined, stage: undefined, progress: undefined } : previous);
      if (next.signal.aborted || currentOwner.current !== owner) throw new DOMException('Stopped', 'AbortError');
      // Failed asset downloads also take the automatic server fallback path.
      const outcome = { type: 'unsupported' as const, reason: 'runtime_failed', detail: String(error).slice(0, 500) };
      setView((previous) => ({ ...previous, reasons: { ...previous.reasons, [problem.id]: outcome.reason }, diagnostics: { ...previous.diagnostics, [problem.id]: outcome.detail } }));
      return outcome;
    }
  }
  function clear(problem: string) {
    setView((previous) => {
      if (previous.owner !== owner) return previous;
      const reports = { ...previous.reports };
      delete reports[problem];
      const reasons = { ...previous.reasons }, diagnostics = { ...previous.diagnostics }, calibrations = { ...previous.calibrations }, syntaxErrors={...previous.syntaxErrors};
      delete reasons[problem]; delete diagnostics[problem]; delete calibrations[problem];
      delete syntaxErrors[problem];
      return { ...previous, reports, reasons, diagnostics, calibrations, syntaxErrors };
    });
  }
  return { ...visible, active: !!visible.problem, run, clear, stop: () => controller.current?.abort() };
}
