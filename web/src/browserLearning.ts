import { withPracticeReport } from './practiceRecords';
import { browserLearningPassed, isPassed, type Problem, type Report, type Saved } from './types';

export type BrowserLearning = {owner: string; id: string; suite: string; version: string; updated: string; report?: Report; acceptedReport?: Report};
export const learningKey = (id:string,suite:string) => JSON.stringify([id,suite]);
export const learningOwner = (owner: string) => owner === 'anonymous' || owner.startsWith('user:') && owner.length > 5 && owner.length < 256;
export function mergeBrowserLearning(previous: BrowserLearning | undefined, owner: string, problem: Problem, report: Report, version: string): BrowserLearning {
  const suite = problem.browser_judge?.bundle_url;
  if (!learningOwner(owner) || !suite || report.problem !== problem.id || report.runtime !== 'browser-numpy-wasm') throw Error('Invalid browser learning result');
  const before = previous?.owner === owner && previous.id === problem.id && previous.suite === suite ? previous : undefined;
  const saved = withPracticeReport(before || {},problem,report);
  return {owner,id:problem.id,suite,version,updated:new Date().toISOString(),report:saved.report,acceptedReport:saved.acceptedReport};
}
/** Learning-only projection. Never pass this to draft synchronization or account APIs. */
export function browserLearningView(owner: string, problems: Problem[], saved: Record<string,Saved>, records: Record<string,BrowserLearning>) {
  const workspace = {...saved}, current: Record<string,BrowserLearning> = {};
  // A bundle URL identifies an executable runtime, not the learner's history.
  // Keep past full-suite passes when that runtime or the maintained tests change,
  // while showing a report only when it belongs to the currently loaded bundle.
  const completed = new Set(Object.values(records).filter(row =>
    row.owner === owner && row.acceptedReport?.runtime === 'browser-numpy-wasm' &&
    isPassed({id:row.id},row.acceptedReport)).map(row => row.id));
  let localPasses = 0;
  for (const problem of problems) {
    const row = records[learningKey(problem.id,problem.browser_judge?.bundle_url || '')], original = saved[problem.id] || {};
    if (!problem.browser_judge) continue;
    const active = row?.owner === owner && row.id === problem.id && row.suite === problem.browser_judge.bundle_url ? row : undefined;
    const completedLocally = completed.has(problem.id);
    if (!active && !completedLocally) continue;
    if (active) current[problem.id] = active;
    const formalAccepted = isPassed(problem,original.acceptedReport) ? original.acceptedReport : isPassed(problem,original.report) ? original.report : undefined;
    const formal = !!formalAccepted;
    const passed = isPassed(problem,active?.acceptedReport);
    if (completedLocally && !formal) localPasses++;
    workspace[problem.id] = {...original,report:original.report || active?.report,
      acceptedReport:formalAccepted || (passed ? active?.acceptedReport : original.acceptedReport),
      updated:original.updated || active?.updated,
      ...(completedLocally ? {[browserLearningPassed]:true as const} : {})};
  }
  return {saved:workspace,records:current,localPasses};
}
