import { withPracticeReport } from './practiceRecords';
import { isPassed, type Problem, type Report, type Saved } from './types';

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
  let localPasses = 0;
  for (const problem of problems) {
    const row = records[learningKey(problem.id,problem.browser_judge?.bundle_url || '')], original = saved[problem.id] || {};
    if (!row || row.owner !== owner || !problem.browser_judge || row.suite !== problem.browser_judge.bundle_url) continue;
    current[problem.id] = row;
    const formalAccepted = isPassed(problem,original.acceptedReport) ? original.acceptedReport : isPassed(problem,original.report) ? original.report : undefined;
    const formal = !!formalAccepted;
    const passed = isPassed(problem,row.acceptedReport);
    if (passed && !formal) localPasses++;
    workspace[problem.id] = {...original,report:original.report || row.report,
      acceptedReport:formalAccepted || (passed ? row.acceptedReport : original.acceptedReport),
      updated:original.updated || row.updated};
  }
  return {saved:workspace,records:current,localPasses};
}
