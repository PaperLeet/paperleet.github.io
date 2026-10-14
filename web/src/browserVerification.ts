import { fingerprintSource, type ReportSourceState } from './reportSource';
import { isPassed, type Problem, type Report } from './types';

export function verificationReason({ passed, sourceState, signedIn, online, ready, conflict, syncError, busy }: {
  passed: boolean; sourceState: ReportSourceState; signedIn: boolean; online: boolean; ready: boolean; conflict: boolean; syncError: boolean; busy: boolean;
}) {
  if (busy) return 'busy';
  if (!passed) return 'failed';
  if (!signedIn) return 'signin';
  if (!ready) return 'workspace';
  if (conflict) return 'conflict';
  if (syncError) return 'sync';
  if (sourceState !== 'current') return sourceState === 'changed' ? 'changed' : 'version';
  if (!online) return 'offline';
  return 'ready';
}

/** Recheck the exact captured source before a click can create a server job. */
export async function matchesBrowserSubmission(problem: Problem, report: Report, source: string) {
  const expected = report.source_sha256;
  if (!expected || report.runtime !== 'browser-numpy-wasm' || !isPassed(problem, report)) return false;
  return (await fingerprintSource(source)).sha256 === expected;
}
