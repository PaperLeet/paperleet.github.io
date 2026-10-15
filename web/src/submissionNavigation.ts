import type { Problem, Report } from './types';

/** A saved result is available through an explicit link; completion stays in the workspace. */
export function submissionDestination(report: Report, owner: string, entry: Pick<Problem, 'id' | 'number'>) {
  if (!owner.startsWith('user:') || !report.run_id || !/^[a-f0-9]{32}$/.test(report.run_id) || report.problem !== entry.id) return;
  return {
    href: `#/submission/${report.run_id}`,
  };
}
