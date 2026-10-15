import { routePath } from './libraryNavigation';
import type { Problem, Report } from './types';

export function submissionDestination(report: Report, owner: string, entry: Pick<Problem, 'id' | 'number'>, route: string) {
  if (!owner.startsWith('user:') || !report.run_id || !/^[a-f0-9]{32}$/.test(report.run_id) || report.problem !== entry.id) return;
  return {
    href: `#/submission/${report.run_id}`,
    redirect: [entry.id, String(entry.number)].includes(routePath(route)),
  };
}
