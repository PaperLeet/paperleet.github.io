import { draftSnapshotMatches, withPracticeReport, type DraftExpectation } from './practiceRecords';
import type { BrowserBackup } from './browserWorkspaceStore';
import type { Problem, Report, Saved } from './types';

export type GuestOwner = 'local' | 'anonymous';
type GuestView = {
  owner?: GuestOwner;
  saved: Record<string, Saved>;
  conflicts: Record<string, Saved>;
  backups: BrowserBackup[];
  ready: boolean;
  loading: boolean;
  saving: boolean;
  pending: boolean;
  failed: boolean;
  error: string;
};

/** A guest workspace belongs to this mounted page, never browser storage. */
export function createGuestWorkspace(owner?: GuestOwner, isCurrent: () => boolean = () => true) {
  let view: GuestView = {
    owner, saved: {}, conflicts: {}, backups: [], ready: !!owner,
    loading: !owner, saving: false, pending: false, failed: false, error: '',
  };
  let lastUpdated = 0;
  const listeners = new Set<() => void>();
  function change(saved: Record<string, Saved>) {
    view = { ...view, saved };
    for (const listener of listeners) listener();
  }
  function update(id: string, patch: Partial<Saved>, expected?: DraftExpectation) {
    if (!owner || !isCurrent() || expected &&
      (expected.owner !== owner || !draftSnapshotMatches(view.saved[id], expected))) return false;
    lastUpdated = Math.max(Date.now(), lastUpdated + 1);
    change({ ...view.saved, [id]: { ...view.saved[id], ...patch, updated: new Date(lastUpdated).toISOString() } });
    return true;
  }
  function applyReport(problem: Problem, report: Report, expectedOwner: string) {
    if (!owner || !isCurrent() || expectedOwner !== owner) return;
    change({ ...view.saved, [problem.id]: withPracticeReport(view.saved[problem.id] || {}, problem, report) });
  }
  function replaceCode(problem: Problem, code: string, expected: DraftExpectation) {
    if ((view.saved[problem.id]?.code ?? problem.starter) === code) return false;
    return update(problem.id, { code }, expected);
  }
  return {
    getSnapshot: () => view,
    subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener); }; },
    update,
    applyReport,
    replaceCode,
    resetToStarter: (problem: Problem, expected: DraftExpectation) => replaceCode(problem, problem.starter, expected),
    resolveConflict: (_id: string, _keepLocal: boolean) => {},
    restoreBackup: (_backup: BrowserBackup) => {},
    discardBackup: async (_backup: BrowserBackup) => {},
    retry: () => {},
  };
}
