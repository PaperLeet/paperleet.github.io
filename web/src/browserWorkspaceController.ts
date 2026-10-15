import { browserNotifications } from './browserNotifications';
import { draftSnapshotMatches, withPracticeReport, type DraftExpectation } from './practiceRecords';
import { deleteBrowserBackup, legacyBrowserWorkspace, loadBrowserWorkspace, writeBrowserDraft, writeBrowserReport, type BrowserBackup, type BrowserOwner, type BrowserRecord, type BrowserPreservedSource } from './browserWorkspaceStore';
import type { Problem, Report, Saved } from './types';
import { keepDraftSafety as keepSafety, clearDraftSafety as clearSafety, readDraftSafety as readSafety, type BrowserDraftWrite as Write } from './browserDraftSafety';
export { hasUnsavedDraftSafety } from './browserDraftSafety';

export type BrowserWorkspaceView = {owner?: BrowserOwner; saved: Record<string,Saved>; conflicts: Record<string,Saved>; reviewed: Record<string,string>; backups: BrowserBackup[]; ready: boolean; saving: boolean; failed: boolean};
type Runtime = {owner: BrowserOwner; writer: string; revision: number; bases: Map<string,string>; touched: Set<string>; pending: Map<string,Write>; reports: {problem: Problem; report: Report}[]; remote: Record<string,BrowserRecord>; processing: boolean; failed: boolean; initial: boolean; notifications?: ReturnType<typeof browserNotifications>};
const empty = (owner?: BrowserOwner): BrowserWorkspaceView => ({owner,saved:{},conflicts:{},reviewed:{},backups:[],ready:false,saving:false,failed:false});
const rememberKey = (owner: BrowserOwner) => `paperleet.practice.recovery.v2:${owner}`;
function remembered(owner: BrowserOwner): Record<string,string> {
  try {const value = JSON.parse(sessionStorage.getItem(rememberKey(owner)) || '{}'); return value && typeof value === 'object' ? value : {};}
  catch {return {};}
}
function remember(owner: BrowserOwner, id: string, writer?: string) {
  try {const entries = remembered(owner); if (writer) entries[id] = writer; else delete entries[id]; sessionStorage.setItem(rememberKey(owner),JSON.stringify(entries));}
  catch { /* The durable backup remains available in Other versions without tab-specific restoration. */ }
}

/** Durable browser workspace, loaded only after a guest owner is resolved. */
export function createBrowserWorkspace(owner: BrowserOwner, changed: (view: BrowserWorkspaceView) => void, isCurrent: () => boolean) {
  let view = empty(owner), disposed = false;
  const latest = {current: view};
  const runtime: Runtime = {owner,writer:crypto.randomUUID(),revision:0,bases:new Map(),touched:new Set(),pending:new Map(),reports:[],remote:{},processing:false,failed:false,initial:true};
  function current(candidate: Runtime) {return candidate === runtime && !disposed && isCurrent();}
  function change(candidate: Runtime, patch: Partial<BrowserWorkspaceView>) {
    if (!current(candidate)) return;
    view = {...latest.current,...patch}; latest.current = view; changed(view);
  }
  function signal(runtime: Runtime) {runtime.notifications?.notify();}
  function sourceSnapshot(saved: Saved | undefined, base: string): BrowserPreservedSource | undefined {
    return typeof saved?.code === 'string' ? {key:crypto.randomUUID(),base,code:saved.code,updated:saved.updated || new Date().toISOString()} : undefined;
  }
  function queueSource(runtime: Runtime, id: string, code: string, updated: string, displaced?: BrowserPreservedSource) {
    // Keep preservation intents when newer typing coalesces a failed/in-flight
    // version switch. The safety journal carries both sources across refresh.
    const preserve = [...(runtime.pending.get(id)?.preserve || [])];
    if (displaced && displaced.code !== code && !preserve.some(source => source.code === displaced.code)) preserve.push(displaced);
    const write: Write = {code,updated,token:crypto.randomUUID(),...(preserve.length ? {preserve} : {})};
    runtime.touched.add(id); runtime.pending.set(id,write); keepSafety(runtime.owner,id,write,runtime.bases.get(id) || '');
  }
  async function refresh(runtime: Runtime) {
    const revision = runtime.revision;
    const snapshot = await loadBrowserWorkspace(runtime.owner,runtime.initial ? legacyBrowserWorkspace(runtime.owner) : undefined);
    if (!current(runtime) || revision !== runtime.revision) return;
    const saved = {...latest.current.saved}, conflicts: Record<string,Saved> = {};
    const held = runtime.initial ? remembered(runtime.owner) : {};
    for (const [id,record] of Object.entries(snapshot.records)) {
      const seed = snapshot.backups.find(backup => backup.id === id && backup.writer === held[id]);
      if (seed && !runtime.touched.has(id)) {
        saved[id] = {...record.saved,code:seed.code,updated:seed.updated};
        runtime.touched.add(id); runtime.bases.set(id,seed.base);
      }
      const preserve = runtime.touched.has(id) && saved[id]?.code !== record.saved.code;
      const old = saved[id];
      saved[id] = preserve ? {...record.saved,code:old?.code,updated:old?.updated,
        ...(runtime.reports.some(item => item.problem.id === id) ? {report:old?.report,acceptedReport:old?.acceptedReport} : {})} : record.saved;
      if (preserve && record.version !== (runtime.bases.get(id) || '')) {
        conflicts[id] = record.saved;
        if (typeof old?.code === 'string' && !snapshot.backups.some(backup => backup.id === id && backup.writer === runtime.writer && backup.code === old.code)
          && !runtime.pending.has(id)) {
          const write = {code:old.code,updated:old.updated || new Date().toISOString(),token:crypto.randomUUID()};
          runtime.pending.set(id,write); keepSafety(runtime.owner,id,write,runtime.bases.get(id) || '');
        }
      } else if (!preserve) runtime.bases.set(id,record.version);
    }
    runtime.remote = snapshot.records; runtime.initial = false;
    change(runtime,{saved,conflicts,reviewed:Object.fromEntries(Object.entries(snapshot.records).map(([id,record])=>[id,record.version])),backups:snapshot.backups,ready:true});
    if (runtime.pending.size && !runtime.processing && !runtime.failed) void drain(runtime);
  }
  async function drain(runtime: Runtime) {
    if (runtime.processing || runtime.failed) return;
    runtime.processing = true; change(runtime,{saving:true});
    try {
      while (runtime.pending.size || runtime.reports.length) {
        const first = runtime.pending.entries().next().value as [string,Write] | undefined;
        if (first) {
          const [id,write] = first;
          const result = await writeBrowserDraft(runtime.owner,id,runtime.writer,runtime.bases.get(id) || '',write.code,write.updated,write.preserve);
          runtime.revision++;
          clearSafety(runtime.owner,id,write.token);
          if (runtime.pending.get(id) === write) runtime.pending.delete(id);
          runtime.remote[id] = result.record;
          const conflicts = {...latest.current.conflicts};
          if (result.conflict) conflicts[id] = result.record.saved;
          else {delete conflicts[id]; runtime.bases.set(id,result.record.version);}
          if (current(runtime)) remember(runtime.owner,id,runtime.writer);
          change(runtime,{conflicts,reviewed:{...latest.current.reviewed,[id]:result.record.version}}); signal(runtime);
        } else {
          const item = runtime.reports[0];
          const record = await writeBrowserReport(runtime.owner,item.problem,item.report);
          runtime.revision++;
          if (!runtime.bases.has(item.problem.id)) runtime.bases.set(item.problem.id,record.version);
          runtime.reports.shift(); runtime.remote[item.problem.id] = record; signal(runtime);
        }
      }
      change(runtime,{failed:false});
    } catch {runtime.failed = true; change(runtime,{failed:true});}
    finally {
      runtime.processing = false;
      change(runtime,{saving:false});
    }
    if (current(runtime) && !runtime.failed) {
      try {await refresh(runtime);} catch {runtime.failed = true; change(runtime,{failed:true});}
    }
  }
  function update(id: string, patch: Partial<Saved>, expected?: DraftExpectation, displaced?: BrowserPreservedSource) {
    if (!runtime || runtime.owner !== owner || !current(runtime) || !latest.current.ready || expected && (expected.owner !== owner || latest.current.conflicts[id] || !draftSnapshotMatches(latest.current.saved[id],expected))) return false;
    const updated = new Date().toISOString(), saved = {...latest.current.saved,[id]:{...latest.current.saved[id],...patch,updated}};
    runtime.revision++;
    if (typeof patch.code === 'string') {
      queueSource(runtime,id,patch.code,updated,displaced);
    }
    change(runtime,{saved}); runtime.failed = false; void drain(runtime); return true;
  }
  function applyReport(problem: Problem, report: Report, expectedOwner: string) {
    if (!runtime || !current(runtime) || runtime.owner !== expectedOwner) return;
    runtime.revision++;
    change(runtime,{saved:{...latest.current.saved,[problem.id]:withPracticeReport(latest.current.saved[problem.id] || {},problem,report)}});
    runtime.reports.push({problem,report}); runtime.failed = false; void drain(runtime);
  }
  function replaceCode(problem: Problem, code: string, expected: DraftExpectation) {
    if (!runtime || !current(runtime) || runtime.processing || runtime.pending.size || runtime.reports.length || runtime.failed) return false;
    const previous = latest.current.saved[problem.id];
    if ((previous?.code ?? problem.starter) === code) return false;
    const displaced = sourceSnapshot({...previous,code:previous?.code ?? problem.starter},runtime.bases.get(problem.id) || '');
    return update(problem.id,{code},expected,displaced);
  }
  function resetToStarter(problem: Problem, expected: DraftExpectation) {
    return replaceCode(problem,problem.starter,expected);
  }
  function resolveConflict(id: string, keepLocal: boolean, decision: BrowserWorkspaceView = view) {
    const alternative = decision.conflicts[id];
    if (!runtime || runtime.owner !== owner || !current(runtime) || runtime.processing || !alternative) return;
    runtime.revision++;
    const conflicts = {...latest.current.conflicts}; delete conflicts[id];
    const originalBase = runtime.bases.get(id) || '';
    // Use the version rendered for this decision, never a newer async snapshot.
    runtime.bases.set(id,decision.reviewed[id] || '');
    if (keepLocal) {
      const displaced = sourceSnapshot(alternative,decision.reviewed[id] || '');
      change(runtime,{conflicts}); update(id,{code:latest.current.saved[id].code || ''},undefined,displaced);
    } else {
      const displaced = sourceSnapshot(latest.current.saved[id],originalBase);
      change(runtime,{conflicts}); update(id,{code:alternative.code || ''},undefined,displaced);
    }
  }
  function restoreBackup(backup: BrowserBackup) {
    if (!runtime || runtime.owner !== owner || !current(runtime) || runtime.processing || runtime.failed || backup.owner !== owner) return;
    runtime.revision++;
    const displaced = sourceSnapshot(latest.current.saved[backup.id],runtime.bases.get(backup.id) || '');
    runtime.bases.set(backup.id,backup.base); remember(runtime.owner,backup.id,backup.writer);
    runtime.touched.add(backup.id);
    change(runtime,{saved:{...latest.current.saved,[backup.id]:{...latest.current.saved[backup.id],code:backup.code,updated:backup.updated}}});
    queueSource(runtime,backup.id,backup.code,backup.updated,displaced); void drain(runtime);
  }
  async function discardBackup(backup: BrowserBackup) {
    if (!runtime || runtime.owner !== owner || !current(runtime) || runtime.processing || backup.owner !== owner) return;
    try {await deleteBrowserBackup(backup); await refresh(runtime); signal(runtime);} catch {change(runtime,{failed:true});}
  }
  const reload = () => {void refresh(runtime).catch(() => {runtime.failed = true; change(runtime,{failed:true,ready:true});});};
  runtime.notifications = browserNotifications('paperleet.practice.v2',owner,reload);
  // Restore the synchronous journal before IndexedDB opens: a quota/permission
  // failure must not hide unsaved source after refresh or an account switch.
  const saved = legacyBrowserWorkspace(owner);
  for (const safety of readSafety(owner)) {
    saved[safety.id] = {...saved[safety.id],code:safety.code,updated:safety.updated};
    runtime.touched.add(safety.id); runtime.bases.set(safety.id,safety.base);
    runtime.pending.set(safety.id,safety); keepSafety(owner,safety.id,safety,safety.base);
  }
  change(runtime,{saved}); reload();
  return {
    update,applyReport,replaceCode,resetToStarter,resolveConflict,restoreBackup,discardBackup,
    retry:() => {if(current(runtime)){runtime.failed=false;void drain(runtime);}},
    dispose:() => {disposed=true;runtime.notifications?.dispose();},
  };
}
