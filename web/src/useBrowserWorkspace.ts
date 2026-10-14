import { useEffect, useRef, useState } from 'react';
import { useI18n } from './i18n';
import { browserNotifications } from './browserNotifications';
import { draftSnapshotMatches, withPracticeReport, type DraftExpectation } from './practiceRecords';
import { deleteBrowserBackup, legacyBrowserWorkspace, loadBrowserWorkspace, writeBrowserDraft, writeBrowserReport, type BrowserBackup, type BrowserOwner, type BrowserRecord, type BrowserPreservedSource } from './browserWorkspaceStore';
import type { Problem, Report, Saved } from './types';

type View = {owner?: BrowserOwner; saved: Record<string,Saved>; conflicts: Record<string,Saved>; reviewed: Record<string,string>; backups: BrowserBackup[]; ready: boolean; saving: boolean; failed: boolean};
type Write = {code: string; updated: string; token: string; preserve?: BrowserPreservedSource[]};
type Safety = Write & {id: string; base: string};
const safetyMemory = new Map<string,Safety>();
const safetyPrefix = (owner: BrowserOwner) => `paperleet.practice.safety.v2:${owner}:`;
function keepSafety(owner: BrowserOwner, id: string, write: Write, base: string) {
  const key = safetyPrefix(owner) + id, value = {...write,id,base};
  safetyMemory.set(key,value);
  try {sessionStorage.setItem(key,JSON.stringify(value));} catch { /* The unload guard protects the in-memory copy. */ }
}
function clearSafety(owner: BrowserOwner, id: string, token: string) {
  const key = safetyPrefix(owner) + id;
  if (safetyMemory.get(key)?.token === token) safetyMemory.delete(key);
  try {if (JSON.parse(sessionStorage.getItem(key) || 'null')?.token === token) sessionStorage.removeItem(key);} catch { /* A stale safety copy is recoverable rather than destructive. */ }
}
function readSafety(owner: BrowserOwner): Safety[] {
  const values = new Map<string,Safety>(), prefix = safetyPrefix(owner);
  try {
    for (let i=0;i<sessionStorage.length;i++) {
      const key = sessionStorage.key(i);
      if (!key?.startsWith(prefix)) continue;
      const value = JSON.parse(sessionStorage.getItem(key) || 'null');
      if (value && typeof value.id === 'string' && key === prefix + value.id && typeof value.code === 'string' && typeof value.base === 'string' && typeof value.updated === 'string' && typeof value.token === 'string') {
        const preserve = Array.isArray(value.preserve) ? value.preserve.filter((row: unknown): row is BrowserPreservedSource => {
          if (!row || typeof row !== 'object') return false;
          const source = row as Partial<BrowserPreservedSource>;
          return typeof source.key === 'string' && typeof source.code === 'string' && typeof source.base === 'string' && typeof source.updated === 'string';
        }) : undefined;
        values.set(key,{...value,preserve});
      }
    }
  } catch { /* Volatile drafts remain available if session storage is blocked. */ }
  for (const [key,value] of safetyMemory) if (key.startsWith(prefix)) values.set(key,value);
  return [...values.values()];
}
type Runtime = {owner: BrowserOwner; writer: string; revision: number; bases: Map<string,string>; touched: Set<string>; pending: Map<string,Write>; reports: {problem: Problem; report: Report}[]; remote: Record<string,BrowserRecord>; processing: boolean; failed: boolean; initial: boolean; notifications?: ReturnType<typeof browserNotifications>};
const empty = (owner?: BrowserOwner): View => ({owner,saved:{},conflicts:{},reviewed:{},backups:[],ready:false,saving:false,failed:false});
const rememberKey = (owner: BrowserOwner) => `paperleet.practice.recovery.v2:${owner}`;
function remembered(owner: BrowserOwner): Record<string,string> {
  try {const value = JSON.parse(sessionStorage.getItem(rememberKey(owner)) || '{}'); return value && typeof value === 'object' ? value : {};}
  catch {return {};}
}
function remember(owner: BrowserOwner, id: string, writer?: string) {
  try {const entries = remembered(owner); if (writer) entries[id] = writer; else delete entries[id]; sessionStorage.setItem(rememberKey(owner),JSON.stringify(entries));}
  catch { /* The durable backup remains available in Other versions without tab-specific restoration. */ }
}

export function useBrowserWorkspace(owner?: BrowserOwner) {
  const {t} = useI18n();
  const [view,setView] = useState<View>(() => empty(owner));
  const latest = useRef(view), active = useRef<Runtime | undefined>(undefined), currentOwner = useRef(owner);
  currentOwner.current = owner;
  function current(runtime: Runtime) {return active.current === runtime && currentOwner.current === runtime.owner;}
  function change(runtime: Runtime, patch: Partial<View>) {
    if (!current(runtime)) return;
    const next = {...latest.current,...patch}; latest.current = next; setView(next);
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
    if (runtime.initial) for (const safety of readSafety(runtime.owner)) {
      if (runtime.touched.has(safety.id)) continue;
      saved[safety.id] = {...snapshot.records[safety.id]?.saved,code:safety.code,updated:safety.updated};
      runtime.touched.add(safety.id); runtime.bases.set(safety.id,safety.base); runtime.pending.set(safety.id,safety);
      keepSafety(runtime.owner,safety.id,safety,safety.base);
    }
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
          remember(runtime.owner,id,runtime.writer);
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
  useEffect(() => {
    if (!owner) {active.current = undefined; latest.current = empty(); setView(latest.current); return;}
    const runtime: Runtime = {owner,writer:crypto.randomUUID(),revision:0,bases:new Map(),touched:new Set(),pending:new Map(),reports:[],remote:{},processing:false,failed:false,initial:true};
    active.current = runtime; latest.current = empty(owner); setView(latest.current);
    const reload = () => {void refresh(runtime).catch(() => {runtime.failed = true; change(runtime,{failed:true,ready:true});});};
    runtime.notifications = browserNotifications('paperleet.practice.v2',owner,reload);
    // Existing work remains readable even if this browser blocks the new store.
    change(runtime,{saved:legacyBrowserWorkspace(owner)}); reload();
    return () => {runtime.notifications?.dispose(); if (active.current === runtime) active.current = undefined;};
  }, [owner]);
  useEffect(() => {
    const leave = (event: BeforeUnloadEvent) => {if(safetyMemory.size){event.preventDefault();event.returnValue='';}};
    window.addEventListener('beforeunload',leave); return ()=>window.removeEventListener('beforeunload',leave);
  },[]);
  function update(id: string, patch: Partial<Saved>, expected?: DraftExpectation, displaced?: BrowserPreservedSource) {
    const runtime = active.current;
    if (!runtime || runtime.owner !== owner || !current(runtime) || !latest.current.ready || expected && (expected.owner !== owner || latest.current.conflicts[id] || !draftSnapshotMatches(latest.current.saved[id],expected))) return false;
    const updated = new Date().toISOString(), saved = {...latest.current.saved,[id]:{...latest.current.saved[id],...patch,updated}};
    runtime.revision++;
    if (typeof patch.code === 'string') {
      queueSource(runtime,id,patch.code,updated,displaced);
    }
    change(runtime,{saved}); runtime.failed = false; void drain(runtime); return true;
  }
  function applyReport(problem: Problem, report: Report, expectedOwner: string) {
    const runtime = active.current;
    if (!runtime || !current(runtime) || runtime.owner !== expectedOwner) return;
    runtime.revision++;
    change(runtime,{saved:{...latest.current.saved,[problem.id]:withPracticeReport(latest.current.saved[problem.id] || {},problem,report)}});
    runtime.reports.push({problem,report}); runtime.failed = false; void drain(runtime);
  }
  function replaceCode(problem: Problem, code: string, expected: DraftExpectation) {
    const runtime = active.current;
    if (!runtime || !current(runtime) || runtime.processing || runtime.pending.size || runtime.reports.length || runtime.failed) return false;
    const previous = latest.current.saved[problem.id];
    if ((previous?.code ?? problem.starter) === code) return false;
    const displaced = sourceSnapshot({...previous,code:previous?.code ?? problem.starter},runtime.bases.get(problem.id) || '');
    return update(problem.id,{code},expected,displaced);
  }
  function resetToStarter(problem: Problem, expected: DraftExpectation) {
    return replaceCode(problem,problem.starter,expected);
  }
  function resolveConflict(id: string, keepLocal: boolean) {
    const runtime = active.current, alternative = view.conflicts[id];
    if (!runtime || runtime.owner !== owner || !current(runtime) || runtime.processing || !alternative) return;
    runtime.revision++;
    const conflicts = {...latest.current.conflicts}; delete conflicts[id];
    const originalBase = runtime.bases.get(id) || '';
    // Use the version rendered for this decision, never a newer async snapshot.
    runtime.bases.set(id,view.reviewed[id] || '');
    if (keepLocal) {
      const displaced = sourceSnapshot(alternative,view.reviewed[id] || '');
      change(runtime,{conflicts}); update(id,{code:latest.current.saved[id].code || ''},undefined,displaced);
    } else {
      const displaced = sourceSnapshot(latest.current.saved[id],originalBase);
      change(runtime,{conflicts}); update(id,{code:alternative.code || ''},undefined,displaced);
    }
  }
  function restoreBackup(backup: BrowserBackup) {
    const runtime = active.current;
    if (!runtime || runtime.owner !== owner || !current(runtime) || runtime.processing || runtime.failed || backup.owner !== owner) return;
    runtime.revision++;
    const displaced = sourceSnapshot(latest.current.saved[backup.id],runtime.bases.get(backup.id) || '');
    runtime.bases.set(backup.id,backup.base); remember(runtime.owner,backup.id,backup.writer);
    runtime.touched.add(backup.id);
    change(runtime,{saved:{...latest.current.saved,[backup.id]:{...latest.current.saved[backup.id],code:backup.code,updated:backup.updated}}});
    queueSource(runtime,backup.id,backup.code,backup.updated,displaced); void drain(runtime);
  }
  async function discardBackup(backup: BrowserBackup) {
    const runtime = active.current;
    if (!runtime || runtime.owner !== owner || !current(runtime) || runtime.processing || backup.owner !== owner) return;
    try {await deleteBrowserBackup(backup); await refresh(runtime); signal(runtime);} catch {change(runtime,{failed:true});}
  }
  const visible = view.owner === owner ? view : empty(owner);
  return {...visible,loading:!visible.ready,pending:visible.saving,backups:visible.backups,
    error:visible.failed ? t('无法保存本机工作区。请下载代码备份，再重试保存。','Could not save the browser workspace. Download a code backup, then retry saving.') : '',
    update,applyReport,replaceCode,resetToStarter,resolveConflict,restoreBackup,discardBackup,
    retry:() => {const runtime=active.current;if(runtime && runtime.owner === owner && current(runtime)){runtime.failed=false;void drain(runtime);}},
  };
}
