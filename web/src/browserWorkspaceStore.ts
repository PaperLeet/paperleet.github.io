import { parseSaved } from './reports';
import { withPracticeReport } from './practiceRecords';
import type { Problem, Report, Saved } from './types';

export type BrowserOwner = 'local' | 'anonymous';
export type BrowserRecord = {owner: BrowserOwner; id: string; version: string; saved: Saved};
export type BrowserBackup = {owner: BrowserOwner; id: string; writer: string; version: string; base: string; code: string; updated: string};
export type BrowserPreservedSource = {key: string; base: string; code: string; updated: string};
export type BrowserSnapshot = {records: Record<string,BrowserRecord>; backups: BrowserBackup[]};
export const BROWSER_WORKSPACE_DB = 'paperleet.practice.v2';
let database: Promise<IDBDatabase> | undefined;
function open() {
  if (database) return database;
  database = new Promise<IDBDatabase>((resolve,reject) => {
    const request = indexedDB.open(BROWSER_WORKSPACE_DB,1);
    request.onupgradeneeded = () => {
      const db = request.result;
      db.createObjectStore('records',{keyPath:['owner','id']});
      db.createObjectStore('backups',{keyPath:['owner','id','writer']});
    };
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error('Browser workspace upgrade is blocked by another tab'));
    request.onsuccess = () => {
      const db = request.result;
      db.onversionchange = () => {db.close(); database = undefined;};
      resolve(db);
    };
  }).catch(error => {database = undefined; throw error;});
  return database;
}
const requestValue = <T>(request: IDBRequest<T>) => new Promise<T>((resolve,reject) => {
  request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
});
function completed(transaction: IDBTransaction) {
  const promise = new Promise<void>((resolve,reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error || new Error('Browser workspace transaction aborted'));
    transaction.onerror = () => reject(transaction.error);
  });
  void promise.catch(() => {}); // Request failure can reject before the completion await.
  return promise;
}
async function transact<T>(names: string[], action: (transaction: IDBTransaction) => Promise<T>): Promise<T> {
  const db = await open(), transaction = db.transaction(names,'readwrite'), done = completed(transaction);
  try {const result = await action(transaction); await done; return result;}
  catch (error) {try {transaction.abort();} catch { /* It may have aborted already. */ } await done.catch(()=>{}); throw error;}
}
export function legacyBrowserWorkspace(owner: BrowserOwner): Record<string,Saved> {
  try {
    if (owner === 'local') return parseSaved(JSON.parse(localStorage.getItem('paperleet.workspace.v1') || '{}'));
    const key = `paperleet.practice.pending.v1:${encodeURIComponent(location.origin)}:anonymous`;
    const entries = JSON.parse(localStorage.getItem(key) || '{}');
    return parseSaved(Object.fromEntries(Object.entries(entries).flatMap(([id,value]) => {
      const draft = value as {code?: unknown; updated?: unknown} | null;
      return draft && typeof draft.code === 'string' ? [[id,{code:draft.code,updated:typeof draft.updated === 'string' ? draft.updated : undefined}]] : [];
    })));
  } catch {return {};}
}
export async function loadBrowserWorkspace(owner: BrowserOwner, legacy?: Record<string,Saved>): Promise<BrowserSnapshot> {
  return transact(['records','backups'],async transaction => {
  const records = transaction.objectStore('records'), backups = transaction.objectStore('backups');
  // The transaction serializes migration against another tab. Old caches stay as backups.
  const existing = await requestValue<BrowserRecord[]>(records.getAll());
  for (const [id,saved] of Object.entries(legacy || {})) {
    if (!existing.some(row => row.owner === owner && row.id === id)) records.put({owner,id,saved,version:crypto.randomUUID()});
  }
  const [rows,branches] = await Promise.all([requestValue<BrowserRecord[]>(records.getAll()),requestValue<BrowserBackup[]>(backups.getAll())]);
  return {records:Object.fromEntries(rows.filter(row => row.owner === owner).map(row => [row.id,row])),backups:branches.filter(row => row.owner === owner)};
  });
}
export async function writeBrowserDraft(owner: BrowserOwner, id: string, writer: string, expected: string, code: string, updated: string, preserve: BrowserPreservedSource[] = []) {
  return transact(['records','backups'],async transaction => {
  const records = transaction.objectStore('records'), backups = transaction.objectStore('backups');
  const previous = await requestValue<BrowserRecord | undefined>(records.get([owner,id]));
  let record = previous, conflict = false;
  if (previous && previous.version !== expected && previous.saved.code !== code) {
    conflict = true;
  } else {
    record = previous?.saved.code === code ? previous : {owner,id,version:crypto.randomUUID(),saved:{...previous?.saved,code,updated}};
    records.put(record);
  }
  // Keep each document's latest source even after autosave. Another tab may
  // advance the shared record before this one sees its change notification.
  const base = conflict ? expected : record!.version;
  // An explicit version switch must not reuse the only slot holding the source
  // it displaces. Stable intent keys make retries/coalesced edits idempotent.
  for (const source of preserve) {
    if (source.code === code) continue;
    const heldWriter = `preserved:${source.key}`;
    if (!await requestValue(backups.get([owner,id,heldWriter]))) {
      backups.put({owner,id,writer:heldWriter,base:source.base,code:source.code,updated:source.updated,version:crypto.randomUUID()});
    }
  }
  const old = await requestValue<BrowserBackup | undefined>(backups.get([owner,id,writer]));
  if (!old || old.code !== code || old.base !== base) backups.put({owner,id,writer,base,code,updated,version:crypto.randomUUID()});
  return {record:record!,conflict};
  });
}
/** A late report never restores the source that was submitted over a newer draft. */
export async function writeBrowserReport(owner: BrowserOwner, problem: Problem, report: Report) {
  return transact(['records'],async transaction => {
  const records = transaction.objectStore('records');
  const previous = await requestValue<BrowserRecord | undefined>(records.get([owner,problem.id]));
  const record: BrowserRecord = {owner,id:problem.id,version:previous?.version || crypto.randomUUID(),saved:withPracticeReport(previous?.saved || {},problem,report)};
  records.put(record); return record;
  });
}
/** Explicitly discard only the reviewed backup, never a later write to its slot. */
export async function deleteBrowserBackup(backup: BrowserBackup) {
  return transact(['backups'],async transaction => {
  const backups = transaction.objectStore('backups');
  const current = await requestValue<BrowserBackup | undefined>(backups.get([backup.owner,backup.id,backup.writer]));
  if (current?.version === backup.version) backups.delete([backup.owner,backup.id,backup.writer]);
  });
}
