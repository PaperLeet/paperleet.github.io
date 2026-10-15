import type { BrowserOwner, BrowserPreservedSource } from './browserWorkspaceStore';

export type BrowserDraftWrite = {code: string; updated: string; token: string; preserve?: BrowserPreservedSource[]};
type Safety = BrowserDraftWrite & {id: string; base: string};
const memory = new Map<string,Safety>();
const prefix = (owner: BrowserOwner) => `paperleet.practice.safety.v2:${owner}:`;

// The shared local cache survives closing the tab before IndexedDB commits.
// A tab also keeps its own journal so another tab cannot replace its recovery source.
export function keepDraftSafety(owner: BrowserOwner, id: string, write: BrowserDraftWrite, base: string) {
  const key = prefix(owner) + id, value = {...write,id,base};
  memory.set(key,value);
  for (const storage of [() => localStorage, () => sessionStorage]) {
    try {storage().setItem(key,JSON.stringify(value));} catch { /* Other stores and the unload guard still protect the draft. */ }
  }
}
export function clearDraftSafety(owner: BrowserOwner, id: string, token: string) {
  const key = prefix(owner) + id;
  if (memory.get(key)?.token === token) memory.delete(key);
  for (const storage of [() => localStorage, () => sessionStorage]) {
    try {
      const store = storage();
      if (JSON.parse(store.getItem(key) || 'null')?.token === token) store.removeItem(key);
    } catch { /* A stale safety copy remains recoverable. */ }
  }
}
export function readDraftSafety(owner: BrowserOwner): Safety[] {
  const values = new Map<string,Safety>(), start = prefix(owner);
  for (const storage of [() => localStorage, () => sessionStorage]) {
    try {
      const store = storage();
      for (let i=0;i<store.length;i++) {
        const key = store.key(i);
        if (!key?.startsWith(start)) continue;
        try {
          const value = JSON.parse(store.getItem(key) || 'null');
          if (!value || typeof value.id !== 'string' || key !== start + value.id || typeof value.code !== 'string'
            || typeof value.base !== 'string' || typeof value.updated !== 'string' || typeof value.token !== 'string') continue;
          const preserve = Array.isArray(value.preserve) ? value.preserve.filter((row: unknown): row is BrowserPreservedSource => {
            if (!row || typeof row !== 'object') return false;
            const source = row as Partial<BrowserPreservedSource>;
            return typeof source.key === 'string' && typeof source.code === 'string' && typeof source.base === 'string' && typeof source.updated === 'string';
          }) : undefined;
          values.set(key,{id:value.id,base:value.base,code:value.code,updated:value.updated,token:value.token,preserve});
        } catch { /* One malformed entry must not hide other modules. */ }
      }
    } catch { /* Try the other journal when browser storage is blocked. */ }
  }
  for (const [key,value] of memory) if (key.startsWith(start)) values.set(key,value);
  return [...values.values()];
}
export function hasUnsavedDraftSafety() {return memory.size > 0;}
