import { parseSaved } from "./reports";
import type { Saved } from "./types";
export type PendingDraft = { code: string; updated: string; base: string; revision: number };
type Pending = Record<string, PendingDraft>;
export function practiceCacheKey(owner: string, origin: string) {
  return `paperleet.practice.pending.v1:${encodeURIComponent(origin)}:${owner}`;
}
function parse(raw: string | null): Pending {
  try {
    const value: unknown = JSON.parse(raw || '{}');
    if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
    return Object.fromEntries(Object.entries(value).filter((entry): entry is [string, PendingDraft] => {
      const draft = entry[1] as Partial<PendingDraft> | null;
      return !!draft && typeof draft.code === 'string' && typeof draft.base === 'string'
        && typeof draft.updated === 'string' && typeof draft.revision === 'number';
    }));
  } catch { return {}; }
}
/** Each edit has an immutable recovery key; a later revision cannot be erased by a peer. */
export function createHostedDraftCache(source: string) {
  const base = (owner: string) => practiceCacheKey(owner, location.origin);
  let sequence = 0;
  const owned = new Map<string, Map<string, string>>();
  const recoveredKeys = new Map<string, Set<string>>();
  const recoveredLegacy = new Map<string, Pending>();
  const key = (owner: string) => `${base(owner)}:window:${source}:${++sequence}`;
  const legacyAckKey = (owner: string, id: string) => `${base(owner)}:legacy-ack:${encodeURIComponent(id)}`;
  const sessionKey = (owner: string) => `${base(owner)}:session`;
  function journalKeys(owner: string) {
    const prefix = `${base(owner)}:window:`;
    return Array.from({length: localStorage.length}, (_, i) => localStorage.key(i))
      .filter((value): value is string => !!value && value.startsWith(prefix));
  }
  return {
    read(owner: string): Pending {
      try {
        // Session snapshots isolate unsynced edits even when a browser duplicates a tab.
        let session: string | null = null;
        try { session = sessionStorage.getItem(sessionKey(owner)); } catch { /* Use the durable recovery journal. */ }
        const legacy = parse(localStorage.getItem(base(owner)));
        recoveredLegacy.set(owner, legacy);
        const recovered = session !== null ? parse(session) : {...legacy};
        if (session === null) for (const [id, draft] of Object.entries(recovered)) {
          if (localStorage.getItem(legacyAckKey(owner, id)) === JSON.stringify(draft)) delete recovered[id];
        }
        const copied = new Set<string>();
        // Track only immutable snapshots seen before the authoritative API read.
        // A later peer edit is never a candidate for this document's cleanup.
        for (const journal of journalKeys(owner)) {
          const entries = Object.entries(parse(localStorage.getItem(journal)));
          if (session !== null) {
            if (entries.every(([id, draft]) => JSON.stringify(recovered[id]) === JSON.stringify(draft))) copied.add(journal);
            continue;
          }
          copied.add(journal);
          for (const [id, draft] of entries) {
            if (!recovered[id] || draft.updated > recovered[id].updated) recovered[id] = draft;
          }
        }
        recoveredKeys.set(owner, copied);
        return recovered;
      } catch { return {}; }
    },
    write(owner: string, pending: Pending) {
      const data = JSON.stringify(pending);
      let sessionSaved = false, durableSaved = false;
      try { sessionStorage.setItem(sessionKey(owner), data); sessionSaved = true; } catch {
        // A stale session copy must not hide a newer durable recovery journal.
        try { sessionStorage.removeItem(sessionKey(owner)); } catch { /* Storage may be disabled entirely. */ }
      }
      try {
        const journals = owned.get(owner) || new Map<string, string>();
        owned.set(owner, journals);
        for (const [id, draft] of Object.entries(pending)) {
          // Never reuse a key: another process may be acknowledging an earlier
          // immutable snapshot while this document writes its next revision.
          const next = key(owner), previous = journals.get(id);
          localStorage.setItem(next, JSON.stringify({[id]: draft}));
          journals.set(id, next);
          if (previous) localStorage.removeItem(previous);
        }
        for (const [id, previous] of journals) {
          if (!(id in pending)) { localStorage.removeItem(previous); journals.delete(id); }
        }
        durableSaved = true;
      } catch { /* Memory and session still preserve current source. */ }
      if (!sessionSaved || !durableSaved) throw Error('Could not persist draft recovery');
    },
    acknowledge(owner: string, workspace: Record<string, {code?: string}>) {
      // Immutable per-problem snapshots can be removed safely: a sibling's
      // next edit always receives a different key, even in another process.
      try {
        const candidates = new Set([...(owned.get(owner)?.values() || []), ...(recoveredKeys.get(owner) || [])]);
        for (const journal of candidates) {
          const pending = Object.entries(parse(localStorage.getItem(journal)));
          if (pending.length && pending.every(([id, draft]) => workspace[id]?.code === draft.code)) {
            localStorage.removeItem(journal);
          }
        }
        // The old application can still be writing the legacy mutable map.
        // Record exactly what was acknowledged; never rewrite/delete its map.
        for (const [id, draft] of Object.entries(recoveredLegacy.get(owner) || {})) {
          if (workspace[id]?.code === draft.code) localStorage.setItem(legacyAckKey(owner, id), JSON.stringify(draft));
        }
      } catch { /* Keeping an acknowledged copy is safe; a future read clears it. */ }
    },
  };
}

export function mergeWorkspaceSnapshot(
  serverValue: unknown,
  pending: Record<string, PendingDraft>,
) {
  const server = parseSaved(serverValue),
    records = { ...server },
    conflicts: Record<string, Saved> = {},
    dirty = { ...pending };
  for (const [id, draft] of Object.entries(pending)) {
    const saved = server[id] || {};
    if (saved.code === draft.code) {
      delete dirty[id];
      continue;
    }
    records[id] = { ...saved, code: draft.code };
    if ((saved.updated || "") !== draft.base) conflicts[id] = saved;
  }
  return { records, pending: dirty, conflicts };
}
