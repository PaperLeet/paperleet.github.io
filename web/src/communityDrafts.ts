export type PostDraft = { title: string; body: string; category: string };
export type DraftStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;
const categories = new Set(["papers", "implementation", "questions", "career", "lounge"]);

export function draftKey(owner: string, scope: string): string {
  return `paperleet.community.draft.v1:${encodeURIComponent(owner)}:${encodeURIComponent(scope)}`;
}

export function hasDraft(draft: PostDraft): boolean {
  return !!(draft.title.trim() || draft.body.trim());
}

export function readPostDraft(storage: DraftStorage | undefined, owner: string, scope: string): PostDraft | undefined {
  if (!owner || !scope || !storage) return undefined;
  try {
    const raw = storage.getItem(draftKey(owner, scope));
    if (!raw || raw.length > 100_000) return undefined;
    const value = JSON.parse(raw);
    if (value?.version !== 1 || value.owner !== owner || value.scope !== scope) return undefined;
    const draft = value.draft;
    if (typeof draft?.title !== "string" || draft.title.length > 160 ||
        typeof draft.body !== "string" || draft.body.length > 20_000 ||
        !categories.has(draft.category)) return undefined;
    return { title: draft.title, body: draft.body, category: draft.category };
  } catch { return undefined; }
}

export function writePostDraft(storage: DraftStorage | undefined, owner: string, scope: string, draft: PostDraft): boolean {
  if (!storage || !owner || !scope) return false;
  try {
    if (hasDraft(draft)) {
      storage.setItem(draftKey(owner, scope), JSON.stringify({ version: 1, owner, scope, draft }));
    } else storage.removeItem(draftKey(owner, scope));
    return true;
  } catch { return false; }
}

/** A delayed publish must not erase a newer draft created after navigation. */
export function clearPublishedDraft(storage: DraftStorage | undefined, owner: string, scope: string, submitted: PostDraft): boolean {
  const saved = readPostDraft(storage, owner, scope);
  if (!saved || saved.title !== submitted.title || saved.body !== submitted.body || saved.category !== submitted.category) return false;
  try { storage!.removeItem(draftKey(owner, scope)); return true; }
  catch { return false; }
}

export function tabDraftStorage(): DraftStorage | undefined {
  try { return window.sessionStorage; } catch { return undefined; }
}
