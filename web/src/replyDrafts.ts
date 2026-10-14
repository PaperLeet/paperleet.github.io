import type { DraftStorage } from "./communityDrafts";

export type ReplyDraft = { body: string; parentId?: string };
const replyId = /^[a-f0-9]{32}$/;
export const replyDraftKey = (owner: string, post: string) =>
  `paperleet.community.reply-draft.v1:${encodeURIComponent(owner)}:${encodeURIComponent(post)}`;
export const hasReplyDraft = (draft: ReplyDraft) => !!(draft.body.trim() || draft.parentId);
export const sameReplyDraft = (a: ReplyDraft, b: ReplyDraft) => a.body === b.body && a.parentId === b.parentId;

function validDraft(value: unknown): value is ReplyDraft {
  if (!value || typeof value !== "object") return false;
  const draft = value as ReplyDraft;
  return typeof draft.body === "string" && draft.body.length <= 10_000 &&
    (draft.parentId === undefined || typeof draft.parentId === "string" && replyId.test(draft.parentId));
}

export function readReplyDraft(storage: DraftStorage | undefined, owner: string, post: string): ReplyDraft | undefined {
  if (!storage || !owner || !post) return undefined;
  try {
    const raw = storage.getItem(replyDraftKey(owner, post));
    if (!raw || raw.length > 100_000) return undefined;
    const value = JSON.parse(raw);
    if (value?.version !== 1 || value.owner !== owner || value.post !== post || !validDraft(value.draft)) return undefined;
    return { body: value.draft.body, ...(value.draft.parentId ? { parentId: value.draft.parentId } : {}) };
  } catch { return undefined; }
}

export function writeReplyDraft(storage: DraftStorage | undefined, owner: string, post: string, draft: ReplyDraft): boolean {
  if (!storage || !owner || !post || !validDraft(draft)) return false;
  try {
    if (hasReplyDraft(draft)) storage.setItem(replyDraftKey(owner, post), JSON.stringify({ version: 1, owner, post, draft }));
    else storage.removeItem(replyDraftKey(owner, post));
    return true;
  } catch { return false; }
}

/** A late response may clear only the exact account/post/source that was sent. */
export function clearPublishedReplyDraft(storage: DraftStorage | undefined, owner: string, post: string, submitted: ReplyDraft): boolean {
  const saved = readReplyDraft(storage, owner, post);
  if (!saved || !sameReplyDraft(saved, submitted)) return false;
  try { storage!.removeItem(replyDraftKey(owner, post)); return true; }
  catch { return false; }
}
