import type { DraftStorage } from './communityDrafts';

export type ProposalDraft = { title: string; paperUrl: string; codeUrl: string; excerpt: string; focus: string };
export type ProposalDraftRecord = { version: 1; owner: string; revision: string; draft: ProposalDraft };
export const proposalDraftLimits = { title: 240, paperUrl: 2048, codeUrl: 2048, excerpt: 60000, focus: 4000 } as const;
const fields = Object.keys(proposalDraftLimits) as (keyof ProposalDraft)[];
const validOwner = (owner: string | undefined): owner is string => !!owner && owner.length <= 256;
const validRevision = (revision: unknown): revision is string => typeof revision === 'string' && /^[a-z0-9-]{1,128}$/i.test(revision);
export const emptyProposalDraft = (): ProposalDraft => ({ title: '', paperUrl: '', codeUrl: '', excerpt: '', focus: '' });
export const proposalDraftKey = (owner: string) => `paperleet.submission.draft.v1:${encodeURIComponent(owner)}`;
export const proposalDraftOwner = (mode?: string, userId?: string) => mode === 'local' ? 'local' : mode === 'hosted' && userId ? `account:${userId}` : undefined;
export const hasProposalDraft = (draft: ProposalDraft) => fields.some(field => !!draft[field].trim());
export const sameProposalDraft = (a: ProposalDraft, b: ProposalDraft) => fields.every(field => a[field] === b[field]);
function validDraft(draft: unknown): draft is ProposalDraft {
  return !!draft && typeof draft === 'object' && !Array.isArray(draft) && fields.every(field => {
    const value = (draft as Record<string, unknown>)[field];
    return typeof value === 'string' && value.length <= proposalDraftLimits[field];
  });
}
function decode(raw: string, owner: string): ProposalDraftRecord | undefined {
  if (raw.length > 500000) return;
  const value = JSON.parse(raw);
  if (value?.version !== 1 || value.owner !== owner || !validRevision(value.revision) || !validDraft(value.draft)) return;
  const draft = value.draft;
  return { version: 1, owner, revision: value.revision, draft: { title: draft.title, paperUrl: draft.paperUrl, codeUrl: draft.codeUrl, excerpt: draft.excerpt, focus: draft.focus } };
}
export function readProposalDraft(storage: DraftStorage | undefined, owner: string | undefined): { available: boolean; record?: ProposalDraftRecord } {
  if (!storage || !validOwner(owner)) return { available: false };
  let raw;
  try { raw = storage.getItem(proposalDraftKey(owner)); }
  catch { return { available: false }; }
  if (!raw) return { available: true };
  try { return { available: true, record: decode(raw, owner) }; }
  catch { return { available: true }; }
}
function newRevision() {
  try { const revision = globalThis.crypto?.randomUUID?.(); if (revision) return revision; }
  catch { /* Older browsers can still retain a local draft. */ }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
}
export function writeProposalDraft(storage: DraftStorage | undefined, owner: string | undefined, draft: ProposalDraft): { saved: boolean; revision?: string } {
  if (!storage || !validOwner(owner) || !validDraft(draft)) return { saved: false };
  try {
    if (!hasProposalDraft(draft)) { storage.removeItem(proposalDraftKey(owner)); return { saved: true }; }
    const revision = newRevision();
    const record: ProposalDraftRecord = { version: 1, owner, revision, draft: { title: draft.title, paperUrl: draft.paperUrl, codeUrl: draft.codeUrl, excerpt: draft.excerpt, focus: draft.focus } };
    storage.setItem(proposalDraftKey(owner), JSON.stringify(record));
    return { saved: true, revision };
  } catch { return { saved: false }; }
}
/** Clear only the acknowledged source revision, including whitespace and ABA edits. */
export function clearSubmittedProposalDraft(storage: DraftStorage | undefined, owner: string | undefined, revision: string | undefined, submitted: ProposalDraft): 'cleared' | 'kept' | 'unavailable' {
  if (!storage || !validOwner(owner)) return 'unavailable';
  try {
    const key = proposalDraftKey(owner), raw = storage.getItem(key);
    if (raw === null) return 'cleared';
    const record = decode(raw, owner);
    if (!record || !revision || record.revision !== revision || !sameProposalDraft(record.draft, submitted)) return 'kept';
    storage.removeItem(key);
    return 'cleared';
  } catch { return 'unavailable'; }
}
