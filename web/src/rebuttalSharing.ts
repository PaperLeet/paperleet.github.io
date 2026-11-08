import { useEffect, useMemo, useRef, useSyncExternalStore } from 'react';
import { socialApi } from './social';
import { rebuttalRequestId } from './rebuttalCloud';
import type { RebuttalPerson, RebuttalRequest, RebuttalRole } from './rebuttalCloud';
import { makeRebuttalShareToken } from './rebuttalShareHandoff';

export type ShareRole = 'viewer' | 'editor';
export type RebuttalShareLink = { role: ShareRole; revision: number; active: boolean; id: string | null; created_at: string | null; expires_at: string | null };
export type RebuttalSharePreview = { user_id: string; link: { id: string; paper_id: string; title: string; venue: string; owner: RebuttalPerson; role: ShareRole; expires_at: string }; member_role: RebuttalRole | null };
export type SharingError = { status: number; code: string };
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const id = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z0-9_-]{1,150}$/.test(value);
const shareRole = (value: unknown): value is ShareRole => value === 'viewer' || value === 'editor';
const date = (value: unknown): value is string => typeof value === 'string' && Number.isFinite(Date.parse(value));
const failure = (error: unknown): SharingError => ({ status: record(error) && typeof error.status === 'number' ? error.status : 0, code: record(error) && typeof error.code === 'string' ? error.code : 'request_failed' });
const definite = (error: SharingError) => error.status >= 400 && error.status < 500 && error.status !== 408 && error.status !== 429;
export function sharingErrorText(error: SharingError, t: (zh: string, en: string) => string) {
  if (error.status === 404) return t('邀请链接已失效。', 'This invitation link is unavailable.');
  if (error.code === 'account_changed' || error.status === 401) return t('账号已变化，请重新登录。', 'Your account changed. Sign in again.');
  if (error.code === 'member_limit') return t('协作者人数已达上限。', 'This paper has reached its collaborator limit.');
  if (error.status === 409) return t('邀请链接已更新，请刷新后重试。', 'This invitation link changed. Refresh and try again.');
  if (error.status === 403) return t('只有论文所有者可以管理邀请链接。', 'Only the paper owner can manage invitation links.');
  return t('操作未完成，请重试。', 'Could not complete this action. Try again.');
}
function person(value: unknown): RebuttalPerson {
  if (!record(value) || !id(value.id) || typeof value.username !== 'string' || typeof value.display_name !== 'string') throw Error('Invalid invitation owner');
  return { id: value.id, username: value.username, display_name: value.display_name };
}
export function readRebuttalShareLink(value: unknown): RebuttalShareLink {
  if (!record(value) || !shareRole(value.role) || !Number.isSafeInteger(value.revision) || Number(value.revision) < 0 || typeof value.active !== 'boolean' ||
    value.id !== null && !id(value.id) || value.created_at !== null && !date(value.created_at) || value.expires_at !== null && !date(value.expires_at) ||
    value.active && (!id(value.id) || !date(value.created_at) || !date(value.expires_at))) throw Error('Invalid invitation link');
  return { role: value.role, revision: Number(value.revision), active: value.active, id: value.id as string | null, created_at: value.created_at as string | null, expires_at: value.expires_at as string | null };
}
export function readRebuttalSharePreview(value: unknown, owner: string): RebuttalSharePreview {
  if (!record(value) || value.user_id !== owner || !record(value.link) || !id(value.link.id) || !id(value.link.paper_id) || typeof value.link.title !== 'string' || typeof value.link.venue !== 'string' || !shareRole(value.link.role) || !date(value.link.expires_at) ||
    value.member_role !== null && !['owner', 'editor', 'viewer'].includes(String(value.member_role))) throw Error('Invalid invitation preview');
  return { user_id: owner, link: { id: value.link.id, paper_id: value.link.paper_id, title: value.link.title, venue: value.link.venue, role: value.link.role, expires_at: value.link.expires_at, owner: person(value.link.owner) }, member_role: value.member_role as RebuttalRole | null };
}
function transport(options: { request?: RebuttalRequest; canAct?: () => boolean }) {
  let active = true, epoch = 0;
  const controllers = new Set<AbortController>();
  const owns = (generation = epoch) => active && generation === epoch && (options.canAct?.() ?? true);
  return {
    owns, epoch: () => epoch,
    async call(path: string, body?: unknown) {
      if (!owns()) throw Error('Account changed');
      const controller = new AbortController(); controllers.add(controller);
      const timeout = setTimeout(() => controller.abort(), 20000);
      try { return await (options.request || socialApi)(path, body, controller.signal); }
      finally { clearTimeout(timeout); controllers.delete(controller); }
    },
    activate() { active = true; },
    dispose() { active = false; epoch++; controllers.forEach(controller => controller.abort()); controllers.clear(); },
  };
}
export function createRebuttalShareLinks(options: { owner: string; paperId: string; request?: RebuttalRequest; canAct?: () => boolean }) {
  type Command = { action: 'create' | 'revoke'; role: ShareRole; token?: string; expected_revision: number; expected_user_id: string; request_id: string };
  let state: { loading: boolean; ready: boolean; busy: boolean; pending: boolean; links: RebuttalShareLink[]; tokens: Partial<Record<ShareRole, { id: string; token: string }>>; error?: SharingError } =
    { loading: false, ready: false, busy: false, pending: false, links: [], tokens: {} };
  const path = `rebuttal/papers/${encodeURIComponent(options.paperId)}/links`, io = transport(options), listeners = new Set<() => void>();
  let command: Command | undefined, reading = 0;
  const emit = (patch: Partial<typeof state>) => { if (io.owns()) { state = { ...state, ...patch }; listeners.forEach(listener => listener()); } };
  async function load() {
    if (!io.owns() || state.busy || command) return;
    const generation = io.epoch(), read = ++reading; emit({ loading: true, error: undefined });
    try {
      const response = await io.call(path);
      if (!io.owns(generation) || read !== reading) return;
      if (!record(response) || response.user_id !== options.owner || response.paper_id !== options.paperId || !Array.isArray(response.links)) throw Error('Invalid invitation links response');
      const links = response.links.map(readRebuttalShareLink);
      if (links.length !== 2 || new Set(links.map(link => link.role)).size !== 2) throw Error('Invalid invitation links response');
      const tokens = { ...state.tokens };
      for (const link of links) if (!link.active || tokens[link.role]?.id !== link.id) delete tokens[link.role];
      emit({ links, tokens, ready: true });
    } catch (error) { if (io.owns(generation) && read === reading) emit({ error: failure(error) }); }
    finally { if (io.owns(generation) && read === reading) emit({ loading: false }); }
  }
  async function change(role?: ShareRole, action?: 'create' | 'revoke') {
    if (!io.owns() || !state.ready || state.busy || state.loading || command && role) return;
    if (!command) {
      const link = state.links.find(item => item.role === role); if (!link || !action || !role) return;
      try { command = { action, role, ...(action === 'create' ? { token: makeRebuttalShareToken() } : {}), expected_revision: link.revision, expected_user_id: options.owner, request_id: rebuttalRequestId() }; }
      catch (error) { emit({ error: failure(error) }); return; }
    }
    const sent = command, generation = io.epoch(); reading++; emit({ busy: true, pending: true, error: undefined });
    try {
      const response = await io.call(path, sent);
      if (!io.owns(generation)) return;
      if (!record(response) || response.user_id !== options.owner || response.paper_id !== options.paperId) throw Error('Invalid invitation link response');
      const link = readRebuttalShareLink(response.link);
      if (link.role !== sent.role || link.revision !== sent.expected_revision + 1 || link.active !== (sent.action === 'create')) throw Error('Unconfirmed invitation link');
      const observed = state.links.find(item => item.role === sent.role)!;
      if (link.revision < observed.revision) throw Object.assign(Error('Link changed'), { status: 409, code: 'link_revision_conflict' });
      const tokens = { ...state.tokens };
      if (sent.action === 'create') tokens[sent.role] = { id: link.id!, token: sent.token! }; else delete tokens[sent.role];
      command = undefined; emit({ links: state.links.map(item => item.role === link.role ? link : item), tokens, pending: false });
      return { link, token: sent.token };
    } catch (error) {
      if (io.owns(generation)) { const value = failure(error); if (definite(value)) command = undefined; emit({ error: value, pending: !!command }); }
    } finally { if (io.owns(generation)) emit({ busy: false }); }
  }
  return { getSnapshot: () => state, subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener); }; }, load, change,
    activate() { io.activate(); }, dispose() { io.dispose(); reading++; state = { ...state, busy: false, loading: false }; } };
}
export function useRebuttalShareLinks(owner: string, paperId: string, available: boolean, visible: boolean) {
  const current = useRef({ owner, paperId, available }); current.current = { owner, paperId, available };
  const store = useMemo(() => createRebuttalShareLinks({ owner, paperId, canAct: () => current.current.available && current.current.owner === owner && current.current.paperId === paperId }), [owner, paperId]);
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
  useEffect(() => { store.activate(); if (available && visible) void store.load(); }, [store, available, visible]);
  useEffect(() => () => store.dispose(), [store]);
  return { state, store };
}
export function createRebuttalJoin(options: { owner: string; token: string; request?: RebuttalRequest; canAct?: () => boolean }) {
  let state: { loading: boolean; busy: boolean; pending: boolean; preview?: RebuttalSharePreview; error?: SharingError; invalid: boolean } = { loading: true, busy: false, pending: false, invalid: false };
  let command: { action: 'accept'; token: string; expected_user_id: string; request_id: string } | undefined, reading = 0;
  const io = transport(options), listeners = new Set<() => void>();
  const emit = (patch: Partial<typeof state>) => { if (io.owns()) { state = { ...state, ...patch }; listeners.forEach(listener => listener()); } };
  async function load() {
    if (!io.owns() || state.busy || command) return;
    const generation = io.epoch(), read = ++reading; emit({ loading: true, error: undefined });
    try {
      const preview = readRebuttalSharePreview(await io.call('rebuttal/join', { action: 'preview', token: options.token, expected_user_id: options.owner }), options.owner);
      if (io.owns(generation) && read === reading) emit({ preview, invalid: false });
    } catch (error) { if (io.owns(generation) && read === reading) { const value = failure(error); emit({ error: value, invalid: value.status === 404, preview: undefined }); } }
    finally { if (io.owns(generation) && read === reading) emit({ loading: false }); }
  }
  async function accept() {
    if (!io.owns() || state.busy || state.loading || !state.preview || state.invalid) return;
    command ||= { action: 'accept', token: options.token, expected_user_id: options.owner, request_id: rebuttalRequestId() };
    const sent = command, generation = io.epoch(); reading++; emit({ busy: true, pending: true, error: undefined });
    try {
      const result = await io.call('rebuttal/join', sent);
      if (!io.owns(generation)) return;
      if (!record(result) || result.user_id !== options.owner || result.id !== state.preview.link.id || result.paper_id !== state.preview.link.paper_id || result.status !== 'accepted' || !['owner', 'editor', 'viewer'].includes(String(result.role))) throw Error('Unconfirmed invitation acceptance');
      command = undefined; emit({ pending: false }); return { paperId: String(result.paper_id), role: result.role as RebuttalRole };
    } catch (error) { if (io.owns(generation)) { const value = failure(error); if (definite(value)) command = undefined; emit({ error: value, pending: !!command, invalid: value.status === 404 }); } }
    finally { if (io.owns(generation)) emit({ busy: false }); }
  }
  return { getSnapshot: () => state, subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener); }; }, load, accept,
    activate() { io.activate(); }, dispose() { io.dispose(); reading++; state = { ...state, busy: false }; } };
}
