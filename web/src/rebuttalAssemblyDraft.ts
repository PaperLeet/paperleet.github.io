import { useEffect, useMemo, useRef, useSyncExternalStore } from 'react';
import { makeId } from './rebuttalModel';

export type AssemblyDocument = { user_id: string; paper_id: string; scope: string; source: string | null; revision: number; updated_at: string | null };
export type AssemblyRequest = (path: string, body?: unknown, signal?: AbortSignal) => Promise<unknown>;
type Storage = Pick<globalThis.Storage, 'getItem' | 'setItem' | 'removeItem'>;
type Command = { source: string | null; expected_revision: number; request_id: string; expected_user_id: string };
type Status = 'loading' | 'saved' | 'saving' | 'error' | 'conflict';
export type AssemblyState = { source: string | null; revision: number; status: Status; error: string; dirty: boolean; ready: boolean; blocked: boolean; recoveryError: boolean };
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const identifier = (value: unknown): value is string => typeof value === 'string' && /^[a-zA-Z0-9_-]{1,160}$/.test(value);
const sourceValue = (value: unknown): value is string | null => value === null || typeof value === 'string';
const limit = 200 * 1024;
const problem = (error: unknown) => error instanceof Error ? error.message : 'Could not save personal draft';
const statusOf = (error: unknown) => record(error) && typeof error.status === 'number' ? error.status : 0;
const codeOf = (error: unknown) => record(error) && typeof error.code === 'string' ? error.code : '';
export const assemblyScope = (reviewer: string) => reviewer === 'all' ? 'all' : `reviewer-${reviewer}`;
export function assemblyErrorText(state: AssemblyState, locale: 'zh' | 'en') {
  const t = (zh: string, en: string) => locale === 'zh' ? zh : en;
  if (/account/i.test(state.error)) return t('账号已变化，请刷新后重试。', 'Your account changed. Refresh and try again.');
  if (/limit|exceed|too.large|200 KiB/i.test(state.error)) return t('个人草稿超出保存限额，请精简或下载。', 'Your personal draft exceeds the save limit. Shorten it or download a copy.');
  if (state.status === 'conflict') return t('其他窗口已更新个人草稿，当前文本已保留。', 'Your personal draft changed in another window. Your current text is preserved.');
  if (/recovered|unconfirmed/i.test(state.error)) return t('未确认的个人草稿已保留，请重试保存。', 'Your unconfirmed personal draft is preserved. Retry saving.');
  if (state.recoveryError || /storage|recovery/i.test(state.error)) return t('无法保存本机备份，请下载个人草稿。', 'Browser draft recovery is unavailable. Download your personal draft.');
  if (state.blocked) return t('暂时无法访问个人草稿，请刷新后重试。', 'Your personal draft is unavailable. Refresh and try again.');
  return state.ready ? t('个人草稿保存失败，当前文本已保留。', 'Your personal draft could not be saved. Your current text is preserved.') : t('无法读取个人草稿，请重试。', 'Could not load your personal draft. Try again.');
}
export function readAssemblyDocument(value: unknown, owner: string, paperId: string, scope: string): AssemblyDocument {
  if (!record(value) || value.user_id !== owner || value.paper_id !== paperId || value.scope !== scope || !sourceValue(value.source) ||
    !Number.isSafeInteger(value.revision) || Number(value.revision) < 0 || value.updated_at !== null && (typeof value.updated_at !== 'string' || !Number.isFinite(Date.parse(value.updated_at)))) throw Error('Invalid personal draft response or account changed');
  return { user_id: owner, paper_id: paperId, scope, source: value.source, revision: Number(value.revision), updated_at: value.updated_at as string | null };
}
export function createAssemblyDraft(options: { owner: string; paperId: string; scope: string; cloud: boolean; request?: AssemblyRequest; storage?: Storage; canAct?: () => boolean; debounceMs?: number }) {
  const { owner, paperId, scope, cloud, storage } = options;
  const key = `paperleet.rebuttal.assembly.${cloud ? 'recovery' : 'local'}.${owner}.${paperId}.${scope}.v1`;
  const path = `rebuttal/papers/${encodeURIComponent(paperId)}/drafts/${encodeURIComponent(scope)}`;
  let state: AssemblyState = { source: null, revision: 0, status: 'loading', error: '', dirty: false, ready: false, blocked: false, recoveryError: false };
  let active = true, epoch = 0, reading = false, saved: string | null = null, command: Command | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined, saving: Promise<boolean> | undefined;
  const listeners = new Set<() => void>(), controllers = new Set<AbortController>();
  const owns = (generation = epoch) => active && generation === epoch && (options.canAct?.() ?? true);
  const emit = (patch: Partial<AssemblyState>) => { if (owns()) { state = { ...state, ...patch }; listeners.forEach(listener => listener()); } };
  const stopTimer = () => { if (timer) clearTimeout(timer); timer = undefined; };
  function persist() {
    if (!storage) { emit({ recoveryError: true }); return false; }
    try {
      if (cloud && !state.dirty && !command) storage.removeItem(key);
      else storage.setItem(key, JSON.stringify({ version: 1, source: state.source, revision: state.revision, command }));
      emit({ recoveryError: false }); return true;
    } catch { emit({ recoveryError: true }); return false; }
  }
  async function call(body?: Command) {
    if (!options.request || !owns()) throw Error('Personal draft service unavailable');
    const controller = new AbortController(); controllers.add(controller);
    const timeout = setTimeout(() => controller.abort(), 20000);
    try { return await options.request(path, body, controller.signal); }
    finally { clearTimeout(timeout); controllers.delete(controller); }
  }
  function recover() {
    if (!storage) { emit({ recoveryError: true }); return; }
    const raw = storage.getItem(key); if (!raw) return;
    const cached: unknown = JSON.parse(raw);
    if (!record(cached) || cached.version !== 1 || !sourceValue(cached.source) || !Number.isSafeInteger(cached.revision) || Number(cached.revision) < 0) throw Error('Personal draft storage could not be read');
    if (!cloud) { saved = cached.source; emit({ source: saved, revision: Number(cached.revision), ready: true, status: 'saved' }); return; }
    if (cached.command !== undefined) {
      const pending = cached.command;
      if (!record(pending) || !sourceValue(pending.source) || pending.expected_user_id !== owner || !identifier(pending.request_id) || !Number.isSafeInteger(pending.expected_revision) || Number(pending.expected_revision) < 0) throw Error('Personal draft recovery could not be read');
      command = { source: pending.source, expected_revision: Number(pending.expected_revision), request_id: pending.request_id, expected_user_id: owner };
    }
    if (cached.source === saved && !command) { storage.removeItem(key); return; }
    emit({ source: cached.source, dirty: cached.source !== saved, status: !command && cached.revision !== state.revision ? 'conflict' : 'error',
      error: 'Your unsaved personal draft was recovered. Retry saving or download it before loading the saved version.' });
  }
  async function load(discard = false) {
    if (!owns() || reading || saving || state.ready && !discard) return false;
    const generation = epoch; reading = true;
    emit({ status: 'loading', error: '' });
    try {
      const document = cloud ? readAssemblyDocument(await call(), owner, paperId, scope) : { source: null, revision: 0 };
      if (!owns(generation)) return false;
      if (discard) command = undefined;
      saved = document.source;
      emit({ source: saved, revision: document.revision, ready: true, dirty: false, status: 'saved', error: '', blocked: false });
      if (discard) persist(); else {
        try { recover(); } catch (error) { emit({ status: 'error', error: problem(error), blocked: true, recoveryError: true }); return false; }
      }
      return true;
    } catch (error) {
      if (owns(generation)) emit({ status: 'error', error: problem(error), blocked: !cloud || [401, 403, 404].includes(statusOf(error)), recoveryError: !cloud });
      return false;
    } finally { if (owns(generation)) reading = false; }
  }
  function update(source: string | null) {
    if (!owns() || reading || !state.ready || state.blocked || !sourceValue(source)) return false;
    if (source === state.source) return true;
    emit({ source, dirty: source !== saved, status: state.status === 'conflict' ? 'conflict' : 'saving', error: '' });
    stopTimer();
    if (!cloud) {
      const ok = persist();
      if (ok) saved = source;
      emit({ status: ok ? 'saved' : 'error', dirty: !ok, error: ok ? '' : 'Browser storage is unavailable. Download your personal draft.' });
      return ok;
    }
    persist();
    if (state.status !== 'conflict') timer = setTimeout(() => void flush(), options.debounceMs ?? 600);
    return true;
  }
  async function drain(generation: number) {
    while (owns(generation) && (state.dirty || command)) {
      if (!state.ready || state.blocked || state.status === 'conflict') return false;
      if (!command) {
        if (state.source !== null && new TextEncoder().encode(state.source).length > limit) {
          emit({ status: 'error', error: 'Personal draft exceeds 200 KiB. Shorten it or download a copy.' }); return false;
        }
        command = { source: state.source, expected_revision: state.revision, request_id: makeId().replaceAll('-', ''), expected_user_id: owner };
      }
      const sent = command; persist(); emit({ status: 'saving', error: '' });
      try {
        const document = readAssemblyDocument(await call(sent), owner, paperId, scope);
        if (!owns(generation)) return false;
        if (document.revision !== sent.expected_revision + 1 || document.source !== sent.source) throw Error('Personal draft save is unconfirmed. Retry the original save.');
        if (document.revision < state.revision) {
          command = undefined;
          emit({ status: 'conflict', dirty: state.source !== saved, error: 'Personal draft changed in another window. Your current text is preserved.' }); persist(); return false;
        }
        command = undefined; saved = document.source;
        emit({ revision: document.revision, dirty: state.source !== saved, status: state.source === saved ? 'saved' : 'saving', error: '' }); persist();
      } catch (error) {
        if (owns(generation)) {
          const status = statusOf(error), code = codeOf(error);
          if ([400, 413, 422].includes(status)) command = undefined;
          emit({ status: status === 409 ? 'conflict' : 'error', error: problem(error), blocked: [401, 403, 404].includes(status) || code === 'account_changed' }); persist();
        }
        return false;
      }
    }
    return owns(generation);
  }
  async function flush(): Promise<boolean> {
    stopTimer();
    if (!owns() || reading || !state.ready || state.blocked || state.status === 'conflict') return false;
    if (!cloud) {
      if (!state.dirty) return true;
      const ok = persist(); if (ok) saved = state.source;
      emit({ dirty: !ok, status: ok ? 'saved' : 'error' }); return ok;
    }
    if (saving) return saving;
    const result = drain(epoch); saving = result;
    try { return await result; } finally { if (saving === result) saving = undefined; }
  }
  return {
    getSnapshot: () => state,
    subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener); }; },
    load, update, flush, hasPending: () => state.dirty || !!command,
    activate() { active = true; },
    dispose() { active = false; epoch++; stopTimer(); controllers.forEach(controller => controller.abort()); controllers.clear(); reading = false; saving = undefined; },
  };
}

export function useAssemblyDraft(options: { owner: string; paperId: string; scope: string; cloud: boolean; available: boolean; enabled: boolean; request?: AssemblyRequest }) {
  const current = useRef(options); current.current = options;
  const library = useMemo(() => {
    const stores = new Map<string, ReturnType<typeof createAssemblyDraft>>();
    let failedScope: string | undefined;
    let storage: Storage | undefined;
    try { storage = options.cloud ? sessionStorage : localStorage; } catch { /* Drafts remain downloadable in memory. */ }
    return {
      store(scope: string) {
        let store = stores.get(scope);
        if (!store) {
          store = createAssemblyDraft({ ...options, scope, storage, request: (path, body, signal) => current.current.request!(path, body, signal),
            canAct: () => current.current.available && current.current.cloud === options.cloud && current.current.owner === options.owner && current.current.paperId === options.paperId });
          stores.set(scope, store);
        }
        return store;
      },
      pending: () => [...stores.values()].some(store => store.hasPending()),
      flush: async () => { failedScope = undefined; for (const [scope, store] of stores) if (store.hasPending() && !await store.flush()) { failedScope = scope; return false; } return true; },
      failedScope: () => failedScope,
      activate: () => stores.forEach(store => store.activate()),
      dispose: () => stores.forEach(store => store.dispose()),
    };
  }, [options.owner, options.paperId, options.cloud]);
  const store = library.store(options.scope);
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
  useEffect(() => { library.activate(); if (options.enabled) void store.load(); }, [library, store, options.enabled]);
  useEffect(() => {
    library.activate();
    const beforeUnload = (event: BeforeUnloadEvent) => { if (library.pending()) { event.preventDefault(); event.returnValue = ''; } };
    const followLink = (event: MouseEvent) => {
      if (options.cloud) return; // The hosted workspace coordinates shared and personal saves together.
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || !library.pending()) return;
      const link = (event.target as Element | null)?.closest<HTMLAnchorElement>('a[href]');
      if (!link || link.target === '_blank' || link.hasAttribute('download') || link.href === location.href) return;
      event.preventDefault(); event.stopPropagation();
      const href = link.href;
      void library.flush().then(ok => { if (ok && current.current.owner === options.owner && current.current.paperId === options.paperId) location.assign(href); });
    };
    window.addEventListener('beforeunload', beforeUnload); window.addEventListener('click', followLink, true);
    return () => { window.removeEventListener('beforeunload', beforeUnload); window.removeEventListener('click', followLink, true); library.dispose(); };
  }, [library]);
  return { state, store, flushAll: library.flush, hasPending: library.pending, failedScope: library.failedScope };
}
