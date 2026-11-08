import { useEffect, useMemo, useRef, useSyncExternalStore } from 'react';
import { socialApi } from './social';
import { makeId, validateWorkspace } from './rebuttalModel';
import type { RebuttalProject } from './rebuttalModel';

export type RebuttalRole = 'owner' | 'editor' | 'viewer';
export type RebuttalPerson = { id: string; display_name: string; username: string };
export type RebuttalMember = { user: RebuttalPerson; role: RebuttalRole };
export type RebuttalPendingMember = { id: string; user: RebuttalPerson; role: 'editor' | 'viewer'; created_at: string };
export type RebuttalInvitation = {
  id: string; paper_id: string; title: string; venue: string; owner: RebuttalPerson; role: 'editor' | 'viewer'; created_at: string;
};
export type RebuttalInvitationResult = { id: string; paper_id: string; status: 'accepted' | 'declined' };
export type RebuttalPaperRow = {
  id: string; title: string; venue: string; updatedAt: string; archived: boolean;
  reviewerCount: number; issueCount: number; readyCount: number; role: RebuttalRole; owner: RebuttalPerson;
};
export type RebuttalPaper = {
  project: RebuttalProject; revision: number; role: RebuttalRole; owner: RebuttalPerson; members: RebuttalMember[]; invitations: RebuttalPendingMember[];
};
export type RebuttalSaveStatus = 'saved' | 'saving' | 'error' | 'conflict';
export type RebuttalCloudState = {
  loading: boolean; project?: RebuttalProject; document?: RebuttalPaper;
  status: RebuttalSaveStatus; error: string; blocked: boolean; dirty: boolean;
  reloadVersion: number; memberBusy: boolean; recoveryUnavailable: boolean;
};
export type RebuttalRequest = (path: string, body?: unknown, signal?: AbortSignal) => Promise<unknown>;
type Storage = Pick<globalThis.Storage, 'getItem' | 'setItem' | 'removeItem'>;
type SaveCommand = { project: RebuttalProject; expected_revision: number; request_id: string; expected_user_id: string };
type MemberChange = { username: string; role: 'editor' | 'viewer' } | { action: 'remove'; user_id: string };
type MemberCommand = MemberChange & { expected_revision: number; request_id: string; expected_user_id: string };
const projectLimit = 200 * 1024;
const identifier = (value: unknown): value is string => typeof value === 'string' && /^[a-zA-Z0-9_-]{1,150}$/.test(value);
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const integer = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) >= 0;
const role = (value: unknown): value is RebuttalRole => value === 'owner' || value === 'editor' || value === 'viewer';
const message = (error: unknown) => error instanceof Error ? error.message : 'Request failed';
const statusCode = (error: unknown) => record(error) && typeof error.status === 'number' ? error.status : undefined;
const errorCode = (error: unknown) => record(error) && typeof error.code === 'string' ? error.code : undefined;
// Timestamps are assigned by the server; changing them alone is not an author edit.
const projectSignature = ({ updatedAt: _updatedAt, createdAt: _createdAt, ...content }: RebuttalProject) => JSON.stringify(content);
export const rebuttalRequestId = () => makeId().replaceAll('-', '');
export const rebuttalHref = (id?: string) => id ? `#/rebuttal/${encodeURIComponent(id)}` : '#/rebuttal';
export function readRebuttalRoute(route: string): { paperId?: string; tab?: 'invitations' } | undefined {
  const path = route.replace(/^#\/?/, '');
  if (path === 'rebuttal' || path === 'rebuttal/') return {};
  if (path === 'rebuttal/invitations' || path === 'rebuttal/invitations/') return { tab: 'invitations' };
  const match = /^rebuttal\/([a-zA-Z0-9_-]{1,150})\/?$/.exec(path);
  return match ? { paperId: match[1] } : undefined;
}
export function validateCloudProject(value: unknown, enforceLimit = true): RebuttalProject {
  if (!record(value) || !identifier(value.id)) throw Error('Invalid paper response');
  const project = validateWorkspace({ version: 1, activeProjectId: value.id, projects: [value] }).projects[0];
  if (enforceLimit && new TextEncoder().encode(JSON.stringify(project)).length > projectLimit) throw Error('Paper exceeds the 200 KiB limit. Download your draft and split it into smaller papers.');
  return project;
}
function person(value: unknown): RebuttalPerson {
  if (!record(value) || !identifier(value.id) || typeof value.display_name !== 'string' || typeof value.username !== 'string') throw Error('Invalid paper member');
  return { id: value.id, display_name: value.display_name, username: value.username };
}
export function readRebuttalPaper(value: unknown, id?: string): RebuttalPaper {
  if (!record(value) || !integer(value.revision) || value.revision < 1 || !role(value.role) || !Array.isArray(value.members)) throw Error('Invalid paper response');
  const project = validateCloudProject(value.project);
  if (id && project.id !== id) throw Error('The response belongs to another paper');
  const owner = person(value.owner);
  const members = value.members.map(item => {
    if (!record(item) || !role(item.role)) throw Error('Invalid paper membership');
    return { user: person(item.user), role: item.role };
  });
  if (new Set(members.map(item => item.user.id)).size !== members.length || !members.some(item => item.user.id === owner.id && item.role === 'owner')) throw Error('Invalid paper membership');
  if (value.invitations !== undefined && !Array.isArray(value.invitations)) throw Error('Invalid pending invitations');
  const invitations = ((value.invitations || []) as unknown[]).map<RebuttalPendingMember>(item => {
    if (!record(item) || !identifier(item.id) || (item.role !== 'editor' && item.role !== 'viewer') ||
      typeof item.created_at !== 'string' || !Number.isFinite(Date.parse(item.created_at))) throw Error('Invalid pending invitation');
    return { id: item.id, user: person(item.user), role: item.role, created_at: item.created_at };
  });
  if (new Set(invitations.map(item => item.id)).size !== invitations.length || new Set(invitations.map(item => item.user.id)).size !== invitations.length ||
    invitations.some(item => members.some(member => member.user.id === item.user.id))) throw Error('Invalid pending invitations');
  return { project, revision: value.revision, role: value.role, owner, members, invitations };
}
export function readRebuttalInvitations(value: unknown, owner?: string): RebuttalInvitation[] {
  if (!record(value) || !Array.isArray(value.invitations)) throw Error('Invalid invitations response');
  if (owner && value.user_id !== owner) throw Error('Your account changed. Refresh your account and try again.');
  const invitations = value.invitations.map<RebuttalInvitation>(item => {
    if (!record(item) || !identifier(item.id) || !identifier(item.paper_id) || typeof item.title !== 'string' || typeof item.venue !== 'string' ||
      (item.role !== 'editor' && item.role !== 'viewer') || typeof item.created_at !== 'string' || !Number.isFinite(Date.parse(item.created_at))) throw Error('Invalid invitation');
    return { id: item.id, paper_id: item.paper_id, title: item.title, venue: item.venue, owner: person(item.owner), role: item.role, created_at: item.created_at };
  });
  if (new Set(invitations.map(item => item.id)).size !== invitations.length) throw Error('Duplicate invitation identifiers');
  return invitations;
}

/** Accept/decline retries retain their exact command until its result is confirmed. */
export function createRebuttalInvitations(options: { owner: string; request?: RebuttalRequest; canAct?: () => boolean }) {
  type Command = { id: string; paper_id: string; body: { action: 'accept' | 'decline'; request_id: string; expected_user_id: string } };
  let state: { invitations: RebuttalInvitation[]; loading: boolean; busy: boolean; error: string; pending: boolean; unavailable: boolean } =
    { invitations: [], loading: true, busy: false, error: '', pending: false, unavailable: false };
  let active = true, generation = 0, reading = 0, command: Command | undefined;
  const listeners = new Set<() => void>(), controllers = new Set<AbortController>();
  const owns = (epoch = generation) => active && epoch === generation && (options.canAct?.() ?? true);
  const emit = (patch: Partial<typeof state>) => { if (owns()) { state = { ...state, ...patch }; listeners.forEach(listener => listener()); } };
  async function call(path: string, body?: unknown) {
    const controller = new AbortController(); controllers.add(controller);
    const timeout = setTimeout(() => controller.abort(), 20000);
    try { return await (options.request || socialApi)(path, body, controller.signal); }
    finally { clearTimeout(timeout); controllers.delete(controller); }
  }
  async function load() {
    if (!owns() || state.busy || command || state.unavailable) return;
    const epoch = generation, read = ++reading;
    emit({ loading: true, error: '' });
    try {
      const invitations = readRebuttalInvitations(await call('rebuttal/invitations'), options.owner);
      if (owns(epoch) && read === reading) emit({ invitations });
    } catch (error) {
      if (owns(epoch) && read === reading) emit(statusCode(error) === 404 ? { unavailable: true, invitations: [] } : { error: message(error) });
    } finally { if (owns(epoch) && read === reading) emit({ loading: false }); }
  }
  async function respond(invitation?: RebuttalInvitation, action?: 'accept' | 'decline'): Promise<RebuttalInvitationResult | undefined> {
    if (!owns() || state.busy || state.unavailable || command && invitation) return;
    if (!command) {
      if (!invitation || !action || !state.invitations.some(item => item.id === invitation.id)) return;
      command = { id: invitation.id, paper_id: invitation.paper_id, body: { action, request_id: rebuttalRequestId(), expected_user_id: options.owner } };
    }
    const sent = command, epoch = generation;
    reading++; emit({ busy: true, pending: true, loading: false, error: '' });
    try {
      const result = await call(`rebuttal/invitations/${encodeURIComponent(sent.id)}`, sent.body);
      if (!owns(epoch)) return;
      const expectedStatus = sent.body.action === 'accept' ? 'accepted' : 'declined';
      if (!record(result) || result.id !== sent.id || result.paper_id !== sent.paper_id || result.status !== expectedStatus) throw Error('The invitation result is unconfirmed. Retry the original operation.');
      command = undefined;
      emit({ invitations: state.invitations.filter(item => item.id !== sent.id), pending: false });
      return { id: sent.id, paper_id: sent.paper_id, status: expectedStatus };
    } catch (error) {
      if (owns(epoch)) {
        const code = statusCode(error);
        if (code && code >= 400 && code < 500 && code !== 408 && code !== 429) command = undefined;
        emit({ error: message(error), pending: !!command,
          ...(code === 404 ? { invitations: state.invitations.filter(item => item.id !== sent.id) } : {}) });
      }
    } finally { if (owns(epoch)) emit({ busy: false }); }
  }
  return {
    getSnapshot: () => state,
    subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener); }; },
    load, respond,
    activate() { active = true; },
    dispose() { active = false; generation++; reading++; controllers.forEach(controller => controller.abort()); controllers.clear(); state = { ...state, busy: false }; },
  };
}
export function useRebuttalInvitations(owner: string, enabled: boolean) {
  const current = useRef({ owner, enabled }); current.current = { owner, enabled };
  const store = useMemo(() => createRebuttalInvitations({ owner, canAct: () => current.current.owner === owner && current.current.enabled }), [owner]);
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
  useEffect(() => {
    if (!enabled) return;
    store.activate(); void store.load();
    const interval = setInterval(() => { if (document.visibilityState === 'visible' && navigator.onLine) void store.load(); }, 45000);
    return () => { clearInterval(interval); store.dispose(); };
  }, [store, enabled]);
  return { state, store };
}
export function readRebuttalPapers(value: unknown): RebuttalPaperRow[] {
  if (!record(value) || !Array.isArray(value.papers)) throw Error('Invalid paper library');
  const papers = value.papers.map(item => {
    if (!record(item) || !identifier(item.id) || typeof item.title !== 'string' || typeof item.venue !== 'string' ||
      typeof item.updatedAt !== 'string' || !Number.isFinite(Date.parse(item.updatedAt)) || typeof item.archived !== 'boolean' ||
      !role(item.role) || !integer(item.reviewerCount) || !integer(item.issueCount) || !integer(item.readyCount) || item.readyCount > item.issueCount) throw Error('Invalid paper library');
    return { id: item.id, title: item.title, venue: item.venue, updatedAt: item.updatedAt, archived: item.archived,
      reviewerCount: item.reviewerCount, issueCount: item.issueCount, readyCount: item.readyCount, role: item.role, owner: person(item.owner) };
  });
  if (new Set(papers.map(item => item.id)).size !== papers.length) throw Error('Duplicate paper identifiers');
  return papers;
}
export function downloadRebuttalProject(project: RebuttalProject) {
  const content = JSON.stringify({ version: 1, activeProjectId: project.id, projects: [project] }, null, 2);
  const url = URL.createObjectURL(new Blob([content], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = 'paperleet-rebuttal-draft.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** A single paper's revisioned queue. Failed requests keep their original idempotency key. */
export function createRebuttalCloud(options: {
  owner: string; paperId: string; request?: RebuttalRequest; storage?: Storage; canAct?: () => boolean; debounceMs?: number;
}) {
  const { owner, paperId, storage } = options;
  const request: RebuttalRequest = options.request || socialApi;
  const path = `rebuttal/papers/${encodeURIComponent(paperId)}`;
  const recoveryKey = `paperleet.rebuttal.cloud.${owner}.${paperId}`;
  let state: RebuttalCloudState = { loading: true, status: 'saved', error: '', blocked: false, dirty: false, reloadVersion: 0, memberBusy: false, recoveryUnavailable: false };
  let active = true, generation = 0, saving: Promise<boolean> | undefined, reading = false, localProblem = false, membersPreparing = false, timer: ReturnType<typeof setTimeout> | undefined;
  let command: SaveCommand | undefined, memberCommand: MemberCommand | undefined;
  let savedSignature = '', draftSignature = '';
  const controllers = new Set<AbortController>(), listeners = new Set<() => void>();
  const owns = () => active && (options.canAct?.() ?? true);
  const ownsGeneration = (epoch: number) => owns() && epoch === generation;
  const emit = (patch: Partial<RebuttalCloudState>) => { if (!owns()) return; state = { ...state, ...patch }; listeners.forEach(fn => fn()); };
  const cancelTimer = () => { if (timer) clearTimeout(timer); timer = undefined; };
  const persist = () => {
    if (!storage || !state.project || !state.document) return;
    try {
      if (!state.dirty && !command && !memberCommand) storage.removeItem(recoveryKey);
      else storage.setItem(recoveryKey, JSON.stringify({ project: state.project, baseRevision: state.document.revision, command, memberCommand }));
    } catch { emit({ recoveryUnavailable: true }); }
  };
  const call = async (body?: unknown, suffix = '') => {
    if (!owns()) throw Error('Account changed');
    const controller = new AbortController(); controllers.add(controller);
    const timeout = setTimeout(() => controller.abort(), 20000);
    try { return await request(path + suffix, body, controller.signal); }
    finally { clearTimeout(timeout); controllers.delete(controller); }
  };
  const reject = (error: unknown) => {
    const code = statusCode(error);
    emit({ loading: false, status: code === 409 && errorCode(error) !== 'member_limit' ? 'conflict' : 'error', error: message(error),
      blocked: code === 401 || code === 403 || code === 404 && errorCode(error) !== 'member_not_found' || errorCode(error) === 'account_changed' });
    persist();
  };
  const applyRemote = (document: RebuttalPaper, reload: boolean, preserveRecovery = false) => {
    localProblem = false;
    savedSignature = draftSignature = projectSignature(document.project);
    emit({ document, project: document.project, loading: false, dirty: false, status: 'saved', error: '', blocked: false,
      reloadVersion: state.reloadVersion + (reload ? 1 : 0) });
    if (!preserveRecovery) persist();
  };
  const restoreRecovery = () => {
    if (!storage || !state.document) return;
    try {
      const raw = storage.getItem(recoveryKey); if (!raw) return;
      const cached: unknown = JSON.parse(raw);
      if (!record(cached) || !integer(cached.baseRevision)) throw Error('Invalid local draft');
      const project = validateCloudProject(cached.project, false);
      if (project.id !== paperId) throw Error('Invalid local draft');
      draftSignature = projectSignature(project);
      if (record(cached.command) && cached.command.expected_user_id === owner && integer(cached.command.expected_revision) &&
        identifier(cached.command.request_id)) {
        const sent = validateCloudProject(cached.command.project);
        if (sent.id === paperId) command = { project: sent, expected_revision: cached.command.expected_revision, request_id: cached.command.request_id, expected_user_id: owner };
      }
      const member = cached.memberCommand;
      if (record(member) && member.expected_user_id === owner && integer(member.expected_revision) &&
        identifier(member.request_id)) {
        const change: MemberChange | undefined = member.action === 'remove' && identifier(member.user_id) ? { action: 'remove', user_id: member.user_id } :
          typeof member.username === 'string' && /^[a-z0-9_]{3,24}$/.test(member.username) && (member.role === 'editor' || member.role === 'viewer') ? { username: member.username, role: member.role } : undefined;
        if (change) memberCommand = { ...change, expected_revision: member.expected_revision, request_id: member.request_id, expected_user_id: owner };
      }
      if (draftSignature === savedSignature && !command && !memberCommand) { storage.removeItem(recoveryKey); return; }
      emit({ project, dirty: draftSignature !== savedSignature, status: command || memberCommand ? 'error' : cached.baseRevision === state.document.revision ? 'error' : 'conflict',
        error: command || memberCommand ? 'A previous operation is unconfirmed. Retry it to confirm the result.' : 'A local draft was recovered. Save it or download it before loading the server version.' });
    } catch { emit({ recoveryUnavailable: true }); }
  };
  async function load() {
    if (!owns() || reading) return;
    const epoch = generation;
    reading = true;
    try {
      const document = readRebuttalPaper(await call(), paperId);
      if (!ownsGeneration(epoch)) return;
      applyRemote(document, false, true);
      restoreRecovery();
    } catch (error) { if (ownsGeneration(epoch)) reject(error); }
    finally { if (ownsGeneration(epoch)) reading = false; }
  }
  function update(project: RebuttalProject) {
    if (!owns() || !state.document || state.document.role === 'viewer' || state.blocked) return false;
    let normalized: RebuttalProject;
    try { normalized = validateCloudProject(project, false); }
    catch (error) { emit({ status: 'error', error: message(error) }); return false; }
    if (normalized.id !== paperId) return false;
    const signature = projectSignature(normalized);
    if (signature === draftSignature) return true;
    draftSignature = signature;
    const dirty = signature !== savedSignature;
    const oversized = new TextEncoder().encode(JSON.stringify(normalized)).length > projectLimit;
    const recovering = localProblem && !oversized; localProblem = oversized;
    emit({ project: normalized, dirty,
      status: oversized ? 'error' : state.status === 'conflict' ? 'conflict' : command || state.status === 'error' && !recovering ? state.status : dirty ? 'saving' : 'saved',
      ...(oversized ? { error: 'Paper exceeds the 200 KiB limit. Download your draft and split it into smaller papers.' } : recovering ? { error: '' } : {}) });
    persist(); cancelTimer();
    if (dirty && !command && !memberCommand && state.status !== 'conflict' && state.status !== 'error') timer = setTimeout(() => void flush(), options.debounceMs ?? 700);
    return true;
  }
  async function drain(epoch: number): Promise<boolean> {
    while (ownsGeneration(epoch) && state.document && (command || state.dirty)) {
      if (state.document.role === 'viewer' || state.blocked || state.status === 'conflict' || state.memberBusy) return false;
      if (!command) {
        try { validateCloudProject(state.project); }
        catch (error) { localProblem = true; reject(error); return false; }
        command = { project: structuredClone(state.project!), expected_revision: state.document.revision, request_id: rebuttalRequestId(), expected_user_id: owner };
      }
      const sent = command, signature = projectSignature(sent.project);
      emit({ status: 'saving', error: '' }); persist();
      try {
        const document = readRebuttalPaper(await call(sent), paperId);
        if (!ownsGeneration(epoch)) return false;
        if (document.revision !== sent.expected_revision + 1) throw Error('The save result is unconfirmed. Retry the original save.');
        command = undefined;
        savedSignature = projectSignature(document.project);
        const noNewerEdits = draftSignature === signature;
        if (noNewerEdits) draftSignature = savedSignature;
        emit({ document, project: noNewerEdits ? document.project : state.project, dirty: draftSignature !== savedSignature,
          status: draftSignature !== savedSignature ? 'saving' : 'saved', error: '', blocked: false });
        persist();
      } catch (error) {
        if (ownsGeneration(epoch)) {
          if ([400, 413, 422].includes(statusCode(error) || 0)) { command = undefined; localProblem = true; }
          reject(error);
        }
        return false;
      }
    }
    return ownsGeneration(epoch) && !state.dirty;
  }
  async function flush(): Promise<boolean> {
    cancelTimer();
    if (!owns() || state.status === 'conflict' || state.memberBusy || memberCommand) return false;
    if (saving) return saving;
    const pending = drain(generation);
    saving = pending;
    try { return await pending; } finally { if (saving === pending) saving = undefined; }
  }
  async function refresh(discard = false) {
    if (!owns() || reading || saving || state.memberBusy || memberCommand && !discard) return false;
    const epoch = generation;
    const startingRevision = state.document?.revision;
    if (discard) emit({ loading: true });
    reading = true;
    try {
      const document = readRebuttalPaper(await call(), paperId);
      if (!ownsGeneration(epoch)) return false;
      if (!discard && (saving || state.memberBusy || startingRevision !== state.document?.revision)) return false;
      if (!discard && (state.dirty || command)) {
        if (state.document?.revision !== document.revision || document.role === 'viewer') emit({ status: 'conflict', error: 'This paper changed on the server. Your local draft is preserved.',
          document: state.document && { ...state.document, role: document.role, members: document.members } });
        return false;
      }
      if (discard || state.document?.revision !== document.revision || state.document?.role !== document.role) {
        command = undefined; memberCommand = undefined; cancelTimer(); applyRemote(document, true);
      } else emit({ document, status: 'saved', error: '', blocked: false });
      return true;
    } catch (error) { if (ownsGeneration(epoch)) reject(error); return false; }
    finally { if (ownsGeneration(epoch)) reading = false; }
  }
  async function changeMember(change?: MemberChange) {
    if (!owns() || state.document?.role !== 'owner' || state.memberBusy || membersPreparing || state.status === 'conflict' || state.blocked) return false;
    const epoch = generation;
    membersPreparing = true;
    const flushed = memberCommand ? true : await flush();
    if (!ownsGeneration(epoch)) return false;
    membersPreparing = false;
    if (!flushed || !state.document) return false;
    if (!memberCommand) {
      if (!change) return false;
      memberCommand = { ...change, expected_revision: state.document.revision, request_id: rebuttalRequestId(), expected_user_id: owner };
    }
    emit({ memberBusy: true, error: '' });
    persist();
    const sent = memberCommand;
    let succeeded = false;
    try {
      const document = readRebuttalPaper(await call(sent, '/members'), paperId);
      if (!ownsGeneration(epoch)) return false;
      if (document.revision !== sent.expected_revision + 1) throw Error('The membership change is unconfirmed. Retry the original operation.');
      memberCommand = undefined;
      if (state.dirty) {
        savedSignature = projectSignature(document.project);
        emit({ document, dirty: draftSignature !== savedSignature, status: 'saving', error: '' });
        persist();
      } else applyRemote(document, false);
      succeeded = true; return true;
    } catch (error) {
      if (ownsGeneration(epoch)) {
        if ([400, 413, 422].includes(statusCode(error) || 0) || errorCode(error) === 'member_not_found' || errorCode(error) === 'member_limit') memberCommand = undefined;
        reject(error);
      }
      return false;
    }
    finally {
      if (ownsGeneration(epoch)) {
        emit({ memberBusy: false });
        if (succeeded && state.dirty) timer = setTimeout(() => void flush(), options.debounceMs ?? 700);
      }
    }
  }
  return {
    getSnapshot: () => state,
    subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener); }; },
    load, update, flush, refresh, changeMember,
    hasPendingMember: () => !!memberCommand,
    hasPending: () => state.dirty || !!command || !!memberCommand,
    activate() { active = true; },
    dispose() {
      active = false; generation++; cancelTimer();
      controllers.forEach(controller => controller.abort()); controllers.clear();
      reading = false; saving = undefined; membersPreparing = false;
    },
  };
}
export function useRebuttalCloud(owner: string, paperId: string) {
  const current = useRef({ owner, paperId }); current.current = { owner, paperId };
  const store = useMemo(() => {
    let storage: Storage | undefined; try { storage = sessionStorage; } catch { /* Memory and download remain available. */ }
    return createRebuttalCloud({ owner, paperId, storage, canAct: () => current.current.owner === owner && current.current.paperId === paperId });
  }, [owner, paperId]);
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
  useEffect(() => {
    store.activate();
    void store.load();
    const interval = setInterval(() => { if (document.visibilityState === 'visible') void store.refresh(); }, 20000);
    const beforeUnload = (event: BeforeUnloadEvent) => { if (store.hasPending()) { event.preventDefault(); event.returnValue = ''; } };
    window.addEventListener('beforeunload', beforeUnload);
    return () => { clearInterval(interval); window.removeEventListener('beforeunload', beforeUnload); store.dispose(); };
  }, [store]);
  return { state, store };
}
