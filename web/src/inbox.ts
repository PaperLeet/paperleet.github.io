import { useEffect, useRef, useState } from 'react';
import { errorMessage, socialApi, type PublicUser } from './social';

export type InboxCounts = { user_id: string; unread_count: number; notification_unread_count: number; message_unread_count: number };
export type InboxNotification = {
  id: string; kind: 'like' | 'reply' | 'follow' | 'repost' | 'review'; latest_event_id: number; created_at: string;
  unread: boolean; actor_count: number; actors: PublicUser[]; post: {id: string; title: string} | null;
  reply_id: string | null; preview: string; review?:unknown;
};
export type InboxNotificationKind='all'|InboxNotification['kind'];
export type DirectMessage = { id: number; sender_id: string; body: string; created_at: string; request_id?: string };
export type Conversation = { id: number; peer: PublicUser; last_message: DirectMessage; unread_count: number; can_send: boolean; awaiting_reply: boolean; is_mutual_follow: boolean };
export type InboxSnapshot = InboxCounts & {
  notification_kind?:InboxNotificationKind;reviews_included?:boolean; observed_event_id: number; observed_message_id?: number; notifications: InboxNotification[]; notifications_next_before: number | null;
  conversations: Conversation[]; conversations_next_before: number | null;
};
export type MessageThread = {
  user_id: string; peer: PublicUser; conversation_id: number | null; messages: DirectMessage[];
  next_before: number | null; observed_message_id: number; unread_count: number;
  can_send: boolean; awaiting_reply: boolean; is_mutual_follow: boolean;
};

export const messagesHref = (username?: string) => '#/messages' + (username ? `/${encodeURIComponent(username)}` : '');
export const messagePath = (username: string) => `messages/${encodeURIComponent(username)}`;
export const newMessageRequestId = () => crypto.randomUUID().replaceAll('-', '');
export const inboxVisible = () => document.visibilityState === 'visible' && navigator.onLine;
const inboxChangedEvent = 'paperleet:inbox-changed';

/** Events contain only the owner ID; private message text never enters browser storage. */
export function inboxChanged(owner: string) {
  window.dispatchEvent(new CustomEvent(inboxChangedEvent, {detail: owner}));
}
export async function inboxRequest<T extends {user_id: string}>(owner: string, path: string, body?: object, signal?: AbortSignal): Promise<T> {
  const result = await socialApi<T>(path, body ? {...body, expected_user_id: owner} : undefined, signal);
  if (result.user_id !== owner) throw new Error('Your account changed. Please refresh your account and try again.');
  return result;
}

export type InboxReadCutoff = {through_event_id: number; through_message_id: number;include_reviews?:true};
export function inboxReadCutoff(snapshot?: InboxSnapshot): InboxReadCutoff | undefined {
  if (!snapshot || !Number.isSafeInteger(snapshot.observed_event_id) || snapshot.observed_event_id < 0
    || !Number.isSafeInteger(snapshot.observed_message_id) || snapshot.observed_message_id! < 0) return;
  return {through_event_id: snapshot.observed_event_id, through_message_id: snapshot.observed_message_id!,...(snapshot.reviews_included===true?{include_reviews:true as const}:{})};
}

/** Only rows covered by the acknowledged snapshot change; newer arrivals stay unread. */
export function applyInboxReadAll(snapshot: InboxSnapshot, counts: InboxCounts, cutoff: InboxReadCutoff): InboxSnapshot {
  return {...snapshot, ...counts,
    notifications: snapshot.notifications.map(item => item.latest_event_id <= cutoff.through_event_id && (item.kind!=='review'||cutoff.include_reviews===true) ? {...item, unread: false} : item),
    conversations: snapshot.conversations.map(item => item.last_message.id <= cutoff.through_message_id ? {...item, unread_count: 0} : item)};
}

export function useInboxReadAll({owner, snapshot, available, blocked, onRead, canAct}: {
  owner: string; snapshot?: InboxSnapshot; available: boolean; blocked: boolean;canAct?:()=>boolean;
  onRead: (counts: InboxCounts, cutoff: InboxReadCutoff) => void;
}) {
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [result, setResult] = useState<InboxCounts>();
  const pending = useRef<InboxReadCutoff | undefined>(undefined), request = useRef<AbortController | undefined>(undefined), running = useRef(false);
  useEffect(() => () => { request.current?.abort(); }, [owner]);
  useEffect(() => {
    const cutoff = inboxReadCutoff(snapshot), attempted = pending.current;
    if (busy || !attempted || !cutoff || snapshot?.user_id !== owner || snapshot.unread_count !== 0
      || cutoff.through_event_id < attempted.through_event_id || cutoff.through_message_id < attempted.through_message_id) return;
    // A later authoritative empty snapshot confirms delivery even if the write response was lost.
    pending.current = undefined; setError('');
    setResult({user_id: snapshot.user_id, unread_count: 0, notification_unread_count: snapshot.notification_unread_count, message_unread_count: snapshot.message_unread_count});
  }, [owner, snapshot, busy]);
  const ready = available && !!inboxReadCutoff(snapshot);
  async function mark() {
    if (!ready || blocked || running.current || !snapshot?.unread_count || canAct?.()===false) return;
    const cutoff = pending.current || inboxReadCutoff(snapshot)!;
    pending.current = cutoff;
    running.current = true; setBusy(true); setError(''); setResult(undefined);
    const controller = new AbortController(); request.current = controller;
    try {
      const counts = await inboxRequest<InboxCounts>(owner, 'inbox/read-all', cutoff, controller.signal);
      if (controller.signal.aborted) return;
      pending.current = undefined; if(canAct?.()!==false){onRead(counts, cutoff); setResult(counts);}
      inboxChanged(owner);
    } catch (e) { if (!controller.signal.aborted) setError(errorMessage(e)); }
    finally { running.current = false; if (!controller.signal.aborted) setBusy(false); }
  }
  return {busy, error, result, mark, disabled: !ready || blocked || busy || !snapshot?.unread_count};
}

/** Pause background refreshes when hidden/offline and abort every obsolete read. */
export function useInboxPolling(owner: string, refresh: (signal: AbortSignal) => Promise<void>, enabled = true, scope = '') {
  const latest = useRef(refresh);
  latest.current = refresh;
  useEffect(() => {
    if (!enabled) return;
    let controller: AbortController | undefined;
    const run = () => {
      controller?.abort();
      if (!inboxVisible()) return;
      controller = new AbortController();
      void latest.current(controller.signal).catch(() => { /* The caller renders request errors. */ });
    };
    const changed = (event: Event) => { if ((event as CustomEvent).detail === owner) run(); };
    run();
    const interval = window.setInterval(run, 45_000);
    document.addEventListener('visibilitychange', run);
    window.addEventListener('focus', run);
    window.addEventListener('online', run);
    window.addEventListener('offline', run);
    window.addEventListener(inboxChangedEvent, changed);
    return () => {
      controller?.abort();
      window.clearInterval(interval);
      document.removeEventListener('visibilitychange', run);
      window.removeEventListener('focus', run);
      window.removeEventListener('online', run);
      window.removeEventListener('offline', run);
      window.removeEventListener(inboxChangedEvent, changed);
    };
  }, [owner, enabled, scope]);
}
