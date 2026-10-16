import { useEffect, useRef } from 'react';
import { socialApi, type PublicUser } from './social';

export type InboxCounts = { user_id: string; unread_count: number; notification_unread_count: number; message_unread_count: number };
export type InboxNotification = {
  id: string; kind: 'like' | 'reply' | 'follow' | 'repost'; latest_event_id: number; created_at: string;
  unread: boolean; actor_count: number; actors: PublicUser[]; post: {id: string; title: string} | null;
  reply_id: string | null; preview: string;
};
export type DirectMessage = { id: number; sender_id: string; body: string; created_at: string; request_id?: string };
export type Conversation = { id: number; peer: PublicUser; last_message: DirectMessage; unread_count: number; can_send: boolean; awaiting_reply: boolean; is_mutual_follow: boolean };
export type InboxSnapshot = InboxCounts & {
  observed_event_id: number; notifications: InboxNotification[]; notifications_next_before: number | null;
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

/** Pause background refreshes when hidden/offline and abort every obsolete read. */
export function useInboxPolling(owner: string, refresh: (signal: AbortSignal) => Promise<void>, enabled = true) {
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
  }, [owner, enabled]);
}
