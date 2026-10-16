import { useEffect, useId, useRef, useState } from 'react';
import { Bell, CheckCheck, Heart, MessageCircle, Repeat2, UserPlus } from 'lucide-react';
import { Avatar, DateLabel, errorMessage, profileHref } from './social';
import { useI18n } from './i18n';
import { communityPostHref } from './communityNavigation';
import { inboxChanged, inboxRequest, messagesHref, useInboxPolling, type InboxCounts, type InboxNotification, type InboxSnapshot } from './inbox';

export type InboxPanelProps = {owner: string; notifications: boolean; messages: boolean; onCounts: (counts: InboxCounts) => void};
export default function InboxPanel({owner, notifications, messages, onCounts}: InboxPanelProps) {
  const {t} = useI18n();
  const panelId = useId();
  const [tab, setTab] = useState<'notifications' | 'messages'>(notifications ? 'notifications' : 'messages');
  const [data, setData] = useState<InboxSnapshot>(), [error, setError] = useState(''), [busy, setBusy] = useState(false), [more, setMore] = useState(false);
  const alive = useRef(true), manual = useRef<AbortController | undefined>(undefined), revision = useRef(0);
  useEffect(() => { alive.current = true; return () => { alive.current = false; manual.current?.abort(); }; }, []);
  async function refresh(signal?: AbortSignal) {
    const version = ++revision.current;
    try {
      const next = await inboxRequest<InboxSnapshot>(owner, 'inbox?limit=20', undefined, signal);
      if (signal?.aborted || !alive.current || version !== revision.current) return;
      setData(next); onCounts(next); setError(''); setMore(false);
    } catch (e) { if (!signal?.aborted && alive.current && version === revision.current) setError(errorMessage(e)); }
  }
  useInboxPolling(owner, refresh);
  const retry = () => { manual.current?.abort(); manual.current = new AbortController(); setMore(false); void refresh(manual.current.signal); };
  async function read(notification?: InboxNotification) {
    if (!data || busy) return;
    const cutoff = notification?.latest_event_id ?? data.observed_event_id;
    ++revision.current;
    setBusy(true);
    try {
      const counts = await inboxRequest<InboxCounts>(owner, 'inbox/read', {through_event_id: cutoff, ...(notification ? {notification_ids: [notification.id]} : {})});
      if (alive.current) {
        ++revision.current;
        onCounts(counts); setError('');
        setData(current => current ? {...current, ...counts, notifications: current.notifications.map(item =>
          item.latest_event_id <= cutoff && (!notification || item.id === notification.id) ? {...item, unread: false} : item)} : current);
      }
      inboxChanged(owner);
    } catch (e) { if (alive.current) setError(errorMessage(e)); }
    finally { if (alive.current) setBusy(false); }
  }
  async function loadMore() {
    const before = tab === 'notifications' ? data?.notifications_next_before : data?.conversations_next_before;
    if (!before || more) return;
    manual.current?.abort(); const controller = new AbortController(); manual.current = controller;
    const version = ++revision.current;
    setMore(true);
    try {
      const next = await inboxRequest<InboxSnapshot>(owner, `inbox?limit=20&${tab === 'notifications' ? 'notifications' : 'conversations'}_before=${before}`, undefined, controller.signal);
      if (controller.signal.aborted || !alive.current || version !== revision.current) return;
      setData(current => current ? tab === 'notifications' ? {...current, notifications: [...current.notifications, ...next.notifications.filter(item => !current.notifications.some(old => old.id === item.id))], notifications_next_before: next.notifications_next_before}
        : {...current, conversations: [...current.conversations, ...next.conversations.filter(item => !current.conversations.some(old => old.id === item.id))], conversations_next_before: next.conversations_next_before} : next);
      setError('');
    } catch (e) { if (!controller.signal.aborted && alive.current && version === revision.current) setError(errorMessage(e)); }
    finally { if (alive.current && version === revision.current) setMore(false); }
  }
  return <>
    <header className="inbox-panel-heading"><h2>{t('收件箱', 'Inbox')}</h2>{messages && <a href={messagesHref()}>{t('打开私信', 'Open messages')}</a>}</header>
    <div className="inbox-tabs" role="tablist" aria-label={t('消息类型', 'Inbox sections')} onKeyDown={event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      const tabs = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]')], current = tabs.indexOf(event.target as HTMLButtonElement);
      if (current < 0) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      tabs[next]?.focus(); tabs[next]?.click();
    }}>
      {notifications && <button type="button" role="tab" id={`${panelId}-notifications`} aria-controls={panelId} tabIndex={tab === 'notifications' ? 0 : -1} aria-selected={tab === 'notifications'} onClick={() => setTab('notifications')}>{t('通知', 'Notifications')}{!!data?.notification_unread_count && <span>{data.notification_unread_count}</span>}</button>}
      {messages && <button type="button" role="tab" id={`${panelId}-messages`} aria-controls={panelId} tabIndex={tab === 'messages' ? 0 : -1} aria-selected={tab === 'messages'} onClick={() => setTab('messages')}>{t('私信', 'Messages')}{!!data?.message_unread_count && <span>{data.message_unread_count}</span>}</button>}
    </div>
    {error && <div className="inbox-error" role="alert"><p>{error}</p><button type="button" onClick={retry}>{t('重试', 'Retry')}</button></div>}
    {!data && !error && <p className="inbox-empty" role="status">{t('正在加载…', 'Loading…')}</p>}
    {data && <div className="inbox-tab-content" role="tabpanel" id={panelId} aria-labelledby={`${panelId}-${tab}`}>
      {tab === 'notifications' ? <>
        {data.notification_unread_count > 0 && <button type="button" className="inbox-mark-read" disabled={busy} onClick={() => void read()}><CheckCheck size={14}/>{busy ? t('正在标记…', 'Marking…') : t('全部标为已读', 'Mark all as read')}</button>}
        {data.notifications.length ? <ul className="inbox-items">{data.notifications.map(item => <li key={item.id}><NotificationRow notification={item} onRead={() => { if (item.unread) void read(item); }}/></li>)}</ul>
          : <div className="inbox-empty"><Bell size={26}/><p>{t('还没有通知', 'No notifications yet')}</p><small>{t('回复、点赞和新关注会显示在这里。', 'Replies, likes and new followers will appear here.')}</small></div>}
      </> : data.conversations.length ? <ul className="inbox-items">{data.conversations.map(item => <li key={item.id}><a className={`inbox-item${item.unread_count ? ' unread' : ''}`} href={messagesHref(item.peer.username)}>
        <Avatar user={item.peer} size={36}/><span className="inbox-item-copy"><b>{item.peer.display_name}</b><span className="inbox-message-preview">{item.last_message.sender_id === owner ? t('你：', 'You: ') : ''}{item.last_message.body}</span><DateLabel value={item.last_message.created_at}/></span>
        {item.unread_count > 0 && <span className="inbox-count" aria-label={t(`${item.unread_count} 条未读`, `${item.unread_count} unread`)}>{item.unread_count}</span>}
      </a></li>)}</ul> : <div className="inbox-empty"><MessageCircle size={26}/><p>{t('还没有私信', 'No messages yet')}</p><small>{t('可以从对方的个人主页发起私信。', 'Start a conversation from someone’s profile.')}</small></div>}
      {(tab === 'notifications' ? data.notifications_next_before : data.conversations_next_before) && <button type="button" className="inbox-more" disabled={more} onClick={() => void loadMore()}>{more ? t('正在加载…', 'Loading…') : t('加载更多', 'Load more')}</button>}
    </div>}
  </>;
}

export function NotificationRow({notification: item, onRead}: {notification: InboxNotification; onRead: () => void}) {
  const {t} = useI18n();
  const actor = item.actors[0];
  const name = actor?.anonymous ? t('匿名用户', 'Anonymous') : actor?.display_name || t('一位用户', 'Someone');
  const people = item.actor_count > 1 ? t(`${name} 等 ${item.actor_count} 人`, `${name} and ${item.actor_count - 1} ${item.actor_count === 2 ? 'other' : 'others'}`) : name;
  const action = item.kind === 'like' ? t('赞了你的帖子', 'liked your post') : item.kind === 'follow' ? t('关注了你', 'followed you') : item.kind === 'repost' ? t('转发了你的帖子', 'reposted your post') : t('回复了你', 'replied to you');
  const Icon = item.kind === 'like' ? Heart : item.kind === 'follow' ? UserPlus : item.kind === 'repost' ? Repeat2 : MessageCircle;
  const href = item.post ? communityPostHref(item.post.id, undefined, item.reply_id || undefined) : actor && !actor.anonymous ? profileHref(actor) : '#/community';
  return <a href={href} className={`inbox-item${item.unread ? ' unread' : ''}`} onClick={onRead}>
    <span className={`inbox-event-icon inbox-event-${item.kind}`}><Icon size={18}/></span>
    <span className="inbox-item-copy"><span><b>{people}</b>{' '}{action}</span>{item.post && <strong className="inbox-post-title">{item.post.title}</strong>}{item.preview && item.kind === 'reply' && <span className="inbox-message-preview">{item.preview}</span>}<DateLabel value={item.created_at}/></span>
    {item.unread && <span className="inbox-row-dot" aria-label={t('未读', 'Unread')}/>}
  </a>;
}
