import InboxListRefresh from './InboxListRefresh';
import {inboxListUpdate} from './inboxListUpdate';
import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { Bell, CheckCheck, Heart, MessageCircle, Repeat2, UserPlus } from 'lucide-react';
import { Avatar, DateLabel, errorMessage, profileHref, useAccount } from './social';
import { useI18n } from './i18n';
import { communityPostHref } from './communityNavigation';
import { applyInboxReadAll, inboxChanged, inboxRequest, messagesHref, useInboxPolling, useInboxReadAll, type InboxCounts, type InboxNotification, type InboxSnapshot } from './inbox';
import InboxReadAll from './InboxReadAll';
import {navigateTabs} from './tabNavigation';

export type InboxPanelProps = {owner: string; notifications: boolean; messages: boolean; onCounts: (counts: InboxCounts) => void};
export default function InboxPanel({owner, notifications, messages, onCounts}: InboxPanelProps) {
  const {t} = useI18n();
  const {capabilities} = useAccount();
  const readAllAvailable = capabilities?.inbox_read_all === true;
  const panelId = useId();
  const [tab, setTab] = useState<'notifications' | 'messages'>(notifications ? 'notifications' : 'messages');
  const [data, setData] = useState<InboxSnapshot>(), [error, setError] = useState(''), [busy, setBusy] = useState(false), [more, setMore] = useState(false), [snapshotLoading, setSnapshotLoading] = useState(true), [readingId,setReadingId]=useState<string>(),[marked,setMarked]=useState(false),[readError,setReadError]=useState('');
  const alive = useRef(true), manual = useRef<AbortController | undefined>(undefined), revision = useRef(0);
  const currentData=useRef(data),replacing=useRef(false);
  const [changed,setChanged]=useState(false);
  useLayoutEffect(()=>{currentData.current=data;},[data]);
  useEffect(() => { alive.current = true; return () => { alive.current = false; manual.current?.abort(); }; }, []);
  async function refresh(signal?: AbortSignal,replace=false) {
    if(!replace&&replacing.current)return;
    const version = ++revision.current;
    if(!currentData.current||replace)setSnapshotLoading(true);
    try {
      const next = await inboxRequest<InboxSnapshot>(owner, 'inbox?limit=20', undefined, signal);
      if (signal?.aborted || !alive.current || version !== revision.current) return;
      const current=currentData.current;
      if(current&&!replace){const update=inboxListUpdate(current,next);setData(update.snapshot);setChanged(update.changed);}
      else{setData(next);setChanged(false);}
      onCounts(next); setError('');
    } catch (e) { if (!signal?.aborted && alive.current && version === revision.current) setError(errorMessage(e)); }
    finally { if(replace&&!signal?.aborted)replacing.current=false; if (!signal?.aborted && alive.current && version === revision.current) setSnapshotLoading(false); }
  }
  const readAll = useInboxReadAll({owner, snapshot: data, available: readAllAvailable, blocked: snapshotLoading || busy || more, onRead: (counts, cutoff) => {
    ++revision.current; replacing.current=false;manual.current?.abort(); setSnapshotLoading(false); setMore(false); onCounts(counts);
    setData(current => current ? applyInboxReadAll(current, counts, cutoff) : current);
  }});
  useInboxPolling(owner, refresh, !readAll.busy && !busy && !more);
  const retry = () => { manual.current?.abort(); manual.current = new AbortController(); setMore(false);replacing.current=true; void refresh(manual.current.signal,true); };
  const reading=useRef(false),readAttempt=useRef<{notification?:Pick<InboxNotification,'id'|'latest_event_id'>;cutoff:number}|undefined>(undefined);
  async function read(notification?: Pick<InboxNotification,'id'|'latest_event_id'>,through?:number) {
    if (!data || busy || readAll.busy || snapshotLoading || replacing.current || more || reading.current) return;
    const cutoff = through ?? notification?.latest_event_id ?? data.observed_event_id;
    ++revision.current;
    reading.current=true;readAttempt.current={notification,cutoff};setReadError('');setBusy(true);setReadingId(notification?.id);setMarked(false);
    try {
      const counts = await inboxRequest<InboxCounts>(owner, 'inbox/read', {through_event_id: cutoff, ...(notification ? {notification_ids: [notification.id]} : {})});
      if (alive.current) {
        ++revision.current;
        onCounts(counts); setError('');setReadError('');setMarked(true);
        setData(current => current ? {...current, ...counts, notifications: current.notifications.map(item =>
          item.latest_event_id <= cutoff && (!notification || item.id === notification.id) ? {...item, unread: false} : item)} : current);
      }
      inboxChanged(owner);
    } catch (e) { if (alive.current) setReadError(errorMessage(e)); }
    finally { reading.current=false;if (alive.current){setBusy(false);setReadingId(undefined);} }
  }
  useEffect(()=>{
    const attempt=readAttempt.current;if(!readError||busy||!attempt?.notification)return;
    const item=data?.notifications.find(row=>row.id===attempt.notification!.id);
    if(item?.latest_event_id===attempt.notification.latest_event_id&&!item.unread){setReadError('');setMarked(true);}
  },[data,readError,busy]);
  async function loadMore() {
    const before = tab === 'notifications' ? data?.notifications_next_before : data?.conversations_next_before;
    if (!before || more || busy || readAll.busy || snapshotLoading || replacing.current) return;
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
    <header className="inbox-panel-heading"><div className="inbox-panel-title"><h2>{t('收件箱', 'Inbox')}</h2>{messages && <a href={messagesHref()}>{t('打开私信', 'Open messages')}</a>}</div>{readAllAvailable && <InboxReadAll action={readAll}/>}</header>
    <div className="inbox-tabs" role="tablist" aria-label={t('消息类型', 'Inbox sections')} onKeyDown={event => {
      if(event.defaultPrevented||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229)return;
      navigateTabs(event);
    }}>
      {notifications && <button type="button" role="tab" id={`${panelId}-notifications`} aria-controls={panelId} tabIndex={tab === 'notifications' ? 0 : -1} aria-selected={tab === 'notifications'} onClick={() => setTab('notifications')}>{t('通知', 'Notifications')}{!!data?.notification_unread_count && <span>{data.notification_unread_count}</span>}</button>}
      {messages && <button type="button" role="tab" id={`${panelId}-messages`} aria-controls={panelId} tabIndex={tab === 'messages' ? 0 : -1} aria-selected={tab === 'messages'} onClick={() => setTab('messages')}>{t('私信', 'Messages')}{!!data?.message_unread_count && <span>{data.message_unread_count}</span>}</button>}
    </div>
    {data&&<InboxListRefresh changed={changed} busy={snapshotLoading||busy||more||readAll.busy} onRefresh={retry}/>}
    <span className="sr-only" role="status">{marked?t('通知已标为已读。','Notifications marked as read.'):null}</span>
    {error && <div className="inbox-error" role="alert"><p>{error}</p><button type="button" onClick={retry}>{t('重试', 'Retry')}</button></div>}
    {readError&&<div className="inbox-error inbox-notification-read-error" role="alert"><p>{readError}</p><button type="button" disabled={busy||readAll.busy||snapshotLoading||more} onClick={()=>void read(readAttempt.current?.notification,readAttempt.current?.cutoff)}>{t('重试标记已读','Retry mark as read')}</button></div>}
    {!data && !error && <p className="inbox-empty" role="status">{t('正在加载…', 'Loading…')}</p>}
    {data && <div className="inbox-tab-content" role="tabpanel" id={panelId} aria-labelledby={`${panelId}-${tab}`}>
      {tab === 'notifications' ? <>
        {!readAllAvailable && <button type="button" className="inbox-mark-read" disabled={busy || snapshotLoading || !data.notification_unread_count} onClick={() => void read()}><CheckCheck size={14}/>{busy ? t('正在标记…', 'Marking…') : t('将通知标为已读', 'Mark notifications as read')}</button>}
        {data.notifications.length ? <ul className="inbox-items">{data.notifications.map(item => <li key={item.id}><NotificationRow notification={item} onRead={() => { if (item.unread) void read(item); }} onMarkRead={()=>void read(item)} disabled={busy||readAll.busy||snapshotLoading||more} marking={readingId===item.id}/></li>)}</ul>
          : <div className="inbox-empty"><Bell size={26}/><p>{changed?t('当前列表没有通知','No notifications in this list'):t('还没有通知', 'No notifications yet')}</p><small>{t('回复、点赞和新关注会显示在这里。', 'Replies, likes and new followers will appear here.')}</small></div>}
      </> : data.conversations.length ? <ul className="inbox-items">{data.conversations.map(item => <li key={item.id}><a className={`inbox-item${item.unread_count ? ' unread' : ''}`} href={messagesHref(item.peer.username)}>
        <Avatar user={item.peer} size={36}/><span className="inbox-item-copy"><b>{item.peer.display_name}</b><span className="inbox-message-preview">{item.last_message.sender_id === owner ? t('你：', 'You: ') : ''}{item.last_message.body}</span><DateLabel value={item.last_message.created_at}/></span>
        {item.unread_count > 0 && <span className="inbox-count" aria-label={t(`${item.unread_count} 条未读`, `${item.unread_count} unread`)}>{item.unread_count}</span>}
      </a></li>)}</ul> : <div className="inbox-empty"><MessageCircle size={26}/><p>{changed?t('当前列表没有私信','No messages in this list'):t('还没有私信', 'No messages yet')}</p><small>{t('可以从对方的个人主页发起私信。', 'Start a conversation from someone’s profile.')}</small></div>}
      {(tab === 'notifications' ? data.notifications_next_before : data.conversations_next_before) && <button type="button" className="inbox-more" disabled={more || busy || readAll.busy || snapshotLoading} onClick={() => void loadMore()}>{more ? t('正在加载…', 'Loading…') : t('加载更多', 'Load more')}</button>}
    </div>}
  </>;
}

export function NotificationRow({notification: item, onRead,onMarkRead,disabled=false,marking=false}: {notification: InboxNotification; onRead: () => void;onMarkRead?:()=>void;disabled?:boolean;marking?:boolean}) {
  const {t} = useI18n(),rowId=useId();
  const actor = item.actors[0];
  const name = actor?.anonymous ? t('匿名用户', 'Anonymous') : actor?.display_name || t('一位用户', 'Someone');
  const people = item.actor_count > 1 ? t(`${name} 等 ${item.actor_count} 人`, `${name} and ${item.actor_count - 1} ${item.actor_count === 2 ? 'other' : 'others'}`) : name;
  const action = item.kind === 'like' ? t('赞了你的帖子', 'liked your post') : item.kind === 'follow' ? t('关注了你', 'followed you') : item.kind === 'repost' ? t('转发了你的帖子', 'reposted your post') : t('回复了你', 'replied to you');
  const Icon = item.kind === 'like' ? Heart : item.kind === 'follow' ? UserPlus : item.kind === 'repost' ? Repeat2 : MessageCircle;
  const href = item.post ? communityPostHref(item.post.id, undefined, item.reply_id || undefined) : actor && !actor.anonymous ? profileHref(actor) : '#/community';
  return <div className="inbox-notification-row"><a id={rowId} href={href} className={`inbox-item${item.unread ? ' unread' : ''}`} onClick={onRead}>
    <span className={`inbox-event-icon inbox-event-${item.kind}`}><Icon size={18}/></span>
    <span className="inbox-item-copy"><span><b>{people}</b>{' '}{action}</span>{item.post && <strong className="inbox-post-title">{item.post.title}</strong>}{item.preview && item.kind === 'reply' && <span className="inbox-message-preview">{item.preview}</span>}<DateLabel value={item.created_at}/></span>
    {item.unread && <span className="inbox-row-dot" aria-label={t('未读', 'Unread')}/>}
  </a>{onMarkRead&&<button type="button" className="inbox-row-read" aria-describedby={rowId} aria-disabled={disabled||!item.unread} aria-busy={marking} aria-label={marking?t('正在标记通知已读…','Marking notification read…'):item.unread?t('将此通知标为已读','Mark this notification as read'):t('此通知已读','This notification is read')} title={marking?t('正在标记…','Marking…'):item.unread?t('标为已读','Mark as read'):t('已读','Read')} onClick={()=>{if(!disabled&&item.unread)onMarkRead();}}><span aria-hidden="true">{marking?<span className="spinner"/>:<CheckCheck size={16}/>}</span></button>}</div>;
}
