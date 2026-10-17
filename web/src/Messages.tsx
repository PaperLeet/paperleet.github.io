import InboxListRefresh from './InboxListRefresh';
import {inboxListUpdate} from './inboxListUpdate';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowLeft, ArrowRight, MessageCircle, Send } from 'lucide-react';
import { Avatar, DateLabel, errorMessage, profileHref, SocialError, useAccount } from './social';
import { useI18n } from './i18n';
import { signInHref } from './accountNavigation';
import { applyInboxReadAll, inboxChanged, inboxRequest, inboxVisible, messagePath, messagesHref, newMessageRequestId, useInboxPolling, useInboxReadAll, type Conversation, type DirectMessage, type InboxCounts, type InboxSnapshot, type MessageThread } from './inbox';
import InboxReadAll from './InboxReadAll';
import './inbox.css';

export default function Messages({username}: {username?: string}) {
  const {user, loading, capabilities, error, refresh} = useAccount();
  const {t} = useI18n();
  if (loading) return <main className="messages-page"><p role="status">{t('正在检查账号…', 'Checking account…')}</p></main>;
  if (error) return <main className="messages-page"><div className="inbox-error" role="alert"><p>{error}</p><button type="button" onClick={() => void refresh()}>{t('重试', 'Retry')}</button></div></main>;
  if (!user) return <main className="messages-page">
    <a className="social-back messages-welcome-back" href="#/community"><ArrowLeft size={15}/>{t('返回社区', 'Back to community')}</a>
    <section className="messages-welcome" aria-labelledby="messages-welcome-title">
      <div className="messages-welcome-icon" aria-hidden="true"><MessageCircle size={34}/></div>
      <span className="messages-welcome-eyebrow">{t('社区交流', 'Community conversations')}</span>
      <h1 id="messages-welcome-title">{t('私信', 'Messages')}</h1>
      <p>{t('与社区成员讨论论文和实现。', 'Discuss papers and implementations with the community.')}</p>
      <a className="button primary" href={signInHref(messagesHref(username))}>{t('登录', 'Sign in')}<ArrowRight size={15} aria-hidden="true"/></a>
      <small>{username ? t('登录后回到当前对话。', 'Return to this conversation after signing in.') : t('登录后回到消息页。', 'Return here after signing in.')}</small>
    </section>
  </main>;
  if (!capabilities?.direct_messages) return <main className="messages-page"><h1>{t('私信', 'Messages')}</h1><p>{t('私信暂时不可用，请稍后再试。', 'Messages are currently unavailable. Please try again later.')}</p></main>;
  return <AccountMessages key={user.id} owner={user.id} username={username} readAllAvailable={capabilities.inbox_read_all === true}/>;
}

function AccountMessages({owner, username, readAllAvailable}: {owner: string; username?: string; readAllAvailable: boolean}) {
  const {t} = useI18n();
  const [data, setData] = useState<InboxSnapshot>(), [error, setError] = useState(''), [more, setMore] = useState(false), [snapshotLoading, setSnapshotLoading] = useState(true);
  const alive = useRef(true), manual = useRef<AbortController | undefined>(undefined), revision = useRef(0);
  const currentData=useRef(data),replacing=useRef(false);
  const [changed,setChanged]=useState(false);
  useLayoutEffect(()=>{currentData.current=data;},[data]);
  useEffect(() => { alive.current = true; return () => { alive.current = false; manual.current?.abort(); }; }, []);
  async function refresh(signal: AbortSignal,replace=false) {
    if(!replace&&replacing.current)return;
    const version = ++revision.current;
    if(!currentData.current||replace)setSnapshotLoading(true);
    try {
      const next = await inboxRequest<InboxSnapshot>(owner, 'inbox?limit=20', undefined, signal);
      if (signal.aborted || !alive.current || version !== revision.current) return;
      const current=currentData.current;
      if(current&&!replace){const update=inboxListUpdate(current,next);setData(update.snapshot);setChanged(update.changed);}
      else{setData(next);setChanged(false);}
      setError('');
    } catch (e) { if (!signal.aborted && alive.current && version === revision.current) setError(errorMessage(e)); }
    finally { if(replace&&!signal?.aborted)replacing.current=false; if (!signal?.aborted && alive.current && version === revision.current) setSnapshotLoading(false); }
  }
  const readAll = useInboxReadAll({owner, snapshot: data, available: readAllAvailable, blocked: snapshotLoading || more, onRead: (counts, cutoff) => {
    ++revision.current; replacing.current=false;manual.current?.abort(); setSnapshotLoading(false); setMore(false);
    setData(current => current ? applyInboxReadAll(current, counts, cutoff) : current);
  }});
  useInboxPolling(owner, refresh, !readAll.busy && !more);
  const retry = () => { manual.current?.abort(); manual.current = new AbortController(); setMore(false);replacing.current=true; void refresh(manual.current.signal,true); };
  async function loadMore() {
    if (!data?.conversations_next_before || more || readAll.busy || snapshotLoading || replacing.current) return;
    manual.current?.abort(); const controller = new AbortController(); manual.current = controller;
    const version = ++revision.current;
    setMore(true);
    try {
      const next = await inboxRequest<InboxSnapshot>(owner, `inbox?limit=20&conversations_before=${data.conversations_next_before}`, undefined, controller.signal);
      if (controller.signal.aborted || !alive.current || version !== revision.current) return;
      setData(current => current ? {...current, conversations: [...current.conversations, ...next.conversations.filter(item => !current.conversations.some(old => old.id === item.id))], conversations_next_before: next.conversations_next_before} : next);
      setError('');
    } catch (e) { if (!controller.signal.aborted && alive.current && version === revision.current) setError(errorMessage(e)); }
    finally { if (alive.current && version === revision.current) setMore(false); }
  }
  return <main className={`messages-page${username ? ' has-conversation' : ''}`}>
    <div className="messages-page-heading"><a className="social-back" href="#/community"><ArrowLeft size={15}/>{t('返回社区', 'Back to community')}</a><h1>{t('私信', 'Messages')}</h1></div>
    <div className="messages-layout">
      <aside className="messages-sidebar" aria-label={t('对话列表', 'Conversations')}>
        <header className="messages-sidebar-heading"><h2>{t('对话', 'Conversations')}</h2>{readAllAvailable && <InboxReadAll action={readAll}/>}</header>
        {data&&<InboxListRefresh changed={changed} busy={snapshotLoading||more||readAll.busy} onRefresh={retry}/>}
        {error && <div className="inbox-error" role="alert"><p>{error}</p><button type="button" onClick={retry}>{t('重试', 'Retry')}</button></div>}
        {!data && !error && <p className="inbox-empty" role="status">{t('正在加载…', 'Loading…')}</p>}
        {data && (data.conversations.length ? <ul className="inbox-items">{data.conversations.map(item => <li key={item.id}><ConversationLink conversation={item} owner={owner} selected={item.peer.username === username}/></li>)}</ul>
          : <p className="inbox-empty">{changed?t('当前列表没有对话，可查看最新消息。','No conversations in this list. Check the latest messages.'):t('还没有对话。前往对方的个人主页发起私信。', 'No conversations yet. Visit someone’s profile to send a message.')}</p>)}
        {data?.conversations_next_before && <button type="button" className="inbox-more" disabled={more || readAll.busy || snapshotLoading} onClick={() => void loadMore()}>{more ? t('正在加载…', 'Loading…') : t('更多对话', 'More conversations')}</button>}
      </aside>
      {username ? <ConversationThread key={username} owner={owner} username={username}/>
        : <section className="messages-placeholder"><MessageCircle size={36}/><h2>{t('选择一段对话', 'Choose a conversation')}</h2><p>{t('互相关注可以直接聊天。其他人在收到回复前只能发送一条消息。', 'Mutual followers can chat freely. Everyone else can send one message until they get a reply.')}</p></section>}
    </div>
  </main>;
}

function ConversationLink({conversation: item, owner, selected}: {conversation: Conversation; owner: string; selected: boolean}) {
  const {t} = useI18n();
  return <a className={`inbox-item${item.unread_count ? ' unread' : ''}`} href={messagesHref(item.peer.username)} aria-current={selected ? 'page' : undefined}>
    <Avatar user={item.peer} size={38}/><span className="inbox-item-copy"><b>{item.peer.display_name}</b><span className="inbox-message-preview">{item.last_message.sender_id === owner ? t('你：', 'You: ') : ''}{item.last_message.body}</span><DateLabel value={item.last_message.created_at}/></span>
    {item.unread_count > 0 && <span className="inbox-count" aria-label={t(`${item.unread_count} 条未读`, `${item.unread_count} unread`)}>{item.unread_count}</span>}
  </a>;
}

export function mergeMessages(previous: DirectMessage[], next: DirectMessage[]) {
  return [...new Map([...previous, ...next].map(message => [message.id, message])).values()].sort((a, b) => a.id - b.id);
}

function ConversationThread({owner, username}: {owner: string; username: string}) {
  const {t, locale} = useI18n();
  const [thread, setThread] = useState<MessageThread>(), [error, setError] = useState(''), [sendError, setSendError] = useState(''), [readError, setReadError] = useState('');
  const [draft, setDraft] = useState(''), [pending, setPending] = useState<{body: string; requestId: string}>(), [sending, setSending] = useState(false), [more, setMore] = useState(false);
  const alive = useRef(true), manual = useRef<AbortController | undefined>(undefined), sendController = useRef<AbortController | undefined>(undefined), readController = useRef<AbortController | undefined>(undefined);
  const revision = useRef(0), loadedOlder = useRef(false), lastRead = useRef(0), readingThrough = useRef(0), scroll = useRef<HTMLDivElement>(null), shouldScroll = useRef(true);
  const editor = useRef<HTMLTextAreaElement>(null);
  const [atLatest, setAtLatest] = useState(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; manual.current?.abort(); sendController.current?.abort(); readController.current?.abort(); }; }, []);
  function receive(next: MessageThread) {
    setThread(current => ({...next, messages: mergeMessages(current?.messages || [], next.messages), next_before: current && loadedOlder.current ? current.next_before : next.next_before}));
    if (pending && next.messages.some(message => message.sender_id === owner && message.request_id === pending.requestId)) {
      setPending(undefined); setDraft(''); setSendError('');
    }
  }
  async function refresh(signal: AbortSignal) {
    const version = ++revision.current;
    try {
      const next = await inboxRequest<MessageThread>(owner, `${messagePath(username)}?limit=50`, undefined, signal);
      if (signal.aborted || !alive.current || version !== revision.current) return;
      receive(next); setError(''); setMore(false);
    } catch (e) { if (!signal.aborted && alive.current && version === revision.current) setError(errorMessage(e)); }
  }
  useInboxPolling(owner, refresh, !sending);
  function retry() { manual.current?.abort(); manual.current = new AbortController(); setMore(false); void refresh(manual.current.signal); }
  async function readVisible(snapshot: MessageThread) {
    const box = scroll.current;
    if (!inboxVisible() || !box || box.clientHeight <= 0 || box.scrollHeight - box.scrollTop - box.clientHeight > 1 || !snapshot.unread_count || snapshot.observed_message_id <= lastRead.current) return;
    if (readController.current && !readController.current.signal.aborted && readingThrough.current >= snapshot.observed_message_id) return;
    readController.current?.abort(); const controller = new AbortController(); readController.current = controller;
    readingThrough.current = snapshot.observed_message_id;
    try {
      await inboxRequest<InboxCounts>(owner, `${messagePath(username)}/read`, {through_message_id: snapshot.observed_message_id}, controller.signal);
      if (controller.signal.aborted || !alive.current) return;
      lastRead.current = snapshot.observed_message_id; setReadError('');
      setThread(current => current && current.observed_message_id === snapshot.observed_message_id ? {...current, unread_count: 0} : current);
      inboxChanged(owner);
    } catch (e) { if (!controller.signal.aborted && alive.current) setReadError(errorMessage(e)); }
    finally { if (readController.current === controller) readController.current = undefined; }
  }
  useEffect(() => {
    if (!thread) return;
    if (shouldScroll.current && scroll.current) { scroll.current.scrollTop = scroll.current.scrollHeight; setAtLatest(true); }
    void readVisible(thread);
  }, [thread]);
  function historyScrolled() {
    const box = scroll.current;
    if (!box) return;
    const remaining = box.scrollHeight - box.scrollTop - box.clientHeight;
    shouldScroll.current = remaining < 80; setAtLatest(remaining <= 1);
    if (thread) void readVisible(thread);
  }
  function jumpToLatest() {
    const box = scroll.current;
    if (!box) return;
    shouldScroll.current = true; box.scrollTop = box.scrollHeight; setAtLatest(true);
    box.focus({preventScroll: true});
    if (thread) void readVisible(thread);
  }
  async function loadOlder() {
    if (!thread?.next_before || more) return;
    manual.current?.abort(); const controller = new AbortController(); manual.current = controller;
    const version = ++revision.current, height = scroll.current?.scrollHeight || 0;
    shouldScroll.current = false; setAtLatest(false); setMore(true);
    try {
      const next = await inboxRequest<MessageThread>(owner, `${messagePath(username)}?limit=50&before=${thread.next_before}`, undefined, controller.signal);
      if (controller.signal.aborted || !alive.current || version !== revision.current) return;
      loadedOlder.current = true;
      setThread(current => current ? {...current, messages: mergeMessages(next.messages, current.messages), next_before: next.next_before} : next);
      setError('');
      requestAnimationFrame(() => { if (alive.current && scroll.current) scroll.current.scrollTop += scroll.current.scrollHeight - height; });
    } catch (e) { if (!controller.signal.aborted && alive.current && version === revision.current) setError(errorMessage(e)); }
    finally { if (alive.current && version === revision.current) setMore(false); }
  }
  async function send() {
    if (!thread || (!pending && !thread.can_send) || sending || (!pending && (!draft.trim() || Array.from(draft.trim()).length > 2000))) return;
    const request = pending || {body: draft.trim(), requestId: newMessageRequestId()};
    setPending(request); setSending(true); setSendError('');
    const controller = new AbortController(); sendController.current = controller;
    manual.current?.abort(); setMore(false);
    ++revision.current;
    try {
      const next = await inboxRequest<MessageThread>(owner, messagePath(username), {body: request.body, request_id: request.requestId}, controller.signal);
      if (controller.signal.aborted || !alive.current) return;
      ++revision.current; shouldScroll.current = true; receive(next); setDraft(''); setPending(undefined); setError('');
      inboxChanged(owner); editor.current?.focus({preventScroll: true});
    } catch (e) {
      if (controller.signal.aborted || !alive.current) return;
      if (e instanceof SocialError && e.code === 'awaiting_reply') {
        setThread(current => current ? {...current, can_send: false, awaiting_reply: true} : current); setPending(undefined);
        setSendError(t('请等待对方回复后再继续发送。', 'Wait for a reply before sending another message.'));
      } else {
        // A rejected request can be edited. Uncertain delivery keeps its original retry key.
        if (e instanceof SocialError && ([400, 403, 404].includes(e.status) || e.code === 'message_changed')) setPending(undefined);
        setSendError(errorMessage(e));
      }
    } finally { if (!controller.signal.aborted && alive.current) setSending(false); }
  }
  const length = Array.from(draft).length;
  return <section className="message-thread" aria-label={t('私信对话', 'Private conversation')}>
    <header className="message-thread-heading"><a className="messages-back" href={messagesHref()} aria-label={t('返回对话列表', 'Back to conversations')}><ArrowLeft size={18}/></a>
      {thread ? <a className="message-peer" href={profileHref(thread.peer)}><Avatar user={thread.peer} size={36}/><span><b>{thread.peer.display_name}</b><small>@{thread.peer.username}</small></span></a> : <h2>@{username}</h2>}
    </header>
    {error && <div className="inbox-error" role="alert"><p>{error}</p><button type="button" onClick={retry}>{t('重试', 'Retry')}</button></div>}
    {readError && <div className="inbox-error" role="alert"><p>{t('未能标为已读。', 'Could not mark messages as read.')}</p><button type="button" onClick={() => thread && void readVisible(thread)}>{t('重试', 'Retry')}</button></div>}
    <div className="message-history" ref={scroll} onScroll={historyScrolled} role="region" aria-label={t('对话消息', 'Conversation messages')} tabIndex={0}>
      {!thread && !error && <p className="inbox-empty" role="status">{t('正在加载对话…', 'Loading conversation…')}</p>}
      {thread?.next_before && <button type="button" className="inbox-more" disabled={more} onClick={() => void loadOlder()}>{more ? t('正在加载…', 'Loading…') : t('加载更早的消息', 'Load earlier messages')}</button>}
      {thread?.messages.length === 0 && <div className="inbox-empty"><MessageCircle size={28}/><p>{t('发送第一条消息', 'Send the first message')}</p></div>}
      <ol className="message-list" aria-label={t('消息记录', 'Message history')} aria-live="polite" aria-relevant="additions">{thread?.messages.map(message => <li className={message.sender_id === owner ? 'outgoing' : 'incoming'} key={message.id}>
        <p>{message.body}</p><time dateTime={message.created_at}>{new Date(message.created_at).toLocaleString(locale === 'zh' ? 'zh-CN' : 'en-US', {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'})}</time>
      </li>)}</ol>
    </div>
    {thread && thread.messages.length > 0 && !atLatest && <div className="message-jump"><button type="button" onClick={jumpToLatest}><ArrowDown size={15} aria-hidden="true"/>{thread.unread_count > 0 ? t('有新消息 · 查看最新消息', 'New messages · Jump to latest') : t('回到最新消息', 'Back to latest messages')}</button></div>}
    {thread && <form className="message-composer" onSubmit={event => { event.preventDefault(); void send(); }}>
      {!thread.is_mutual_follow && <p className="message-permission" role="status">{thread.awaiting_reply ? t('消息已发出，等待对方回复后即可继续聊天。', 'Message sent. You can send more after they reply.') : thread.messages.length === 0 ? t('你们尚未互相关注。在对方回复前，只能发送一条消息。', 'You are not mutual followers. You can send one message until they reply.') : ''}</p>}
      <label className="sr-only" htmlFor="message-body">{t('消息内容', 'Message')}</label>
      <textarea ref={editor} id="message-body" value={draft} onChange={event => setDraft(event.target.value)} readOnly={!!pending} disabled={!thread.can_send || sending} rows={3} maxLength={4000} placeholder={thread.awaiting_reply ? t('等待回复…', 'Waiting for a reply…') : t('写一条消息…', 'Write a message…')} onKeyDown={event => {
        if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !event.nativeEvent.isComposing) { event.preventDefault(); void send(); }
      }}/>
      {sendError && <div className="inbox-error" role="alert"><p>{sendError}</p>{pending && <small>{t('消息可能已送达。重试会使用同一请求，不会重复发送。', 'The message may have arrived. Retrying uses the same request and will not send a duplicate.')}</small>}</div>}
      <footer><span className={length > 2000 ? 'message-too-long' : ''}>{length}/2000</span><button type="submit" className="button primary" disabled={(!pending && !thread.can_send) || sending || (!pending && (!draft.trim() || length > 2000))}><Send size={15}/>{sending ? t('正在发送…', 'Sending…') : pending ? t('重试发送', 'Retry send') : t('发送', 'Send')}</button></footer>
    </form>}
  </section>;
}
