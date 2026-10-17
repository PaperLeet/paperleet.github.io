import { useEffect, useId, useRef, useState } from 'react';
import { Bell,X } from 'lucide-react';
import { useAccount } from './social';
import { useI18n } from './i18n';
import { deferredFeature } from './deferredFeature';
import { inboxRequest, useInboxPolling, type InboxCounts } from './inbox';
import './inbox.css';

const InboxPanel = deferredFeature(() => import('./InboxPanel'), ['通知', 'notifications']);

export default function InboxBell() {
  const {user, loading, capabilities} = useAccount();
  if (!user || loading || (!capabilities?.inbox_notifications && !capabilities?.direct_messages)) return null;
  return <AccountInboxBell key={user.id} owner={user.id} notifications={capabilities.inbox_notifications === true} messages={capabilities.direct_messages === true}/>;
}

function AccountInboxBell({owner, notifications, messages}: {owner: string; notifications: boolean; messages: boolean}) {
  const {t} = useI18n();
  const [open, setOpen] = useState(false), [counts, setCounts] = useState<InboxCounts>();
  const root = useRef<HTMLDivElement>(null), trigger = useRef<HTMLButtonElement>(null), panel = useRef<HTMLElement>(null);
  const id = useId();
  useInboxPolling(owner, async signal => {
    try {
      const data = await inboxRequest<InboxCounts>(owner, 'inbox?summary=1', undefined, signal);
      if (!signal.aborted) setCounts(data);
    } catch { /* Keep the last confirmed count; the open panel has retry controls. */ }
  });
  const unread = (notifications ? counts?.notification_unread_count || 0 : 0) + (messages ? counts?.message_unread_count || 0 : 0);
  useEffect(() => {
    if (!open) return;
    const href=location.href,opener=document.activeElement;
    const frame = requestAnimationFrame(() => {
      if(location.href!==href||document.activeElement!==opener)return;
      panel.current?.focus({preventScroll:true});
    });
    const close = () => setOpen(false);
    const outside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) close(); };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented || event.repeat || event.isComposing || event.keyCode===229 || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || location.href!==href || !root.current?.contains(document.activeElement)) return;
      event.preventDefault(); close(); trigger.current?.focus({preventScroll: true});
    };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape);
    window.addEventListener('hashchange', close);
    return () => { cancelAnimationFrame(frame); document.removeEventListener('pointerdown', outside); document.removeEventListener('keydown', escape); window.removeEventListener('hashchange', close); };
  }, [open]);
  const label = unread ? t(`通知，${unread} 条未读`, `Notifications, ${unread} unread`) : t('通知', 'Notifications');
  return <div className="inbox-bell" ref={root} onBlur={event => {
    if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget as Node)) setOpen(false);
  }}>
    <button type="button" className="inbox-bell-trigger" ref={trigger} aria-haspopup="dialog" aria-expanded={open} aria-controls={id} aria-label={label} title={label}
      onClick={() => setOpen(value => !value)} onKeyDown={event => { if (event.key === 'ArrowDown'&&!event.defaultPrevented&&!event.repeat&&!event.nativeEvent.isComposing&&event.nativeEvent.keyCode!==229&&!event.altKey&&!event.ctrlKey&&!event.metaKey&&!event.shiftKey) { event.preventDefault(); setOpen(true); } }}>
      <Bell size={20}/>{unread > 0 && <span className="inbox-unread-dot" aria-hidden="true"/>}
    </button>
    {open && <section className="inbox-popover" id={id} ref={panel} tabIndex={-1} role="dialog" aria-label={t('通知与私信', 'Notifications and messages')}>
      <button type="button" className="inbox-close" aria-label={t('关闭收件箱','Close inbox')} onClick={()=>{setOpen(false);trigger.current?.focus({preventScroll:true});}}><X size={17} aria-hidden="true"/></button>
      <InboxPanel owner={owner} notifications={notifications} messages={messages} onCounts={setCounts}/>
    </section>}
  </div>;
}
