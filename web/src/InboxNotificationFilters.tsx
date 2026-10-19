import {useId,useRef} from 'react';
import {ListFilter,X} from 'lucide-react';
import {useI18n} from './i18n';
import type {InboxNotificationKind} from './inbox';
import {isNotificationKind,notificationKinds} from './inboxNotificationQuery';
import './inbox-notification-filters.css';
const names:Record<InboxNotificationKind,[string,string]>={all:['全部通知','All notifications'],reply:['回复','Replies'],like:['点赞','Likes'],follow:['关注','Followers'],repost:['转发','Reposts'],review:['审核结果','Review outcomes']};
export default function InboxNotificationFilters({kind,reviews,disabled,controls,readAll,canAct,onChange}:{kind:InboxNotificationKind;reviews:boolean;disabled:boolean;controls:string;readAll:boolean;canAct:()=>boolean;onChange:(kind:InboxNotificationKind)=>void}){
 const {t}=useI18n(),id=useId(),select=useRef<HTMLSelectElement>(null),root=useRef<HTMLDivElement>(null);
 function active(){return canAct()&&!!root.current?.isConnected&&!root.current.closest('[hidden],[inert],[aria-hidden="true"]');}
 return <div ref={root} className="inbox-notification-filters">
  <div className="inbox-filter-controls"><label htmlFor={id}><ListFilter size={16} aria-hidden="true"/><span>{t('通知类型','Notification type')}</span></label><select id={id} ref={select} value={kind} disabled={disabled} aria-controls={controls} aria-describedby={kind!=='all'?id+'-scope':undefined} onChange={event=>{if(active()&&isNotificationKind(event.target.value))onChange(event.target.value);}} onKeyDown={event=>{if(active()&&event.key==='Escape'&&!event.defaultPrevented&&!event.repeat&&!event.nativeEvent.isComposing&&event.nativeEvent.keyCode!==229&&!event.altKey&&!event.ctrlKey&&!event.metaKey&&!event.shiftKey)event.stopPropagation();}}>{notificationKinds.filter(value=>value!=='review'||reviews).map(value=><option key={value} value={value}>{t(...names[value])}</option>)}</select>
   {kind!=='all'&&<button type="button" disabled={disabled} aria-label={t('清除通知类型筛选','Clear notification type filter')} onClick={()=>{if(disabled||!active())return;select.current?.focus({preventScroll:true});onChange('all');}}><X size={15} aria-hidden="true"/><span>{t('全部','All')}</span></button>}
  </div>
  {kind!=='all'&&<p id={id+'-scope'}>{readAll?t('筛选仅影响列表；未读计数与「全部标为已读」仍涵盖所有通知和私信。','This filters the list. Unread counts and “Mark all as read” still cover all notifications and messages.'):t('筛选仅影响列表；未读计数与「将通知标为已读」仍涵盖所有通知。','This filters the list. Unread counts and “Mark notifications as read” still cover all notifications.')}</p>}
 </div>;
}
