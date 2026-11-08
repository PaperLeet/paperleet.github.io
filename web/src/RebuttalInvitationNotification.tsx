import {useId} from 'react';
import {CheckCheck,UserPlus} from 'lucide-react';
import AppLink from './AppLink';
import {DateLabel} from './social';
import {useI18n} from './i18n';
import type {InboxNotification} from './inbox';

type Notice={id:string;paper_id:string;title:string;venue:string;role:'viewer'|'editor';status:'pending'|'accepted'|'declined'|'cancelled'};
export function rebuttalInvitationNotice(value:unknown):Notice|undefined{
 if(!value||typeof value!=='object')return;
 const item=value as Record<string,unknown>;
 if(typeof item.id!=='string'||!/^[a-zA-Z0-9_-]{1,150}$/.test(item.id)||typeof item.paper_id!=='string'||!/^[a-zA-Z0-9_-]{1,150}$/.test(item.paper_id)
  ||typeof item.title!=='string'||[...item.title].length>240||typeof item.venue!=='string'||[...item.venue].length>120
  ||typeof item.role!=='string'||!['viewer','editor'].includes(item.role)||typeof item.status!=='string'||!['pending','accepted','declined','cancelled'].includes(item.status))return;
 return item as Notice;
}
export default function RebuttalInvitationNotification({notification:item,onRead,onMarkRead,disabled=false,marking=false}:{notification:InboxNotification;onRead:()=>void;onMarkRead?:()=>void;disabled?:boolean;marking?:boolean}){
 const {t}=useI18n(),id=useId(),notice=rebuttalInvitationNotice(item.rebuttal_invitation);
 const statuses={pending:t('待接受','Pending'),accepted:t('已接受','Accepted'),declined:t('已拒绝','Declined'),cancelled:t('已撤销','Cancelled')};
 const content=<><span className="inbox-event-icon"><UserPlus size={18} aria-hidden="true"/></span><span className="inbox-item-copy"><span>{notice?<><b>{item.actors[0]?.display_name||t('论文所有者','Paper owner')}</b>{t(' 邀请你协作',' invited you to collaborate')}</>:t('论文邀请暂不可用','Paper invitation unavailable')}</span>{notice&&<><strong className="inbox-post-title">{notice.title}</strong><span>{notice.role==='editor'?t('可编辑','Editor'):t('只读','Viewer')} · {statuses[notice.status]}</span></>}<DateLabel value={item.created_at}/></span>{item.unread&&<span className="inbox-row-dot" aria-label={t('未读','Unread')}/>}</>;
 return <div className="inbox-notification-row inbox-rebuttal-invitation">{notice?<AppLink id={id} href="#/rebuttal/invitations" onClick={onRead} className={`inbox-item${item.unread?' unread':''}`}>{content}</AppLink>:<div id={id} className="inbox-item">{content}</div>}{onMarkRead&&<button type="button" className="inbox-row-read" aria-describedby={id} aria-disabled={disabled||!item.unread} aria-busy={marking} aria-label={t('将此邀请通知标为已读','Mark this invitation notification as read')} onClick={()=>{if(!disabled&&item.unread)onMarkRead();}}><CheckCheck size={16} aria-hidden="true"/></button>}</div>;
}
