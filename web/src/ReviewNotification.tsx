import {useId} from 'react';
import {CheckCheck,FileCheck,Flag,ShieldCheck} from 'lucide-react';
import AppLink from './AppLink';
import {DateLabel} from './social';
import {useI18n} from './i18n';
import {reviewNotice} from './reviewNoticeModel';
import {reportHistoryHref} from './reportHistoryNavigation';
import type {InboxNotification} from './inbox';
import './review-notifications.css';

export default function ReviewNotification({notification:item,onRead,onMarkRead,disabled=false,marking=false}:{notification:InboxNotification;onRead:()=>void;onMarkRead?:()=>void;disabled?:boolean;marking?:boolean}){
 const {t}=useI18n(),id=useId(),notice=reviewNotice(item.review);
 const paper=notice?.kind==='paper_review',reported=notice?.kind==='report_result';
 const stage=notice?.phase==='candidate'?t(`候选版本 v${notice.version}`,`Candidate v${notice.version}`):t('投稿审核','Submission review');
 const title=!notice?t('审核通知暂不可用','Review notification unavailable'):paper?stage+' · '+(notice.decision==='approve'?t('审核通过','Approved'):notice.decision==='changes'?t('需要修改','Changes requested'):t('未通过','Not approved')):reported?t('举报已处理','Report reviewed'):t('帖子已移除','Post removed');
 const text=!notice?t('请刷新收件箱后重试。','Refresh your inbox and try again.'):paper?t('这是当时版本的审核结果，请打开投稿查看最新状态。','This decision records the reviewed version. Open the submission for its current status.'):reported?notice.decision==='remove'?t('你举报的帖子已被移除。','The post you reported was removed.'):t('审核后帖子予以保留。','The post was kept after review.'):t('你的帖子已由管理员移除。','Your post was removed by an administrator.');
 const href=notice?paper?'#/submit?proposal='+notice.resource_id:reported?reportHistoryHref('reviewed'):'#/community':undefined;
 const Icon=paper?FileCheck:reported?Flag:ShieldCheck;
 const content=<><span className="inbox-event-icon inbox-event-review"><Icon size={18} aria-hidden="true"/></span><span className="inbox-item-copy"><span className="inbox-review-sender">{t('PaperLeet 审核通知','PaperLeet review')}</span><b>{title}</b>{notice?.title&&<strong className="inbox-post-title">{notice.title}</strong>}<span className="inbox-review-description">{text}</span><DateLabel value={item.created_at}/></span>{item.unread&&<span className="inbox-row-dot" aria-label={t('未读','Unread')}/>}</>;
 return <div className="inbox-notification-row inbox-review-notification">{href?<AppLink id={id} href={href} onClick={onRead} className={`inbox-item${item.unread?' unread':''}`}>{content}</AppLink>:<div id={id} className={`inbox-item${item.unread?' unread':''}`}>{content}</div>}
  {notice?.feedback&&<details className="inbox-review-feedback"><summary>{t('查看审核意见','Read review feedback')}</summary><p>{notice.feedback}</p></details>}
  {onMarkRead&&<button type="button" className="inbox-row-read" aria-describedby={id} aria-disabled={disabled||!item.unread} aria-busy={marking} aria-label={item.unread?t('将此审核通知标为已读','Mark this review notification as read'):t('此审核通知已读','This review notification is read')} onClick={()=>{if(!disabled&&item.unread)onMarkRead();}}><CheckCheck size={16} aria-hidden="true"/></button>}
 </div>;
}
