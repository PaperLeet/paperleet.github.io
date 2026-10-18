import AppLink from './AppLink';
import {Check,Copy,Link2} from 'lucide-react';
import {useI18n} from './i18n';
import {communityPostHref,communityShareUrl} from './communityNavigation';
import {useClipboardCopy} from './useClipboardCopy';
import './discussion-link.css';

type Props={postId:string;replyId?:string;returnTo?:string};

export default function DiscussionLink(props:Props) {
  return <ScopedDiscussionLink key={JSON.stringify([props.postId,props.replyId,props.returnTo])} {...props}/>;
}

function ScopedDiscussionLink({postId,replyId,returnTo}:Props) {
  const {t}=useI18n(),{status,value,copy}=useClipboardCopy();
  let localReplyHref:string|undefined;
  try {if(replyId&&/^[a-f0-9]{32}$/.test(postId))localReplyHref=communityPostHref(postId,returnTo,replyId);}catch {/* Invalid targets cannot become local reply links. */}
  return <span className="discussion-share">
    {localReplyHref&&<AppLink href={localReplyHref} aria-label={t('定位这条回复','Link to this reply')}><Link2 size={15} aria-hidden="true"/>{t('定位回复','View reply')}</AppLink>}
    <button className="text-button discussion-copy" type="button" aria-disabled={status==='copying'} aria-busy={status==='copying'} onClick={()=>copy(()=>communityShareUrl(location.href,postId,replyId))}>
      {status==='copied'?<Check size={15} aria-hidden="true"/>:<Copy size={15} aria-hidden="true"/>}
      {status==='copied'?t('已复制链接','Link copied'):status==='copying'?t('正在复制…','Copying…'):replyId?t('复制回复链接','Copy reply link'):t('复制讨论链接','Copy discussion link')}
    </button>
    <span className="sr-only" role="status">{status==='copying'?t('正在复制链接…','Copying link…'):status==='copied'?t('链接已复制。','Link copied.'):status==='failed'&&value?t('自动复制不可用，请使用下方的手动复制链接。','Automatic copy is unavailable. Use the manual link below.'):null}</span>
    {status==='failed'&&(value?<label className="discussion-link-fallback">{t('自动复制不可用，请选择并复制链接。','Automatic copy is unavailable. Select and copy the link.')}<input aria-label={t('可手动复制的讨论链接','Discussion link to copy manually')} readOnly value={value} onFocus={event=>event.currentTarget.select()}/></label>:<span className="discussion-link-error" role="alert">{t('当前链接无法分享，请重新打开帖子后重试。','This link cannot be shared. Reopen the discussion and try again.')}</span>)}
  </span>;
}
