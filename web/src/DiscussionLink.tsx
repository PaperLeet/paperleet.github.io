import { useEffect, useRef, useState } from 'react';
import { useI18n } from './i18n';
import { communityPostHref, communityShareUrl } from './communityNavigation';

export default function DiscussionLink({postId,replyId,returnTo}:{postId:string;replyId?:string;returnTo?:string}) {
  const {t}=useI18n();
  const [status,setStatus]=useState<''|'copying'|'copied'|'failed'>('');
  const [fallback,setFallback]=useState('');
  const mounted=useRef(true);
  useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;};},[]);
  async function copy() {
    const url=communityShareUrl(location.href,postId,replyId);
    setFallback(url);setStatus('copying');
    try {await navigator.clipboard.writeText(url);if(mounted.current)setStatus('copied');}
    catch {if(mounted.current)setStatus('failed');}
  }
  return <span className="discussion-share">
    {replyId&&<a href={communityPostHref(postId,returnTo,replyId)} aria-label={t('定位这条回复','Link to this reply')}>{t('链接','Link')}</a>}
    <button className="text-button" type="button" disabled={status==='copying'} onClick={()=>void copy()}>{status==='copied'?t('已复制链接','Link copied'):replyId?t('复制回复链接','Copy reply link'):t('复制讨论链接','Copy discussion link')}</button>
    {status==='copied'&&<span className="sr-only" role="status">{t('链接已复制。','Link copied.')}</span>}
    {status==='failed'&&<label className="discussion-link-fallback">{t('自动复制不可用，请选择并复制链接。','Automatic copy is unavailable. Select and copy the link.')}<input aria-label={t('可手动复制的讨论链接','Discussion link to copy manually')} readOnly value={fallback} onFocus={event=>event.currentTarget.select()}/></label>}
  </span>;
}
