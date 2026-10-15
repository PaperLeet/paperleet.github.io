import {useId,useState} from 'react';
import {ArrowRight,BookOpen,Heart,MessageCircle,RefreshCw} from 'lucide-react';
import {useI18n} from './i18n';
import {DateLabel,PersonLink} from './social';
import {communityPostHref} from './communityNavigation';
import PostBookmark from './PostBookmark';
import InterviewMeta from './InterviewMeta';
import type {InterviewPost} from './interviewModels';

/** The feed keys cards by account and request, so confirmations cannot cross either boundary. */
export default function InterviewReportCard({post,owner,canSave,returnTo,savedView,onReload,onAccountRefresh}:{post:InterviewPost;owner:string;canSave:boolean;returnTo:string;savedView:boolean;onReload:()=>void;onAccountRefresh?:()=>Promise<void>}){
 const {t}=useI18n(),titleId=useId();
 const [bookmarked,setBookmarked]=useState(post.viewer?.bookmarked),[busy,setBusy]=useState(false);
 const href=communityPostHref(post.id,returnTo);
 return <article className="interview-report-card" aria-labelledby={titleId}>
  <div className="interview-report-author"><PersonLink user={post.author} compact/><DateLabel value={post.created_at}/></div>
  <InterviewMeta details={post.interview} compact/>
  <a id={titleId} className="interview-report-title" href={href}>{post.title}</a><p>{post.body.replace(/[#*`>\[\]]/g,'').slice(0,210)}</p>
  <div className="interview-report-footer"><span><MessageCircle size={13}/>{post.counts.replies} {t('条讨论','replies')}</span><span><Heart size={13}/>{post.counts.likes}</span>{post.interview.questions.length>0&&<span><BookOpen size={13}/>{post.interview.questions.length} {t('个知识点','concepts')}</span>}</div>
  <div className="interview-report-actions">{canSave&&typeof bookmarked==='boolean'&&<PostBookmark postId={post.id} owner={owner} bookmarked={bookmarked} enabled={canSave} returnTo={href} disabled={false} onChange={setBookmarked} onBusyChange={setBusy} onAccountRefresh={onAccountRefresh}/>}<a href={href}>{t('阅读复盘','Read experience')}<ArrowRight size={13}/></a></div>
  {savedView&&bookmarked===false&&!busy&&<div className="interview-unsaved-notice" role="status"><span>{t('已取消收藏。你仍可以重新收藏或继续阅读；刷新后将从列表移除。','Unsaved. You can save it again or keep reading; refreshing removes it from this list.')}</span><button type="button" onClick={onReload}>{t('刷新收藏列表','Refresh saved experiences')}<RefreshCw size={12}/></button></div>}
 </article>;
}
