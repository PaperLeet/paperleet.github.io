import {useId,useState} from 'react';
import {ArrowRight,RefreshCw} from 'lucide-react';
import {useI18n} from './i18n';
import {DateLabel,useAccount} from './social';
import {communityPostHref} from './communityNavigation';
import CommunityAuthor from './CommunityAuthor';
import PostBookmark from './PostBookmark';
import InterviewMeta from './InterviewMeta';
import {InlinePostBody} from './InlineCommentBody';
import CommunityCardActions from './CommunityCardActions';
import type {InterviewPost} from './interviewModels';
import type {Problem} from './types';
import InterviewReportPractice from './InterviewReportPractice';

/** The feed keys cards by account and request, so confirmations cannot cross either boundary. */
export default function InterviewReportCard({post,owner,canSave,returnTo,savedView,resumed,resumedPractice,problems,catalogError,onRetryCatalog,onReload,onAccountRefresh}:{post:InterviewPost;owner:string;canSave:boolean;returnTo:string;savedView:boolean;resumed?:boolean;resumedPractice?:string;problems?:Problem[];catalogError?:string;onRetryCatalog?:()=>void;onReload:()=>void;onAccountRefresh?:()=>Promise<void>}){
 const {t}=useI18n(),titleId=useId();
 const {loading:accountLoading}=useAccount();
 const [bookmarked,setBookmarked]=useState(post.viewer?.bookmarked),[busy,setBusy]=useState(false);
 const href=communityPostHref(post.id,returnTo);
 return <article data-reading-post={post.id} data-reading-resumed={resumed||undefined} className="interview-report-card" aria-labelledby={titleId}>
  <div className="interview-report-author"><CommunityAuthor user={post.author} compact/><DateLabel value={post.created_at}/></div>
  <InterviewMeta details={post.interview} compact/>
  <a data-reading-link="title" id={titleId} className="interview-report-title" href={href}>{post.title}</a><InlinePostBody postId={post.id} body={post.body}/>
  <InterviewReportPractice key={post.id} questions={post.interview.questions} revealPractice={resumedPractice} problems={problems} returnTo={returnTo} catalogError={catalogError} onRetryCatalog={onRetryCatalog}/>
  <CommunityCardActions post={post} returnTo={returnTo}>
   <div className="interview-report-actions">{canSave&&typeof bookmarked==='boolean'&&<PostBookmark postId={post.id} owner={owner} bookmarked={bookmarked} enabled={canSave} returnTo={href} disabled={accountLoading} hidden={accountLoading} onChange={setBookmarked} onBusyChange={setBusy} onAccountRefresh={onAccountRefresh}/>}<a data-reading-link="read" href={href}>{t('阅读复盘','Read experience')}<ArrowRight size={13}/></a></div>
  </CommunityCardActions>
  {savedView&&bookmarked===false&&!busy&&<div className="interview-unsaved-notice" role="status"><span>{t('已取消收藏。你仍可以重新收藏或继续阅读；刷新后将从列表移除。','Unsaved. You can save it again or keep reading; refreshing removes it from this list.')}</span><button type="button" onClick={onReload}>{t('刷新收藏列表','Refresh saved experiences')}<RefreshCw size={12}/></button></div>}
 </article>;
}
