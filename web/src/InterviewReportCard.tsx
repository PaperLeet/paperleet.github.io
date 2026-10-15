import {useId,useState} from 'react';
import {ArrowRight,BookOpen,Code2,RefreshCw} from 'lucide-react';
import {useI18n} from './i18n';
import {DateLabel} from './social';
import {communityPostHref} from './communityNavigation';
import CommunityAuthor from './CommunityAuthor';
import PostBookmark from './PostBookmark';
import InterviewMeta from './InterviewMeta';
import PostPreview from './PostPreview';
import CommunityCardActions from './CommunityCardActions';
import type {InterviewPost} from './interviewModels';
import type {Problem} from './types';
import {withLibraryContext} from './libraryNavigation';

/** The feed keys cards by account and request, so confirmations cannot cross either boundary. */
export default function InterviewReportCard({post,owner,canSave,returnTo,savedView,resumed,problems,onReload,onAccountRefresh}:{post:InterviewPost;owner:string;canSave:boolean;returnTo:string;savedView:boolean;resumed?:boolean;problems?:Problem[];onReload:()=>void;onAccountRefresh?:()=>Promise<void>}){
 const {t,locale}=useI18n(),titleId=useId();
 const [bookmarked,setBookmarked]=useState(post.viewer?.bookmarked),[busy,setBusy]=useState(false);
 const href=communityPostHref(post.id,returnTo);
 const related=post.interview.questions.map(id=>problems?.find(problem=>problem.id===id)).filter((problem):problem is Problem=>!!problem).slice(0,2);
 return <article data-reading-post={post.id} data-reading-resumed={resumed||undefined} className="interview-report-card" aria-labelledby={titleId}>
  <div className="interview-report-author"><CommunityAuthor user={post.author} compact/><DateLabel value={post.created_at}/></div>
  <InterviewMeta details={post.interview} compact/>
  <a data-reading-link="title" id={titleId} className="interview-report-title" href={href}>{post.title}</a><PostPreview body={post.body}/>
  {related.length>0&&<div className="interview-report-topics" aria-label={t('面经配套练习','Practice linked to this experience')}><b>{t('配套练习','Practice')}</b>{related.map(problem=>{const title=locale==='en'?problem.translations?.en?.title||problem.title:problem.title;return <a key={problem.id} data-reading-link="practice" data-reading-practice={problem.id} href={withLibraryContext(problem.id,returnTo)} title={title}><Code2 size={13}/><span>{title}</span></a>;})}{post.interview.questions.length>related.length&&<a href={href}>+{post.interview.questions.length-related.length} {t('个知识点','concepts')}</a>}</div>}
  <CommunityCardActions post={post} returnTo={returnTo}>
   <div className="interview-report-actions">{post.interview.questions.length>0&&!related.length&&<span className="interview-report-concepts"><BookOpen size={13}/>{post.interview.questions.length} {t('个知识点','concepts')}</span>}{canSave&&typeof bookmarked==='boolean'&&<PostBookmark postId={post.id} owner={owner} bookmarked={bookmarked} enabled={canSave} returnTo={href} disabled={false} onChange={setBookmarked} onBusyChange={setBusy} onAccountRefresh={onAccountRefresh}/>}<a data-reading-link="read" href={href}>{t('阅读复盘','Read experience')}<ArrowRight size={13}/></a></div>
  </CommunityCardActions>
  {savedView&&bookmarked===false&&!busy&&<div className="interview-unsaved-notice" role="status"><span>{t('已取消收藏。你仍可以重新收藏或继续阅读；刷新后将从列表移除。','Unsaved. You can save it again or keep reading; refreshing removes it from this list.')}</span><button type="button" onClick={onReload}>{t('刷新收藏列表','Refresh saved experiences')}<RefreshCw size={12}/></button></div>}
 </article>;
}
