import {activeSourcePreview,sourcePreviewEscape} from './sourcePreviewKeyboard';
import {appLocation} from './appLocation';
import AppLink from './AppLink';
import {useEffect,useId,useLayoutEffect,useRef,useState} from 'react';
import {ArrowRight,ArrowUpRight,ChevronDown,ChevronUp,RefreshCw} from 'lucide-react';
import {useI18n} from './i18n';
import {DateLabel} from './social';
import {useInterviewResource} from './interviewApi';
import {communityPostHref} from './communityNavigation';
import {interviewHref,readInterviewRoute} from './interviewNavigation';
import {interviewSourceRequest,readInterviewSourcePreview,interviewPracticePreviewScope,libraryInterviewPreviewScope,libraryExercisePreviewScope,type LibraryExerciseSourceScope} from './interviewSourcePreviewModel';
import {InlinePostBody} from './InlineCommentBody';
import {keepReadingTriggerVisible} from './readingViewport';
import CommunityAuthor from './CommunityAuthor';
import InterviewMeta from './InterviewMeta';
import './interview-source-preview.css';

type Props={concept:string;name:string;reports:number;contributors:number;sourceHref:string;returnTo:string;owner:string;accountLoading?:boolean;contextAvailable?:boolean;practice?:{href:string;number?:number};library?:{href:string;exercise?:LibraryExerciseSourceScope};returnId?:number;onReturnReady?:(id:number)=>void;onReturnPending?:(id:number)=>void;onReturnCancel?:(id:number)=>void};
const currentScope=()=>{const filters=readInterviewRoute(appLocation.hash);return filters?interviewHref(filters):undefined;};
const inSourceScope=({library,practice,concept,sourceHref,returnTo}:Pick<Props,'library'|'practice'|'concept'|'sourceHref'|'returnTo'>)=>library?(library.exercise?(concept===library.exercise.concept&&libraryExercisePreviewScope(appLocation.hash,sourceHref,library.exercise)):libraryInterviewPreviewScope(appLocation.hash,sourceHref))===library.href:practice?interviewPracticePreviewScope(appLocation.hash,concept,practice.number)===practice.href:currentScope()===returnTo;

export default function InterviewSourcePreview(props:Props){
  return <SourcePreview key={JSON.stringify([props.owner,props.returnTo,props.sourceHref])} {...props}/>;
}

function SourcePreview(props:Props){
  const {t}=useI18n(),id=useId(),toggle=useRef<HTMLButtonElement>(null),panel=useRef<HTMLDivElement>(null);
  const [view,setView]=useState<{open:boolean;returned?:number}>({open:false});
  const reveal=props.returnId!==undefined&&view.returned!==props.returnId;
  const expanded=view.open||reveal;
  useLayoutEffect(()=>{if(reveal)setView({open:true,returned:props.returnId});},[reveal,props.returnId]);
  const close=()=>{
    if(!activeSourcePreview(panel.current)||!inSourceScope(props))return;
    if(panel.current?.contains(document.activeElement)&&toggle.current){toggle.current.focus({preventScroll:true});keepReadingTriggerVisible(toggle.current,100);}
    setView({open:false,returned:props.returnId??view.returned});
    if(props.returnId!==undefined)props.onReturnCancel?.(props.returnId);
  };
  return <section className="interview-source-preview interview-ranking-evidence">
    <div className="interview-source-toolbar"><span>{t('榜单统计：','Ranking evidence: ')}<b>{props.contributors}</b> {t('位分享者','contributors')} · <b>{props.reports}</b> {t('篇面经','reports')}</span><div>
      {props.reports>0&&<button ref={toggle} type="button" className="interview-source-preview-toggle" aria-expanded={expanded} aria-controls={id} aria-label={expanded?t(`收起${props.name}的来源预览`,`Collapse source preview for ${props.name}`):t(`预览${props.name}的来源面经`,`Preview source experiences for ${props.name}`)} onClick={event=>{if(activeSourcePreview(event.currentTarget)&&inSourceScope(props))expanded?close():setView({open:true,returned:view.returned});}}>{expanded?<ChevronUp size={14} aria-hidden="true"/>:<ChevronDown size={14} aria-hidden="true"/>}{expanded?t('收起预览','Collapse preview'):t('预览来源','Preview sources')}</button>}
      <AppLink href={props.sourceHref}>{t(`查看 ${props.reports} 篇来源`,`Read ${props.reports} source ${props.reports===1?'report':'reports'}`)}<ArrowRight size={13} aria-hidden="true"/></AppLink>
    </div></div>
    <div id={id} ref={panel} hidden={!expanded} className="interview-source-panel" role="region" aria-label={t(`${props.name}的来源预览`,`Source preview for ${props.name}`)} onKeyDown={event=>{
      if(!inSourceScope(props)||!sourcePreviewEscape(event))return;
      event.preventDefault();event.stopPropagation();close();
    }}>{expanded&&<><InterviewSourceReports key={`${props.owner}:${!!props.accountLoading}`} {...props}/><div className="interview-source-reading-footer"><button type="button" className="text-button" aria-controls={id} aria-expanded aria-label={t(`收起${props.name}的来源预览`,`Close source preview for ${props.name}`)} onClick={close}><ChevronUp size={14} aria-hidden="true"/>{t('收起来源预览','Close source preview')}</button></div></>}</div>
  </section>;
}

export function InterviewSourceReports({concept,sourceHref,returnTo,owner,accountLoading=false,contextAvailable=false,practice,library,returnId,onReturnReady,onReturnPending}:Props){
  const {t}=useI18n(),[,setRouteRevision]=useState(0),[retry,setRetry]=useState(0),busy=useRef(false);
  const inPractice=!!practice,newTab=!!practice||!!library;
  const path=interviewSourceRequest(sourceHref,contextAvailable),isHere=()=>inSourceScope({library,practice,concept,sourceHref,returnTo}),here=isHere();
  const resource=useInterviewResource<unknown>(path||'',!!path&&!accountLoading&&here,retry,JSON.stringify([owner,returnTo,practice?.href]));
  const feed=readInterviewSourcePreview(resource.data,sourceHref,contextAvailable),loading=accountLoading||resource.loading;
  const error=!path?t('来源筛选暂不可用，请查看全部来源。','Source filters are unavailable. Open all source reports.'):resource.error||(resource.data&&!feed?t('暂时无法确认来源面经，请刷新预览。','Could not confirm the source experiences. Refresh this preview.'):undefined);
  busy.current=loading;
  useEffect(()=>{if(inPractice)return;const changed=()=>setRouteRevision(value=>value+1);window.addEventListener('hashchange',changed);window.addEventListener('popstate',changed);return()=>{window.removeEventListener('hashchange',changed);window.removeEventListener('popstate',changed);};},[inPractice]);
  const returnCallbacks=useRef({onReturnReady,onReturnPending});returnCallbacks.current={onReturnReady,onReturnPending};
  useEffect(()=>{if(returnId===undefined||!here)return;if(loading)returnCallbacks.current.onReturnPending?.(returnId);else if(feed||error)returnCallbacks.current.onReturnReady?.(returnId);},[returnId,here,loading,feed,error]);
  if(!here)return null;
  return <>
    <div className="interview-source-panel-heading"><p>{t('当前知识点的公开来源，最多预览 3 篇。','Public source experiences for this concept, up to 3 reports.')} {library&&t('题库搜索、难度与学习状态只筛选练习，不筛选面经正文。','Library search, difficulty and progress select exercises, not report text.')} {(readInterviewRoute(returnTo)?.saved||readInterviewRoute(returnTo)?.q)&&t('收藏与知识点搜索仅用于原列表。','Saved and concept-search filters apply to the original list.')}</p><button type="button" className="text-button" aria-disabled={loading||!path} onClick={event=>{if(busy.current||!path||!isHere()||!activeSourcePreview(event.currentTarget))return;busy.current=true;setRetry(value=>value+1);}}><RefreshCw size={13} aria-hidden="true"/>{error?t('重试来源','Retry sources'):t('刷新来源','Refresh sources')}</button></div>
    <div aria-busy={loading}>
      {loading?<p className="interview-source-status" role="status"><span className="spinner"/>{accountLoading?t('正在确认账号…','Checking your account…'):t('正在读取来源面经…','Loading source experiences…')}</p>:error?<p className="interview-source-error" role="alert">{error}</p>:feed?.items.length?<><p className="interview-source-status" role="status">{t(`预览 ${feed.items.length} / ${feed.total} 篇公开面经`,`Previewing ${feed.items.length} of ${feed.total} public ${feed.total===1?'experience':'experiences'}`)}</p><div className="interview-source-items">{feed.items.map(post=>{
        return <article key={post.id} className="interview-source-item" data-reading-post={post.id} data-reading-concept={concept}><div className="interview-source-author"><CommunityAuthor user={post.author} compact/><DateLabel value={post.created_at}/></div><AppLink data-reading-link="title" className="interview-source-title" href={communityPostHref(post.id,returnTo)} onClick={event=>{if(!isHere()||!activeSourcePreview(event.currentTarget))event.preventDefault();}} target={newTab?'_blank':undefined} rel={newTab?'noopener noreferrer':undefined} aria-label={newTab?t(`${post.title} · 在新标签页阅读完整面经`,`${post.title} · Read the full experience in a new tab`):undefined}>{post.title}{newTab&&<ArrowUpRight size={13} aria-hidden="true"/>}</AppLink><div className="interview-source-excerpt" onKeyDown={event=>{if(event.key==='Escape'&&event.target instanceof HTMLElement&&event.target.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"])'))event.stopPropagation();}}><InlinePostBody postId={post.id} body={post.body} limit={140}/></div><InterviewMeta details={post.interview} compact browseFrom={sourceHref} newTab={!!library}/></article>;
      })}</div></>:<p className="interview-source-status" role="status">{t('当前预览没有匹配的面经，内容可能已修改或删除。仍可刷新预览、查看全部来源或继续练习。','No experiences match this preview. Content may have changed or been removed. Refresh, open all sources or keep practicing.')}</p>}
    </div>
  </>;
}
