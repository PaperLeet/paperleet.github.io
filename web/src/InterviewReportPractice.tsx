import {useId,useLayoutEffect,useRef,useState} from 'react';
import {ArrowRight,CheckCircle2,ChevronDown,ChevronUp,Clock3,Code2,RefreshCw} from 'lucide-react';
import {useI18n} from './i18n';
import {withLibraryContext} from './libraryNavigation';
import type {Problem} from './types';
import {topicFromProblem,type ProblemTopic} from './communityTopics';
import {usePracticeProgress} from './PracticeProgressContext';
import {keepReadingTriggerVisible} from './readingViewport';

/** An expansion is local reading state; it never fetches or starts a practice session. */
export default function InterviewReportPractice({questions,problems=[],topics,returnTo,catalogLoading,catalogError,onRetryCatalog,revealPractice,revealRequest=0}:{questions:string[];revealRequest?:number;problems?:Problem[];topics?:ProblemTopic[];returnTo:string;catalogLoading?:boolean;catalogError?:string;onRetryCatalog?:()=>void;revealPractice?:string}){
 const {t,locale}=useI18n(),id=useId(),workspace=usePracticeProgress();
 const [view,setView]=useState<{expanded:boolean;revealed?:string}>({expanded:false});
 const [focusedNext,setFocusedNext]=useState<{owner:string;source:string;id:string}>();
 const progressRoot=useRef<HTMLDivElement>(null),nextLink=useRef<HTMLAnchorElement>(null);
 const progressRetry=useRef<{owner:string;source:string;href:string;opener:HTMLButtonElement}|undefined>(undefined);
 const list=useRef<HTMLDivElement>(null),toggle=useRef<HTMLButtonElement>(null);
 const pending=useRef<{href:string;opener:Element|null;kind:'expand'|'collapse'|'retry';keyboard:boolean;question?:string}|undefined>(undefined);
 const sources=topics||problems.map(topicFromProblem),waiting=catalogLoading??(!sources.length&&!catalogError);
 const retired=new Set(sources.filter(topic=>topic.retired).map(topic=>topic.id));
 const ids=[...new Set(questions)],byId=new Map(sources.filter(topic=>!topic.retired).map(topic=>[topic.id,topic]));
 const revealKey=revealPractice?revealPractice+':'+revealRequest:undefined,fromReturn=!!revealPractice&&view.revealed!==revealKey&&ids.indexOf(revealPractice)>1;
 const expanded=fromReturn||view.expanded;
 // Keep a revealed origin mounted if the parent later yields to newer focus.
 useLayoutEffect(()=>{if(fromReturn)setView({expanded:true,revealed:revealKey});},[fromReturn,revealKey]);
 const shown=expanded?ids:ids.slice(0,2),missing=shown.some(id=>!byId.has(id));
 const progress=ids.flatMap(id=>{const value=byId.has(id)?workspace?.read(id):undefined;return value?[{id,...value}]:[];});
 const passed=progress.filter(row=>row.state==='done').length,complete=progress.length===ids.length&&passed===ids.length;
 const started=progress.filter(row=>row.state==='started').sort((a,b)=>b.updated.localeCompare(a.updated));
 const sourceKey=ids.join(','),pinned=focusedNext?.owner===workspace?.owner&&focusedNext?.source===sourceKey?progress.find(row=>row.id===focusedNext.id):undefined;
 const next=pinned||started[0]||progress.find(row=>row.state==='todo')||(complete?progress[0]:undefined);
 const nextTopic=next?byId.get(next.id):undefined;
 const nextTitle=nextTopic?(locale==='en'?nextTopic.title_en||nextTopic.title:nextTopic.title):'';
 const stateLabel=(state:'todo'|'started'|'done')=>state==='done'?t('已通过','Passed'):state==='started'?t('进行中','In progress'):t('未开始','Not started');
 useLayoutEffect(()=>{
  if(focusedNext&&(focusedNext.owner!==workspace?.owner||focusedNext.source!==sourceKey||!workspace?.ready||document.activeElement!==nextLink.current))setFocusedNext(undefined);
 },[focusedNext,workspace?.owner,workspace?.ready,sourceKey]);
 useLayoutEffect(()=>{
  const intent=progressRetry.current;if(!intent)return;
  const active=document.activeElement;
  if(workspace?.owner!==intent.owner||sourceKey!==intent.source||location.href!==intent.href||active!==intent.opener&&!(active===document.body&&!intent.opener.isConnected)){progressRetry.current=undefined;return;}
  if(!workspace.ready)return;
  progressRetry.current=undefined;const target=nextLink.current||progressRoot.current;
  target?.focus({preventScroll:true});if(target)keepReadingTriggerVisible(target,100);
 });
 useLayoutEffect(()=>{
  const request=pending.current;
  if(!request)return;
  if(request.href!==location.href||document.activeElement!==request.opener&&!(document.activeElement===document.body&&!request.opener?.isConnected)){pending.current=undefined;return;}
  if(request.kind==='retry'){
   if(request.question&&(byId.has(request.question)||retired.has(request.question))){
    const row=list.current?.querySelector<HTMLElement>(`[data-practice-index="${ids.indexOf(request.question)}"]`);
    (row?.querySelector<HTMLElement>('a')||row)?.focus({preventScroll:true});pending.current=undefined;
   }return;
  }
  pending.current=undefined;
  if(request.kind==='expand'&&request.keyboard&&request.opener===toggle.current){
   const row=list.current?.querySelector<HTMLElement>('[data-practice-index="2"]');
   (row?.querySelector<HTMLElement>('a')||row)?.focus({preventScroll:true});
  }else if(request.kind==='collapse'&&!request.opener?.isConnected){toggle.current?.focus({preventScroll:true});}
 });
 if(!ids.length)return null;
 return <div className="interview-report-practice">
  <div id={id} ref={list} className="interview-report-topics" role="group" aria-label={t('面经配套练习','Practice linked to this experience')}>
   <b>{t('配套练习','Practice')}</b>{shown.map((question,index)=>{
    const problem=byId.get(question);
    if(!problem)return <span key={question} data-practice-index={index} tabIndex={-1} className="interview-report-practice-item interview-report-practice-missing"><code>{question}</code><small>{waiting&&!retired.has(question)?t('信息待加载','Details pending'):t('练习暂不可用','Practice unavailable')}</small></span>;
    const title=locale==='en'?problem.title_en||problem.title:problem.title;
    const state=progress.find(row=>row.id===question)?.state;
    return <span key={question} data-practice-index={index} tabIndex={-1} className="interview-report-practice-item"><a data-reading-link="practice" data-reading-practice={question} data-practice-state={state} href={withLibraryContext(problem.id,returnTo)} title={title}>{state==='done'?<CheckCircle2 size={13} aria-hidden="true"/>:state==='started'?<Clock3 size={13} aria-hidden="true"/>:<Code2 size={13} aria-hidden="true"/>}<span>{title}</span>{state&&<small className="interview-report-practice-state">{stateLabel(state)}</small>}</a></span>;
   })}
  </div>
  {ids.length>2&&<button ref={toggle} type="button" className="interview-report-practice-toggle" aria-expanded={expanded} aria-controls={id} onClick={event=>{pending.current={href:location.href,opener:document.activeElement,kind:expanded?'collapse':'expand',keyboard:event.detail===0};setView({expanded:!expanded,revealed:revealKey||view.revealed});}}>{expanded?<ChevronUp size={13} aria-hidden="true"/>:<ChevronDown size={13} aria-hidden="true"/>}{expanded?t('收起配套练习','Collapse practice'):t(`再展开 ${ids.length-2} 个知识点`,`Show ${ids.length-2} more ${ids.length-2===1?'concept':'concepts'}`)}</button>}
  {missing&&catalogError&&<p className="interview-report-practice-retry" role="status">{t('部分练习信息暂未加载，面经和评论仍可阅读。','Some practice details could not load. The experience and comments remain readable.')} {onRetryCatalog&&<button type="button" onClick={event=>{pending.current={href:location.href,opener:event.currentTarget,kind:'retry',keyboard:false,question:shown.find(question=>!byId.has(question)&&!retired.has(question))||shown.find(question=>!byId.has(question))};onRetryCatalog();}}><RefreshCw size={12} aria-hidden="true"/>{t('重试练习信息','Retry practice details')}</button>}</p>}
  {workspace&&<div ref={progressRoot} tabIndex={-1} className="interview-report-progress" aria-label={t('本篇面经的练习进度','Practice progress for this experience')}>
   <span className="interview-report-progress-label">{t('我的练习','My practice')}</span>
   {workspace.ready?progress.length>0?<>
    <span className="interview-report-progress-count" role="status">{t('已通过','Passed')} <b>{passed} / {progress.length}</b>{progress.length<ids.length&&<small>{t(`已加载的练习 · ${ids.length-progress.length} 道暂不可用`,`loaded concepts · ${ids.length-progress.length} unavailable`)}</small>}</span>
    <progress aria-label={t('已加载配套练习的完成情况','Completion of loaded linked practice')} value={passed} max={progress.length}/>
    {next&&nextTopic&&<a ref={nextLink} className="interview-report-practice-next" data-reading-link="practice" data-reading-practice={next.id} href={withLibraryContext(next.id,returnTo)} onFocus={()=>setFocusedNext({owner:workspace.owner,source:sourceKey,id:next.id})} onBlur={()=>setFocusedNext(undefined)}><span>{complete?t('复习这些知识点','Review these concepts'):next.state==='done'?t('复习这道题','Review this concept'):next.state==='started'?t('继续练习','Resume practice'):t('开始练习','Start practice')}</span><b>{nextTitle}</b><ArrowRight size={13} aria-hidden="true"/></a>}
   </>:<span role="status">{t('练习信息就绪后显示进度。','Progress appears when practice details are available.')}</span>:<>
    <span role="status">{workspace.failed?t('学习进度暂不可用，仍可打开练习。','Learning progress is unavailable. You can still open practice.'):t('正在读取学习进度…','Loading learning progress…')}</span>
    {workspace.failed&&workspace.retry&&<button type="button" className="text-button" onClick={event=>{progressRetry.current={owner:workspace.owner,source:sourceKey,href:location.href,opener:event.currentTarget};workspace.retry?.();}}><RefreshCw size={12} aria-hidden="true"/>{t('重试学习进度','Retry learning progress')}</button>}
   </>}
  </div>}
 </div>;
}
