import {useId,useLayoutEffect,useRef,useState} from 'react';
import {ChevronDown,ChevronUp,Code2,RefreshCw} from 'lucide-react';
import {useI18n} from './i18n';
import {withLibraryContext} from './libraryNavigation';
import type {Problem} from './types';
import {topicFromProblem,type ProblemTopic} from './communityTopics';

/** An expansion is local reading state; it never fetches or starts a practice session. */
export default function InterviewReportPractice({questions,problems=[],topics,returnTo,catalogLoading,catalogError,onRetryCatalog,revealPractice}:{questions:string[];problems?:Problem[];topics?:ProblemTopic[];returnTo:string;catalogLoading?:boolean;catalogError?:string;onRetryCatalog?:()=>void;revealPractice?:string}){
 const {t,locale}=useI18n(),id=useId();
 const [view,setView]=useState<{expanded:boolean;revealed?:string}>({expanded:false});
 const list=useRef<HTMLDivElement>(null),toggle=useRef<HTMLButtonElement>(null);
 const pending=useRef<{href:string;opener:Element|null;kind:'expand'|'collapse'|'retry';keyboard:boolean;question?:string}|undefined>(undefined);
 const sources=topics||problems.map(topicFromProblem),waiting=catalogLoading??(!sources.length&&!catalogError);
 const retired=new Set(sources.filter(topic=>topic.retired).map(topic=>topic.id));
 const ids=[...new Set(questions)],byId=new Map(sources.filter(topic=>!topic.retired).map(topic=>[topic.id,topic]));
 const fromReturn=!!revealPractice&&view.revealed!==revealPractice&&ids.indexOf(revealPractice)>1;
 const expanded=fromReturn||view.expanded;
 // Keep a revealed origin mounted if the parent later yields to newer focus.
 useLayoutEffect(()=>{if(fromReturn)setView({expanded:true,revealed:revealPractice});},[fromReturn,revealPractice]);
 const shown=expanded?ids:ids.slice(0,2),missing=shown.some(id=>!byId.has(id));
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
    return <span key={question} data-practice-index={index} tabIndex={-1} className="interview-report-practice-item"><a data-reading-link="practice" data-reading-practice={question} href={withLibraryContext(problem.id,returnTo)} title={title}><Code2 size={13} aria-hidden="true"/><span>{title}</span></a></span>;
   })}
  </div>
  {ids.length>2&&<button ref={toggle} type="button" className="interview-report-practice-toggle" aria-expanded={expanded} aria-controls={id} onClick={event=>{pending.current={href:location.href,opener:document.activeElement,kind:expanded?'collapse':'expand',keyboard:event.detail===0};setView({expanded:!expanded,revealed:revealPractice||view.revealed});}}>{expanded?<ChevronUp size={13} aria-hidden="true"/>:<ChevronDown size={13} aria-hidden="true"/>}{expanded?t('收起配套练习','Collapse practice'):t(`再展开 ${ids.length-2} 个知识点`,`Show ${ids.length-2} more ${ids.length-2===1?'concept':'concepts'}`)}</button>}
  {missing&&catalogError&&<p className="interview-report-practice-retry" role="status">{t('部分练习信息暂未加载，面经和评论仍可阅读。','Some practice details could not load. The experience and comments remain readable.')} {onRetryCatalog&&<button type="button" onClick={event=>{pending.current={href:location.href,opener:event.currentTarget,kind:'retry',keyboard:false,question:shown.find(question=>!byId.has(question)&&!retired.has(question))||shown.find(question=>!byId.has(question))};onRetryCatalog();}}><RefreshCw size={12} aria-hidden="true"/>{t('重试练习信息','Retry practice details')}</button>}</p>}
 </div>;
}
