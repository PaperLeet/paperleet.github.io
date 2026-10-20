import {useId,useRef} from 'react';
import {BookOpen,ChevronDown,ExternalLink} from 'lucide-react';
import AppLink from './AppLink';
import {useI18n} from './i18n';
import {controlIsActive} from './activeControls';
import {withLibraryContext} from './libraryNavigation';
import {resolveResourcePractice} from './resourcePractice';
import type {Problem} from './types';

export default function ResourceConceptPractice({resource,problems,returnTo,canAct,error,onRetry}:{resource:string;problems:Problem[];returnTo?:string;canAct:()=>boolean;error?:string;onRetry?:()=>void}){
 const {t,locale}=useI18n(),id=useId(),details=useRef<HTMLDetailsElement>(null);
 const practice=resolveResourcePractice(resource,problems);
 if(!practice||!returnTo)return null;
 const current=()=>canAct()&&controlIsActive(details.current)&&details.current?.open===true;
 return <details ref={details} className="resource-concept-practice">
  <summary onClick={event=>{if(!canAct())event.preventDefault();}}><BookOpen size={15} aria-hidden="true"/><span>{t('相关概念练习','Related concept practice')}</span><small>{practice.missing?`${practice.modules.length}/${practice.total}`:practice.total}</small><ChevronDown size={15} aria-hidden="true"/></summary>
  <div className="resource-concept-body">
   <p id={`${id}-note`}>{locale==='en'?practice.en:practice.zh}</p>
   <small id={`${id}-context`}>{t('平台整理 · 新标签页打开，保留当前资料页','Curated by PaperLeet · opens a new tab, keeping this resource page')}</small>
   {practice.modules.map(problem=>{
    const title=locale==='en'?problem.translations?.en?.title||problem.title:problem.title;
    const difficulty=({E:t('简单','Easy'),M:t('中等','Medium'),H:t('困难','Hard')} as Record<string,string>)[problem.difficulty];
    const minutes=typeof problem.estimated_minutes==='number'&&Number.isFinite(problem.estimated_minutes)&&problem.estimated_minutes>0?problem.estimated_minutes:undefined;
    const meta=[difficulty,minutes?t(`预计 ${minutes} 分钟`,`About ${minutes} min`):''].filter(Boolean).join(' · '),metaId=`${id}-${problem.number}-meta`;
    return <AppLink key={problem.id} data-resource-practice={problem.id} href={withLibraryContext(problem.number,returnTo)} target="_blank" rel="noopener noreferrer" aria-label={t(`${problem.number}. ${title}（新标签页打开）`,`${problem.number}. ${title} (opens in a new tab)`)} aria-describedby={`${id}-note ${id}-context${meta?' '+metaId:''}`} onClick={event=>{if(!current())event.preventDefault();}}><span><span><b>{problem.number}.</b> {title}</span>{meta&&<small id={metaId} className="resource-practice-meta">{meta}</small>}</span><ExternalLink size={14} aria-hidden="true"/></AppLink>;
   })}
   {practice.missing>0&&<div className="resource-practice-unavailable"><p>{error?t('相关练习目录暂时无法读取，官方资料仍可打开。','Related exercises are unavailable. You can still open the official resource.'):!problems.length?t('相关练习会在题库目录加载后显示。','Related exercises appear when the catalog finishes loading.'):t(`${practice.missing} 道相关练习尚未加载，已保留当前资料范围。`,`${practice.missing} related ${practice.missing===1?'exercise is':'exercises are'} not loaded. Your resource view is kept.`)}</p>{onRetry&&<button type="button" className="text-button" onClick={()=>{if(current())onRetry();}}>{t('重试练习目录','Retry exercise catalog')}</button>}</div>}
  </div>
 </details>;
}
