import AppLink from './AppLink';
import {ArrowRight,BookOpen,CheckCircle2,Clock3} from 'lucide-react';
import {useI18n} from './i18n';
import {interviewHref,type InterviewFilters} from './interviewNavigation';
import {preparationProgress,rolePreparationPaths} from './interviewPreparation';
import {withLibraryContext} from './libraryNavigation';
import {status,type Problem,type Saved} from './types';

export default function InterviewStudyPaths({filters,problems,saved,ready:loaded,error,onRetry}:{filters:InterviewFilters;problems:Problem[];saved:Record<string,Saved>;ready:boolean;error?:string;onRetry?:()=>void}){
 const {t,locale}=useI18n(),en=locale==='en';
 const ready=loaded&&!error;
 // Lead with the requested path, including when revisiting under another role.
 const available=rolePreparationPaths(filters.role);
 const selected=preparationProgress(filters.track,problems)?.path;
 const paths=selected?[selected,...available.filter(path=>path.id!==selected.id)]:available;
 return <>
  <p className="interview-progress-context">{t('进度沿用当前学习工作区，通过测试后自动更新。','Progress follows your current learning workspace and updates after passing tests.')}</p>
  {!ready&&<p className="interview-progress-status" role="status">{error?t('学习进度暂不可用，仍可打开题目练习。','Learning progress is unavailable. You can still open practice modules.'):t('正在读取学习进度…','Loading learning progress…')}{error&&onRetry&&<button type="button" className="text-button" onClick={onRetry}>{t('重试学习进度','Retry learning progress')}</button>}</p>}
  <div className="interview-path-grid">{paths.map(path=>{
   const progress=preparationProgress(path.id,problems,saved)!;
   const active=filters.track===path.id;
   const returnTo=interviewHref({...filters,track:path.id});
   const title=(problem:Problem)=>en?problem.translations?.en?.title||problem.title:problem.title;
   const links=progress.modules.map(problem=>{
    const state=ready?status(problem,saved[problem.id]):undefined;
    return <AppLink key={problem.id} href={withLibraryContext(problem.number,returnTo)}><span className="interview-path-module-title">{title(problem)}</span><span className={`interview-path-module-state ${state||''}`} aria-label={!ready?t('进度未就绪','Progress not ready'):state==='done'?t('已通过','Passed'):state==='started'?t('进行中','In progress'):t('未开始','Not started')}>{state==='done'?<CheckCircle2 size={13}/>:state==='started'?<Clock3 size={13}/>:<ArrowRight size={12}/>}</span></AppLink>;
   });
   return <article key={path.id} data-selected={active||undefined} aria-labelledby={`interview-path-${path.id}`}>
    <span className="interview-path-icon"><BookOpen size={19}/></span>{active&&<span className="interview-path-selected">{t('当前路径','Current path')}</span>}
    <h3 id={`interview-path-${path.id}`}>{en?path.name_en:path.name}</h3><p>{en?path.description_en:path.description}</p>
    <div className="interview-path-progress" aria-busy={!ready}>{ready?<><span>{t('已通过','Passed')} <b>{progress.completed} / {progress.total}</b></span><progress aria-label={t(`${path.name}进度`,`${path.name_en} progress`)} value={progress.completed} max={progress.total}/></>:<span>{t('等待学习进度','Waiting for progress')}</span>}</div>
    {ready&&progress.next?<AppLink className="interview-path-start button subtle" href={withLibraryContext(progress.next.number,returnTo)}><span>{status(progress.next,saved[progress.next.id])==='started'?t('继续准备','Resume preparation'):t('开始下一题','Start next module')}<small>{title(progress.next)}</small></span><ArrowRight size={15}/></AppLink>:ready&&progress.completed===progress.total?<AppLink className="interview-path-start button subtle" href={interviewHref({...filters,track:path.id,view:'reports',page:1})}><CheckCircle2 size={15}/>{t('路径已完成 · 返回复盘','Path complete · review experiences')}</AppLink>:null}
    <div className="interview-path-modules">{links.slice(0,3)}{links.length>3&&<details className="interview-path-module-disclosure" open={active}><summary><span className="interview-path-module-expand-label">{t(`再看 ${links.length-3} 道练习`,`Show ${links.length-3} more modules`)}</span><span className="interview-path-module-collapse-label">{t('收起练习列表','Show fewer modules')}</span><span className="interview-path-module-chevron" aria-hidden="true"><ArrowRight size={13}/></span></summary><div className="interview-path-module-extra">{links.slice(3)}</div></details>}</div>
    {progress.missing>0&&<small role="status">{t(`${progress.missing} 道练习暂不可用；补全题库后再确认路径完成。`,`${progress.missing} ${progress.missing===1?'module is':'modules are'} unavailable. Completion waits for the full catalog.`)}</small>}
   </article>;
  })}</div>
 </>;
}
