import AppLink from './AppLink';
import {useId,useLayoutEffect,useRef} from 'react';
import {ArrowRight,BookOpen,CheckCircle2,RefreshCw} from 'lucide-react';
import {useI18n} from './i18n';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import {interviewHref} from './interviewNavigation';
import {companyPreparationProgress} from './companyPreparationModel';
import type {InterviewSummary} from './interviewModels';
import type {InterviewFilters} from './interviewNavigation';
import type {Problem,Saved} from './types';
import PracticeMetadata from './PracticeMetadata';
import './company-preparation-progress.css';

export type CompanyPreparationProps={filters:InterviewFilters;summary?:InterviewSummary;summaryLoading?:boolean;summaryError?:string;problems:Problem[];saved:Record<string,Saved>;progressReady:boolean;progressError?:string;catalogError?:string;owner?:string;onRetrySummary?:()=>void;onRetryProgress?:()=>void;onRetryCatalog?:()=>void};
export default function CompanyPreparationProgress({filters,summary,summaryLoading,summaryError,problems,saved,progressReady,progressError,catalogError,owner='',onRetrySummary,onRetryProgress,onRetryCatalog}:CompanyPreparationProps){
 const {t,locale}=useI18n(),id=useId();
 const intent=useWorkspaceIntent(JSON.stringify([owner,interviewHref(filters)])),recovery=useRef<()=>boolean>(undefined),start=useRef<HTMLAnchorElement>(null),refresh=useRef<HTMLButtonElement>(null);
 const state=companyPreparationProgress(filters,summaryLoading||summaryError?undefined:summary,problems,saved,progressReady&&!progressError,!!catalogError);
 const resolved=state.phase==='ready',name=resolved?(locale==='en'?state.next.translations?.en?.title||state.next.title:state.next.title):'';
 function retry(action:()=>void){recovery.current=intent.create();action();}
 useLayoutEffect(()=>{if(summaryLoading||(!resolved&&state.phase!=='empty')||!recovery.current)return;const claim=recovery.current;recovery.current=undefined;if(claim())(start.current||refresh.current)?.focus({preventScroll:true});},[resolved,state.phase,summaryLoading,owner]);
 return <section className="company-preparation-progress" aria-labelledby={id} aria-busy={!!summaryLoading||state.phase==='progress'&&!progressError}>
  <div className="company-preparation-progress-heading"><BookOpen size={18} aria-hidden="true"/><h3 id={id}>{t('我的关联练习','My linked practice')}</h3>{resolved&&state.done===state.total&&<span className="company-preparation-complete"><CheckCircle2 size={15} aria-hidden="true"/>{t('全部通过','All passed')}</span>}{onRetrySummary&&<button ref={refresh} type="button" className="text-button" aria-disabled={!!summaryLoading} onClick={()=>{if(!summaryLoading)onRetrySummary();}}><RefreshCw size={14} aria-hidden="true"/>{!summaryLoading&&state.phase==='unavailable'?t('重试面经统计','Retry report counts'):t('刷新关联题目','Refresh linked exercises')}</button>}</div>
  {resolved?<>
   <div className="company-preparation-progress-body"><div><p className="company-preparation-progress-count"><strong>{state.done}<span> / {state.total}</span></strong><span>{t('道已通过','passed')}</span></p><progress value={state.done} max={state.total} aria-label={t('当前公司关联练习的通过进度','Passed progress for this company’s linked exercises')}/><p className="company-preparation-progress-states">{t(`${state.started} 道进行中 · ${state.todo} 道未开始`,`${state.started} in progress · ${state.todo} not started`)}</p></div><div className="company-preparation-next"><AppLink ref={start} className="button" data-company-practice="start" href={state.href}>{state.action==='review'?t('复习关联练习','Review linked practice'):state.action==='continue'?t('继续练习','Continue practice'):t('开始关联练习','Start linked practice')}<ArrowRight size={15} aria-hidden="true"/></AppLink><span>{name}</span><PracticeMetadata difficulty={state.next.difficulty} estimatedMinutes={state.next.estimated_minutes}/><AppLink data-company-practice="list" href={state.viewHref}>{t(`查看全部 ${state.total} 道关联题`,`See all ${state.total} linked exercises`)}</AppLink></div></div>
  </>:<div className="company-preparation-progress-status" role="status">
   <p>{summaryLoading?t('正在确认关联练习…','Checking linked exercises…'):state.phase==='empty'?t('当前面经范围还没有关联题目，可以先阅读面经或查看下方推荐路径。','No exercises are linked in this report view yet. Read experiences or explore the preparation paths below.'):state.phase==='catalog'?catalogError?t('题库暂不可用，个人练习进度尚未确认。','The catalog is unavailable. Personal practice progress is unconfirmed.'):t(`已载入 ${state.loaded} / ${state.total} 道关联题，暂不汇总完成情况。`,`Loaded ${state.loaded} of ${state.total} linked exercises. Completion is unconfirmed.`):state.phase==='progress'?progressError?t('个人练习进度暂不可用，恢复后可继续上次的练习。','Personal practice progress is unavailable. Retry to resume your practice.'):t('正在同步个人练习进度…','Syncing your practice progress…'):t('暂时无法确认关联练习，请重试面经统计。','Linked exercises could not be confirmed. Retry the report counts.')}</p>
   {!summaryLoading&&(state.phase==='catalog'?onRetryCatalog&&<button type="button" className="text-button" onClick={()=>retry(onRetryCatalog)}>{t('重试题库','Retry catalog')}</button>:state.phase==='progress'&&progressError&&onRetryProgress&&<button type="button" className="text-button" onClick={()=>retry(onRetryProgress)}>{t('重试个人进度','Retry my progress')}</button>)}
  </div>}
  <p className="company-preparation-progress-scope">{t('按当前公司与面经筛选汇总；正文搜索、面经收藏不影响题目范围，提及不代表实际考频。','Uses this company and report filters. Text search and saved reports do not change the practice scope; mentions are not measured frequency.')}</p>
 </section>;
}
