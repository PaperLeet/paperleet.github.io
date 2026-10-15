import {ArrowRight,CheckCircle2,Clock3,Code2,RefreshCw} from 'lucide-react';
import {useI18n} from './i18n';
import {interviewHref,type InterviewFilters} from './interviewNavigation';
import {withLibraryContext} from './libraryNavigation';
import {status,type Problem,type Saved} from './types';
import {hasPreparationGuide} from './preparationGuides';
import {openPreparationHandbook} from './interviewPreparation';

export default function InterviewConceptContext({filters,problems,saved,ready,error,catalogError,onRetryCatalog,onRetryProgress}:{filters:InterviewFilters;problems:Problem[];saved:Record<string,Saved>;ready:boolean;error?:string;catalogError?:string;onRetryCatalog?:()=>void;onRetryProgress?:()=>void}){
 const {t,locale}=useI18n();
 if(!filters.concept)return null;
 const problem=problems.find(problem=>problem.id===filters.concept);
 const clear=interviewHref({...filters,concept:'',page:1});
 if(!problem)return <section className="interview-concept-context" aria-labelledby="interview-concept-heading">
  <div className="interview-concept-copy"><span className="interview-panel-kicker">{t('关联练习','LINKED PRACTICE')}</span><h2 id="interview-concept-heading">{catalogError?t('练习信息暂不可用','Practice details are unavailable'):!problems.length?t('正在加载关联练习…','Loading linked practice…'):t('这道练习已调整','This practice module has changed')}</h2><p role="status">{t('仍可阅读当前知识点关联的面经，或移除筛选继续浏览。','You can still read reports linked to this concept, or remove the filter to keep browsing.')}</p><code>{filters.concept}</code></div>
  <div className="interview-concept-actions">{catalogError&&onRetryCatalog&&<button type="button" className="button subtle" onClick={onRetryCatalog}><RefreshCw size={14}/>{t('重试加载练习','Retry practice details')}</button>}<a className="text-button" href={clear}>{t('移除知识点筛选','Remove concept filter')}</a></div>
 </section>;
 const resolved=ready&&!error,state=resolved?status(problem,saved[problem.id]):undefined;
 const label=state==='done'?t('已通过','Passed'):state==='started'?t('进行中','In progress'):state==='todo'?t('未开始','Not started'):error?t('学习进度暂不可用','Learning progress unavailable'):t('正在读取学习进度…','Loading learning progress…');
 return <section className="interview-concept-context" aria-labelledby="interview-concept-heading">
  <div className="interview-concept-copy"><span className="interview-panel-kicker"><Code2 size={14}/>{t(`关联练习 · #${problem.number}`,`LINKED PRACTICE · #${problem.number}`)}</span><h2 id="interview-concept-heading">{locale==='en'?problem.translations?.en?.title||problem.title:problem.title}</h2><p>{t(problem.module.goal,problem.module.goal_en)}</p>{catalogError&&<small className="interview-concept-cached" role="status">{t('正在显示已加载的练习信息。','Showing the practice details already loaded.')}</small>}<div className="interview-concept-meta"><span className={`interview-concept-state ${state||''}`} role="status">{state==='done'?<CheckCircle2 size={13}/>:<Clock3 size={13}/>}<span>{label}</span></span><span>{({E:t('简单','Easy'),M:t('中等','Medium'),H:t('困难','Hard')} as Record<string,string>)[problem.difficulty]} · {problem.estimated_minutes||25} min</span></div></div>
  <div className="interview-concept-actions"><a className="button" href={withLibraryContext(problem.number,interviewHref(filters))}>{state==='done'?t('复习这道题','Review this module'):state==='started'?t('继续练习','Resume practice'):state==='todo'?t('开始练习','Start practice'):t('打开练习','Open practice')}<ArrowRight size={14}/></a>{hasPreparationGuide(problem.id)&&<button type="button" className="text-button" onClick={openPreparationHandbook}>{t('练习前先自测','Try the study prompts')}</button>}{catalogError&&onRetryCatalog&&<button type="button" className="text-button" onClick={onRetryCatalog}>{t('重试加载练习','Retry practice details')}</button>}{error&&onRetryProgress?<button type="button" className="text-button" onClick={onRetryProgress}>{t('重试学习进度','Retry learning progress')}</button>:<small>{t('代码与成绩沿用当前工作区','Uses your current workspace code and results')}</small>}</div>
 </section>;
}
