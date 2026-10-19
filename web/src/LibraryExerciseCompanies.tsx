import {activeSourcePreview,sourcePreviewEscape} from './sourcePreviewKeyboard';
import {useId,useLayoutEffect,useRef,useState} from 'react';
import {Building2,ChevronUp,X} from 'lucide-react';
import AppLink from './AppLink';
import {appLocation} from './appLocation';
import {hostedEdition} from './deployment';
import {useI18n} from './i18n';
import {useAccount} from './social';
import {useInterviewResource} from './interviewApi';
import {interviewLabel} from './interviewModels';
import {interviewHref} from './interviewNavigation';
import {libraryHref,readLibraryRoute} from './libraryNavigation';
import {problemInterviewPeriods,problemInterviewFilters,readProblemInterviewEvidence,type ProblemInterviewPeriod} from './problemInterviewEvidence';
import {keepReadingTriggerVisible} from './readingViewport';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import OrganizationLogo from './OrganizationLogo';
import LibraryExerciseSourcePreview from './LibraryExerciseSourcePreview';
import LibraryExerciseFavorite from './LibraryExerciseFavorite';
import {interviewSourceRequest} from './interviewSourcePreviewModel';
import type {LibraryCompanyGeneralRowProps} from './libraryCompanyScope';
import './library-exercise-companies.css';

export default function LibraryExerciseCompanies(props:LibraryCompanyGeneralRowProps){
 const {user,loading,capabilities}=useAccount(),view=readLibraryRoute(props.returnTo);
 if(!hostedEdition||loading||!view||view.filters.company)return null;
 return <Companies key={JSON.stringify([user?.id||'guest',props.returnTo,props.problem.id])} {...props} owner={user?.id||'guest'} favoriteAvailable={capabilities?.problem_bookmarks===true}/>;
}
function Companies({problem,returnTo,owner,favoriteAvailable}:LibraryCompanyGeneralRowProps&{owner:string;favoriteAvailable:boolean}){
 const {t,locale}=useI18n(),id=useId(),root=useRef<HTMLDivElement>(null),toggle=useRef<HTMLButtonElement>(null),panel=useRef<HTMLElement>(null),heading=useRef<HTMLHeadingElement>(null);
 const [open,setOpen]=useState(false),[visited,setVisited]=useState(false),[days,setDays]=useState<ProblemInterviewPeriod>('365'),[retry,setRetry]=useState(0),[previewCompany,setPreviewCompany]=useState('');
 const scope=JSON.stringify([owner,returnTo,problem.id]),intent=useWorkspaceIntent(scope),pending=useRef<(()=>boolean)|undefined>(undefined);
 const current=readLibraryRoute(appLocation.hash),sameView=!!current&&libraryHref(current.collectionId,current.filters)===returnTo,view=readLibraryRoute(returnTo)!;
 const resource=useInterviewResource<unknown>('community/interviews/summary?'+new URLSearchParams({days,problem_id:problem.id}),visited&&sameView,retry,scope);
 const evidence=readProblemInterviewEvidence(resource.data,problem.id,days),error=resource.error||(resource.data&&!evidence?t('暂时无法确认本题的公司来源，请重试。','Could not confirm company sources for this exercise. Please retry.'):'');
 const name=locale==='en'?problem.translations?.en?.title||problem.title:problem.title;
 const here=()=>sameView&&activeSourcePreview(root.current)&&!!matchesCurrentView();
 function matchesCurrentView(){const now=readLibraryRoute(appLocation.hash);return now&&libraryHref(now.collectionId,now.filters)===returnTo;}
 useLayoutEffect(()=>{
  const claim=pending.current;if(!claim)return;pending.current=undefined;
  if(open&&heading.current?.isConnected&&here()&&claim()){heading.current.focus({preventScroll:true});keepReadingTriggerVisible(heading.current,100);}else intent.cancel();
 },[open]);
 function close(){if(!here()||!open)return;pending.current=undefined;intent.cancel();if(panel.current?.contains(document.activeElement)){toggle.current?.focus({preventScroll:true});if(toggle.current)keepReadingTriggerVisible(toggle.current,100);}setPreviewCompany('');setOpen(false);}
 if(!sameView)return null;
 return <div ref={root} className="library-exercise-companies" data-company-problem={problem.id} onKeyDown={event=>{
  if(!open||!here()||!sourcePreviewEscape(event))return;
  event.preventDefault();event.stopPropagation();close();
 }}>
  <div className="library-exercise-row-actions" role="group" aria-label={t(`${name}的题目操作`,`Exercise actions for ${name}`)} data-favorite-available={favoriteAvailable||undefined}>{favoriteAvailable&&<LibraryExerciseFavorite problemId={problem.id} problemName={name} returnTo={returnTo} showStudyLists={false} canAct={()=>!!here()}/>}
  <button ref={toggle} type="button" className="library-exercise-company-trigger" aria-expanded={open} aria-controls={id} aria-keyshortcuts={open?'Escape':undefined} aria-label={t(`查看${name}的公司来源`,`View company sources for ${name}`)} title={t('公司来源','Company sources')} onClick={()=>{if(!here())return;if(open)close();else{pending.current=intent.create();setVisited(true);setOpen(true);}}}><Building2 size={17} aria-hidden="true"/></button>
  </div>
  <section id={id} ref={panel} className="library-exercise-company-panel" hidden={!open} aria-label={t(`${name}的公司来源`,`Company sources for ${name}`)}>{open&&<>
   <header><h3 ref={heading} tabIndex={-1}>{t('本题的公司来源','Company sources for this exercise')}</h3><button type="button" className="library-exercise-company-close" aria-label={t('收起公司来源','Collapse company sources')} onClick={close}><X size={16} aria-hidden="true"/></button></header>
   <label className="library-exercise-company-period">{t('面经发布时间','Report publication period')}<select value={days} aria-label={t(`${name}的面经发布时间`,`Report period for ${name}`)} onChange={event=>{const value=event.target.value;if(here()&&open&&problemInterviewPeriods.includes(value as ProblemInterviewPeriod)){setPreviewCompany('');setDays(value as ProblemInterviewPeriod);}}}>{[['30','近 30 天','Past 30 days'],['90','近 90 天','Past 90 days'],['365','近一年','Past year'],['all','全部时间','All time']].map(([value,zh,en])=><option key={value} value={value}>{t(zh,en)}</option>)}</select></label>
   <p className="library-exercise-company-basis">{t('依据公开面经的发布时间；提及次数不代表实际考频。','Based on public report publication dates; mentions are not actual interview frequency.')}</p>
   <div className="library-exercise-company-results" aria-busy={resource.loading}>{resource.loading?<p role="status">{t('正在读取公司来源…','Loading company sources…')}</p>:error?<div role="status"><p>{error}</p><button type="button" className="text-button" onClick={event=>{if(!here()||!open)return;if(document.activeElement===event.currentTarget){heading.current?.focus({preventScroll:true});if(heading.current)keepReadingTriggerVisible(heading.current,100);}setRetry(value=>value+1);}}>{t('重试公司来源','Retry company sources')}</button></div>:evidence?.companies.length?<ul>{evidence.companies.map(company=>{
    const label=interviewLabel('companies',company.company,locale==='en'),filter=libraryHref(view.collectionId,{...view.filters,company:company.company,company_days:days==='365'?undefined:days,sort:'default',page:undefined}),source=interviewHref({...problemInterviewFilters(problem.id,days,company.company),view:'reports'});
    return <li key={company.company}><AppLink className="library-exercise-company-filter" href={filter} onClick={event=>{if(!here())event.preventDefault();}} aria-label={t(`用${label}筛选当前题库；${company.contributors} 位分享者，${company.reports} 篇公开面经`,`Filter this library view by ${label}; ${company.reports} public reports from ${company.contributors} contributors`)}><OrganizationLogo id={company.company} size={20}/><span><b>{label}</b><small>{t(`${company.contributors} 位分享者 · ${company.reports} 篇面经`,`${company.contributors} ${company.contributors===1?'contributor':'contributors'} · ${company.reports} ${company.reports===1?'report':'reports'}`)}</small></span></AppLink>{interviewSourceRequest(source)?<LibraryExerciseSourcePreview cohort={{concept:problem.id,company:company.company,days}} sourceHref={source} returnTo={returnTo} owner={scope} label={label} name={`${label} · ${name}`} reports={company.reports} contributors={company.contributors} open={previewCompany===company.company} here={here} onOpen={()=>setPreviewCompany(company.company)} onClose={()=>setPreviewCompany('')}/>:<AppLink className="library-exercise-company-source" href={source} onClick={event=>{if(!here())event.preventDefault();}}>{t('来源面经','Source experiences')}</AppLink>}</li>;
   })}</ul>:<p role="status">{t('这个时间范围暂无本题的关联公开面经，可以调整发布时间。','No public reports link this exercise in this period. Try another publication period.')}</p>}</div>
   <footer><span>{t('点击公司名称，保留其他筛选条件查看公司练习。','Choose a company to filter practice with your other filters kept.')}</span><button type="button" className="text-button" onClick={close}><ChevronUp size={14} aria-hidden="true"/>{t('收起来源','Collapse sources')}</button></footer>
  </>}</section>
 </div>;
}
