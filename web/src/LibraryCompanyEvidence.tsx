import AppLink from './AppLink';
import {useEffect} from 'react';
import {Building2,ArrowUpRight,X} from 'lucide-react';
import {useI18n} from './i18n';
import {useInterviewResource} from './interviewApi';
import {hostedEdition,hostedAppAvailable,hostedAppHref} from './deployment';
import {interviewCatalog,interviewLabel,type InterviewSummary} from './interviewModels';
import {emptyInterviewFilters,interviewHref,type InterviewFilters} from './interviewNavigation';
import {libraryInterviewEvidence} from './libraryInterviewEvidence';
import type {LibraryCompanyData,LibraryCompanyScope,LibraryCompanyToolsProps} from './libraryCompanyScope';
import type {Problem} from './types';
import './library-company.css';
import LibraryCompanyRowTools from './LibraryCompanyRowTools';
export {LibraryCompanyRowTools};
export function LibraryCompanyTools(props:LibraryCompanyToolsProps){return 'problem' in props?<LibraryCompanyRowTools {...props}/>:'scope' in props?<LibraryCompanyStatus {...props}/>:<LibraryCompanyControls {...props}/>;}

/** Confirm the complete selected public cohort before it can direct practice. */
export function readLibraryCompanyEvidence(summary:InterviewSummary|undefined,problems:Problem[],company:string,days:string){
 if(!interviewCatalog.companies.some(row=>row.id===company)||!['30','90','365','all'].includes(days))return;
 const result=libraryInterviewEvidence(summary,problems,company,days);
 if(!summary||!result||summary.companies.some(row=>row.company!==company)||summary.questions.some(row=>row.companies.some(source=>source.company!==company)))return;
 if(summary.reports===0?(summary.contributors!==0||summary.companies.length!==0||summary.questions.length!==0):(summary.companies.length!==1||summary.companies[0].reports!==summary.reports||summary.companies[0].contributors!==summary.contributors))return;
 const known=new Set(problems.map(problem=>problem.id)),missing=summary.questions.filter(row=>row.reports>0&&!known.has(row.problem_id)).length;
 return {...result,missing};
}

export default function LibraryCompanyBridge({company,days,scope,accountLoading,problems,retry,onChange,enabled}:{company:string;days:string;scope:string;accountLoading:boolean;problems:Problem[];retry:number;onChange:(data:LibraryCompanyData)=>void;enabled:boolean}){
 const known=interviewCatalog.companies.some(row=>row.id===company);
 const resource=useInterviewResource<InterviewSummary>('community/interviews/summary?'+new URLSearchParams({days,company}),enabled&&!accountLoading&&known,retry,scope);
 useEffect(()=>{
  const base={scope,company,days,ids:new Set<string>()};
  if(!enabled){onChange({...base,phase:'unavailable'});return;}
  if(!known){onChange({...base,phase:'error'});return;}
  if(accountLoading||resource.loading){onChange({...base,phase:'loading'});return;}
  const result=readLibraryCompanyEvidence(resource.data,problems,company,days);
  if(resource.error||!result){onChange({...base,phase:'error',error:resource.error});return;}
  if(!problems.length&&result.missing){onChange({...base,phase:'loading'});return;}
  if(result.missing){onChange({...base,phase:'error',missing:result.missing});return;}
  onChange({...base,phase:'ready',ids:new Set(result.questions.map(row=>row.problem.id)),summary:resource.data});
 },[scope,company,days,enabled,known,accountLoading,resource.loading,resource.error,resource.data,problems,onChange]);
 return null;
}

export function LibraryCompanyControls({company,days,onChange}:{company:string;days:string;onChange:(company:string,days:string)=>void}){
 const {t,locale}=useI18n();
 if(!hostedAppAvailable&&!company)return null;
 return <div className="library-company-controls" role="group" aria-label={t('公司练习筛选','Company practice filters')}>
  <Building2 size={17} aria-hidden="true"/>
  <label>{t('公司','Company')}<select aria-label={t('题库公司筛选','Filter library by company')} value={company} onChange={event=>onChange(event.target.value,days)}><option value="">{t('全部公司','All companies')}</option>{company&&!interviewCatalog.companies.some(row=>row.id===company)&&<option value={company}>{company}</option>}{interviewCatalog.companies.map(row=><option key={row.id} value={row.id}>{interviewLabel('companies',row.id,locale==='en')}</option>)}</select></label>
  <label>{t('面经发布时间','Report publication period')}<select aria-label={t('题库面经发布时间','Library report publication period')} value={days} disabled={!company} onChange={event=>onChange(company,event.target.value)}>{[['30','近 30 天','Past 30 days'],['90','近 90 天','Past 90 days'],['365','近一年','Past year'],['all','全部时间','All time']].map(([id,zh,en])=><option key={id} value={id}>{t(zh,en)}</option>)}</select></label>
  {company&&<button type="button" className="text-button" onClick={()=>onChange('','365')}><X size={14} aria-hidden="true"/>{t('清除公司','Clear company')}</button>}
 </div>;
}

export function LibraryCompanyStatus({company,days,scope,data,retry,reload,returnTo}:{company:string;days:string;returnTo:string}&LibraryCompanyScope){
 const {t,locale}=useI18n();
 const current=data?.scope===scope?data:undefined,phase=current?.phase||'loading';
 const label=interviewLabel('companies',company,locale==='en');
 const hub=interviewHref({...emptyInterviewFilters(),view:'questions',company,days:days as InterviewFilters['days']});
 return <section className="library-company-status" aria-label={t('公司练习依据','Company practice evidence')} data-phase={phase}>
  <div><b tabIndex={-1}><Building2 size={16} aria-hidden="true"/>{label}</b><p role="status">{phase==='loading'?t('正在确认公司练习范围…','Confirming company practice scope…'):phase==='unavailable'?t('此版本无法读取公司面经范围。请打开在线版，或清除公司筛选。','Company evidence is unavailable in this edition. Open the hosted app or clear this filter.'):phase==='error'?current?.missing?t(`${current.missing} 道关联题目尚未载入，范围暂未确认。`,`${current.missing} linked exercises are unavailable; the scope is unconfirmed.`):t('公司练习范围暂时无法确认，请重试。','The company practice scope could not be confirmed. Try again.'):t(`${current!.ids.size} 道关联题 · ${current!.summary!.contributors} 位分享者 · ${current!.summary!.reports} 篇面经`,`${current!.ids.size} linked exercises · ${current!.summary!.contributors} contributors · ${current!.summary!.reports} reports`)}</p><small>{t('依据公开面经的发布时间筛选，按分享者数排序；提及次数不代表真实面试频率。','Filtered by public report publication date and ranked by contributors. Mentions do not establish interview frequency.')}</small></div>
  {phase==='error'&&<button type="button" className="button subtle" onClick={event=>{event.currentTarget.closest('section')?.querySelector<HTMLElement>('b')?.focus({preventScroll:true});current?.missing?reload():retry();}}>{current?.missing?t('重新加载题库','Reload catalog'):t('重试公司范围','Retry company scope')}</button>}
  {phase==='unavailable'?hostedAppAvailable&&<AppLink href={hostedAppHref(returnTo)}>{t('打开在线版','Open hosted app')}<ArrowUpRight size={14}/></AppLink>:<AppLink href={hostedAppHref(hub)} target="_blank" rel="noreferrer">{t('查看面经依据','Explore source reports')}<ArrowUpRight size={14}/></AppLink>}
 </section>;
}
