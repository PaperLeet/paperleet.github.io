import AppLink from './AppLink';
import {useId,type ReactNode} from 'react';
import {ChevronDown} from 'lucide-react';
import {useI18n} from './i18n';
import OrganizationLogo from './OrganizationLogo';
import {interviewLabel,type InterviewSummary} from './interviewModels';
import {emptyInterviewFilters,interviewHref,type InterviewFilters} from './interviewNavigation';
import './interview-company-sources.css';
import {sourcePublicationTime} from './interviewEvidenceOrder';

type CompanySource=InterviewSummary['questions'][number]['companies'][number];
type Preview={panelId:string;company:string;open:boolean;onSelect:(company:string,opener:HTMLButtonElement)=>void};

/** Company counts are source evidence; disclosure changes only local reading state. */
export default function InterviewCompanySources({companies,problemId,name,filters=emptyInterviewFilters(),preview,renderLabel,preferredCompany}:{companies:CompanySource[];problemId:string;name?:string;filters?:InterviewFilters;preview?:Preview;renderLabel?:(label:string)=>ReactNode;preferredCompany?:string}){
 const {t,locale}=useI18n(),extraId=useId();
 const available=companies.filter(company=>company.reports>0&&company.contributors>0),preferred=available.find(company=>company.company===preferredCompany);
 const sources=preferred?[preferred,...available.filter(company=>company!==preferred)]:available;
 if(!sources.length)return null;
 const links=(rows:CompanySource[])=>rows.map(company=>{
  const label=interviewLabel('companies',company.company,locale==='en');
  const stamp=sourcePublicationTime(company.latest),recency=stamp===undefined?'':t(` · 最新关联分享 ${new Date(stamp).toISOString().slice(0,10)}`,` · Latest linked report ${new Date(stamp).toISOString().slice(0,10)}`);
  const title=t(`${company.reports} 篇公开面经关联此知识点，来自 ${company.contributors} 位分享者`,`${company.reports} public ${company.reports===1?'report links':'reports link'} this concept, from ${company.contributors} ${company.contributors===1?'contributor':'contributors'}`);
  const content=<><OrganizationLogo id={company.company} size={16}/><span>{renderLabel?renderLabel(label):label}</span><small>{company.reports}</small></>;
  const selected=company.company===preferredCompany,selection=selected?t(' · 当前来源公司',' · Current source company'):'';
  return preview?<button key={company.company} type="button" className="interview-company-source" data-company-source={company.company} data-source-selected={selected||undefined} aria-controls={preview.panelId} aria-expanded={preview.open&&preview.company===company.company} title={title+recency+selection} aria-label={t(`预览${label}关于${name||'此知识点'}的 ${company.reports} 篇来源面经，来自 ${company.contributors} 位分享者`,`Preview ${company.reports} ${label} source ${company.reports===1?'report':'reports'} about ${name||'this concept'}, from ${company.contributors} ${company.contributors===1?'contributor':'contributors'}`)+recency+selection} onClick={event=>preview.onSelect(company.company,event.currentTarget)}>{content}</button>:<AppLink key={company.company} className="interview-company-source" href={interviewHref({...filters,view:'reports',company:company.company,concept:problemId,q:'',page:1,saved:false})} title={title+recency} aria-label={t(`阅读${label}关于${name||'此知识点'}的 ${company.reports} 篇来源面经，来自 ${company.contributors} 位分享者`,`Read ${company.reports} ${label} source ${company.reports===1?'report':'reports'} about ${name||'this concept'}, from ${company.contributors} ${company.contributors===1?'contributor':'contributors'}`)+recency}>{content}</AppLink>;
 });
 return <div className="interview-company-tags company-source-list" role="group" aria-label={name?t(`${name}的公司来源`,`Company sources for ${name}`):t('此知识点的公司来源','Company sources for this concept')}>
  {links(sources.slice(0,3))}
  {sources.length>3&&<details className="interview-company-more">
   <summary aria-controls={extraId} onClick={event=>{
    const control=event.currentTarget,details=control.parentElement as HTMLDetailsElement,active=document.activeElement,href=location.href;
    // Native disclosure handles keyboard order; only repair a now-hidden link.
    queueMicrotask(()=>{if(control.isConnected&&!details.open&&location.href===href&&active&&document.activeElement===active&&control.nextElementSibling?.contains(active))control.focus({preventScroll:true});});
   }}><span className="company-source-more-label">{t(`另外 ${sources.length-3} 家公司`,`${sources.length-3} more ${sources.length===4?'company':'companies'}`)}</span><span className="company-source-less-label">{t('收起公司列表','Show fewer companies')}</span><ChevronDown size={14} aria-hidden="true"/></summary>
   <div id={extraId} className="interview-company-extra-links">{links(sources.slice(3))}</div>
  </details>}
 </div>;
}
