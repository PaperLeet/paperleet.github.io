import {useId} from 'react';
import {ArrowRight,Flame,MessageCircle} from 'lucide-react';
import {useI18n} from './i18n';
import OrganizationLogo from './OrganizationLogo';
import {interviewCatalog,type InterviewSummary} from './interviewModels';
import {interviewHref,type InterviewFilters,type InterviewCompanyOrder} from './interviewNavigation';
import {companyResources} from './interviewPreparation';
import {matchesInterviewCompany} from './interviewSearch';
import './interview-company-directory.css';

export default function InterviewCompanyDirectory({filters,summary,loading,error,onOrder,onClearSearch}:{filters:InterviewFilters;summary?:InterviewSummary;loading:boolean;error?:string;onOrder:(order:InterviewCompanyOrder)=>void;onClearSearch:()=>void}){
 const {t,locale}=useI18n(),hint=useId(),hasCounts=!!summary;
 const name=(company:typeof interviewCatalog.companies[number])=>locale==='en'?company.name_en:company.name;
 const rows=interviewCatalog.companies.filter(company=>matchesInterviewCompany(company,filters.q)).map((company,index)=>({company,index,evidence:summary?.companies.find(row=>row.company===company.id)}));
 const requested=filters.company_order,waiting=(requested==='contributors'||requested==='reports')&&!hasCounts,active=waiting?'catalog':requested;
 rows.sort((a,b)=>{
  if(active==='contributors')return (b.evidence?.contributors||0)-(a.evidence?.contributors||0)||(b.evidence?.reports||0)-(a.evidence?.reports||0)||a.index-b.index;
  if(active==='reports')return (b.evidence?.reports||0)-(a.evidence?.reports||0)||(b.evidence?.contributors||0)-(a.evidence?.contributors||0)||a.index-b.index;
  return active==='name'?name(a.company).localeCompare(name(b.company),locale==='en'?'en':'zh-CN')||a.index-b.index:a.index-b.index;
 });
 return <section className="interview-company-directory" aria-label={t('公司准备索引','Company preparation directory')}>
  <div className="company-directory-toolbar"><p role="status">{t(`共 ${rows.length} 家公司`,`${rows.length} ${rows.length===1?'company':'companies'}`)}</p><label><span>{t('排序','Order')}</span><select aria-label={t('公司索引排序','Company directory order')} aria-describedby={hint} value={requested} onChange={event=>onOrder(event.target.value as InterviewCompanyOrder)}><option value="catalog">{t('默认顺序','Default order')}</option><option value="contributors" disabled={!hasCounts}>{t('分享者最多','Most contributors')}</option><option value="reports" disabled={!hasCounts}>{t('面经最多','Most experiences')}</option><option value="name">{t('公司名称','Company name')}</option></select></label></div>
  <p id={hint} className="company-directory-hint">{waiting?loading?t('正在等待面经统计，暂按默认顺序展示；已保留所选排序。','Waiting for report counts. Showing the default order until your chosen order is ready.'):t('面经统计暂不可用，暂按默认顺序展示；已保留所选排序。','Report counts are unavailable. Showing the default order and keeping your chosen order.'):t('分享统计沿用当前筛选；分享者与面经数量都相同时，沿用默认顺序。','Counts follow the current filters. Ties in both counts keep the default order.')}{!waiting&&error&&<> {t('统计暂不可用，仍可浏览公司与准备资料。','Counts are unavailable. Company preparation remains available.')}</>}</p>
  {rows.length?<div className="interview-company-grid">{rows.map(({company,evidence})=>{
   const title=name(company),resources=companyResources(company.id),concepts=summary?new Set(summary.questions.filter(row=>row.companies.some(source=>source.company===company.id&&source.reports>0&&source.contributors>0)).map(row=>row.problem_id)).size:undefined;
   const href=(view:InterviewFilters['view'])=>interviewHref({...filters,company:company.id,view,q:'',page:1,saved:false});
   return <article key={company.id} className="interview-company-entry" data-company={company.id}>
    <a href={href('companies')} className="interview-company-card"><span className="interview-company-mark" aria-hidden="true"><OrganizationLogo id={company.id} size={32}/></span><b>{title}</b><small>{evidence&&evidence.reports>0?t(`${evidence.reports} 篇面经 · ${evidence.contributors} 位分享者`,`${evidence.reports} ${evidence.reports===1?'report':'reports'} · ${evidence.contributors} ${evidence.contributors===1?'contributor':'contributors'}`):hasCounts?t('期待第一份分享','Awaiting the first experience'):t('查看面经与准备资料','Explore experiences and resources')}</small>{concepts!==undefined&&concepts>0&&<small className="company-directory-concepts">{t(`${concepts} 个关联知识点`,`${concepts} linked ${concepts===1?'concept':'concepts'}`)}</small>}<span className="interview-company-resource-count">{resources.length?t(`${resources.length} 份官方资料`,`${resources.length} official resources`):t('按岗位开始准备','Prepare by role')}</span><ArrowRight size={15} aria-hidden="true"/></a>
    <nav className="company-directory-actions" aria-label={t(`${title}的准备入口`,`Preparation entries for ${title}`)}><a data-company-view="reports" href={href('reports')}><MessageCircle size={14} aria-hidden="true"/>{t('面经','Experiences')}</a><a data-company-view="questions" href={href('questions')}><Flame size={14} aria-hidden="true"/>{t('知识点与练习','Concepts & practice')}</a></nav>
   </article>;
  })}</div>:<div className="interview-empty"><h3>{t('没有匹配的公司','No matching companies')}</h3><p>{t('可以清空公司搜索，保留当前岗位、时间与排序。','Clear the company search while keeping your role, period and order.')}</p><button type="button" className="button subtle" onClick={onClearSearch}>{t('清空公司搜索','Clear company search')}</button></div>}
 </section>;
}
