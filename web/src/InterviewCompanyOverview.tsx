import {ArrowRight,BookOpen,Building2,Flame,MessageCircle} from 'lucide-react';
import {useI18n} from './i18n';
import OrganizationLogo from './OrganizationLogo';
import {interviewCatalog,type InterviewSummary} from './interviewModels';
import {interviewHref,type InterviewFilters} from './interviewNavigation';
import {companyResources,focusInterviewSection} from './interviewPreparation';
import CompanyPreparationProgress from './CompanyPreparationProgress';
import type {CompanyPreparationProps} from './CompanyPreparationProgress';

export default function InterviewCompanyOverview({filters,summary,practice}:{filters:InterviewFilters;summary?:InterviewSummary;practice?:Omit<CompanyPreparationProps,'filters'|'summary'>}){
 const {t,locale}=useI18n();
 const company=interviewCatalog.companies.find(company=>company.id===filters.company);
 if(!company)return null;
 const evidence=summary?.companies.find(row=>row.company===company.id),resources=companyResources(company.id);
 const reports=summary?evidence?.reports||0:'—',contributors=summary?evidence?.contributors||0:'—';
 return <section className="interview-company-overview" aria-labelledby="interview-company-heading">
  <header><span className="interview-company-mark" aria-hidden="true"><OrganizationLogo id={company.id} size={40}/></span><div><span className="interview-panel-kicker">{t('公司准备空间','COMPANY PREPARATION')}</span><h2 id="interview-company-heading">{locale==='en'?company.name_en:company.name}</h2><p><span>{reports} {t('篇面经','reports')}</span><span>{contributors} {t('位分享者','contributors')}</span><span>{resources.length} {t('份官方资料','official resources')}</span></p></div></header>
  <nav className="interview-company-actions" aria-label={t('公司准备导航','Company preparation navigation')}><a href={interviewHref({...filters,view:'reports',page:1,q:''})}><MessageCircle size={14}/>{t('阅读面经','Read experiences')}</a><a href={interviewHref({...filters,view:'questions',page:1,q:''})}><Flame size={14}/>{t('查看高频知识点','Explore frequent concepts')}</a><button type="button" onClick={()=>focusInterviewSection('interview-preparation-heading')}><BookOpen size={14}/>{t('开始准备','Start preparing')}<ArrowRight size={13}/></button></nav>
  {practice&&filters.view!=='questions'&&<CompanyPreparationProgress filters={filters} summary={summary} {...practice}/>}
  {filters.view==='companies'&&<ol className="interview-company-checklist"><li><Building2 size={17}/><div><b>{t('先核对目标岗位','Understand the role')}</b><p>{t('从官方资料确认职责、要求和流程，整理需要进一步询问招聘团队的问题。','Check responsibilities, requirements and process in the official sources, then note questions for the recruiting team.')}</p><button type="button" className="text-button" onClick={()=>focusInterviewSection('interview-resources-heading')}>{t('查看该公司资料','Read company resources')}<ArrowRight size={12}/></button></div></li><li><BookOpen size={17}/><div><b>{t('按岗位完成一条路径','Practice for your role')}</b><p>{t('结合岗位方向选择站内练习，用实现、测试与实验解释算法行为。','Choose role-related exercises and explain algorithm behavior through implementation, tests and experiments.')}</p></div></li><li><MessageCircle size={17}/><div><b>{t('结合真实样本复盘','Reflect with real experiences')}</b><p>{t('面经提及用于发现准备重点；推荐题单是学习建议，不计为公司考频。','Reports can reveal preparation topics. Curated paths are learning recommendations and do not count as company interview frequency.')}</p></div></li></ol>}
 </section>;
}
