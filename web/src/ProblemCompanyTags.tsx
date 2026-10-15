import {useState} from 'react';
import {useI18n} from './i18n';
import OrganizationLogo from './OrganizationLogo';
import {hostedEdition} from './deployment';
import {useInterviewResource} from './interviewApi';
import {interviewLabel,readInterviewSummary,type InterviewSummary} from './interviewModels';
import {interviewHref,emptyInterviewFilters} from './interviewNavigation';
import './interviews.css';
export default function ProblemCompanyTags({problemId}:{problemId:string}){
 const {t,locale}=useI18n();
 const [retry,setRetry]=useState(0);
 const resource=useInterviewResource<InterviewSummary>('community/interviews/summary?days=365&problem_id='+encodeURIComponent(problemId),hostedEdition,retry);
 const summary=readInterviewSummary(resource.data);
 if(resource.error||resource.data&&!summary)return <div className="problem-company-evidence problem-company-retry" role="status"><small>{t('公司面经暂时无法读取，仍可继续练习。','Company experiences are unavailable. You can keep practicing.')}</small><button type="button" onClick={()=>setRetry(value=>value+1)}>{t('重试面经','Retry experiences')}</button></div>;
 const companies=summary?.questions.find(row=>row.problem_id===problemId)?.companies.filter(company=>company.reports>0&&company.contributors>0);
 if(!companies?.length)return null;
 return <div className="problem-company-evidence"><small>{t('近一年面经提及','In reports from the past year')}</small><div className="interview-company-tags">{companies.map(company=><a key={company.company} href={interviewHref({...emptyInterviewFilters(),company:company.company,concept:problemId})} title={t(`${company.reports} 篇公开面经关联此知识点，来自 ${company.contributors} 位分享者`,`${company.reports} public ${company.reports===1?'report links':'reports link'} this concept, from ${company.contributors} ${company.contributors===1?'contributor':'contributors'}`)}><OrganizationLogo id={company.company} size={14}/>{interviewLabel('companies',company.company,locale==='en')}<small>{company.reports}</small></a>)}</div><a href={interviewHref({...emptyInterviewFilters(),view:'questions',concept:problemId})}>{t('查看统计依据','Explore report-based counts')}</a></div>;
}
