import {useI18n} from './i18n';
import {hostedEdition} from './deployment';
import {useInterviewResource} from './interviewApi';
import {interviewLabel,readInterviewSummary,type InterviewSummary} from './interviewModels';
import {interviewHref,emptyInterviewFilters} from './interviewNavigation';
import './interviews.css';
export default function ProblemCompanyTags({problemId}:{problemId:string}){
 const {t,locale}=useI18n();
 const resource=useInterviewResource<InterviewSummary>('community/interviews/summary?days=365&problem_id='+encodeURIComponent(problemId),hostedEdition);
 const row=readInterviewSummary(resource.data)?.questions.find(row=>row.problem_id===problemId);
 if(!row?.companies?.length)return null;
 return <div className="problem-company-evidence"><small>{t('近一年面经提及','Mentioned in reports this year')}</small><div className="interview-company-tags">{row.companies.map(company=><a key={company.company} href={interviewHref({...emptyInterviewFilters(),company:company.company,view:'questions',q:problemId})} title={t(`${company.reports} 篇公开面经关联此知识点，来自 ${company.contributors} 位分享者`,`${company.reports} public reports link this concept, from ${company.contributors} contributors`)}>{interviewLabel('companies',company.company,locale==='en')}<small>{company.reports}</small></a>)}</div><a href={interviewHref({...emptyInterviewFilters(),view:'questions',q:problemId})}>{t('查看统计依据','Explore report-based counts')}</a></div>;
}
