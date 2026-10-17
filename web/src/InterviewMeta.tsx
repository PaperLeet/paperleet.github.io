import {BriefcaseBusiness,CalendarDays,MapPin} from 'lucide-react';
import {useI18n} from './i18n';
import OrganizationLogo from './OrganizationLogo';
import {interviewLabel,type InterviewDetails} from './interviewModels';
import {emptyInterviewFilters,interviewHref} from './interviewNavigation';
import InterviewReportPractice from './InterviewReportPractice';
import type {ProblemTopic} from './communityTopics';
import './interviews.css';
export default function InterviewMeta({details,compact=false,returnTo,topics,catalogLoading,catalogError,onRetryCatalog}:{details:InterviewDetails;compact?:boolean;returnTo?:string;topics?:ProblemTopic[];catalogLoading?:boolean;catalogError?:string;onRetryCatalog?:()=>void}){
 const {t,locale}=useI18n(),en=locale==='en';
 return <div className={`interview-meta ${compact?'compact':''}`}>
   <div className="interview-meta-tags"><a href={interviewHref({...emptyInterviewFilters(),company:details.company})}><OrganizationLogo id={details.company} size={14}/>{interviewLabel('companies',details.company,en)}</a><span><BriefcaseBusiness size={13}/>{interviewLabel('roles',details.role,en)}</span><span>{interviewLabel('stages',details.stage,en)}</span>{details.outcome!=='unknown'&&<span className="interview-outcome" data-outcome={details.outcome}>{interviewLabel('outcomes',details.outcome,en)}</span>}
   {!compact&&details.interview_month&&<span><CalendarDays size={13}/>{details.interview_month}</span>}{!compact&&details.location&&<span><MapPin size={13}/>{details.location}</span>}</div>
   {!compact&&<InterviewReportPractice questions={details.questions} topics={topics} returnTo={returnTo||interviewHref({...emptyInterviewFilters(),company:details.company})} catalogLoading={catalogLoading} catalogError={catalogError} onRetryCatalog={onRetryCatalog}/>}
 </div>;
}
