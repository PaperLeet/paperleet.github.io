import {BriefcaseBusiness,CalendarDays,Code2,MapPin} from 'lucide-react';
import {useI18n} from './i18n';
import OrganizationLogo from './OrganizationLogo';
import {interviewLabel,type InterviewDetails} from './interviewModels';
import {emptyInterviewFilters,interviewHref} from './interviewNavigation';
import {withLibraryContext} from './libraryNavigation';
import './interviews.css';
export default function InterviewMeta({details,compact=false,returnTo}:{details:InterviewDetails;compact?:boolean;returnTo?:string}){
 const {t,locale}=useI18n(),en=locale==='en';
 return <div className={`interview-meta ${compact?'compact':''}`}>
   <div className="interview-meta-tags"><a href={interviewHref({...emptyInterviewFilters(),company:details.company})}><OrganizationLogo id={details.company} size={14}/>{interviewLabel('companies',details.company,en)}</a><span><BriefcaseBusiness size={13}/>{interviewLabel('roles',details.role,en)}</span><span>{interviewLabel('stages',details.stage,en)}</span>{details.outcome!=='unknown'&&<span className="interview-outcome" data-outcome={details.outcome}>{interviewLabel('outcomes',details.outcome,en)}</span>}
   {!compact&&details.interview_month&&<span><CalendarDays size={13}/>{details.interview_month}</span>}{!compact&&details.location&&<span><MapPin size={13}/>{details.location}</span>}</div>
   {!compact&&details.questions.length>0&&<div className="interview-meta-questions"><b>{t('面经关联知识点 · 进入配套练习','Linked concepts · practice')}</b>{details.questions.map(id=><a key={id} href={withLibraryContext(id,returnTo||interviewHref({...emptyInterviewFilters(),company:details.company}))}><Code2 size={12}/>{id.split('/')[1].replaceAll('_',' ')}</a>)}</div>}
 </div>;
}
