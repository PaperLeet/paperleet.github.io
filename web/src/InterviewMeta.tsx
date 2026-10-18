import AppLink from './AppLink';
import type {ReactNode} from 'react';
import {BriefcaseBusiness,CalendarDays,MapPin} from 'lucide-react';
import {useI18n} from './i18n';
import OrganizationLogo from './OrganizationLogo';
import {interviewLabel,type InterviewDetails} from './interviewModels';
import {emptyInterviewFilters,interviewHref} from './interviewNavigation';
import {interviewFacetHref,type InterviewFacet} from './interviewFacetNavigation';
import InterviewReportPractice from './InterviewReportPractice';
import type {ProblemTopic} from './communityTopics';
import './interviews.css';
export default function InterviewMeta({details,compact=false,returnTo,browseFrom,topics,catalogLoading,catalogError,onRetryCatalog,revealPractice,revealRequest}:{details:InterviewDetails;revealPractice?:string;revealRequest?:number;compact?:boolean;returnTo?:string;browseFrom?:string;topics?:ProblemTopic[];catalogLoading?:boolean;catalogError?:string;onRetryCatalog?:()=>void}){
 const {t,locale}=useI18n(),en=locale==='en';
 const company=interviewLabel('companies',details.company,en),role=interviewLabel('roles',details.role,en),stage=interviewLabel('stages',details.stage,en);
 const facets=[
   {id:'company',text:company,icon:<OrganizationLogo id={details.company} size={14}/>,label:t(`浏览${company}的面经`,`Browse experiences at ${company}`)},
   {id:'role',text:role,icon:<BriefcaseBusiness size={13}/>,label:t(`浏览${company}的${role}面经`,`Browse ${role} experiences at ${company}`)},
   {id:'stage',text:stage,icon:null,label:t(`浏览${company}的${role} · ${stage}面经`,`Browse ${role} · ${stage} experiences at ${company}`)},
 ] satisfies {id:InterviewFacet;text:string;icon:ReactNode;label:string}[];
 return <div className={`interview-meta ${compact?'compact':''}`}>
   <div className="interview-meta-tags">{facets.map(facet=>{
     const href=interviewFacetHref(details,facet.id,browseFrom),content=<>{facet.icon}{facet.text}</>;
     return href?<AppLink key={facet.id} data-interview-facet={facet.id} href={href} aria-label={facet.label} title={facet.label}>{content}</AppLink>:<span key={facet.id}>{content}</span>;
   })}{details.outcome!=='unknown'&&<span className="interview-outcome" data-outcome={details.outcome}>{interviewLabel('outcomes',details.outcome,en)}</span>}
   {details.interview_month&&<span className="interview-event-month" title={t('分享者填写的实际面试月份','Actual interview month supplied by the author')}><CalendarDays size={13}/>{t('面试：','Interview: ')}{details.interview_month}</span>}{details.location&&<span className="interview-event-location" title={details.location}><MapPin size={13}/>{details.location}</span>}</div>
   {!compact&&<InterviewReportPractice revealPractice={revealPractice} revealRequest={revealRequest} questions={details.questions} topics={topics} returnTo={returnTo||interviewHref({...emptyInterviewFilters(),company:details.company})} catalogLoading={catalogLoading} catalogError={catalogError} onRetryCatalog={onRetryCatalog}/>}
 </div>;
}
