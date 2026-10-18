import {companyPracticeProgress} from './companyPracticeProgress';
import {libraryInterviewEvidence} from './libraryInterviewEvidence';
import {interviewCatalog,readInterviewSummary,type InterviewSummary} from './interviewModels';
import {interviewHref,type InterviewFilters} from './interviewNavigation';
import {interviewPracticeHref} from './interviewPracticeBatch';
import {status,type Problem,type Saved} from './types';

/** Company preparation uses the public report cohort, independently of report text search and saved reports. */
export function companyPreparationProgress(filters:InterviewFilters,summary:InterviewSummary|undefined,problems:Problem[],saved:Record<string,Saved>,ready:boolean,catalogError=false){
 const confirmed=readInterviewSummary(summary,filters.outcome,filters);
 const evidence=libraryInterviewEvidence(confirmed,problems,filters.company,filters.days);
 if(filters.view==='questions'||!interviewCatalog.companies.some(row=>row.id===filters.company)||!confirmed||!evidence||confirmed.companies.some(row=>row.company!==filters.company))return {phase:'unavailable'} as const;
 const progress=companyPracticeProgress(confirmed,problems,saved,ready,catalogError)?.get(filters.company);
 if(!progress)return {phase:'empty'} as const;
 if(progress.phase!=='ready')return progress;
 const candidates=evidence.questions.map(row=>row.problem);
 const started=candidates.filter(problem=>status(problem,saved[problem.id])==='started').sort((a,b)=>(saved[b.id]?.updated||'').localeCompare(saved[a.id]?.updated||''));
 const next=started[0]||candidates.find(problem=>status(problem,saved[problem.id])==='todo')||candidates[0];
 if(!next)return {phase:'unavailable'} as const;
 const practice={...filters,view:'questions' as const,q:'',saved:false,study:'all' as const,queue:'',page:1};
 return {...progress,next,action:status(next,saved[next.id])==='done'?'review':status(next,saved[next.id])==='started'?'continue':'start',href:interviewPracticeHref(next,practice,candidates),viewHref:interviewHref(practice)} as const;
}
