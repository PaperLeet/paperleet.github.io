import {libraryReturnHref,readLibraryRoute} from './libraryNavigation';
import {readInterviewRoute} from './interviewNavigation';
import {interviewCatalog} from './interviewModels';
import {interviewPracticePreviewScope} from './interviewSourcePreviewModel';
import {problemInterviewPeriods,type ProblemInterviewPeriod} from './problemInterviewEvidence';
export type PracticeEvidenceOrigin={company:string;days:ProblemInterviewPeriod};

/** Inherit only public company/date controls from a validated practice continuation. */
export function practiceEvidenceOrigin(href:string,problemId:string,number?:number):PracticeEvidenceOrigin|undefined{
 if(!interviewPracticePreviewScope(href,problemId,number))return;
 const back=libraryReturnHref(href);if(!back)return;
 const library=readLibraryRoute(back),interview=readInterviewRoute(back);
 if(!library&&!interview)return;
 const company=(library?.filters.company??interview?.company)||'',days=library?.filters.company_days||interview?.days||'365';
 if(company&&!interviewCatalog.companies.some(row=>row.id===company)||!problemInterviewPeriods.includes(days as ProblemInterviewPeriod)||!company&&days==='365')return;
 return {company,days:days as ProblemInterviewPeriod};
}
