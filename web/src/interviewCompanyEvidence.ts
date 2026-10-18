import {readInterviewSummary,type InterviewSummary} from './interviewModels';

type Count=InterviewSummary['companies'][number];
/** One validation boundary for source counts used by comparison and personal practice. */
export function readCompanyConceptEvidence(summary?:InterviewSummary){
 if(!summary||!readInterviewSummary(summary,summary.outcome,summary.context))return;
 const companies=new Map<string,Count>();
 for(const row of summary.companies){if(companies.has(row.company)||row.reports>summary.reports||row.contributors>summary.contributors||(row.reports===0)!==(row.contributors===0))return;companies.set(row.company,row);}
 const questions:{id:string;sources:Map<string,Count>}[]=[],seen=new Set<string>();
 for(const question of summary.questions){
  if(seen.has(question.problem_id))return;seen.add(question.problem_id);
  const sources=new Map<string,Count>();
  for(const source of question.companies){
   if(sources.has(source.company))return;sources.set(source.company,source);
   const total=companies.get(source.company);
   if(source.reports>question.reports||source.contributors>question.contributors||source.reports>(total?.reports||0)||source.contributors>(total?.contributors||0))return;
  }
  questions.push({id:question.problem_id,sources:new Map([...sources].filter(([,source])=>source.reports>0&&source.contributors>0))});
 }
 return {companies,questions};
}
