import {readCompanyConceptEvidence} from './interviewCompanyEvidence';
import type {InterviewSummary} from './interviewModels';
import type {Problem} from './types';

/** Counts describe one public report cohort; a report can link several exercises. */
export function libraryInterviewEvidence(summary:InterviewSummary|undefined,problems:Problem[],company=''){
 const evidence=readCompanyConceptEvidence(summary);
 if(!summary||!evidence||summary.days!=='365'||summary.questions.some(row=>row.reports>summary.reports||row.contributors>summary.contributors||row.reports!==row.companies.reduce((sum,source)=>sum+source.reports,0)||row.contributors>row.companies.reduce((sum,source)=>sum+source.contributors,0)||row.companies.some(source=>source.problem_id!==row.problem_id||(source.reports===0)!==(source.contributors===0))))return;
 const byId=new Map(problems.map(problem=>[problem.id,problem]));
 const questions=summary.questions.flatMap(row=>{
  const problem=byId.get(row.problem_id),sources=evidence.questions.find(value=>value.id===row.problem_id)!.sources;
  const selected=company?sources.get(company):row;
  return problem&&sources.size&&selected&&selected.reports>0&&selected.contributors>0?[{problem,reports:selected.reports,contributors:selected.contributors,companies:[...sources.values()]}]:[];
 }).sort((a,b)=>b.contributors-a.contributors||b.reports-a.reports||a.problem.number-b.problem.number);
 const companies=new Map<string,number>();
 for(const row of evidence.questions)if(byId.has(row.id))for(const id of row.sources.keys())companies.set(id,(companies.get(id)||0)+1);
 return {questions,companies};
}
