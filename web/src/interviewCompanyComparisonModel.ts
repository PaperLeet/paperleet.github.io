import {interviewCatalog,readInterviewSummary,type InterviewSummary} from './interviewModels';

type Count=InterviewSummary['companies'][number];
export type CompanyComparisonRow={id:string;left?:Count;right?:Count};
export type CompanyComparisonView='shared'|'different'|'all';

/** Use per-company source counts, never an addition of overlapping contributor totals. */
export function compareInterviewCompanies(summary:InterviewSummary,left:string,right:string){
 if(left===right||![left,right].every(id=>interviewCatalog.companies.some(company=>company.id===id))||!readInterviewSummary(summary,summary.outcome))return;
 const companies=new Map<string,Count>();
 for(const row of summary.companies){if(companies.has(row.company)||row.reports>summary.reports||row.contributors>summary.contributors||(row.reports===0)!==(row.contributors===0))return;companies.set(row.company,row);}
 const counts=(id:string)=>companies.get(id)||{company:id,reports:0,contributors:0};
 const rows:CompanyComparisonRow[]=[],seen=new Set<string>();
 for(const question of summary.questions){
  if(seen.has(question.problem_id))return;seen.add(question.problem_id);
  const sources=new Map<string,Count>();
  for(const source of question.companies){
   if(sources.has(source.company))return;sources.set(source.company,source);
   const total=companies.get(source.company);
   if(source.reports>question.reports||source.contributors>question.contributors||source.reports>(total?.reports||0)||source.contributors>(total?.contributors||0))return;
  }
  const a=sources.get(left),b=sources.get(right);
  const row={id:question.problem_id,left:a&&a.reports>0&&a.contributors>0?a:undefined,right:b&&b.reports>0&&b.contributors>0?b:undefined};
  if(row.left||row.right)rows.push(row);
 }
 return {left:counts(left),right:counts(right),rows,shared:rows.filter(row=>row.left&&row.right).length};
}
export const comparisonRows=(rows:CompanyComparisonRow[],view:CompanyComparisonView)=>rows.filter(row=>view==='all'||(view==='shared'?!!row.left&&!!row.right:!row.left||!row.right));

export function companyComparisonPair(value?:string){
 const parts=value?.split(',')||[],[left='',right='']=parts;
 return parts.length===2&&left!==right&&[left,right].every(id=>interviewCatalog.companies.some(company=>company.id===id))?{left,right}:{left:'',right:''};
}
