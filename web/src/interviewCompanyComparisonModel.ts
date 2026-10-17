import {interviewCatalog,type InterviewSummary} from './interviewModels';
import {readCompanyConceptEvidence} from './interviewCompanyEvidence';

type Count=InterviewSummary['companies'][number];
export type CompanyComparisonRow={id:string;left?:Count;right?:Count};
export type CompanyComparisonView='shared'|'different'|'all';

/** Use per-company source counts, never an addition of overlapping contributor totals. */
export function compareInterviewCompanies(summary:InterviewSummary,left:string,right:string){
 if(left===right||![left,right].every(id=>interviewCatalog.companies.some(company=>company.id===id)))return;
 const evidence=readCompanyConceptEvidence(summary);if(!evidence)return;
 const counts=(id:string)=>evidence.companies.get(id)||{company:id,reports:0,contributors:0};
 const rows:CompanyComparisonRow[]=[];
 for(const question of evidence.questions){
  const row={id:question.id,left:question.sources.get(left),right:question.sources.get(right)};
  if(row.left||row.right)rows.push(row);
 }
 return {left:counts(left),right:counts(right),rows,shared:rows.filter(row=>row.left&&row.right).length};
}
export const comparisonRows=(rows:CompanyComparisonRow[],view:CompanyComparisonView)=>rows.filter(row=>view==='all'||(view==='shared'?!!row.left&&!!row.right:!row.left||!row.right));

export function companyComparisonPair(value?:string){
 const parts=value?.split(',')||[],[left='',right='']=parts;
 return parts.length===2&&left!==right&&[left,right].every(id=>interviewCatalog.companies.some(company=>company.id===id))?{left,right}:{left:'',right:''};
}
