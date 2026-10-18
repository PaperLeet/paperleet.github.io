import {readInterviewSummary} from './interviewModels';
import {emptyInterviewFilters} from './interviewNavigation';
export const problemInterviewPeriods=['30','90','365','all'] as const;
export type ProblemInterviewPeriod=typeof problemInterviewPeriods[number];
export const problemInterviewFilters=(problemId:string,days:ProblemInterviewPeriod,company='')=>({...emptyInterviewFilters(),days,company,concept:problemId});

/** A problem-scoped summary counts the same matching reports as its requested concept. */
export function readProblemInterviewEvidence(raw:unknown,problemId:string,days:ProblemInterviewPeriod){
 const summary=readInterviewSummary(raw);
 if(!summary||summary.days!==days||!problemInterviewPeriods.includes(days)||!/^[a-z0-9_-]+\/[a-z0-9_-]+$/.test(problemId)||new Set(summary.questions.map(row=>row.problem_id)).size!==summary.questions.length||new Set(summary.companies.map(row=>row.company)).size!==summary.companies.length)return;
 const evidence=summary.questions.find(row=>row.problem_id===problemId);
 if(summary.reports===0)return !summary.contributors&&!summary.companies.length&&!summary.questions.length?{summary,companies:[],evidence:undefined}:undefined;
 if(!evidence||evidence.reports!==summary.reports||evidence.contributors!==summary.contributors||new Set(evidence.companies.map(row=>row.company)).size!==evidence.companies.length||evidence.companies.length!==summary.companies.length)return;
 if(evidence.companies.some(row=>row.reports<1||row.contributors<1||!summary.companies.some(company=>company.company===row.company&&company.reports===row.reports&&company.contributors===row.contributors))||evidence.companies.reduce((sum,row)=>sum+row.reports,0)!==summary.reports)return;
 return {summary,companies:evidence.companies,evidence};
}
