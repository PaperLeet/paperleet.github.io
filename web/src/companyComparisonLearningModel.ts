import {status,type Problem,type Saved} from './types';
import type {CompanyComparisonRow} from './interviewCompanyComparisonModel';
import type {InterviewStudyState} from './interviewNavigation';

/** Personal states never alter public source counts or contributor order. */
export function companyComparisonLearning(rows:CompanyComparisonRow[],problems:Problem[],saved:Record<string,Saved>,ready:boolean,selected:InterviewStudyState){
 const byId=new Map(problems.map(problem=>[problem.id,problem]));
 const counts={all:rows.length,todo:0,started:0,done:0};
 const items=rows.map(row=>{
  const problem=byId.get(row.id),state=problem&&ready?status(problem,saved[problem.id]):undefined;
  if(state)counts[state]++;
  return {row,problem,state};
 });
 const loaded=items.filter(item=>!!item.problem).length,pending=selected!=='all'&&!ready;
 return {items,counts,loaded,unavailable:rows.length-loaded,pending,
  matching:pending?[]:items.filter(item=>selected==='all'||item.state===selected)};
}
export type CompanyComparisonLearningState=ReturnType<typeof companyComparisonLearning>;
