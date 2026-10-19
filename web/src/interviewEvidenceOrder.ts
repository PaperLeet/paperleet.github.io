import type {InterviewSummary} from './interviewModels';
export type EvidenceOrder='contributors'|'recent';
const key='paperleet.interview.question-order.v1';
let memory:EvidenceOrder='contributors';
export function readEvidenceOrder():EvidenceOrder{try{const value=sessionStorage.getItem(key);return value==='recent'||value==='contributors'?value:memory;}catch{return memory;}}
export function keepEvidenceOrder(value:EvidenceOrder){if(value!=='contributors'&&value!=='recent')return;memory=value;try{sessionStorage.setItem(key,value);}catch{/* Reading order remains available for this page session. */}}

/** Source timestamps are report publication times in UTC, never interview dates. */
export function sourcePublicationTime(value:unknown):number|undefined{
 if(typeof value!=='string'||value.length>40||!/^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00))?$/.test(value))return;
 const stamp=Date.parse(value);
 if(Number.isFinite(stamp)&&new Date(stamp).toISOString().slice(0,10)===value.slice(0,10))return stamp;
}
export function latestQuestionPublication(row:InterviewSummary['questions'][number]):number|undefined{
 let latest:number|undefined;
 for(const source of row.companies){
  if(source.problem_id!==row.problem_id||!Number.isSafeInteger(source.reports)||source.reports<1||!Number.isSafeInteger(source.contributors)||source.contributors<1||source.contributors>source.reports)continue;
  const stamp=sourcePublicationTime(source.latest);if(stamp!==undefined&&(latest===undefined||stamp>latest))latest=stamp;
 }
 return latest;
}
export function orderInterviewEvidence(rows:InterviewSummary['questions'],order:EvidenceOrder){
 if(order==='contributors')return rows;
 return rows.map((row,index)=>({row,index,stamp:latestQuestionPublication(row)})).sort((a,b)=>a.stamp===undefined?b.stamp===undefined?a.index-b.index:1:b.stamp===undefined?-1:b.stamp-a.stamp||a.index-b.index).map(value=>value.row);
}
