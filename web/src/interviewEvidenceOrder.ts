import type {InterviewSummary} from './interviewModels';
import type {Problem} from './types';
export type EvidenceOrder='contributors'|'recent'|'difficulty'|'duration';
export function isEvidenceOrder(value:unknown):value is EvidenceOrder{return value==='contributors'||value==='recent'||value==='difficulty'||value==='duration';}
export type CuratedOrder='curriculum'|'difficulty'|'duration';
export function isCuratedOrder(value:unknown):value is CuratedOrder{return value==='curriculum'||value==='difficulty'||value==='duration';}
const curatedKey='paperleet.interview.curated-order.v1';
let curatedMemory:CuratedOrder='curriculum';
export function readCuratedOrder():CuratedOrder{try{const value=sessionStorage.getItem(curatedKey);return isCuratedOrder(value)?value:curatedMemory;}catch{return curatedMemory;}}
export function keepCuratedOrder(value:CuratedOrder){if(!isCuratedOrder(value))return;curatedMemory=value;try{sessionStorage.setItem(curatedKey,value);}catch{/* Keep the local reading preference available. */}}
const key='paperleet.interview.question-order.v1';
let memory:EvidenceOrder='contributors';
export function readEvidenceOrder():EvidenceOrder{try{const value=sessionStorage.getItem(key);return isEvidenceOrder(value)?value:memory;}catch{return memory;}}
export function keepEvidenceOrder(value:EvidenceOrder){if(!isEvidenceOrder(value))return;memory=value;try{sessionStorage.setItem(key,value);}catch{/* Reading order remains available for this page session. */}}

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
/** Practice ordering changes presentation only; source counts and contributor ties stay intact. */
export function orderInterviewEvidence(rows:InterviewSummary['questions'],order:EvidenceOrder,problems?:ReadonlyMap<string,Pick<Problem,'difficulty'|'estimated_minutes'>>){
 if(order==='contributors')return rows;
 if(order==='difficulty'||order==='duration')return orderPracticeConcepts(rows,order,row=>row.problem_id,problems);
 return rows.map((row,index)=>({row,index,stamp:latestQuestionPublication(row)})).sort((a,b)=>a.stamp===undefined?b.stamp===undefined?a.index-b.index:1:b.stamp===undefined?-1:b.stamp-a.stamp||a.index-b.index).map(value=>value.row);
}

/** Source-neutral practice ordering preserves the supplied order for ties and unknown values. */
export function orderPracticeConcepts<T>(rows:readonly T[],order:'difficulty'|'duration',id:(row:T)=>string,problems?:ReadonlyMap<string,Pick<Problem,'difficulty'|'estimated_minutes'>>):T[]{
 const value=(row:T)=>{const problem=problems?.get(id(row));if(order==='difficulty')return problem?.difficulty==='E'?0:problem?.difficulty==='M'?1:problem?.difficulty==='H'?2:undefined;const minutes=problem?.estimated_minutes;return typeof minutes==='number'&&Number.isFinite(minutes)&&minutes>0?minutes:undefined;};
 return rows.map((row,index)=>({row,index,value:value(row)})).sort((a,b)=>a.value===undefined?b.value===undefined?a.index-b.index:1:b.value===undefined?-1:a.value-b.value||a.index-b.index).map(value=>value.row);
}
