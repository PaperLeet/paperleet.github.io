import {status,type Problem,type Saved} from './types';
import type {InterviewSummary} from './interviewModels';
import {readCompanyConceptEvidence} from './interviewCompanyEvidence';

export type CompanyPracticeProgress={total:number;loaded:number}&({phase:'ready';done:number;started:number;todo:number}|{phase:'catalog'|'progress'});
/** Join confirmed public associations to this owner's real workspace grades, without changing evidence order. */
export function companyPracticeProgress(summary:InterviewSummary|undefined,problems:Problem[],saved:Record<string,Saved>,ready:boolean,catalogError=false){
 const evidence=readCompanyConceptEvidence(summary);if(!evidence)return;
 const ids=new Map<string,string[]>(),byId=new Map(problems.map(problem=>[problem.id,problem]));
 for(const row of evidence.questions)for(const company of row.sources.keys()){const list=ids.get(company)||[];list.push(row.id);ids.set(company,list);}
 const result=new Map<string,CompanyPracticeProgress>();
 for(const [company,concepts]of ids){
  const available=concepts.flatMap(id=>byId.has(id)?[byId.get(id)!]:[]),base={total:concepts.length,loaded:available.length};
  if(catalogError||available.length!==concepts.length){result.set(company,{...base,phase:'catalog'});continue;}
  if(!ready){result.set(company,{...base,phase:'progress'});continue;}
  const states=available.map(problem=>status(problem,saved[problem.id])),done=states.filter(state=>state==='done').length,started=states.filter(state=>state==='started').length;
  result.set(company,{...base,phase:'ready',done,started,todo:concepts.length-done-started});
 }
 return result;
}
