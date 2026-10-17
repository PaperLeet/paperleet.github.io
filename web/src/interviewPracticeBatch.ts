import {interviewHref,readInterviewRoute,interviewBatchView,type InterviewFilters} from './interviewNavigation';
import {withLibraryContext} from './libraryNavigation';
import {status,type Problem,type Saved} from './types';

/** Capture the ordered six-module group at selection time, independently of later grades. */
export function interviewPracticeHref(current:Problem,filters:InterviewFilters,candidates:Problem[]){
 const seen=new Set<number>();
 const ordered=candidates.filter(problem=>{if(seen.has(problem.number))return false;seen.add(problem.number);return true;});
 const index=ordered.findIndex(problem=>problem.id===current.id);
 const start=Math.floor(Math.max(0,index)/6)*6;
 const batch=index<0?[current]:ordered.slice(start,start+6);
 return withLibraryContext(current.number,interviewHref({...filters,view:interviewBatchView(filters)?filters.view:'questions',queue:batch.map(problem=>problem.number).join(',')}));
}

/** The snapshot selects navigation only; real workspace evidence owns all grades. */
export function interviewPracticeBatch(returnTo:string|undefined,problems:Problem[],saved:Record<string,Saved>,current:Problem){
 const filters=returnTo?readInterviewRoute(returnTo):undefined;
 if(!filters?.queue||!interviewBatchView(filters))return;
 const numbers=filters.queue.split(',').map(Number),byNumber=new Map(problems.map(problem=>[problem.number,problem]));
 const modules=numbers.flatMap(number=>byNumber.has(number)?[byNumber.get(number)!]:[]);
 const currentIncluded=modules.some(problem=>problem.id===current.id);
 const completed=modules.filter(problem=>status(problem,saved[problem.id])==='done').length;
 const remaining=modules.filter(problem=>problem.id!==current.id&&status(problem,saved[problem.id])!=='done');
 const index=modules.findIndex(problem=>problem.id===current.id);
 const next=currentIncluded?remaining.find(problem=>modules.indexOf(problem)>index)||remaining[0]:undefined;
 return {modules,total:numbers.length,missing:numbers.length-modules.length,completed,currentIncluded,next};
}
