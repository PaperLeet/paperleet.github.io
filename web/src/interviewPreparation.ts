import {interviewCatalog} from './interviewModels';
import {status,type Problem,type Saved} from './types';

export const companyResources=(company='')=>interviewCatalog.resources.filter(resource=>!company||resource.company===company);
export const rolePreparationPaths=(role='')=>interviewCatalog.paths.filter(path=>!role||path.roles.includes(role));

/** Catalog order defines a preparation path, independently of paper order or difficulty. */
export function preparationProgress(track:string,problems:Problem[],saved:Record<string,Saved>={},current?:Problem){
 const path=interviewCatalog.paths.find(path=>path.id===track);
 if(!path)return;
 const byId=new Map(problems.map(problem=>[problem.id,problem]));
 const modules=path.modules.flatMap(id=>byId.has(id)?[byId.get(id)!]:[]);
 const completed=modules.filter(problem=>status(problem,saved[problem.id])==='done').length;
 const remaining=modules.filter(problem=>problem.id!==current?.id&&status(problem,saved[problem.id])!=='done');
 const index=current?modules.findIndex(problem=>problem.id===current.id):-1;
 const next=current?remaining.find(problem=>modules.indexOf(problem)>index)||remaining[0]
  :remaining.filter(problem=>status(problem,saved[problem.id])==='started').sort((a,b)=>(saved[b.id]?.updated||'').localeCompare(saved[a.id]?.updated||''))[0]||remaining[0];
 return {path,modules,completed,total:path.modules.length,missing:path.modules.length-modules.length,next};
}

/** Hash routing owns the URL; in-page actions scroll and focus without changing it. */
export function focusInterviewSection(id:string){
 const heading=document.getElementById(id);
 if(!heading)return;
 heading.scrollIntoView?.({behavior:window.matchMedia?.('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
 heading.focus({preventScroll:true});
}

export function openPreparationHandbook(){
 const reader=document.querySelector<HTMLDetailsElement>('#preparation-handbook .preparation-guide');
 if(reader)reader.open=true;
 focusInterviewSection('preparation-handbook-heading');
}
