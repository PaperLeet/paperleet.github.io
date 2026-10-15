import {interviewCatalog} from './interviewModels';

export const companyResources=(company='')=>interviewCatalog.resources.filter(resource=>!company||resource.company===company);
export const rolePreparationPaths=(role='')=>interviewCatalog.paths.filter(path=>!role||path.roles.includes(role));

/** Hash routing owns the URL; in-page actions scroll and focus without changing it. */
export function focusInterviewSection(id:string){
 const heading=document.getElementById(id);
 if(!heading)return;
 heading.scrollIntoView?.({behavior:window.matchMedia?.('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
 heading.focus({preventScroll:true});
}
