import {interviewCatalog,type InterviewDetails} from './interviewModels';
import {emptyInterviewFilters,interviewHref,readInterviewRoute} from './interviewNavigation';

export type InterviewFacet = 'company'|'role'|'stage';

/** Carry applied browse filters, never a profile identity or a nested post URL. */
export function interviewFacetHref(details:Pick<InterviewDetails,'company'|'role'|'stage'>,facet:InterviewFacet,source?:string):string|undefined {
  if(!['company','role','stage'].includes(facet)||!interviewCatalog.companies.some(row=>row.id===details.company))return;
  const candidate=source?readInterviewRoute(source):undefined;
  const previous=candidate&&([
    ['company','companies'],['role','roles'],['stage','stages'],['outcome','outcomes'],
  ] as const).every(([field,group])=>!candidate[field]||interviewCatalog[group].some(row=>row.id===candidate[field]))?candidate:undefined;
  const next={...(previous||{...emptyInterviewFilters(),days:'all' as const}),company:details.company,view:'reports' as const,page:1,queue:'',study:'all' as const,company_order:'catalog' as const};
  if(facet==='role'||facet==='stage'){
    if(!interviewCatalog.roles.some(row=>row.id===details.role))return;
    next.role=details.role;
  }
  if(facet==='stage'){
    if(!interviewCatalog.stages.some(row=>row.id===details.stage))return;
    next.stage=details.stage;
  }
  return interviewHref(next);
}
