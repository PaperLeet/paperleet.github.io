import {interviewCatalog} from './interviewModels';

export type InterviewResource=typeof interviewCatalog.resources[number];
export type ResourceKind='all'|'guide'|'careers';
export const resourceBrowserQuery=(value:string)=>Array.from(value.replace(/[\u0000-\u001f\u007f]/g,' ').trim()).slice(0,100).join('');
export const resourceBrowserTerms=(value:string)=>resourceBrowserQuery(value).normalize('NFKC').toLowerCase().replace(/[\s/_-]+/gu,' ').trim();

/** Search only maintained public labels; never query a provider or alter the resource URL. */
export function matchingInterviewResources(resources:InterviewResource[],query:string){
 const terms=resourceBrowserTerms(query).split(' ').filter(Boolean);
 if(!terms.length)return resources;
 return resources.filter(resource=>{
  const company=interviewCatalog.companies.find(row=>row.id===resource.company);
  // Only the input is bounded; full public descriptions and labels remain searchable.
  const full=[resource.id,resource.name,resource.name_en,resource.description,resource.description_en,resource.url,company?.id,company?.name,company?.name_en].join('\n').normalize('NFKC').toLowerCase().replace(/[\s/_-]+/gu,' ');
  return terms.every(term=>full.includes(term));
 });
}
