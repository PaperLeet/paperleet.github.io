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

/** Explain matched public terms that the active language and visible source label do not show. */
export function resourceSearchContexts(resource:InterviewResource,query:string,en:boolean){
 const terms=resourceBrowserTerms(query).split(' ').filter(Boolean);if(!terms.length)return [];
 const company=interviewCatalog.companies.find(row=>row.id===resource.company);
 const shown=[en?resource.name_en:resource.name,en?resource.description_en:resource.description,en?company?.name_en:company?.name,new URL(resource.url).hostname.replace(/^www\./,'')].join('\n').normalize('NFKC').toLowerCase();
 let missing=terms.filter(term=>!shown.includes(term));
 const fields=[{key:'name',kind:'translation' as const,text:en?resource.name:resource.name_en},{key:'description',kind:'translation' as const,text:en?resource.description:resource.description_en},{key:'url',kind:'url' as const,text:resource.url},{key:'identifier',kind:'identifier' as const,text:resource.id}];
 return fields.filter(field=>{
  const text=field.text.normalize('NFKC').toLowerCase(),matched=missing.filter(term=>text.includes(term));
  if(!matched.length)return false;
  missing=missing.filter(term=>!matched.includes(term));return true;
 });
}
