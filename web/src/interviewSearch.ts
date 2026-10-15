import {interviewCatalog,type InterviewSummary} from './interviewModels';
import type {Problem} from './types';

type Company=typeof interviewCatalog.companies[number];
export type InterviewMatchField='title'|'module'|'paper'|'goal'|'company';
/** Match literal words, including full-width input, without treating input as a pattern. */
const normalize=(value:string)=>value.normalize('NFKC').toLowerCase().replace(/[\s/_\-]+/gu,' ').trim();
const terms=(query:string)=>[...new Set(normalize(query).split(' ').filter(Boolean))];
const companyText=(company:Company)=>`${company.id} ${company.name} ${company.name_en}`;
export function matchingInterviewCompanies(query:string){
 const words=terms(query);
 return words.length?interviewCatalog.companies.filter(company=>words.every(word=>normalize(companyText(company)).includes(word))):[];
}
export function matchesInterviewCompany(company:Company,query:string){
 const text=normalize(companyText(company));
 return terms(query).every(word=>text.includes(word));
}

/** Company matches require actual evidence for this concept, never a curated path. */
export function interviewConceptMatch(id:string,problem:Problem|undefined,evidence:InterviewSummary['questions'][number]|undefined,query:string):InterviewMatchField[]|undefined {
 const words=terms(query);
 if(!words.length)return [];
 const companies=evidence?.companies.filter(row=>row.reports>0&&row.contributors>0).flatMap(row=>{
  const company=interviewCatalog.companies.find(company=>company.id===row.company);
  return company?[companyText(company)]:[];
 }).join(' ')||'';
 const fields:[InterviewMatchField,string][]=[
  ['title',`${problem?.title||''} ${problem?.translations?.en?.title||''}`],
  ['module',`${id} ${problem?.short_id||''} ${problem?.number||''}`],
  ['paper',problem?.paper?.title||''],
  ['goal',`${problem?.module?.goal||''} ${problem?.module?.goal_en||''}`],
  ['company',companies],
 ];
 const searchable=fields.map(([field,value])=>[field,normalize(value)] as const);
 const matched=new Set<InterviewMatchField>();
 for(const word of words){
  const field=searchable.find(([,value])=>value.includes(word));
  if(!field)return;
  matched.add(field[0]);
 }
 return fields.flatMap(([field])=>matched.has(field)?[field]:[]);
}
