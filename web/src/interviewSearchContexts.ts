import {interviewCatalog,type InterviewSummary} from './interviewModels';
import {interviewConceptMatch,interviewSearchTerms,normalizeInterviewSearch,type InterviewMatchField} from './interviewSearch';
import type {Problem} from './types';

type Evidence=InterviewSummary['questions'][number];
export type InterviewSearchContext={key:string;kind:'title'|'goal'|'module'|'company';text:string;lang?:'en'|'zh-CN'};

/** Explain only maintained public fields outside the row’s primary labels. */
export function interviewSearchContexts({id,problem,evidence,query,en,matches=interviewConceptMatch(id,problem,evidence,query)||[]}:{id:string;problem?:Problem;evidence?:Evidence;query:string;en:boolean;matches?:InterviewMatchField[]}):InterviewSearchContext[]{
 const terms=interviewSearchTerms(query);if(!terms.length||!matches.length)return [];
 const companies=(evidence?.companies||[]).filter(row=>row.reports>0&&row.contributors>0).flatMap(row=>{const company=interviewCatalog.companies.find(company=>company.id===row.company);return company?[company]:[];});
 const title=problem?(en?problem.translations?.en?.title||problem.title:problem.title):id;
 const goal=en?problem?.module?.goal_en||problem?.module?.goal||'':problem?.module?.goal||'';
 const shown=normalizeInterviewSearch([title,matches.includes('paper')?problem?.paper?.title:'',!evidence||matches.includes('goal')?goal:'',...companies.slice(0,3).map(company=>en?company.name_en:company.name)].filter(Boolean).join('\n'));
 let missing=terms.filter(term=>!shown.includes(term));
 const other=en?'zh-CN':'en';
 const fields:InterviewSearchContext[]=[
  {key:'module',kind:'module',text:id},
  {key:'short-id',kind:'module',text:problem?.short_id||''},
  {key:'number',kind:'module',text:problem?.number?String(problem.number):''},
  {key:'title',kind:'title',text:en?problem?.title||'':problem?.translations?.en?.title||'',lang:other},
  {key:'goal',kind:'goal',text:en?problem?.module?.goal||'':problem?.module?.goal_en||'',lang:other},
  ...companies.flatMap<InterviewSearchContext>(company=>[
   {key:`company:${company.id}:shown`,kind:'company' as const,text:en?company.name_en:company.name,lang:en?'en' as const:'zh-CN' as const},
   {key:`company:${company.id}:other`,kind:'company' as const,text:en?company.name:company.name_en,lang:other},
   {key:`company:${company.id}:id`,kind:'company' as const,text:company.id},
  ]),
 ];
 return fields.filter(field=>{
  const text=normalizeInterviewSearch(field.text),matched=missing.filter(term=>text.includes(term));
  if(!matched.length)return false;
  missing=missing.filter(term=>!matched.includes(term));return true;
 });
}
