import type {Problem} from './types';

export const preparationConceptQuery=(value:string)=>Array.from(value.replace(/[\u0000-\u001f\u007f]/g,' ').trim()).slice(0,200).join('');
const fold=(value:string)=>value.normalize('NFKC').toLowerCase();
/** Search loaded public metadata; study hints remain behind their deferred import. */
export function searchPreparationConcepts(problems:Problem[],query:string){
 const terms=fold(preparationConceptQuery(query)).split(/\s+/u).filter(Boolean);
 if(!terms.length)return problems;
 return problems.filter(problem=>{
  const text=fold([problem.number,problem.id,problem.title,problem.translations?.en?.title,...(problem.module?.required_functions||[])].join('\n'));
  return terms.every(term=>text.includes(term));
 });
}
