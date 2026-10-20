import type {Problem} from './types';
export const PATH_PREVIEW_SIZE=3;
export const pathSearchDraft=(value:string)=>Array.from(value).slice(0,100).join('').replace(/[\u0000-\u001f\u007f]/g,' ');
const normalize=(value:string)=>value.normalize('NFKC').toLowerCase().replace(/\s+/g,' ').trim();
const queryTerms=(query:string)=>[...new Set(normalize(pathSearchDraft(query)).split(' ').filter(Boolean))];
const moduleText=(id:string,module?:Problem)=>normalize([id,module?.title||'',module?.translations?.en?.title||'',module?String(module.number):''].join(' '));
type Path={id:string;name:string;name_en:string;description:string;description_en:string;modules:string[]};
/** Search maintained path metadata and loaded public module labels, never drafts or reports. */
export function matchesPreparationPath(path:Path,modules:ReadonlyMap<string,Problem>,query:string){
 const terms=queryTerms(query);if(!terms.length)return true;
 const text=normalize([path.id,path.name,path.name_en,path.description,path.description_en,...path.modules.map(id=>moduleText(id,modules.get(id)))].join(' '));
 return terms.every(term=>text.includes(term));
}
/** Reading shortcuts mention at least one query term; path filtering still requires every term. */
export function matchingPreparationModules(path:Path,modules:ReadonlyMap<string,Problem>,query:string):Problem[]{
 const terms=queryTerms(query);if(!terms.length)return [];
 return path.modules.flatMap(id=>{const problem=modules.get(id);if(!problem)return [];const text=moduleText(id,problem),score=terms.filter(term=>text.includes(term)).length;return score?[{problem,score}]:[];})
  .sort((a,b)=>b.score-a.score).map(match=>match.problem);
}
