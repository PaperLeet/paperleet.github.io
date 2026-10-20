import type {Problem} from './types';
export const PATH_PREVIEW_SIZE=3;
export const pathSearchDraft=(value:string)=>Array.from(value).slice(0,100).join('').replace(/[\u0000-\u001f\u007f]/g,' ');
const normalize=(value:string)=>value.normalize('NFKC').toLowerCase().replace(/\s+/g,' ').trim();
type Path={id:string;name:string;name_en:string;description:string;description_en:string;modules:string[]};
/** Search maintained path metadata and loaded public module labels, never drafts or reports. */
export function matchesPreparationPath(path:Path,modules:ReadonlyMap<string,Problem>,query:string){
 const terms=normalize(pathSearchDraft(query)).split(' ').filter(Boolean);if(!terms.length)return true;
 const text=normalize([path.id,path.name,path.name_en,path.description,path.description_en,...path.modules.flatMap(id=>{const module=modules.get(id);return [id,module?.title||'',module?.translations?.en?.title||'',module?String(module.number):''];})].join(' '));
 return terms.every(term=>text.includes(term));
}
