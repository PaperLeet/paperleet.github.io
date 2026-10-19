import {interviewCatalog} from './interviewModels';

export type PreparationPrompt={focus:'explain'|'implement'|'verify';question:{zh:string;en:string};hint:{zh:string;en:string}};
export type PreparationFollowup={id:string;question:{zh:string;en:string};hint:{zh:string;en:string}};
export type PreparationGuide={id:string;source:string;functions:string[];prompts:PreparationPrompt[];followups?:PreparationFollowup[]};
// Every published exercise belongs to a maintained preparation path.
export const preparationGuideIds=[...new Set(interviewCatalog.paths.flatMap(path=>path.modules))];
export const hasPreparationGuide=(id:string)=>preparationGuideIds.includes(id);
export function readPreparationGuides(raw:unknown):PreparationGuide[]|undefined{
 const value=raw as {version:number;reviewed_on:string;modules:PreparationGuide[]};
 if(!value||value.version!==1||!/^\d{4}-\d{2}-\d{2}$/.test(value.reviewed_on)||!Array.isArray(value.modules)||value.modules.length!==preparationGuideIds.length)return;
 const ids=new Set<string>();
 for(const guide of value.modules){
  if(!guide||!hasPreparationGuide(guide.id)||ids.has(guide.id)||guide.source!==`problems/${guide.id}/README.md`||!Array.isArray(guide.functions)||!guide.functions.length||guide.functions.some(name=>typeof name!=='string'||!/^\w+$/.test(name))||!Array.isArray(guide.prompts)||guide.prompts.length!==3)return;
  if(guide.followups!==undefined){
   if(!Array.isArray(guide.followups)||guide.followups.length<1||guide.followups.length>4)return;
   const followups=new Set<string>();
   for(const item of guide.followups){
    if(!item||typeof item.id!=='string'||!/^[a-z][a-z0-9-]{1,63}$/.test(item.id)||followups.has(item.id))return;
    followups.add(item.id);
    for(const field of ['question','hint'] as const)for(const language of ['zh','en'] as const){const text=item[field]?.[language];if(typeof text!=='string'||text.trim().length<12||text.length>1200)return;}
   }
  }
  ids.add(guide.id);
  for(const [index,prompt] of guide.prompts.entries()){
   if(!prompt||prompt.focus!==['explain','implement','verify'][index])return;
   for(const field of ['question','hint'] as const)for(const language of ['zh','en'] as const){const text=prompt[field]?.[language];if(typeof text!=='string'||text.trim().length<12||text.length>1200)return;}
  }
 }
 return value.modules;
}
let catalog:Promise<PreparationGuide[]>|undefined;
/** The larger bilingual content is loaded only when a reader opens a guide. */
export async function loadPreparationGuide(id:string){
 if(!hasPreparationGuide(id))throw Error('Guide unavailable');
 if(!catalog)catalog=import('./data/preparation-guides.json').then(module=>{
  const guides=readPreparationGuides(module.default);if(!guides)throw Error('Guide unavailable');return guides;
 }).catch(error=>{catalog=undefined;throw error;});
 const guide=(await catalog).find(guide=>guide.id===id);if(!guide)throw Error('Guide unavailable');return guide;
}
