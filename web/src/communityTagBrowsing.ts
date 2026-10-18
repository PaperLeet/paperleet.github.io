import type {CommunityTerm} from './communityTaxonomy';

const words=(value:string)=>value.normalize('NFKC').toLowerCase().trim().split(/\s+/).filter(Boolean);
/** Search only public metadata; a selected paused tag stays available for historical reading. */
export function matchingCommunityTags(tags:readonly CommunityTerm[],query:string,selected?:string|readonly string[]){
 const terms=words(query),retained=new Set(typeof selected==='string'?[selected]:selected||[]);
 return tags.filter(tag=>{if(!tag.enabled&&!retained.has(tag.id))return false;const text=words([tag.id,tag.zh,tag.en,tag.description].join(' ')).join(' ');return terms.every(term=>text.includes(term));});
}
export function initialCommunityTag(tags:readonly CommunityTerm[],selected?:string){
 return tags.find(tag=>tag.id===selected&&tag.enabled)?.id;
}
