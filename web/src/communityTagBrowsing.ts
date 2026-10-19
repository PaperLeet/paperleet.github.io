import type {CommunityTerm} from './communityTaxonomy';

export const COMMUNITY_TAG_PAGE_SIZE=8;
export const communityTagQuery=(value:string)=>Array.from(value.replace(/[\u0000-\u001f\u007f]/g,' ')).slice(0,100).join('');

const words=(value:string)=>value.normalize('NFKC').toLowerCase().trim().split(/\s+/).filter(Boolean);
/** Search only public metadata; a selected paused tag stays available for historical reading. */
export function matchingCommunityTags(tags:readonly CommunityTerm[],query:string,selected?:string|readonly string[]){
 const terms=words(communityTagQuery(query)),retained=new Set(typeof selected==='string'?[selected]:selected||[]);
 return tags.filter(tag=>{if(!tag.enabled&&!retained.has(tag.id))return false;const text=words([tag.id,tag.zh,tag.en,tag.description].join(' ')).join(' ');return terms.every(term=>text.includes(term));});
}
export function initialCommunityTag(tags:readonly CommunityTerm[],selected?:string){
 return tags.find(tag=>tag.id===selected&&tag.enabled)?.id;
}
