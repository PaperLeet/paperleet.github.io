import type {CommunityTerm} from './communityTaxonomy';

/** Metadata helps authors resolve selections; the server still checks membership and write eligibility. */
export function unavailableCommunityTags(tags:readonly CommunityTerm[],selected:readonly string[],existing:readonly string[]=[]){
 const definitions=new Map(tags.map(tag=>[tag.id,tag])),retained=new Set(existing);
 return selected.filter(id=>{const tag=definitions.get(id);return !tag||!tag.enabled&&!retained.has(id);});
}
