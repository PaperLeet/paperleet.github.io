import type {MarkdownSection} from './markdownOutline';
import {postSearchIndex,postSearchQuery} from './postSearchModel';

/** Match rendered heading labels, retaining their heading-depth ancestors as context. */
export function filterReadingSections(sections:MarkdownSection[],value:string) {
  const query=postSearchQuery(value),matched=new Set<string>(),included=new Set<string>(),parents:MarkdownSection[]=[];
  if(!query)return {rows:sections.map(section=>({section,match:true})),matched:sections.length};
  for(const section of sections) {
    while(parents.length&&parents[parents.length-1].depth>=section.depth)parents.pop();
    if(postSearchIndex(section.label,query)>=0) {
      matched.add(section.id);included.add(section.id);
      for(const parent of parents)included.add(parent.id);
    }
    parents.push(section);
  }
  return {rows:sections.filter(section=>included.has(section.id)).map(section=>({section,match:matched.has(section.id)})),matched:matched.size};
}
export const headingFilterDraft=(value:string)=>Array.from(value.replace(/[\u0000-\u001f\u007f]/g,' ')).slice(0,200).join('');
