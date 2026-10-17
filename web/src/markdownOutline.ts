import {parsePostMarkdown,type PostMarkdownNode} from './postExcerpt';
export type MarkdownSection={id:string;label:string;depth:number};
export function headingAnchor(scope:string,offset:number|undefined):string|undefined {
  if(!/^reading-[a-f0-9]{32}(?:-instance-(?:[a-f0-9]{4}){1,64})?$/.test(scope)||!Number.isSafeInteger(offset)||offset===undefined||offset<0)return;
  return `${scope}-${offset}`;
}
export function markdownHeadingLabel(root:PostMarkdownNode){
  const stack=[root],pieces:string[]=[];
  while(stack.length){
    const node=stack.pop()!;
    if(node.type==='html')continue;
    if(node.type==='image'||node.type==='imageReference'){if(node.alt)pieces.push(node.alt);continue;}
    if(node.value!==undefined){pieces.push(node.value);continue;}
    if(node.type==='break'){pieces.push(' ');continue;}
    for(const child of [...node.children||[]].reverse())stack.push(child);
  }
  return pieces.join('').replace(/\s+/gu,' ').trim();
}
/** Source offsets give repeated, Unicode and formatted headings distinct renderer-compatible IDs. */
export function markdownOutline(body:string,scope:string):MarkdownSection[]{
  try{
    const stack=[parsePostMarkdown(body)],sections:MarkdownSection[]=[];
    while(stack.length){
      const node=stack.pop()!;
      if(node.type==='heading'){
        const id=headingAnchor(scope,node.position?.start.offset),text=markdownHeadingLabel(node);
        if(id&&text&&node.depth&&node.depth>=1&&node.depth<=6)sections.push({id,label:text,depth:node.depth});
      }
      for(const child of [...node.children||[]].reverse())stack.push(child);
    }
    return sections;
  }catch{return [];}
}
