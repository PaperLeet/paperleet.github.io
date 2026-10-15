import {unified} from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

type PreviewNode={type:string;value?:string;children?:PreviewNode[]};
export type PostContentKind='code'|'math'|'table';
const parser=unified().use(remarkParse).use(remarkGfm).use(remarkMath).freeze();
const segmenter=typeof Intl.Segmenter==='function'?new Intl.Segmenter(undefined,{granularity:'grapheme'}):undefined;
const whitespace=(text:string)=>text.replace(/\s+/gu,' ').trim();

/** Read only visible inline content: link labels survive, destinations and raw markup do not. */
function inlineText(root:PreviewNode,formula:string){
 const stack=[root],pieces:string[]=[];
 while(stack.length){
  const node=stack.pop()!;
  if(node.type==='image'||node.type==='imageReference'){pieces.push(' ');continue;}
  if(node.type==='html')continue;
  if(node.type==='text'||node.type==='inlineCode'){pieces.push(node.value||'');continue;}
  if(node.type==='inlineMath'){if(node.value?.trim())pieces.push(formula);continue;}
  if(node.type==='break'){pieces.push(' ');continue;}
  if((node.type==='link'||node.type==='linkReference')&&node.children?.every(child=>child.type==='text')&&/^(?:https?:\/\/|www\.)\S+$/i.test(node.children.map(child=>child.value||'').join(''))){pieces.push(' ');continue;}
  for(const child of [...node.children||[]].reverse())stack.push(child);
 }
 return whitespace(pieces.join(''));
}

/** Parse the same Markdown dialect as the reader, without rendering HTML, math or external assets. */
export function postPreview(body:string,limit=210,formula='[formula]'):{text:string;truncated:boolean;kinds:PostContentKind[];unavailable?:boolean}{
 const bounded=body.slice(0,20000).replace(/[\uD800-\uDBFF]$/u,'');
 let tree:PreviewNode;
 try{tree=parser.parse(bounded);}catch{return {text:'',truncated:false,kinds:[],unavailable:true};}
 const stack=[tree],paragraphs:string[]=[],kinds=new Set<PostContentKind>();
 while(stack.length){
  const node=stack.pop()!;
  if(node.type==='code'&&node.value?.trim())kinds.add('code');
  if((node.type==='math'||node.type==='inlineMath')&&node.value?.trim())kinds.add('math');
  if(node.type==='table')kinds.add('table');
  if(node.type==='paragraph'){
   const text=inlineText(node,formula);
   if(text)paragraphs.push(text);
  }
  for(const child of [...node.children||[]].reverse())stack.push(child);
 }
 const text=paragraphs.join(' '),length=Number.isSafeInteger(limit)?Math.max(1,Math.min(400,limit)):210;
 const letters=segmenter?Array.from(segmenter.segment(text),part=>part.segment):Array.from(text);
 const truncated=letters.length>length||bounded.length<body.length;
 return {text:letters.slice(0,length).join('').trimEnd()+(truncated&&letters.length?'…':''),truncated,kinds:(['code','math','table'] as const).filter(kind=>kinds.has(kind))};
}
