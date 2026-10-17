import {unified} from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import {postSearchIndex,postSearchQuery,postSearchWindow} from './postSearchModel';

export type PostMarkdownNode={type:string;value?:string;alt?:string|null;depth?:number;position?:{start:{offset?:number}};children?:PostMarkdownNode[]};
type PreviewNode=PostMarkdownNode;
export type PostContentKind='code'|'math'|'table';
export type PostPreviewSummary={text:string;truncated:boolean;kinds:PostContentKind[];rich:boolean;unavailable?:boolean;search?:'text'|PostContentKind|'source'|'outside';searchRange?:{start:number;end:number}};
const parser=unified().use(remarkParse).use(remarkGfm).use(remarkMath).freeze();
export function parsePostMarkdown(body:string):PostMarkdownNode{return parser.parse(body);}
const segmenter=typeof Intl.Segmenter==='function'?new Intl.Segmenter(undefined,{granularity:'grapheme'}):undefined;
const whitespace=(text:string)=>text.replace(/\s+/gu,' ').trim();

/** Read only visible inline content: link labels survive, destinations and raw markup do not. */
function inlineText(root:PreviewNode,formula:string,search=false){
 const stack=[root],pieces:string[]=[];
 while(stack.length){
  const node=stack.pop()!;
  if(node.type==='image'||node.type==='imageReference'){pieces.push(' ');continue;}
  if(node.type==='html')continue;
  if(node.type==='text'||node.type==='inlineCode'){pieces.push(node.value||'');continue;}
  if(node.type==='inlineMath'){if(node.value?.trim())pieces.push(search?node.value:formula);continue;}
  if(node.type==='break'){pieces.push(' ');continue;}
  if((node.type==='link'||node.type==='linkReference')&&node.children?.every(child=>child.type==='text')&&/^(?:https?:\/\/|www\.)\S+$/i.test(node.children.map(child=>child.value||'').join(''))){pieces.push(' ');continue;}
  for(const child of [...node.children||[]].reverse())stack.push(child);
 }
 return whitespace(pieces.join(''));
}

/** Parse the same Markdown dialect as the reader, without rendering HTML, math or external assets. */
export function postPreview(body:string,limit=210,formula='[formula]',search=''):PostPreviewSummary{
 const bounded=body.slice(0,20000).replace(/[\uD800-\uDBFF]$/u,'');
 const query=postSearchQuery(search),chunks:{text:string;kind:'text'|PostContentKind}[]=[];
 let tree:PreviewNode;
 try{tree=parsePostMarkdown(bounded);}catch{return {text:'',truncated:false,kinds:[],rich:true,unavailable:true};}
 const stack=[tree],paragraphs:string[]=[],kinds=new Set<PostContentKind>();
 let rich=(tree.children?.length||0)>1;
 while(stack.length){
  const node=stack.pop()!;
  if(node.type!=='root'&&node.type!=='paragraph'&&node.type!=='text'||node.type==='text'&&whitespace(node.value||'')!==node.value)rich=true;
  if(node.type==='code'&&node.value?.trim())kinds.add('code');
  if((node.type==='math'||node.type==='inlineMath')&&node.value?.trim())kinds.add('math');
  if(node.type==='table')kinds.add('table');
  if(node.type==='paragraph'){
   const text=inlineText(node,formula);
   if(text)paragraphs.push(text);
  }
  if(query){
   if(node.type==='paragraph'||node.type==='heading'){
    const text=inlineText(node,formula,true);if(text)chunks.push({text,kind:'text'});
   }else if((node.type==='code'||node.type==='math')&&node.value?.trim())chunks.push({text:whitespace(node.value),kind:node.type});
   else if(node.type==='tableCell'){
    const text=inlineText(node,formula,true);if(text)chunks.push({text,kind:'table'});
   }
  }
  for(const child of [...node.children||[]].reverse())stack.push(child);
 }
 const text=paragraphs.join(' '),length=Number.isSafeInteger(limit)?Math.max(1,Math.min(400,limit)):210;
 const contentKinds=(['code','math','table'] as const).filter(kind=>kinds.has(kind));
 if(query){
  for(const chunk of chunks){
   const context=postSearchWindow(chunk.text,query,length);
   if(context)return {...context,truncated:context.truncated||chunks.length>1||bounded.length<body.length,kinds:contentKinds,rich,search:chunk.kind};
  }
 }
 const letters=segmenter?Array.from(segmenter.segment(text),part=>part.segment):Array.from(text);
 const truncated=letters.length>length||bounded.length<body.length;
 const rawIndex=query?postSearchIndex(body.slice(0,40000),query):-1;
 return {text:letters.slice(0,length).join('').trimEnd()+(truncated&&letters.length?'…':''),truncated,kinds:contentKinds,rich,...(rawIndex<0?{}:{search:rawIndex+query.length>bounded.length?'outside' as const:'source' as const})};
}
