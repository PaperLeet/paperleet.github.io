import {parsePostMarkdown} from './postExcerpt';
import {markdownHeadingLabel} from './markdownOutline';
export type WritingSection={offset:number;label:string;depth:number};
/** Only real top-level headings become editing destinations; quoted examples stay examples. */
export function writingSections(body:string):WritingSection[]{
 if(body.length>20000)return [];
 try{return (parsePostMarkdown(body).children||[]).flatMap(node=>{
  const offset=node.position?.start.offset,label=node.type==='heading'?markdownHeadingLabel(node):'';
  return label&&Number.isSafeInteger(offset)&&offset!==undefined&&offset>=0&&offset<body.length&&node.depth&&node.depth>=1&&node.depth<=6?[{offset,label,depth:node.depth}]:[];
 });}catch{return [];}
}
export const writingDomOffset=(body:string,offset:number)=>body.slice(0,offset).replace(/\r\n?/g,'\n').length;

/** Measure wrapped native textarea lines only on an explicit section jump. */
export function writingSectionScroll(field:HTMLTextAreaElement,offset:number):number{
 const document=field.ownerDocument,view=document.defaultView;
 if(!view||!document.body||field.clientWidth<=0||field.clientHeight<=0)return field.scrollTop;
 const computed=view.getComputedStyle(field),mirror=document.createElement('div'),marker=document.createElement('span');
 mirror.setAttribute('aria-hidden','true');mirror.style.cssText='position:fixed;left:-100000px;top:0;visibility:hidden;pointer-events:none;box-sizing:border-box;border:0;margin:0;height:auto;min-height:0;max-height:none;overflow:visible;';
 for(const property of ['font-family','font-size','font-weight','font-style','font-stretch','font-variant','font-feature-settings','font-variation-settings','letter-spacing','line-height','text-indent','text-transform','text-align','direction','tab-size','padding-top','padding-right','padding-bottom','padding-left','word-break','overflow-wrap'])mirror.style.setProperty(property,computed.getPropertyValue(property));
 mirror.style.width=field.clientWidth+'px';mirror.style.whiteSpace=field.wrap==='off'?'pre':'pre-wrap';
 mirror.textContent=field.value.slice(0,offset);marker.textContent='\u200b';mirror.append(marker);
 try{
  document.body.append(mirror);const bounds=marker.getClientRects()[0];if(!bounds)return field.scrollTop;
  const top=bounds.top-mirror.getBoundingClientRect().top;
  return Number.isFinite(top)?Math.max(0,top+Math.max(0,bounds.height)/2-field.clientHeight/2):field.scrollTop;
 }catch{return field.scrollTop;}finally{mirror.remove();}
}
