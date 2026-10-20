import {parsePostMarkdown} from './postExcerpt';
import {markdownHeadingLabel} from './markdownOutline';

/** A text-length planning aid, never an estimate of time spent understanding code or math. */
export function postReadingOverview(body:string){
 try{
  const stack=[parsePostMarkdown(body)],text:string[]=[];
  let sections=0,code=0,math=0,tables=0;
  while(stack.length){
   const node=stack.pop()!;
   if(['html','image','imageReference','definition'].includes(node.type))continue;
   if(node.type==='code'){if(node.value?.trim())code++;text.push(' ');continue;}
   if(node.type==='math'||node.type==='inlineMath'){if(node.value?.trim())math++;text.push(' ');continue;}
   if(node.type==='inlineCode'||node.type==='break'){text.push(' ');continue;}
   if(node.type==='heading'&&markdownHeadingLabel(node))sections++;
   if(node.type==='table')tables++;
   if(node.type==='text'){text.push(node.value||'');continue;}
   if((node.type==='link'||node.type==='linkReference')&&node.children?.every(child=>child.type==='text')&&/^(?:https?:\/\/|www\.)\S+$/i.test(node.children.map(child=>child.value||'').join('')))continue;
   if(['paragraph','heading','tableCell','listItem','blockquote'].includes(node.type)){text.push(' ');stack.push({type:'break'});}
   for(const child of [...node.children||[]].reverse())stack.push(child);
  }
  const visible=text.join(''),cjk=/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu;
  const characters=visible.match(cjk)?.length||0,words=visible.replace(cjk,' ').match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu)?.length||0;
  const minutes=characters||words?Math.max(1,Math.ceil(characters/300+words/200)):undefined;
  return {minutes,sections,code,math,tables};
 }catch{return undefined;}
}
