import {parsePostMarkdown,type PostMarkdownNode} from './postExcerpt';
import {communityPostHref} from './communityNavigation';
import type {ReplyDraft} from './replyDrafts';

const identifier=/^[a-f0-9]{32}$/;
const segmenter=typeof Intl.Segmenter==='function'?new Intl.Segmenter(undefined,{granularity:'grapheme'}):undefined;
const clip=(text:string,limit:number)=>{
  const letters=segmenter?Array.from(segmenter.segment(text),part=>part.segment):Array.from(text);
  return letters.slice(0,limit).join('')+(letters.length>limit?'…':'');
};
const escape=(text:string)=>text.replace(/[\\`*_{}\[\]()<>#!|~.+-]/g,'\\$&');
export type ReplyQuote={before:ReplyDraft;after:ReplyDraft;replyId:string};
export type ReplyQuoteResult={ok:true;quote:ReplyQuote}|{ok:false;reason:'unavailable'|'length'};

/** Copy a bounded visible excerpt, never hidden link destinations, images or return filters. */
export function quoteReplyDraft(draft:ReplyDraft,postId:string,reply:{id:string;post_id:string;body:string},name:string,en:boolean):ReplyQuoteResult{
  if(!identifier.test(postId)||!identifier.test(reply.id)||reply.post_id!==postId||typeof reply.body!=='string'||typeof draft.body!=='string')return {ok:false,reason:'unavailable'};
  let verbatim=false;
  function visible(node:PostMarkdownNode):string{
    if(['html','image','imageReference','definition','footnoteDefinition'].includes(node.type))return '';
    if(node.type==='code'||node.type==='math'||node.type==='inlineMath')verbatim=true;
    if(node.value!==undefined)return node.value+(['code','math'].includes(node.type)?'\n\n':'');
    if(node.type==='break')return '\n';
    const text=(node.children||[]).map(visible).join(node.type==='tableRow'?' | ':'');
    return text+(['paragraph','heading','code','math','listItem','tableRow'].includes(node.type)?'\n\n':'');
  }
  let excerpt:string;
  try{excerpt=clip(visible(parsePostMarkdown(reply.body.slice(0,20000))).trim().replace(/\n{3,}/g,'\n\n'),500);}
  catch{return {ok:false,reason:'unavailable'};}
  if(!excerpt)excerpt=en?'Open the source reply to view its content.':'点击原回复查看完整内容。';
  const author=escape(clip(name.replace(/[\u0000-\u001f\u007f]/g,' ').replace(/\s+/gu,' ').trim(),80));
  const label=en?`Quoted reply from ${author}`:`引用 ${author} 的回复`;
  const href=communityPostHref(postId,undefined,reply.id);
  // A variable fence keeps quoted code/math literal even when an excerpt
  // contains its own Markdown fences or has been clipped mid-expression.
  const fence='`'.repeat(Math.max(3,...[...excerpt.matchAll(/`+/g)].map(match=>match[0].length+1)));
  const content=verbatim?`${fence}text\n${excerpt}\n${fence}`:excerpt.split('\n').map(escape).join('\n');
  const prefix=`> [${label}](${href})\n>\n${content.split('\n').map(line=>'> '+line).join('\n')}\n\n`;
  const after={body:prefix+draft.body,parentId:reply.id};
  if(after.body.length>10000)return {ok:false,reason:'length'};
  return {ok:true,quote:{before:{...draft},after,replyId:reply.id}};
}
