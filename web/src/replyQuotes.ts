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
const labelText=(text:string,limit:number)=>escape(clip(text.replace(/[\u0000-\u001f\u007f]/g,' ').replace(/\s+/gu,' ').trim(),limit));
type QuoteBase={before:ReplyDraft;after:ReplyDraft;sourceBody:string};
export type ReplyQuote=QuoteBase&{replyId:string};
export type PostQuote=QuoteBase&{postId:string;replyId?:never};
export type DraftQuote=ReplyQuote|PostQuote;
export type ReplyQuoteSelection={start:number;end:number;direction:'forward'|'backward'|'none';scroll:number;left:number};
export function captureReplyQuoteSelection(field:HTMLTextAreaElement|null,fallback=0):ReplyQuoteSelection{
  return {start:field?.selectionStart??fallback,end:field?.selectionEnd??fallback,direction:field?.selectionDirection||'none',scroll:field?.scrollTop||0,left:field?.scrollLeft||0};
}
export type ReplyQuoteResult={ok:true;quote:ReplyQuote}|{ok:false;reason:'unavailable'|'length'};
export type PostQuoteResult={ok:true;quote:PostQuote}|{ok:false;reason:'unavailable'|'length'|'exists'};

/** Copy visible text only; code/math stays literal even across clipped fences. */
function quotePrefix(body:string,label:string,href:string,fallback:string):string|undefined{
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
  try{excerpt=clip(visible(parsePostMarkdown(body.slice(0,20000))).trim().replace(/\n{3,}/g,'\n\n'),500);}
  catch{return;}
  if(!excerpt)excerpt=fallback;
  const fence='`'.repeat(Math.max(3,...[...excerpt.matchAll(/`+/g)].map(match=>match[0].length+1)));
  const content=verbatim?`${fence}text\n${excerpt}\n${fence}`:excerpt.split('\n').map(escape).join('\n');
  return `> [${label}](${href})\n>\n${content.split('\n').map(line=>'> '+line).join('\n')}\n\n`;
}

/** Never copy hidden destinations, images or private return filters. */
export function quoteReplyDraft(draft:ReplyDraft,postId:string,reply:{id:string;post_id:string;body:string},name:string,en:boolean):ReplyQuoteResult{
  if(!identifier.test(postId)||!identifier.test(reply.id)||reply.post_id!==postId||typeof reply.body!=='string'||typeof draft.body!=='string')return {ok:false,reason:'unavailable'};
  const author=labelText(name,80),label=en?`Quoted reply from ${author}`:`引用 ${author} 的回复`;
  const prefix=quotePrefix(reply.body,label,communityPostHref(postId,undefined,reply.id),en?'Open the source reply to view its content.':'点击原回复查看完整内容。');
  if(!prefix)return {ok:false,reason:'unavailable'};
  const after={body:prefix+draft.body,parentId:reply.id};
  if(after.body.length>10000)return {ok:false,reason:'length'};
  return {ok:true,quote:{before:{...draft},after,replyId:reply.id,sourceBody:reply.body}};
}

/** Add a published post excerpt without changing the draft's reply target. */
export function quotePostDraft(draft:ReplyDraft,postId:string,post:{id:string;title:string;body:string},en:boolean):PostQuoteResult{
  if(!identifier.test(postId)||post.id!==postId||typeof post.title!=='string'||typeof post.body!=='string'||typeof draft.body!=='string')return {ok:false,reason:'unavailable'};
  const title=labelText(post.title,120),href=communityPostHref(postId);
  const english=quotePrefix(post.body,`Quoted post: ${title}`,href,'Open the source post to view its content.');
  const chinese=quotePrefix(post.body,`引用帖子：${title}`,href,'点击原帖查看完整内容。');
  const prefix=en?english:chinese;
  if(!prefix)return {ok:false,reason:'unavailable'};
  if([english,chinese].some(value=>value&&draft.body.includes(value)))return {ok:false,reason:'exists'};
  const after={...draft,body:prefix+draft.body};
  if(after.body.length>10000)return {ok:false,reason:'length'};
  return {ok:true,quote:{before:{...draft},after,postId,sourceBody:post.body}};
}
