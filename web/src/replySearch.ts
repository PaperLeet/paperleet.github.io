import type {Reply} from './communityDiscussion';

const fold=(text:string)=>text.normalize('NFKC').toLowerCase();
export const replySearchQuery=(value:string)=>Array.from(value.replace(/[\u0000-\u001f\u007f]/g,' ').trim()).slice(0,200).join('');
export const replySearchTerms=(query:string)=>fold(query).split(/\s+/u).filter(Boolean);
/** Search only the public source and identity rendered for this reply. Never IDs or private drafts. */
export function replySearchText(reply:Reply){
 const author=reply.author.anonymous?'匿名用户 Anonymous':`${reply.author.display_name} ${reply.author.username}`;
 return fold(`${reply.body}\n${author}`);
}
export const matchesReplySearch=(text:string,terms:string[])=>terms.every(term=>text.includes(term));
