import type {Reply} from './communityDiscussion';
import {resolveAffiliation} from './affiliationNames';

export const foldReplySearch=(text:string)=>text.normalize('NFKC').toLowerCase();
export const replySearchQuery=(value:string)=>Array.from(value.replace(/[\u0000-\u001f\u007f]/g,' ').trim()).slice(0,200).join('');
export const replySearchTerms=(query:string)=>foldReplySearch(query).split(/\s+/u).filter(Boolean);
export function replySearchFields(reply:Reply){
 const author=reply.author.anonymous?'匿名用户 Anonymous':`${reply.author.display_name} ${reply.author.username}`;
 const organization=resolveAffiliation(reply.author.affiliation,reply.author.institution);
 return {body:reply.body,author,affiliation:organization?[...new Set([organization.name,organization.name_en].filter(Boolean))].join(' · '):''};
}
/** Search only the public source and identity rendered for this reply. Never IDs or private drafts. */
export function replySearchText(reply:Reply){
 return foldReplySearch(Object.values(replySearchFields(reply)).join('\n'));
}
export const matchesReplySearch=(text:string,terms:string[])=>terms.every(term=>text.includes(term));
