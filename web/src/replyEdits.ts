import type { DraftStorage } from './communityDrafts';
import type { Reply } from './communityDiscussion';
export type ReplyEditDraft={body:string;baseVersion:string};
const version=/^[a-f0-9]{64}$/;
export const replyEditKey=(owner:string,reply:string)=>`paperleet.community.reply-edit.v1:${encodeURIComponent(owner)}:${encodeURIComponent(reply)}`;
export const canEditReply=(reply:Reply,owner:string)=>!!owner&&reply.author.id===owner&&reply.can_edit===true&&version.test(reply.edit_version||'');
export const newReplyEdit=(reply:Reply):ReplyEditDraft=>({body:reply.body,baseVersion:reply.edit_version||''});
function valid(value:unknown):value is ReplyEditDraft {
  if(!value||typeof value!=='object')return false;
  const d=value as ReplyEditDraft;return typeof d.body==='string'&&d.body.length<=10000&&typeof d.baseVersion==='string'&&version.test(d.baseVersion);
}
export function readReplyEdit(storage:DraftStorage|undefined,owner:string,reply:string):ReplyEditDraft|undefined {
  if(!storage||!owner||!reply)return;
  try {const raw=storage.getItem(replyEditKey(owner,reply));if(!raw||raw.length>100000)return;
    const v=JSON.parse(raw);if(v?.version!==1||v.owner!==owner||v.reply!==reply||!valid(v.draft))return;
    return {body:v.draft.body,baseVersion:v.draft.baseVersion};
  } catch{return;}
}
export function writeReplyEdit(storage:DraftStorage|undefined,owner:string,reply:string,draft:ReplyEditDraft):boolean {
  if(!storage||!owner||!reply||!valid(draft))return false;
  try{storage.setItem(replyEditKey(owner,reply),JSON.stringify({version:1,owner,reply,draft}));return true;}catch{return false;}
}
export function clearSavedReplyEdit(storage:DraftStorage|undefined,owner:string,reply:string,submitted:ReplyEditDraft):boolean {
  const saved=readReplyEdit(storage,owner,reply);
  if(!saved||saved.body!==submitted.body||saved.baseVersion!==submitted.baseVersion)return false;
  try{storage!.removeItem(replyEditKey(owner,reply));return true;}catch{return false;}
}
