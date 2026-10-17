import type { DraftStorage } from './communityDrafts';
import {isThreadAnonymous} from './communityIdentity';
import type { Reply } from './communityDiscussion';
export type ReplyEditDraft={body:string;baseVersion:string};
const version=/^[a-f0-9]{64}$/;
export const replyEditKey=(owner:string,reply:string)=>`paperleet.community.reply-edit.v1:${encodeURIComponent(owner)}:${encodeURIComponent(reply)}`;
export const canEditReply=(reply:Reply,owner:string)=>!!owner&&(reply.author.id===owner||isThreadAnonymous(reply.author,reply.post_id))&&reply.can_edit===true&&version.test(reply.edit_version||'');
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

export type ReplyEditOrigin={href:string;opener:Element|null};
export type ReplyEditConfirmation={origin:ReplyEditOrigin;baseVersion:string;keepEditing:boolean};

/** A save acknowledgement must describe the exact reply and submitted source. */
export function confirmedReplyEdit(original:Reply,owner:string,submitted:ReplyEditDraft,value:unknown):Reply|undefined {
  const row=value as Reply;
  if(!row||typeof row!=='object'||Array.isArray(row)||row.id!==original.id||row.post_id!==original.post_id||row.parent_id!==original.parent_id||row.created_at!==original.created_at)return;
  if(!row.author||typeof row.author.id!=='string'||row.author.id!==original.author.id||!!row.author.anonymous!==!!original.author.anonymous||row.body!==submitted.body.trim()||!canEditReply(row,owner))return;
  if(row.body!==original.body&&row.edit_version===submitted.baseVersion)return;
  return row;
}
