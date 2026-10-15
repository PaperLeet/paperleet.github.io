import type { DraftStorage } from './communityDrafts';
import type { Post } from './communityDiscussion';
import {readInterviewDetails,type InterviewDetails} from './interviewModels';

export type PostEditDraft = { title: string; body: string; category: string; baseVersion: string; interview?:InterviewDetails };
const version = /^[a-f0-9]{64}$/;
const categories = new Set(['papers','implementation','questions','career','lounge']);
export const postEditKey = (owner:string,post:string) =>
  `paperleet.community.post-edit.v1:${encodeURIComponent(owner)}:${encodeURIComponent(post)}`;
export function canEditPost(post:Post,owner:string):boolean {
  return !!owner && post.author.id === owner && post.viewer.can_edit === true && version.test(post.edit_version || '');
}
export function newPostEdit(post:Post):PostEditDraft {
  return {title:post.title,body:post.body,category:post.category,baseVersion:post.edit_version || '',...(post.interview?{interview:{...post.interview,questions:[...post.interview.questions]}}:{})};
}
function valid(value:unknown):value is PostEditDraft {
  if(!value || typeof value !== 'object') return false;
  const draft=value as PostEditDraft;
  return typeof draft.title==='string' && draft.title.length<=180 && typeof draft.body==='string' && draft.body.length<=20000
    && typeof draft.category==='string' && categories.has(draft.category) && typeof draft.baseVersion==='string' && version.test(draft.baseVersion)
    && (draft.interview===undefined||!!readInterviewDetails(draft.interview));
}
export function samePostEdit(a:PostEditDraft,b:PostEditDraft):boolean {
  return a.title===b.title && a.body===b.body && a.category===b.category && a.baseVersion===b.baseVersion && JSON.stringify(a.interview)===JSON.stringify(b.interview);
}
export function readPostEdit(storage:DraftStorage|undefined,owner:string,post:string):PostEditDraft|undefined {
  if(!storage||!owner||!post)return;
  try {
    const raw=storage.getItem(postEditKey(owner,post));
    if(!raw||raw.length>200000)return;
    const value=JSON.parse(raw);
    if(value?.version!==1||value.owner!==owner||value.post!==post||!valid(value.draft))return;
    const {title,body,category,baseVersion}=value.draft;
    return {title,body,category,baseVersion,...(value.draft.interview?{interview:readInterviewDetails(value.draft.interview)}:{})};
  } catch {return;}
}
export function writePostEdit(storage:DraftStorage|undefined,owner:string,post:string,draft:PostEditDraft):boolean {
  if(!storage||!owner||!post||!valid(draft))return false;
  try {storage.setItem(postEditKey(owner,post),JSON.stringify({version:1,owner,post,draft}));return true;}
  catch {return false;}
}
export function clearSavedPostEdit(storage:DraftStorage|undefined,owner:string,post:string,submitted:PostEditDraft):boolean {
  const saved=readPostEdit(storage,owner,post);
  if(!saved||!samePostEdit(saved,submitted))return false;
  try {storage!.removeItem(postEditKey(owner,post));return true;}catch{return false;}
}
