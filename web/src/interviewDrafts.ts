import {readInterviewDetails,type InterviewDetails} from './interviewModels';
import type {DraftStorage} from './communityDrafts';
import type {PublicationPayload} from './publicationRequests';
export type InterviewDraft={title:string;body:string;interview:InterviewDetails;anonymous?:boolean};
const memory=new Map<string,InterviewDraft>();
export const interviewDraftKey=(owner:string)=>'paperleet.interview.draft.v1:'+encodeURIComponent(owner);
export const interviewPublication=(draft:InterviewDraft):PublicationPayload=>({title:draft.title.trim(),body:draft.body.trim(),category:'career',problem_id:null,...(draft.anonymous?{anonymous:true}:{}),interview:JSON.stringify(readInterviewDetails(draft.interview)||draft.interview)});
export function readInterviewDraft(storage:DraftStorage|undefined,owner:string):InterviewDraft|undefined {
  if(memory.has(owner))return structuredClone(memory.get(owner));
  try{
    const raw=storage?.getItem(interviewDraftKey(owner));if(!raw||raw.length>100000)return;
    const value=JSON.parse(raw);if(value.version!==1||value.owner!==owner||!value.draft)return;
    const {title,body}=value.draft,interview=readInterviewDetails(value.draft.interview);
    if(typeof title!=='string'||title.length>180||typeof body!=='string'||body.length>20000||!interview||(value.draft.anonymous!==undefined&&typeof value.draft.anonymous!=='boolean'))return;
    return {title,body,interview,...(value.draft.anonymous?{anonymous:true}:{})};
  }catch{return;}
}
export function writeInterviewDraft(storage:DraftStorage|undefined,owner:string,draft:InterviewDraft){
  if((draft.anonymous!==undefined&&typeof draft.anonymous!=='boolean')||!owner||!readInterviewDetails(draft.interview)||draft.title.length>180||draft.body.length>20000)return false;
  memory.set(owner,structuredClone(draft));
  if(!storage)return false;
  try{storage.setItem(interviewDraftKey(owner),JSON.stringify({version:1,owner,draft}));return true;}catch{return false;}
}
export function clearInterviewDraft(storage:DraftStorage|undefined,owner:string,submitted:InterviewDraft){
  const current=readInterviewDraft(storage,owner);
  if(!current||JSON.stringify(interviewPublication(current))!==JSON.stringify(interviewPublication(submitted)))return false;
  try{storage?.removeItem(interviewDraftKey(owner));memory.delete(owner);return true;}catch{return false;}
}
export function discardInterviewDraft(storage:DraftStorage|undefined,owner:string){
  try{storage?.removeItem(interviewDraftKey(owner));memory.delete(owner);return true;}catch{return false;}
}
