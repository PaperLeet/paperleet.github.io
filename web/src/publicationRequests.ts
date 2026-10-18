import type { DraftStorage, PostDraft } from './communityDrafts';
import type { ReplyDraft } from './replyDrafts';
import {readInterviewDetails} from './interviewModels';
import {canonicalArxivUrl} from './communityPapers';
import { isCommunityCategory, validCommunityTags } from './communityCategories';

export type PublicationKind = 'post' | 'reply' | 'interview';
export type PublicationPayload = Record<string,string|null|boolean|string[]>;
export type PublicationReceipt = {version:1;owner:string;kind:PublicationKind;scope:string;requestId:string;payload:PublicationPayload};
export type PendingPublication = {receipt:PublicationReceipt;durable:boolean};
const memory = new Map<string,PendingPublication>();
/** Inventory only the current account's retained requests, without exposing mutable receipts. */
export function heldPublications(owner:string):PendingPublication[]{
  if(!owner)return [];
  return [...memory.values()].filter(row=>row.receipt.owner===owner&&valid(row.receipt,owner,row.receipt.kind,row.receipt.scope)).map(row=>structuredClone(row));
}
const token = /^[a-f0-9]{32}$/;
const moduleId = /^[a-z0-9_-]+\/[a-z0-9_-]+$/;
export const publicationKey = (owner:string,kind:PublicationKind,scope:string) => `paperleet.community.publication.v1:${encodeURIComponent(owner)}:${kind}:${encodeURIComponent(scope)}`;
export const publicationSignature = (payload:PublicationPayload) => JSON.stringify(payload,Object.keys(payload).sort());
// This is also used during render. Invalid input stays visible in the draft;
// preparePublication rejects it before a request can be sent.
export const postPublication = (draft:PostDraft,topic?:string,solution=false):PublicationPayload => ({...((draft.tagIds?.length)?{tag_ids:[...draft.tagIds]}:{}),title:draft.title.trim(),body:draft.body.trim(),category:solution?'implementation':draft.category,problem_id:topic || (solution?draft.problemId:null) || null,...(solution?{solution:true}:{}),...(draft.anonymous?{anonymous:true}:{}),
  ...(draft.category==='papers'&&draft.paperUrl?.trim()?{paper_url:canonicalArxivUrl(draft.paperUrl)||draft.paperUrl}:{})});
export const replyPublication = (draft:ReplyDraft):PublicationPayload => ({body:draft.body.trim(),parent_id:draft.parentId || null});
export const samePublication = (receipt:PublicationReceipt,payload:PublicationPayload) => publicationSignature(receipt.payload)===publicationSignature(payload);
function valid(receipt:PublicationReceipt,owner:string,kind:PublicationKind,scope:string) {
  if(!owner || owner.length>256 || !scope || !receipt || receipt.version!==1 || receipt.owner!==owner || receipt.kind!==kind || receipt.scope!==scope || typeof receipt.requestId!=='string' || !token.test(receipt.requestId))return false;
  const p=receipt.payload;
  if(!p || typeof p!=='object' || Array.isArray(p) || typeof p.body!=='string' || !p.body.trim() || p.body.length>(kind==='reply'?10000:20000))return false;
  if(Object.hasOwn(p,'anonymous')&&(kind==='reply'||p.anonymous!==true))return false;
  if(Object.hasOwn(p,'solution')&&(kind!=='post'||p.solution!==true||p.category!=='implementation'||typeof p.problem_id!=='string'||!moduleId.test(p.problem_id)))return false;
  if(Object.hasOwn(p,'tag_ids')&&(kind==='reply'||!validCommunityTags(p.tag_ids)))return false;
  const keys=Object.keys(p).filter(key=>key!=='anonymous'&&key!=='solution'&&key!=='tag_ids').sort().join(',');
  if(kind==='interview'){
    if(keys!=='body,category,interview,problem_id,title'||scope!=='general'||p.category!=='career'||p.problem_id!==null||typeof p.title!=='string'||!p.title.trim()||p.title.length>180||typeof p.interview!=='string'||p.interview.length>10000)return false;
    try{return !!readInterviewDetails(JSON.parse(p.interview));}catch{return false;}
  }
  return kind==='post' ? (keys==='body,category,problem_id,title'||keys==='body,category,paper_url,problem_id,title') && typeof p.title==='string' && !!p.title.trim() && p.title.length<=180
    && isCommunityCategory(p.category)
    && (!Object.hasOwn(p,'paper_url') || p.paper_url===null || (p.category==='papers' && typeof p.paper_url==='string' && canonicalArxivUrl(p.paper_url)===p.paper_url))
    && (p.solution===true ? scope==='general:solutions'||scope===p.problem_id+':solutions' : scope==='general'?p.problem_id===null:moduleId.test(scope)&&p.problem_id===scope)
    : keys==='body,parent_id' && token.test(scope) && (p.parent_id===null || typeof p.parent_id==='string' && token.test(p.parent_id));
}
export function readPublication(storage:DraftStorage|undefined,owner:string,kind:PublicationKind,scope:string):PendingPublication|undefined {
  if(!owner||!scope)return;
  const key=publicationKey(owner,kind,scope),held=memory.get(key);if(held)return held;
  try {
    const raw=storage?.getItem(key);if(!raw||raw.length>200000)return;
    const receipt=JSON.parse(raw);if(valid(receipt,owner,kind,scope))return {receipt,durable:true};
  }catch{/* In-page retries can retain their identifier when tab storage is blocked. */}
}
export function preparePublication(storage:DraftStorage|undefined,owner:string,kind:PublicationKind,scope:string,payload:PublicationPayload,previous?:PublicationReceipt):PendingPublication {
  const key=publicationKey(owner,kind,scope),held=readPublication(storage,owner,kind,scope);
  if(previous && held && held.receipt.requestId!==previous.requestId)throw Error('The pending publication changed. Review it before retrying.');
  const receipt=previous || (held&&samePublication(held.receipt,payload)?held.receipt:{version:1 as const,owner,kind,scope,requestId:crypto.randomUUID().replaceAll('-',''),payload:structuredClone(payload)});
  if(!valid(receipt,owner,kind,scope))throw Error('Invalid publication request');
  let durable=false;
  try {if(storage){storage.setItem(key,JSON.stringify(receipt));durable=true;}}catch{/* The same page retains the request ID in memory. */}
  const pending={receipt,durable};memory.set(key,pending);return pending;
}
export function clearPublication(storage:DraftStorage|undefined,receipt:PublicationReceipt) {
  const {owner,kind,scope}=receipt,key=publicationKey(owner,kind,scope),current=readPublication(storage,owner,kind,scope);
  if(current?.receipt.requestId!==receipt.requestId)return false;
  try {
    const raw=storage?.getItem(key),stored=raw?JSON.parse(raw):undefined;
    if(stored && stored.requestId!==receipt.requestId)return false;
    storage?.removeItem(key);memory.delete(key);return true;
  }catch{return false;}
}
