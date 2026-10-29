import {appLocation, routeHash} from './appLocation';
import {communityPostHref,communityContinuation} from './communityNavigation';
import {libraryReturnHref} from './libraryNavigation';
import {replySearchQuery} from './replySearch';

export type DiscussionView={order:'oldest'|'newest';authorOnly:boolean;query:string;limit:number};
export type DiscussionExpansion={reply:string;revision:string};
export type DiscussionReturn={version:1;id:number;owner:string;post:string;practice:string;at:number;phase:'opening'|'practice'|'returning';view:DiscussionView;expanded:DiscussionExpansion[];anchor:{href:string;reply?:string;module?:string;place:'reply'|'topic'|'practice'|'next'|'mentions'|'article'|'other';index:number;top:number;keyboard:boolean}};
export const DISCUSSION_RETURN_KEY='paperleet.discussion.reading.v1';
const postId=/^[a-f0-9]{32}$/,ownerId=/^(?:guest|account:[a-zA-Z0-9_-]{1,160})$/,moduleId=/^(?:foundations|rl|architecture|diffusion|finetuning|optimization|representation|world_models|retrieval|self_evolving|network_biology|multimodal|robotics|ai_infra|diffusion_rl)\/[a-z0-9_-]{1,100}$/;
let pending:DiscussionReturn|undefined,hydrated=false,sequence=0,revision=0,watching=false;
const listeners=new Set<()=>void>();
const notify=()=>{revision++;for(const listener of listeners)listener();};
export const discussionReturnRevision=()=>revision;
export function subscribeDiscussionReturn(listener:()=>void){listeners.add(listener);return()=>{listeners.delete(listener);};}
const keys=(value:object,names:string[])=>Object.keys(value).every(name=>names.includes(name));
export function discussionThreadHere(route:string,post:string){const continuation=communityContinuation(route);return !!continuation&&continuation.split('?')[0]==='community/'+post;}
export function discussionPracticeTarget(route:string,post:string){
 if(libraryReturnHref(route)!==communityPostHref(post))return;
 try{const path=decodeURIComponent(route.replace(/^#\/?/,'').split('?')[0]);return moduleId.test(path)||/^[1-9]\d{0,7}$/.test(path)?path:undefined;}catch{return;}
}
function valid(value:unknown):value is DiscussionReturn{
 if(!value||typeof value!=='object'||Array.isArray(value))return false;
 const row=value as DiscussionReturn,view=row.view,anchor=row.anchor;
 return keys(row,['version','id','owner','post','practice','at','phase','view','expanded','anchor'])&&row.version===1&&Number.isSafeInteger(row.id)&&row.id>0&&typeof row.owner==='string'&&ownerId.test(row.owner)&&typeof row.post==='string'&&postId.test(row.post)
  &&typeof row.practice==='string'&&(moduleId.test(row.practice)||/^[1-9]\d{0,7}$/.test(row.practice))&&Number.isSafeInteger(row.at)&&row.at<=Date.now()&&Date.now()-row.at<=6*60*60*1000&&['opening','practice','returning'].includes(row.phase)
  &&!!view&&typeof view==='object'&&!Array.isArray(view)&&keys(view,['order','authorOnly','query','limit'])&&['oldest','newest'].includes(view.order)&&typeof view.authorOnly==='boolean'&&typeof view.query==='string'&&replySearchQuery(view.query)===view.query&&Number.isSafeInteger(view.limit)&&view.limit>=1&&view.limit<=1_000_000
  &&!!anchor&&typeof anchor==='object'&&!Array.isArray(anchor)&&keys(anchor,['href','reply','module','place','index','top','keyboard'])&&typeof anchor.href==='string'&&anchor.href.length<=1000&&!!discussionPracticeTarget(anchor.href,row.post)&&(!Object.hasOwn(anchor,'reply')||typeof anchor.reply==='string'&&postId.test(anchor.reply))&&(!Object.hasOwn(anchor,'module')||typeof anchor.module==='string'&&moduleId.test(anchor.module))&&['reply','topic','practice','next','mentions','article','other'].includes(anchor.place)&&(anchor.place!=='reply'||!!anchor.reply)&&Number.isSafeInteger(anchor.index)&&anchor.index>=0&&anchor.index<=10000&&Number.isFinite(anchor.top)&&Math.abs(anchor.top)<=1e5&&typeof anchor.keyboard==='boolean'
  &&Array.isArray(row.expanded)&&row.expanded.length<=100&&new Set(row.expanded.map(value=>value?.reply)).size===row.expanded.length&&row.expanded.every(value=>!!value&&typeof value==='object'&&!Array.isArray(value)&&keys(value,['reply','revision'])&&typeof value.reply==='string'&&postId.test(value.reply)&&typeof value.revision==='string'&&/^[a-f0-9]{64}$/.test(value.revision));
}
const copy=(row:DiscussionReturn):DiscussionReturn=>({...row,view:{...row.view},expanded:row.expanded.map(value=>({...value})),anchor:{...row.anchor}});
function persist(){try{if(pending)sessionStorage.setItem(DISCUSSION_RETURN_KEY,JSON.stringify(pending));else sessionStorage.removeItem(DISCUSSION_RETURN_KEY);}catch{/* Only the current owner/tab's in-memory return remains available. */}}
export function discardDiscussionReturn(id?:number){if(id!==undefined&&pending?.id!==id)return;pending=undefined;hydrated=true;if(watching){window.removeEventListener('hashchange',navigation);watching=false;}persist();notify();}
export function currentDiscussionReturn(){
 if(!hydrated){hydrated=true;try{const raw=sessionStorage.getItem(DISCUSSION_RETURN_KEY);if(raw&&raw.length<=24000){const value:unknown=JSON.parse(raw);if(valid(value)){pending=copy(value);sequence=Math.max(sequence,value.id);}}}catch{/* Corrupt storage must not block reading. */}}
 if(pending){const id=pending.id;if(!valid(pending)){discardDiscussionReturn(id);return;}}
 return pending?copy(pending):undefined;
}
export function keepDiscussionOwner(owner:string|undefined){const row=currentDiscussionReturn();if(owner&&row&&row.owner!==owner)discardDiscussionReturn(row.id);}
function watch(){if(!watching&&typeof window!=='undefined'){window.addEventListener('hashchange',navigation);watching=true;}}
function navigation(event:HashChangeEvent){
 const row=currentDiscussionReturn();if(!row)return;
 try{
  const before=routeHash(event.oldURL),after=routeHash(event.newURL);if(after!==appLocation.hash)return;
  const target=discussionPracticeTarget(after,row.post);
  if(target&&(row.phase!=='opening'||discussionThreadHere(before,row.post)))pending={...row,practice:target,phase:'practice'};
  else if(row.phase==='practice'&&discussionPracticeTarget(before,row.post)&&discussionThreadHere(after,row.post))pending={...row,phase:'returning'};
  else if(row.phase==='returning'&&discussionThreadHere(after,row.post))return;
  else{discardDiscussionReturn(row.id);return;}
  persist();notify();
 }catch{discardDiscussionReturn(row.id);}
}
export function rememberDiscussionReturn(value:Omit<DiscussionReturn,'version'|'id'|'at'|'phase'>){
 currentDiscussionReturn();const row:DiscussionReturn={...value,version:1,id:++sequence,at:Date.now(),phase:'opening'};if(!valid(row))return;
 pending=copy(row);hydrated=true;persist();watch();notify();return row.id;
}
/** A real catalog entry must confirm the current workspace after a reload. */
export function confirmDiscussionPractice(owner:string|undefined,problem:{id:string;number:number}|undefined,href:string|undefined,expected:string){
 keepDiscussionOwner(owner);const row=currentDiscussionReturn();if(!owner||!row||row.owner!==owner||!problem||!href||href!==communityPostHref(row.post)||!moduleId.test(problem.id))return;
 const target=discussionPracticeTarget(expected,row.post);if(!target||target!==discussionPracticeTarget(appLocation.hash,row.post)||target!==problem.id&&target!==String(problem.number))return;
 if(row.phase==='returning')return;
 const original=discussionPracticeTarget(row.anchor.href,row.post);
 pending={...row,practice:problem.id,phase:'practice',anchor:!row.anchor.module&&(original===problem.id||original===String(problem.number))?{...row.anchor,module:problem.id}:row.anchor};persist();watch();notify();
}
