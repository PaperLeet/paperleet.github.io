import {communityFeedHref,communityPostReturn,readCommunityFeed} from './communityNavigation';
import {interviewHref,readInterviewRoute} from './interviewNavigation';
import {libraryReturnHref} from './libraryNavigation';

export type ReadingReturn={version:1;id:number;owner:string;href:string;post:string;link:'title'|'read'|'practice';practice?:string;top:number;focus:boolean;at:number;phase:'opening'|'reading'|'returning'};
export const READING_RETURN_KEY='paperleet.reading.return.v1';
const maxAge=6*60*60*1000;
let pending:ReadingReturn|undefined,hydrated=false,sequence=0,revision=0,watching=false;
const listeners=new Set<()=>void>();
export const readingReturnRevision=()=>revision;
export function subscribeReadingReturn(listener:()=>void){listeners.add(listener);return()=>{listeners.delete(listener);};}
const notify=()=>{revision++;for(const listener of listeners)listener();};

export function readingListHref(route:string){
 if(route.length>8192)return;
 const feed=readCommunityFeed(route);
 if(feed)return communityFeedHref(feed.topicId,feed.state);
 const interviews=readInterviewRoute(route);
 return interviews?.view==='reports'?interviewHref(interviews):undefined;
}
const moduleId=/^(?:foundations|rl|architecture|diffusion|finetuning|optimization|representation|world_models|retrieval)\/[a-z0-9_-]{1,100}$/;
/** Only an actual module carrying the exact interview-list scope can extend a reading journey. */
export function readingPracticeTarget(route:string,href:string){
 if(readInterviewRoute(href)?.view!=='reports'||libraryReturnHref(route)!==href)return;
 try{
  const path=decodeURIComponent(route.replace(/^#\/?/,'').split('?')[0]);
  return moduleId.test(path)||/^[1-9]\d{0,7}$/.test(path)?path:undefined;
 }catch{return;}
}
function valid(value:unknown):value is ReadingReturn{
 if(!value||typeof value!=='object'||Array.isArray(value))return false;
 const row=value as ReadingReturn;
 return Object.keys(row).every(key=>['version','id','owner','href','post','link','practice','top','focus','at','phase'].includes(key))&&row.version===1
  &&Number.isSafeInteger(row.id)&&row.id>0&&typeof row.owner==='string'&&/^(?:guest|account:[a-zA-Z0-9_-]{1,160})$/.test(row.owner)
  &&typeof row.href==='string'&&readingListHref(row.href)===row.href&&typeof row.post==='string'&&/^[a-f0-9]{32}$/.test(row.post)
  &&(row.link==='practice'?typeof row.practice==='string'&&moduleId.test(row.practice)&&readInterviewRoute(row.href)?.view==='reports':['title','read'].includes(row.link)&&!Object.hasOwn(row,'practice'))&&Number.isFinite(row.top)&&Math.abs(row.top)<=1e5
  &&typeof row.focus==='boolean'&&Number.isSafeInteger(row.at)&&row.at<=Date.now()&&Date.now()-row.at<=maxAge
  &&['opening','reading','returning'].includes(row.phase);
}
function persist(){
 try{if(pending)sessionStorage.setItem(READING_RETURN_KEY,JSON.stringify(pending));else sessionStorage.removeItem(READING_RETURN_KEY);}catch{/* A blocked store still permits the current tab's in-memory return. */}
}
export function discardReadingReturn(id?:number){
 if(id!==undefined&&pending?.id!==id)return;
 pending=undefined;hydrated=true;
 if(watching){window.removeEventListener('hashchange',navigation);watching=false;}
 persist();notify();
}
export function currentReadingReturn():ReadingReturn|undefined{
 if(!hydrated){
  hydrated=true;
  try{const raw=sessionStorage.getItem(READING_RETURN_KEY);if(raw&&raw.length<=12000){const row:unknown=JSON.parse(raw);if(valid(row)){pending=row;sequence=Math.max(sequence,row.id);}}}catch{/* Malformed storage never blocks a feed. */}
 }
 if(pending){const id=pending.id;if(!valid(pending)){discardReadingReturn(id);return;}}
 return pending?{...pending}:undefined;
}
const isPost=(route:string,row:ReadingReturn)=>communityPostReturn(route,row.post)===row.href;
const isDestination=(route:string,row:ReadingReturn)=>isPost(route,row)||!!readingPracticeTarget(route,row.href);
function navigation(event:HashChangeEvent){
 const row=currentReadingReturn();if(!row)return;
 try{
  const before=new URL(event.oldURL).hash,after=new URL(event.newURL).hash;
  if(isDestination(after,row))pending={...row,phase:'reading'};
  else if(row.phase==='reading'&&isDestination(before,row)&&readingListHref(after)===row.href)pending={...row,phase:'returning'};
  else{discardReadingReturn(row.id);return;}
  persist();notify();
 }catch{discardReadingReturn(row.id);}
}
function watch(){if(!watching&&typeof window!=='undefined'){window.addEventListener('hashchange',navigation);watching=true;}}
export function keepReadingOwner(owner:string|undefined){
 const row=currentReadingReturn();if(!row)return;
 if(owner&&row.owner!==owner){discardReadingReturn(row.id);return;}
 watch();
}
export function rememberReadingReturn(value:Omit<ReadingReturn,'version'|'id'|'at'|'phase'>){
 currentReadingReturn();
 const row:ReadingReturn={...value,version:1,id:++sequence,at:Date.now(),phase:'opening'};
 if(!valid(row))return;
 pending=row;hydrated=true;persist();watch();notify();
}
/** Re-arm navigation after a reload of the actual thread, never after a direct feed visit. */
export function resumeReadingOrigin(owner:string|undefined,post:string,href:string|undefined){
 keepReadingOwner(owner);
 const row=currentReadingReturn();if(!owner||!row)return;
 if(row.post!==post||row.href!==href||!isPost(location.hash,row)){discardReadingReturn(row.id);return;}
 if(row.phase!=='reading'){pending={...row,phase:'reading'};persist();notify();}
 watch();
}
/** Re-arm after a workspace reload only after its real catalog entry confirms the route. */
export function resumePracticeReadingOrigin(owner:string|undefined,problem:{id:string;number:number}|undefined,href:string|undefined,expectedRoute?:string){
 keepReadingOwner(owner);
 const row=currentReadingReturn();if(!owner||!row||!problem||!href)return;
 // An outgoing workspace's passive effect must not erase a newer route or return.
 if(expectedRoute&&readingPracticeTarget(expectedRoute,href)!==readingPracticeTarget(location.hash,href))return;
 const target=readingPracticeTarget(location.hash,row.href);
 if(row.href!==href||!moduleId.test(problem.id)||target!==problem.id&&target!==String(problem.number)){discardReadingReturn(row.id);return;}
 if(row.phase!=='reading'){pending={...row,phase:'reading'};persist();notify();}
 watch();
}
