import {appLocation, routeHash} from './appLocation';
import {readProfileRoute,profileFeedHref} from './profileNavigation';
import {communityFeedHref,communityPostHref,communityPostReturn,communityReplyTarget,communityContinuation,readCommunityFeed} from './communityNavigation';
import {interviewHref,readInterviewRoute} from './interviewNavigation';
import {favoriteHref,readFavoriteRoute,libraryReturnHref} from './libraryNavigation';
import {readReportHistory,reportHistoryHref} from './reportHistoryNavigation';

export type ReadingReturn={version:1;id:number;owner:string;href:string;post:string;link:'title'|'read'|'practice';region?:'highlights';concept?:string;practice?:string;reply?:string;privateProfile?:true;top:number;focus:boolean;at:number;phase:'opening'|'reading'|'returning'};
export const READING_RETURN_KEY='paperleet.reading.return.v1';
const maxAge=6*60*60*1000;
let pending:ReadingReturn|undefined,hydrated=false,sequence=0,revision=0,watching=false;
const listeners=new Set<()=>void>();
export const readingReturnRevision=()=>revision;
export function subscribeReadingReturn(listener:()=>void){listeners.add(listener);return()=>{listeners.delete(listener);};}
const notify=()=>{revision++;for(const listener of listeners)listener();};

export function readingListHref(route:string){
 if(route.length>8192)return;
 const profile=readProfileRoute(route);if(profile&&['posts','solutions','interviews','replies','likes','reposts'].includes(profile.tab))return profileFeedHref(profile.username,profile);
 const favorites=readFavoriteRoute(route);if(favorites)return favoriteHref(favorites);
 const reports=readReportHistory(route);if(reports)return reportHistoryHref(reports.state,reports.page);
 const feed=readCommunityFeed(route);
 if(feed)return communityFeedHref(feed.topicId,feed.state);
 const interviews=readInterviewRoute(route);
 return interviews&&(interviews.view==='reports'||interviews.view==='questions')?interviewHref(interviews):undefined;
}
const moduleId=/^(?:foundations|rl|architecture|diffusion|finetuning|optimization|representation|world_models|retrieval)\/[a-z0-9_-]{1,100}$/;
/** Recommendations belong only to the first, unfiltered community or board feed. */
export function readingHighlightsScope(href:string){
 const feed=readCommunityFeed(href);if(!feed||feed.topicId)return false;
 const f=feed.state;
 return f.page===1&&!f.solutions&&!f.saved&&!f.following&&!f.tag&&!f.search&&!f.sort&&!f.days;
}
/** A source post can occur under several concepts; bind its return to the originating row. */
export function readingSourceConcept(href:string,value:unknown){
 const filters=readInterviewRoute(href);
 return filters?.view==='questions'&&typeof value==='string'&&moduleId.test(value)&&(!filters.concept||filters.concept===value)?value:undefined;
}
function practiceTargetPath(route:string,allowPaper=false){
 try{
  const path=decodeURIComponent(route.replace(/^#\/?/,'').split('?')[0]);
  return moduleId.test(path)||/^[1-9]\d{0,7}$/.test(path)||allowPaper&&/^papers\/[a-z0-9_-]+$/.test(path)?path:undefined;
 }catch{return;}
}
/** Only a module carrying the exact interview/favorite list scope extends its reading journey. */
export function readingPracticeTarget(route:string,href:string){
 if((!readFavoriteRoute(href)&&readInterviewRoute(href)?.view!=='reports')||libraryReturnHref(route)!==href)return;
 return practiceTargetPath(route);
}
function valid(value:unknown):value is ReadingReturn{
 if(!value||typeof value!=='object'||Array.isArray(value))return false;
 const row=value as ReadingReturn;
 const favorite=typeof row.href==='string'&&!!readFavoriteRoute(row.href);
 // Keep the v1 row identifier for stored reading returns. Only favorite views
 // may use a canonical module ID here, bound to the same practice ID and owner.
 const itemValid=typeof row.post==='string'&&(favorite?moduleId.test(row.post)&&row.post===row.practice&&typeof row.owner==='string'&&row.owner.startsWith('account:'):/^[a-f0-9]{32}$/.test(row.post));
 return Object.keys(row).every(key=>['version','id','owner','href','post','link','region','concept','practice','reply','privateProfile','top','focus','at','phase'].includes(key))&&row.version===1
  &&Number.isSafeInteger(row.id)&&row.id>0&&typeof row.owner==='string'&&/^(?:guest|account:[a-zA-Z0-9_-]{1,160})$/.test(row.owner)
  &&typeof row.href==='string'&&readingListHref(row.href)===row.href&&itemValid
  &&(!Object.hasOwn(row,'region')||row.region==='highlights'&&row.link==='title'&&readingHighlightsScope(row.href))
  &&(!readReportHistory(row.href)||row.owner.startsWith('account:'))
  &&(readInterviewRoute(row.href)?.view==='questions'?!!readingSourceConcept(row.href,row.concept):!Object.hasOwn(row,'concept'))
  &&(readProfileRoute(row.href)?.tab==='replies'?typeof row.reply==='string'&&/^[a-f0-9]{32}$/.test(row.reply):!Object.hasOwn(row,'reply'))
  &&(!Object.hasOwn(row,'privateProfile')||row.privateProfile===true&&row.owner.startsWith('account:')&&!!readProfileRoute(row.href))
  &&(row.link==='practice'?typeof row.practice==='string'&&moduleId.test(row.practice)&&(favorite||readInterviewRoute(row.href)?.view==='reports'):['title','read'].includes(row.link)&&!Object.hasOwn(row,'practice'))&&Number.isFinite(row.top)&&Math.abs(row.top)<=1e5
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
export function privateProfilePostRoute(route:string,post:string){
 if(!/^[a-f0-9]{32}$/.test(post))return false;
 return communityContinuation(route)===communityPostHref(post,undefined,communityReplyTarget(route,post)).slice(2);
}
const isPost=(route:string,row:ReadingReturn)=>!readFavoriteRoute(row.href)&&(communityPostReturn(route,row.post)===row.href||(row.privateProfile===true||row.phase==='reading')&&privateProfilePostRoute(route,row.post));
/** Private profile-to-anonymous-thread context never travels in the public URL. */
export function privateProfileReadingReturn(owner:string|undefined,post:string,route:string){
 if(!owner?.startsWith('account:')||!privateProfilePostRoute(route,post))return;
 const row=currentReadingReturn();
 return row?.owner===owner&&row.privateProfile&&row.post===post?row.href:undefined;
}
/** Keep the earlier list private to this owner/tab when practice returns to a source-free post. */
export function sourcePostReadingReturn(owner:string|undefined,post:string,route:string){
 if(!owner||!privateProfilePostRoute(route,post))return;
 const row=currentReadingReturn();
 return row?.owner===owner&&row.post===post&&(row.phase==='reading'||row.privateProfile)?row.href:undefined;
}
function sourcePostPracticeTarget(route:string,post:string){
 if(libraryReturnHref(route)!==communityPostHref(post))return;
 return practiceTargetPath(route,true);
}
const isDestination=(route:string,row:ReadingReturn)=>isPost(route,row)||!!readingPracticeTarget(route,row.href)||!!sourcePostPracticeTarget(route,row.post);
function navigation(event:HashChangeEvent){
 const row=currentReadingReturn();if(!row)return;
 try{
  const before=routeHash(event.oldURL),after=routeHash(event.newURL);
  // A queued hashchange can arrive after a newer navigation or history replacement.
  if(after!==appLocation.hash)return;
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
 if(sourcePostPracticeTarget(appLocation.hash,row.post))return;
 if(row.post!==post||row.href!==href||!isPost(appLocation.hash,row)){discardReadingReturn(row.id);return;}
 if(row.phase!=='reading'){pending={...row,phase:'reading'};persist();notify();}
 watch();
}
/** Re-arm after a workspace reload only after its real catalog entry confirms the route. */
export function resumePracticeReadingOrigin(owner:string|undefined,problem:{id:string;number:number}|undefined,href:string|undefined,expectedRoute?:string){
 keepReadingOwner(owner);
 const row=currentReadingReturn();if(!owner||!row||!problem||!href)return;
 // An outgoing workspace's passive effect must not erase a newer route or return.
 const sourcePost=href===communityPostHref(row.post),targetFor=(route:string)=>sourcePost?sourcePostPracticeTarget(route,row.post):readingPracticeTarget(route,href);
 if(expectedRoute&&(!targetFor(expectedRoute)||targetFor(expectedRoute)!==targetFor(appLocation.hash)))return;
 const target=targetFor(appLocation.hash);
 if(!sourcePost&&row.href!==href||!moduleId.test(problem.id)||target!==problem.id&&target!==String(problem.number)){discardReadingReturn(row.id);return;}
 if(row.phase!=='reading'){pending={...row,phase:'reading'};persist();notify();}
 watch();
}

/** Preserve a previously confirmed journey while its optional workspace observer loads. */
export function replayPracticeReadingOrigin(owner:string,href:string,expected:string,origin:string,events:readonly {before:string;after:string}[],snapshot:string){
 const row=currentReadingReturn();if(!row||row.owner!==owner||row.phase!=='reading'||snapshot.length>12000||events.length<1||events.length>8)return;
 let captured:unknown;try{captured=JSON.parse(snapshot);}catch{return;}
 if(!valid(captured)||JSON.stringify(row)!==JSON.stringify(captured))return;
 const source=href===communityPostHref(row.post),target=(route:string)=>source?sourcePostPracticeTarget(route,row.post):readingPracticeTarget(route,href);
 if(!source&&row.href!==href||!target(origin)||target(expected)!==target(origin)||events.at(-1)?.after!==appLocation.hash)return;
 let previous=origin,phase:ReadingReturn['phase']='reading',retired=false;
 for(const event of events){
  if(typeof event.before!=='string'||typeof event.after!=='string'||event.before.length>8192||event.after.length>8192||event.before!==previous)return;
  previous=event.after;
  if(retired)continue;
  if(isDestination(event.after,row))phase='reading';
  else if(phase==='reading'&&isDestination(event.before,row)&&readingListHref(event.after)===row.href)phase='returning';
  else retired=true;
 }
 if(retired){discardReadingReturn(row.id);return;}
 pending={...row,phase};persist();watch();notify();
}
