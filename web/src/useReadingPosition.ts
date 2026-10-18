import {appLocation, routeHash, sameAppUrl} from './appLocation';
import {useEffect,useLayoutEffect,useRef,useState,useSyncExternalStore} from 'react';
import {communityPostReturn,communityReplyTarget} from './communityNavigation';
import {currentReadingReturn,discardReadingReturn,keepReadingOwner,readingListHref,readingPracticeTarget,readingSourceConcept,privateProfilePostRoute,readingReturnRevision,rememberReadingReturn,resumeReadingOrigin,subscribeReadingReturn,type ReadingReturn} from './readingReturn';
import {readInterviewRoute} from './interviewNavigation';

export type ReadingNotice='restored'|'available'|'missing';
type State={owner:string;href:string;id:number;notice:ReadingNotice;region?:ReadingReturn['region'];post?:string;practice?:string;reply?:string};
export function useReadingOrigin(owner:string|undefined,post:string,href:string|undefined){
 useEffect(()=>resumeReadingOrigin(owner,post,href),[owner,post,href]);
}

/** A return belongs to one owner, exact feed and actual thread/practice traversal. No draft text is retained. */
export function useReadingPosition(owner:string|undefined,href:string,ready:boolean,{highlightsEnabled=false,highlightsReady}:{highlightsEnabled?:boolean;highlightsReady?:number}={}){
 const container=useRef<HTMLElement>(null),live=useRef({owner,href,ready});live.current={owner,href,ready};
 const interactions=useRef(0),previousOwner=useRef(owner);
 const scrollRequest=useRef<{row:ReadingReturn;opener?:HTMLElement;before:number}|undefined>(undefined);
 const [state,setState]=useState<State>();
 const [returning,setReturning]=useState<ReadingReturn>();
 const revision=useSyncExternalStore(subscribeReadingReturn,readingReturnRevision,()=>0);
 const here=()=>live.current.owner===owner&&live.current.href===href&&live.current.ready&&readingListHref(appLocation.hash)===href&&!!container.current?.isConnected;
 const targetFor=(row:ReadingReturn)=>{
  const card=container.current?.querySelector<HTMLElement>(`[data-reading-post="${row.post}"]${row.region?'[data-reading-region="highlights"]':':not([data-reading-region="highlights"])'}${row.concept?`[data-reading-concept="${row.concept}"]`:""}${row.reply?`[data-reading-reply="${row.reply}"]`:""}`);
  if(row.link==='practice')return card?.querySelector<HTMLElement>(`a[data-reading-link="practice"][data-reading-practice="${row.practice}"]`)||card?.querySelector<HTMLElement>('a[data-reading-link="title"]')||card?.querySelector<HTMLElement>('a[data-reading-link="read"]');
  return card?.querySelector<HTMLElement>(`a[data-reading-link="${row.link}"]`)||card?.querySelector<HTMLElement>('a[data-reading-link]');
 };
 const permitted=(before:number,opener?:HTMLElement)=>{
  const active=document.activeElement;
  return interactions.current===before&&(opener?active===opener||!opener.isConnected&&active===document.body:interactions.current===0&&(active===document.body||!active?.isConnected));
 };
 const waitingForHighlights=(row:ReadingReturn)=>row.region==='highlights'&&highlightsEnabled&&(highlightsReady!==row.id||container.current?.querySelector<HTMLDetailsElement>('details[data-reading-region="highlights"]')?.dataset.readingReady!==String(row.id));
 function apply(row:ReadingReturn,opener?:HTMLElement){
  if(!here()||row.owner!==owner||row.href!==href)return;
  if(waitingForHighlights(row))return;
  const before=interactions.current;
  if(!permitted(before,opener)){setState({owner:row.owner,href,id:row.id,region:row.region,notice:'available'});return;}
  scrollRequest.current={row,opener,before};
  setState({owner:row.owner,href,id:row.id,region:row.region,notice:targetFor(row)?'restored':'missing',post:targetFor(row)?row.post:undefined,practice:row.practice,reply:row.reply});
 }
 // Commit the notice first, then measure. Its height must not displace the restored card.
 useLayoutEffect(()=>{
   const request=scrollRequest.current;if(!request)return;
   scrollRequest.current=undefined;
   const {row,opener,before}=request,current=currentReadingReturn();
   if(!here()||!current||current.id!==row.id||current.owner!==owner||current.href!==href||current.phase!=='returning'){
    if(state?.id===row.id)setState(undefined);
    return;
   }
   if(waitingForHighlights(row))return;
   if(!permitted(before,opener)){setState({owner:row.owner,href,id:row.id,region:row.region,notice:'available'});return;}
   if(row.region==='highlights'){
    const disclosure=container.current?.querySelector<HTMLDetailsElement>('details[data-reading-region="highlights"]');
    if(disclosure)disclosure.open=true;
   }
   const target=targetFor(row);
   const notice=target?'restored':'missing';
   if(state?.notice!==notice){scrollRequest.current=request;setState({owner:row.owner,href,id:row.id,notice,post:target?row.post:undefined,practice:row.practice,reply:row.reply});return;}
   if(target){
    window.scrollTo({top:Math.max(0,window.scrollY+target.getBoundingClientRect().top-row.top),behavior:'instant'});
    if(row.focus||opener)target.focus({preventScroll:true});
   }else{
    const list=container.current!.querySelector<HTMLElement>(row.region==='highlights'?'details[data-reading-region="highlights"]>summary':'[data-reading-list]')||container.current!.querySelector<HTMLElement>('[data-reading-list]');
    list?.scrollIntoView?.({block:'start',behavior:'instant'});
    if(row.focus||opener)list?.focus({preventScroll:true});
   }
   discardReadingReturn(row.id);
 },[state]);
 useEffect(()=>{
  interactions.current=0;
  const intent=()=>{interactions.current++;};
  const scroll=()=>{if(window.scrollY>0)intent();};
  for(const name of ['wheel','touchmove','pointerdown','keydown','focusin'])window.addEventListener(name,intent,{passive:true});
  window.addEventListener('scroll',scroll,{passive:true});
  const capture=(event:MouseEvent)=>{
   if(!live.current.owner||live.current.href!==href||!live.current.ready||readingListHref(appLocation.hash)!==href||!container.current?.isConnected||event.defaultPrevented||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey||!(event.target instanceof window.Element))return;
   const link=event.target.closest<HTMLAnchorElement>('a[data-reading-link]'),card=link?.closest<HTMLElement>('[data-reading-post]');
   if(!link||!card||!container.current?.contains(card)||link.target&&link.target!=='_self'||link.hasAttribute('download'))return;
   const destination=new URL(link.href,location.href),post=card.dataset.readingPost;
   if(!sameAppUrl(destination)||!post)return;
   const practice=link.dataset.readingLink==='practice'?readingPracticeTarget(routeHash(destination),href):undefined;
   const reply=card.dataset.readingReply;
   if(reply&&(!/^[a-f0-9]{32}$/.test(reply)||communityReplyTarget(routeHash(destination),post)!==reply))return;
   const privateProfile=card.dataset.readingPrivateProfile==='true'&&live.current.owner.startsWith('account:');
   const sourceConcept=readingSourceConcept(href,card.dataset.readingConcept);
   if(readInterviewRoute(href)?.view==='questions'&&!sourceConcept)return;
   if(link.dataset.readingLink==='practice'?practice!==link.dataset.readingPractice||!practice:privateProfile?!privateProfilePostRoute(routeHash(destination),post):communityPostReturn(routeHash(destination),post)!==href)return;
   rememberReadingReturn({owner:live.current.owner!,href,post,link:practice?'practice':link.dataset.readingLink==='read'?'read':'title',...(card.dataset.readingRegion==='highlights'?{region:'highlights' as const}:{}),...(sourceConcept?{concept:sourceConcept}:{}),...(practice?{practice}:{}),...(reply?{reply}:{}),...(privateProfile?{privateProfile:true as const}:{}),top:link.getBoundingClientRect().top,focus:event.detail===0&&document.activeElement===link});
   setState(undefined);
  };
  document.addEventListener('click',capture);
  return()=>{for(const name of ['wheel','touchmove','pointerdown','keydown','focusin'])window.removeEventListener(name,intent);window.removeEventListener('scroll',scroll);document.removeEventListener('click',capture);scrollRequest.current=undefined;};
 },[href]);
 useEffect(()=>{
  if(previousOwner.current&&previousOwner.current!==owner)interactions.current++;
  previousOwner.current=owner;keepReadingOwner(owner);
 },[owner]);
 useLayoutEffect(()=>{
  const row=currentReadingReturn();
  const next=row&&owner&&row.owner===owner&&row.href===href&&row.phase==='returning'?row:undefined;
  setReturning(previous=>previous?.id===next?.id?previous:next);
 },[owner,href,revision]);
 useLayoutEffect(()=>{
  if(!ready){setState(undefined);return;}
  const row=currentReadingReturn();
  if(row&&owner&&row.owner===owner&&row.href===href&&row.phase==='returning')apply(row);
 },[owner,href,ready,revision,highlightsEnabled,highlightsReady]);
 const visible=ready&&state&&state.owner===owner&&state.href===href&&(state.region!=='highlights'||!highlightsEnabled||highlightsReady===state.id)?state:undefined;
 return {container,notice:visible?.notice,region:visible?.region,highlightReturn:returning&&returning.owner===owner&&returning.href===href&&returning.region==='highlights'?{id:returning.id,post:returning.post}:undefined,resumedPost:visible?.post,resumedReply:visible?.reply,resumedPractice:visible?.post?visible.practice:undefined,restore:(opener:HTMLElement)=>{
  const row=currentReadingReturn();if(row&&owner&&row.owner===owner&&row.href===href&&row.phase==='returning')apply(row,opener);
 }};
}
