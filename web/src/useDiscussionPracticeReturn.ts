import {useEffect,useLayoutEffect,useRef,useState,useSyncExternalStore} from 'react';
import type {useDiscussionReplies} from './useDiscussionReplies';
import {captureReadingAnchor,restoreReadingAnchor,keepReadingTriggerVisible,revealReadingTarget,readingScrollRoot,type ReadingAnchor} from './readingViewport';
import {currentDiscussionReturn,keepDiscussionOwner,rememberDiscussionReturn,discardDiscussionReturn,discussionReturnRevision,subscribeDiscussionReturn,discussionThreadHere,discussionPracticeTarget,type DiscussionReturn} from './discussionPracticeReturn';

export type DiscussionReturnNotice={kind:'available'|'expired'|'restoring'|'restored'|'nearby'|'missing';request:number;row:DiscussionReturn};
type List=ReturnType<typeof useDiscussionReplies>;
type Pending={row:DiscussionReturn;request:number;href:string;before:number;opener?:HTMLElement;stage:'view'|'notice'|'frame'};
export function useDiscussionPracticeReturn({owner,post,list,ready,linkedReply}:{owner:string|undefined;post:string;list:List;ready:boolean;linkedReply?:string}){
 const root=useRef<HTMLElement>(null),live=useRef({owner,post,list,ready,linkedReply});live.current={owner,post,list,ready,linkedReply};
 const interactions=useRef(0),attempted=useRef<number|undefined>(undefined),requests=useRef(0),pending=useRef<Pending|undefined>(undefined),frame=useRef<number|undefined>(undefined);
 const dismissing=useRef<{href:string;before:number;opener:HTMLElement;target?:HTMLElement;anchor?:ReadingAnchor}|undefined>(undefined);
 const [notice,setNotice]=useState<DiscussionReturnNotice>(),[visibility,setVisibility]=useState(0),revision=useSyncExternalStore(subscribeDiscussionReturn,discussionReturnRevision,()=>0);
 const here=()=>!!owner&&live.current.owner===owner&&live.current.post===post&&!!root.current?.isConnected&&root.current.dataset.discussionPost===post&&discussionThreadHere(location.hash,post);
 const allowed=(before:number,opener?:HTMLElement)=>interactions.current===before&&(opener?document.activeElement===opener||document.activeElement===document.body&&!opener.isConnected:interactions.current===0&&(document.activeElement===document.body||!document.activeElement?.isConnected));
 function zone(anchor:Pick<DiscussionReturn['anchor'],'place'|'reply'>){
  if(anchor.place==='reply')return root.current?.querySelector<HTMLElement>(`[data-discussion-reply="${anchor.reply}"]`);
  const selector={topic:'.problem-discussion-context',practice:'.discussion-post .interview-report-practice',next:'.discussion-post .interview-report-practice',mentions:'.discussion-post .practice-mentions',article:'.discussion-post .discussion-reading-body',other:''}[anchor.place];
  return selector?root.current?.querySelector<HTMLElement>(selector):root.current;
 }
 function links(host:HTMLElement,href:string,place:DiscussionReturn['anchor']['place']){return [...host.querySelectorAll<HTMLAnchorElement>('a[href]')].filter(node=>!node.closest('[hidden]')&&new URL(node.href,location.href).hash===href&&(place!=='next'||node.classList.contains('interview-report-practice-next'))&&(place!=='practice'||!node.classList.contains('interview-report-practice-next')));}
 function find(row:DiscussionReturn){
  const host=zone(row.anchor);if(!host)return;
  const found=links(host,row.anchor.href,row.anchor.place),link=found[row.anchor.index];
  return link?{node:link,exact:true}:{node:found[0]||host,exact:false};
 }
 function offer(row:DiscussionReturn){const request=++requests.current;setNotice(value=>value?.row.id===row.id&&value.kind==='available'?value:{row,request,kind:'available'});}
 function restore(opener?:HTMLElement){
  const row=currentDiscussionReturn();if(!here()||!live.current.ready||live.current.list.disabled)return;if(!row||opener&&notice&&row.id!==notice.row.id){if(notice)setNotice({...notice,kind:'expired',request:++requests.current});pending.current=undefined;return;}if(row.owner!==owner||row.post!==post||row.phase==='opening')return;
  if(!allowed(interactions.current,opener)){offer(row);return;}
  const request=++requests.current;attempted.current=row.id;
  if(!live.current.list.restoreView(row.view,row.expanded,request,row.anchor.reply))return;
  pending.current={row,request,href:location.href,before:interactions.current,opener,stage:'view'};setNotice({row,request,kind:'restoring'});
 }
 useEffect(()=>{
  interactions.current=0;
  const intent=()=>{interactions.current++;},scroll=()=>{if(window.scrollY>0)intent();},visible=()=>{if(!document.hidden)setVisibility(value=>value+1);};
  for(const name of ['pointerdown','keydown','wheel','touchmove','focusin'])window.addEventListener(name,intent,{passive:true});
  window.addEventListener('scroll',scroll,{passive:true});document.addEventListener('visibilitychange',visible);
  const capture=(event:MouseEvent)=>{
   if(!here()||!live.current.ready||live.current.list.disabled||event.defaultPrevented||event.button!==0||event.metaKey||event.ctrlKey||event.altKey||event.shiftKey||!(event.target instanceof window.Element))return;
   const link=event.target.closest<HTMLAnchorElement>('a[href]');if(!link||!root.current?.contains(link)||link.target&&link.target!=='_self'||link.hasAttribute('download')||link.closest('[hidden]')||readingScrollRoot(link))return;
   const target=new URL(link.href,location.href),practice=discussionPracticeTarget(target.hash,post);
   if(target.origin!==location.origin||target.pathname!==location.pathname||target.search!==location.search||!practice)return;
   const reply=link.closest<HTMLElement>('[data-discussion-reply]')?.dataset.discussionReply,current=live.current.list;
   const opened=[...root.current.querySelectorAll<HTMLButtonElement>('.reply-search-full[aria-expanded=true]')].flatMap(button=>{const id=button.closest<HTMLElement>('[data-discussion-reply]')?.dataset.discussionReply,row=id?current.byId.get(id):undefined;return row?.edit_version&&/^[a-f0-9]{64}$/.test(row.edit_version)?[{reply:row.id,revision:row.edit_version}]:[];});
   const expanded=[...opened.filter(row=>row.reply===reply),...opened.filter(row=>row.reply!==reply)].slice(0,100);
   const module=link.dataset.readingPractice||practice;
   const place:DiscussionReturn['anchor']['place']=reply?'reply':link.closest('.problem-discussion-context')?'topic':link.classList.contains('interview-report-practice-next')?'next':link.closest('.interview-report-practice')?'practice':link.closest('.practice-mentions')?'mentions':link.closest('.discussion-reading-body')?'article':'other';
   const host=zone({place,reply});if(!host)return;const index=links(host,target.hash,place).indexOf(link);if(index<0)return;
   rememberDiscussionReturn({owner:owner!,post,practice,view:current.snapshotView(),expanded,anchor:{href:target.hash,place,index,...(reply?{reply}:{}),...(!/^\d+$/.test(module)?{module}:{}),top:link.getBoundingClientRect().top,keyboard:event.detail===0&&document.activeElement===link}});
  };
  document.addEventListener('click',capture);
  return()=>{for(const name of ['pointerdown','keydown','wheel','touchmove','focusin'])window.removeEventListener(name,intent);window.removeEventListener('scroll',scroll);document.removeEventListener('visibilitychange',visible);document.removeEventListener('click',capture);pending.current=undefined;if(frame.current!==undefined)window.cancelAnimationFrame(frame.current);};
 },[owner,post]);
 useLayoutEffect(()=>{
  keepDiscussionOwner(owner);if(!here()||!ready)return;
  const row=currentDiscussionReturn();if(!row||row.owner!==owner||row.post!==post||row.phase==='opening'||pending.current)return;
  if(linkedReply||row.phase!=='returning'){offer(row);return;}
  if(attempted.current===row.id||document.hidden)return;
  attempted.current=row.id;restore();
 },[owner,post,ready,linkedReply,revision,visibility]);
 useLayoutEffect(()=>{
  const request=pending.current;if(!request||!notice||notice.request!==request.request||request.stage==='frame')return;
  const current=currentDiscussionReturn();
  if(!here()||!current||current.id!==request.row.id||!live.current.ready||!allowed(request.before,request.opener)||location.href!==request.href){pending.current=undefined;if(current?.id===request.row.id)offer(current);return;}
  if(document.hidden)return;
  if(request.stage==='view'){
   if(list.restoreRequest!==request.request)return;
   const target=find(request.row);request.stage='notice';setNotice({row:request.row,request:request.request,kind:target?target.exact?'restored':'nearby':'missing'});return;
  }
  if(root.current?.querySelector<HTMLElement>('[data-discussion-return-request]')?.dataset.discussionReturnRequest!==String(request.request))return;
  request.stage='frame';frame.current=window.requestAnimationFrame(()=>{
   frame.current=undefined;pending.current=undefined;
   const row=currentDiscussionReturn();if(!here()||!row||row.id!==request.row.id||!live.current.ready||location.href!==request.href||!allowed(request.before,request.opener)){if(row?.id===request.row.id)offer(row);return;}
   if(document.hidden){request.stage='notice';pending.current=request;return;}
   const target=find(request.row),kind=target?target.exact?'restored':'nearby':'missing';if(notice.kind!==kind){request.stage='notice';pending.current=request;setNotice({row:request.row,request:request.request,kind});return;}const node=target?.node||live.current.list.root.current;
   if(node){if(target)restoreReadingAnchor(node,{viewport:null,top:request.row.anchor.top});else revealReadingTarget(node);if(request.row.anchor.keyboard||request.opener)(node.matches('a,[tabindex]')?node:root.current)?.focus({preventScroll:true});keepReadingTriggerVisible(node);}
   discardDiscussionReturn(row.id);
  });
 },[notice,list.restoreRequest,ready,visibility]);
 useLayoutEffect(()=>{
  const request=dismissing.current;if(notice||!request)return;dismissing.current=undefined;
  if(!here()||location.href!==request.href||request.before!==interactions.current||document.activeElement!==request.opener&&!(document.activeElement===document.body&&!request.opener.isConnected))return;
  if(request.target?.isConnected&&request.anchor)restoreReadingAnchor(request.target,request.anchor);
  const target=request.target||root.current;if(target){target.focus({preventScroll:true});if(request.target)keepReadingTriggerVisible(target);}
 },[notice]);
 const visible=notice&&notice.row.owner===owner&&notice.row.post===post?notice:undefined;
 return {root,notice:visible,resumedPractice:visible&&visible.kind!=='available'&&visible.kind!=='expired'?visible.row.anchor.module:undefined,restore,
  dismiss(opener:HTMLElement){
   if(!here())return;
   const header=root.current!.closest('.app')?.querySelector<HTMLElement>(':scope > .topbar'),bounds=header?.getBoundingClientRect(),visual=window.visualViewport,validVisual=!!visual&&Number.isFinite(visual.offsetTop)&&Number.isFinite(visual.height)&&visual.height>0,top=Math.max(12,bounds?.bottom||0,validVisual?visual.offsetTop+12:0),bottom=validVisual?Math.min(window.innerHeight,visual.offsetTop+visual.height):window.innerHeight;
   const fullyVisible=(node:HTMLElement)=>{const rect=node.getBoundingClientRect();return !node.matches(':disabled')&&!node.closest('[hidden],.discussion-practice-return')&&rect.height>0&&rect.top>=top&&rect.bottom<=bottom;};
   const preferred=visible?find(visible.row)?.node:undefined,nodes=[...root.current!.querySelectorAll<HTMLElement>('a[href],button,input,textarea,[data-discussion-reply]')],target=preferred&&fullyVisible(preferred)?preferred:nodes.find(fullyVisible);
   dismissing.current={href:location.href,before:interactions.current,opener,target,anchor:target?captureReadingAnchor(target):undefined};pending.current=undefined;if(frame.current!==undefined)window.cancelAnimationFrame(frame.current);if(visible)discardDiscussionReturn(visible.row.id);setNotice(undefined);
  },
  revealOriginal(){if(!here()||!live.current.ready||live.current.list.disabled||!visible?.row.anchor.reply)return;live.current.list.reveal(visible.row.anchor.reply);setNotice(undefined);},
 };
}
