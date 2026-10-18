import {useEffect,useId,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {ArrowLeft,ChevronDown,UserRound} from 'lucide-react';
import type {Reply} from './communityDiscussion';
import {useI18n} from './i18n';
import {captureReadingAnchor,restoreReadingAnchor,revealReadingTarget,keepReadingTriggerVisible,type ReadingAnchor} from './readingViewport';
import './discussion-replies.css';
import CommunitySearch from './CommunitySearch';
import {matchesReplySearch,replySearchQuery,replySearchTerms,replySearchText} from './replySearch';
import type {DiscussionView,DiscussionExpansion} from './discussionPracticeReturn';

const PAGE_SIZE=20,MAX_RETURN_DEPTH=12;
const EMPTY:Reply[]=[];
type Order='oldest'|'newest';
type Origin=ReadingAnchor&{href:string;trigger:HTMLElement;row?:string;order:Order;authorOnly:boolean;query:string;limit:number;revealed?:string};
type Intent={href:string;opener:Element|null;before:number};
type Jump=Intent&({kind:'reveal';id:string;origin?:Origin}|{kind:'focus';id:string}|{kind:'back'|'restore';origin:Origin});
type ReadingNotice={href:string;kind:'restored'|'missing'};

// Every open discussion shares these four passive listeners. No event contents or
// draft text are retained; the counter only detects a newer navigation intent.
const intentEvents=['pointerdown','keydown','wheel','touchmove'] as const;
let subscribers=0,interactionVersion=0;
const interacted=()=>{interactionVersion++;};
function watchInteractions(){
 const host=window;
 if(++subscribers===1)for(const name of intentEvents)host.addEventListener(name,interacted,{passive:true});
 return()=>{if(--subscribers===0)for(const name of intentEvents)host.removeEventListener(name,interacted);};
}

/** The server supplies the complete thread; only mounted Markdown rows are batched. */
export function useDiscussionReplies({replies=EMPTY,authorId,linkedReply,disabled=false,batchSize=PAGE_SIZE,initialOrder='oldest',rowIdPrefix='reply-',revealBlock='start'}:{replies?:Reply[];authorId?:string;linkedReply?:string;disabled?:boolean;batchSize?:3|20;initialOrder?:Order;rowIdPrefix?:string;revealBlock?:ScrollLogicalPosition}){
 const root=useRef<HTMLElement>(null),located=useRef<string|undefined>(undefined),previousLinked=useRef<string|undefined>(undefined),processed=useRef<Jump|undefined>(undefined);
 const [order,setOrder]=useState<Order>(initialOrder),[authorOnly,setAuthorOnly]=useState(false),[query,setQuery]=useState(''),[limit,setLimit]=useState<number>(batchSize),[jump,setJump]=useState<Jump>();
 const [history,setHistory]=useState<Origin[]>([]),[revealed,setRevealed]=useState<string>(),[notice,setNotice]=useState<ReadingNotice>();
 const [restored,setRestored]=useState<{request:number;query:string;expanded:DiscussionExpansion[]}>();
 useEffect(watchInteractions,[]);
 const ordered=useMemo(()=>[...replies].sort((a,b)=>{
  const left=a.created_at+'\0'+a.id,right=b.created_at+'\0'+b.id;
  return (left<right?-1:left>right?1:0)*(order==='oldest'?1:-1);
 }),[replies,order]);
 const byId=useMemo(()=>new Map(replies.map(reply=>[reply.id,reply])),[replies]);
 const searching=!!query,terms=useMemo(()=>replySearchTerms(query),[query]);
 const searchIndex=useMemo(()=>searching?new Map(replies.map(reply=>[reply.id,replySearchText(reply)])):undefined,[replies,searching]);
 const matched=useMemo(()=>ordered.filter(reply=>(!authorOnly||reply.author.id===authorId)&&(!searchIndex||matchesReplySearch(searchIndex.get(reply.id)||'',terms))),[ordered,authorOnly,authorId,searchIndex,terms]);
 const shown=matched.slice(0,limit);
 const matchPositions=useMemo(()=>new Map((query?matched:[]).map((reply,index)=>[reply.id,{index,previous:matched[index-1]?.id,next:matched[index+1]?.id}])),[matched,query]);
 const rowElement=(id:string)=>{const node=document.getElementById(rowIdPrefix+id);return node&&root.current?.contains(node)?node:undefined;};
 const intent=():Intent=>({href:location.href,opener:document.activeElement,before:interactionVersion});
 function clearReadingReturn(){setHistory(value=>value.length?[]:value);setRevealed(undefined);setNotice(undefined);setJump(undefined);}
 function reveal(id:string,trigger?:HTMLElement){
  if(!byId.has(id))return;
  const request=intent(),element=trigger?.closest<HTMLElement>('[data-discussion-reply]');
  const row=element&&root.current?.contains(element)&&byId.has(element.dataset.discussionReply||'')?element:undefined;
  const origin=trigger?.isConnected&&row?.dataset.discussionReply!==id?{href:request.href,trigger,row:row?.dataset.discussionReply,...captureReadingAnchor(row||trigger),order,authorOnly,query,limit,revealed}:undefined;
  const inFilter=matched.some(reply=>reply.id===id),rows=inFilter?matched:ordered;
  if(!inFilter){setAuthorOnly(false);setQuery('');}
  const index=rows.findIndex(reply=>reply.id===id);
  setLimit(value=>Math.max(value,Math.ceil((index+1)/batchSize)*batchSize));
  setJump({...request,kind:'reveal',id,origin});
 }
 useEffect(()=>{
  if(previousLinked.current!==linkedReply){previousLinked.current=linkedReply;located.current=undefined;clearReadingReturn();}
  if(!linkedReply||located.current===linkedReply||!byId.has(linkedReply))return;
  located.current=linkedReply;reveal(linkedReply);
 },[linkedReply,byId]);
 useLayoutEffect(()=>{
  if(!jump){processed.current=undefined;return;}
  if(processed.current===jump)return;
  processed.current=jump;
  const permitted=jump.href===location.href&&jump.before===interactionVersion&&(document.activeElement===jump.opener||document.activeElement===document.body&&!jump.opener?.isConnected);
  if(!permitted||disabled&&(jump.kind==='back'||jump.kind==='restore')){if(jump.kind==='restore')setNotice(undefined);setJump(undefined);return;}
  if(jump.kind==='reveal'){
   if(!rowElement(jump.id)){setJump(undefined);return;}
   if(jump.origin)setHistory(value=>[...value.filter(origin=>origin.href===jump.href),jump.origin!].slice(-MAX_RETURN_DEPTH));
   setRevealed(jump.id);setNotice(undefined);
   // Commit the return control before measuring or focusing its destination.
   setJump({...jump,kind:'focus'});return;
  }
  if(jump.kind==='back'){
   const origin=jump.origin,rows=order===origin.order?ordered:[...ordered].reverse();
   const restoreAuthor=origin.authorOnly&&(!origin.row||rows.some(row=>row.id===origin.row&&row.author.id===authorId));
   const source=origin.row?byId.get(origin.row):undefined;
   const restoreQuery=origin.query&&(!origin.row||source&&matchesReplySearch(replySearchText(source),replySearchTerms(origin.query)))?origin.query:'';
   const restoredTerms=replySearchTerms(restoreQuery);
   const visible=rows.filter(row=>(!restoreAuthor||row.author.id===authorId)&&(!restoreQuery||matchesReplySearch(replySearchText(row),restoredTerms)));
   const index=origin.row?visible.findIndex(row=>row.id===origin.row):-1;
   setOrder(origin.order);setAuthorOnly(restoreAuthor);setQuery(restoreQuery);setLimit(Math.max(origin.limit,Math.ceil((index+1)/batchSize)*batchSize));
   setHistory(value=>value.at(-1)===origin?value.slice(0,-1):value);
   setRevealed(origin.row&&byId.has(origin.row)?origin.row:origin.revealed&&byId.has(origin.revealed)?origin.revealed:undefined);
   setNotice({href:jump.href,kind:origin.row?byId.has(origin.row)?'restored':'missing':origin.trigger.isConnected?'restored':'missing'});
   // Check intent before changing the view, then restore against committed rows.
   setJump({...jump,kind:'restore'});return;
  }
  setJump(undefined);
  if(jump.kind==='focus'){
   const target=rowElement(jump.id);if(!target)return;
   target.focus({preventScroll:true});revealReadingTarget(target,revealBlock);return;
  }
  const origin=jump.origin,row=origin.row?rowElement(origin.row):undefined;
  const trigger=origin.trigger.isConnected&&!origin.trigger.matches(':disabled')&&(!origin.row||row?.contains(origin.trigger))?origin.trigger:row;
  const anchor=origin.row?row:trigger;
  if(anchor&&trigger&&restoreReadingAnchor(anchor,origin)){
   trigger.focus({preventScroll:true});
   keepReadingTriggerVisible(trigger);
  }else{
   setNotice({href:jump.href,kind:'missing'});
   if(root.current){root.current.focus({preventScroll:true});revealReadingTarget(root.current);}
  }
 },[jump,rowIdPrefix,revealBlock,disabled]);
 const previous=history.at(-1),canReturn=!!previous&&previous.href===location.href;
 return {root,shown,byId,authorOnly,order,query,total:matched.length,reveal,revealed,disabled,canReturn,clearReadingReturn,notice:notice&&notice.href===location.href?notice.kind:undefined,
  snapshotView:():DiscussionView=>({order,authorOnly,query,limit}),restoreRequest:restored?.request||0,
  restoredExpansion:(reply:Reply)=>restored?.query===query&&!!reply.edit_version&&restored.expanded.some(value=>value.reply===reply.id&&value.revision===reply.edit_version),
  restoreView(value:DiscussionView,expanded:DiscussionExpansion[],request:number,anchor?:string){
   if(disabled)return false;
   const rows=order===value.order?ordered:[...ordered].reverse(),wanted=replySearchTerms(value.query),filtered=rows.filter(row=>(!value.authorOnly||row.author.id===authorId)&&(!value.query||matchesReplySearch(replySearchText(row),wanted))),index=anchor?filtered.findIndex(row=>row.id===anchor):-1;
   clearReadingReturn();setOrder(value.order);setAuthorOnly(value.authorOnly);setQuery(value.query);setLimit(Math.min(Math.max(replies.length,batchSize),Math.max(batchSize,value.limit,Math.ceil((index+1)/batchSize)*batchSize)));setRestored({request,query:value.query,expanded:expanded.map(row=>({...row}))});return true;
  },
  firstMatch:matched[0]?.id,matchPosition:(id:string)=>matchPositions.get(id),
  seekSearchReply(id:string){if(disabled||!query)return;const position=matchPositions.get(id);if(!position)return;clearReadingReturn();setLimit(value=>Math.max(value,Math.ceil((position.index+1)/batchSize)*batchSize));setJump({...intent(),kind:'focus',id});},
  returnToOrigin(){if(disabled||!previous||previous.href!==location.href)return;setJump({...intent(),kind:'back',origin:previous});},
  chooseOrder(value:Order){if(disabled)return;clearReadingReturn();setOrder(value);setLimit(batchSize);},
  chooseAuthor(value:boolean){if(disabled)return;clearReadingReturn();setAuthorOnly(value);setLimit(batchSize);},
  chooseQuery(value:string){if(disabled)return;clearReadingReturn();setQuery(replySearchQuery(value));setLimit(batchSize);},
  showMore(){if(disabled||shown.length>=matched.length)return;setJump({...intent(),kind:'focus',id:matched[shown.length].id});setLimit(value=>value+batchSize);},
 };
}

export function ReplyReadingReturn({list,replyId}:{list:ReturnType<typeof useDiscussionReplies>;replyId?:string}){
 const {t}=useI18n();
 if(!list.canReturn||(replyId?list.revealed!==replyId:list.shown.some(row=>row.id===list.revealed)))return null;
 return <button type="button" className="reply-reading-return" disabled={list.disabled} onClick={list.returnToOrigin}><ArrowLeft size={13} aria-hidden="true"/>{t('返回刚才的位置','Return to your previous position')}</button>;
}
export function ReplyReadingNotice({list}:{list:ReturnType<typeof useDiscussionReplies>}){
 const {t}=useI18n();
 return list.notice?<p className="reply-reading-notice" role="status">{list.notice==='restored'?t('已返回刚才的阅读位置。','Returned to your previous reading position.'):t('原位置已不可用，已返回评论列表。','The original position is unavailable. You are back at the replies.')}</p>:null;
}
export function ReplyListControls({list,disabled}:{list:ReturnType<typeof useDiscussionReplies>;disabled:boolean}){
 const {t}=useI18n(),help=useId();
 return <div className="discussion-reply-controls">
  <button type="button" aria-pressed={list.authorOnly} disabled={disabled} onClick={()=>list.chooseAuthor(!list.authorOnly)}><UserRound size={14} aria-hidden="true"/>{t('只看楼主','Author only')}</button>
  <select aria-label={t('回复排序','Sort replies')} value={list.order} disabled={disabled} onChange={event=>list.chooseOrder(event.target.value as Order)}><option value="oldest">{t('最早回复','Oldest first')}</option><option value="newest">{t('最新回复','Newest first')}</option></select>
  <span role="status">{t(`显示 ${list.shown.length} / ${list.total} 条回复`,`${list.shown.length} of ${list.total} replies shown`)}</span>
  {list.query&&list.firstMatch&&<button type="button" className="discussion-reply-first-match" disabled={disabled} onClick={()=>{if(list.firstMatch)list.seekSearchReply(list.firstMatch);}}>{t('跳到首条匹配','Go to first match')}</button>}
  <div className="discussion-reply-search"><CommunitySearch applied={list.query} onApply={list.chooseQuery} disabled={disabled} describedBy={help} label={t('搜索本帖回复','Search replies in this discussion')} placeholder={t('搜索内容、代码、作者或机构…','Search content, code, author or affiliation…')}/></div>
  <p id={help} className="discussion-reply-search-help">{t('搜索本帖全部公开回复，多个关键词用空格分隔。支持代码、公式和公开所属机构；不搜索草稿。','Search all public replies in this discussion. Separate terms with spaces. Includes code, formulas and public affiliations; excludes drafts.')}</p>
 </div>;
}
export function MoreReplies({list,disabled}:{list:ReturnType<typeof useDiscussionReplies>;disabled:boolean}){
 const {t}=useI18n(),count=Math.min(PAGE_SIZE,list.total-list.shown.length);
 return count>0?<button type="button" className="discussion-replies-more" disabled={disabled} onClick={list.showMore}><ChevronDown size={15} aria-hidden="true"/>{t(`再显示 ${count} 条回复`,`Show ${count} more ${count===1?'reply':'replies'}`)}</button>:null;
}
