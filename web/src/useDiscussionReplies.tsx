import {useEffect,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {ChevronDown,UserRound} from 'lucide-react';
import type {Reply} from './communityDiscussion';
import {useI18n} from './i18n';
import './discussion-replies.css';

const PAGE_SIZE=20;
const EMPTY:Reply[]=[];
type Order='oldest'|'newest';
type Jump={id:string;href:string;opener:Element|null};

/** The server supplies the complete thread; only mounted Markdown rows are batched. */
export function useDiscussionReplies({replies=EMPTY,authorId,linkedReply,disabled=false,batchSize=PAGE_SIZE,initialOrder='oldest',rowIdPrefix='reply-'}:{replies?:Reply[];authorId?:string;linkedReply?:string;disabled?:boolean;batchSize?:3|20;initialOrder?:Order;rowIdPrefix?:string}){
  const root=useRef<HTMLElement>(null),located=useRef<string|undefined>(undefined);
  const [order,setOrder]=useState<Order>(initialOrder),[authorOnly,setAuthorOnly]=useState(false),[limit,setLimit]=useState<number>(batchSize),[jump,setJump]=useState<Jump>();
  const ordered=useMemo(()=>[...replies].sort((a,b)=>{
    const left=a.created_at+'\0'+a.id,right=b.created_at+'\0'+b.id;
    return (left<right?-1:left>right?1:0)*(order==='oldest'?1:-1);
  }),[replies,order]);
  const byId=useMemo(()=>new Map(replies.map(reply=>[reply.id,reply])),[replies]);
  const matched=useMemo(()=>authorOnly?ordered.filter(reply=>reply.author.id===authorId):ordered,[ordered,authorOnly,authorId]);
  const shown=matched.slice(0,limit);
  function reveal(id:string){
    if(!byId.has(id))return;
    const inFilter=matched.some(reply=>reply.id===id),rows=inFilter?matched:ordered;
    if(!inFilter)setAuthorOnly(false);
    const index=rows.findIndex(reply=>reply.id===id);
    setLimit(value=>Math.max(value,Math.ceil((index+1)/batchSize)*batchSize));
    setJump({id,href:location.href,opener:document.activeElement});
  }
  useEffect(()=>{
    if(!linkedReply){located.current=undefined;return;}
    if(located.current===linkedReply||!byId.has(linkedReply))return;
    located.current=linkedReply;reveal(linkedReply);
  },[linkedReply,byId]);
  useLayoutEffect(()=>{
    if(!jump)return;setJump(undefined);
    if(jump.href!==location.href)return;
    if(document.activeElement!==jump.opener&&!(document.activeElement===document.body&&!jump.opener?.isConnected))return;
    const target=document.getElementById(rowIdPrefix+jump.id);
    if(!target||!root.current?.contains(target))return;
    target.focus({preventScroll:true});target.scrollIntoView({block:'center'});
  },[jump,rowIdPrefix]);
  return {root,shown,byId,authorOnly,order,total:matched.length,reveal,
    chooseOrder(value:Order){if(disabled)return;setOrder(value);setLimit(batchSize);},
    chooseAuthor(value:boolean){if(disabled)return;setAuthorOnly(value);setLimit(batchSize);},
    showMore(){
      if(disabled||shown.length>=matched.length)return;
      setJump({id:matched[shown.length].id,href:location.href,opener:document.activeElement});
      setLimit(value=>value+batchSize);
    },
  };
}

export function ReplyListControls({list,disabled}:{list:ReturnType<typeof useDiscussionReplies>;disabled:boolean}){
  const {t}=useI18n();
  return <div className="discussion-reply-controls">
    <button type="button" aria-pressed={list.authorOnly} disabled={disabled} onClick={()=>list.chooseAuthor(!list.authorOnly)}><UserRound size={14} aria-hidden="true"/>{t('只看楼主','Author only')}</button>
    <select aria-label={t('回复排序','Sort replies')} value={list.order} disabled={disabled} onChange={event=>list.chooseOrder(event.target.value as Order)}><option value="oldest">{t('最早回复','Oldest first')}</option><option value="newest">{t('最新回复','Newest first')}</option></select>
    <span role="status">{t(`显示 ${list.shown.length} / ${list.total} 条回复`,`${list.shown.length} of ${list.total} replies shown`)}</span>
  </div>;
}

export function MoreReplies({list,disabled}:{list:ReturnType<typeof useDiscussionReplies>;disabled:boolean}){
  const {t}=useI18n(),count=Math.min(PAGE_SIZE,list.total-list.shown.length);
  return count>0?<button type="button" className="discussion-replies-more" disabled={disabled} onClick={list.showMore}><ChevronDown size={15} aria-hidden="true"/>{t(`再显示 ${count} 条回复`,`Show ${count} more ${count===1?'reply':'replies'}`)}</button>:null;
}
