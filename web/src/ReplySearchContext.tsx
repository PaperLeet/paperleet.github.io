import {Fragment,useId,useMemo,useState,type ReactNode} from 'react';
import {ChevronDown,ChevronLeft,ChevronRight,Search} from 'lucide-react';
import type {Reply} from './communityDiscussion';
import {useI18n} from './i18n';
import {replySearchFields} from './replySearch';
import {replySearchPassages,type ReplySearchPassage} from './replySearchPassages';
import type {useDiscussionReplies} from './useDiscussionReplies';
import './reply-search-context.css';

function Passage({value}:{value:ReplySearchPassage}){
 let previous=0;
 return <>{value.before&&<span aria-hidden="true">…</span>}{value.ranges.map(({start,end})=>{const before=value.text.slice(previous,start);previous=end;return <Fragment key={start}>{before}<mark>{value.text.slice(start,end)}</mark></Fragment>;})}{value.text.slice(previous)}{value.after&&<span aria-hidden="true">…</span>}</>;
}
export default function ReplySearchContext({reply,list,children,lazy=false,canAct}:{reply:Reply;list:ReturnType<typeof useDiscussionReplies>;children:ReactNode;lazy?:boolean;canAct?:()=>boolean}){
 const {t}=useI18n(),bodyId=useId(),[source,setSource]=useState<{request:number;opened?:{query:string;body:string}}>(()=>({request:list.restoreRequest,opened:list.restoredExpansion(reply)?{query:list.query,body:reply.body}:undefined}));
 let current=source;if(current.request!==list.restoreRequest){current={request:list.restoreRequest,opened:list.restoredExpansion(reply)?{query:list.query,body:reply.body}:undefined};setSource(current);}
 const opened=current.opened,setOpened=(opened:{query:string;body:string}|undefined)=>setSource({request:list.restoreRequest,opened});
 const contexts=useMemo(()=>list.query?Object.entries(replySearchFields(reply)).flatMap(([kind,text])=>{const passages=replySearchPassages(text,list.query);return passages.length?[{kind,passages}]:[];}):[],[reply,list.query]);
 const position=list.query?list.matchPosition(reply.id):undefined,searching=!!list.query&&!!position,expanded=!searching||opened?.query===list.query&&opened.body===reply.body;
 return <div className="reply-search-reading">
  {searching&&<div className="reply-search-context" role="group" aria-label={t('搜索命中片段','Search match context')}>
   <div className="reply-search-heading"><span><Search size={14} aria-hidden="true"/>{t(`匹配 ${position.index+1} / ${list.total}`,`Match ${position.index+1} of ${list.total}`)}</span><div>
    <button type="button" disabled={list.disabled||!position.previous} aria-label={t('上一条匹配回复','Previous matching reply')} onClick={()=>{if(!list.disabled&&(!canAct||canAct())&&position.previous)list.seekSearchReply(position.previous);}}><ChevronLeft size={16} aria-hidden="true"/>{t('上一条','Previous')}</button>
    <button type="button" disabled={list.disabled||!position.next} aria-label={t('下一条匹配回复','Next matching reply')} onClick={()=>{if(!list.disabled&&(!canAct||canAct())&&position.next)list.seekSearchReply(position.next);}}>{t('下一条','Next')}<ChevronRight size={16} aria-hidden="true"/></button>
   </div></div>
   {contexts.map(({kind,passages})=><div key={kind} className="reply-search-source"><small>{kind==='body'?t('正文原文','Original text'):kind==='author'?t('公开作者','Public author'):t('所属机构','Affiliation')}</small>{passages.map((value,index)=><p key={index}><Passage value={value}/></p>)}</div>)}
   {!contexts.length&&<small className="reply-search-hint">{t('原文命中超出片段预览范围，可展开完整回复。','The match is outside the excerpt preview. Open the full reply below.')}</small>}
  </div>}
  {searching&&<button type="button" className="reply-search-full" disabled={list.disabled} aria-expanded={expanded} aria-controls={bodyId} onClick={()=>{if(list.disabled||canAct&&!canAct())return;if(expanded){setOpened(undefined);}else setOpened({query:list.query,body:reply.body});}}><ChevronDown size={15} aria-hidden="true"/>{expanded?t('收起完整回复','Hide full reply'):t('展开完整回复','Show full reply')}</button>}
  <div id={bodyId} className="reply-search-full-body" hidden={!expanded}>{expanded||!lazy?children:null}</div>
 </div>;
}
