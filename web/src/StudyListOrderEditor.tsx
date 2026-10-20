import {useId,useLayoutEffect,useRef,useState} from 'react';
import {ArrowDown,ArrowDownToLine,ArrowUp,ArrowUpToLine,Check,GripVertical,X} from 'lucide-react';
import {useI18n} from './i18n';
import {controlIsActive} from './activeControls';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import {canSaveStudyListOrder,moveStudyListItem,sameStudyListOrder} from './studyListOrder';
import type {createStudyListStore,StudyList,StudyListsState} from './studyListStore';
import type {Problem} from './types';

type Props={list:StudyList;state:StudyListsState;store?:ReturnType<typeof createStudyListStore>;problems:Problem[];canAct:()=>boolean;onFinish:(focus:boolean)=>void;onDirty:(dirty:boolean)=>void};
type Attempt={request:string;ids:string[];claim:()=>boolean};
const moves=[{kind:'up',Icon:ArrowUp,zh:'上移',en:'Move up'},{kind:'down',Icon:ArrowDown,zh:'下移',en:'Move down'},{kind:'first',Icon:ArrowUpToLine,zh:'置顶',en:'Move to first'},{kind:'last',Icon:ArrowDownToLine,zh:'移到底部',en:'Move to last'}] as const;

export default function StudyListOrderEditor({list,state,store,problems,canAct,onFinish,onDirty}:Props){
 const {t,locale}=useI18n(),id=useId(),root=useRef<HTMLElement>(null),heading=useRef<HTMLHeadingElement>(null),rows=useRef<HTMLOListElement>(null),owner=useRef(state.snapshot?.owner_id),intent=useWorkspaceIntent(JSON.stringify([state.snapshot?.owner_id,list.id]));
 const current=list.items.map(item=>item.problem_id),[draft,setDraft]=useState(()=>({source:[...current],ids:[...current]})),[attempt,setAttempt]=useState<Attempt>(),[receipt,setReceipt]=useState(''),[message,setMessage]=useState<{id:string;position:number}>();
 const moved=useRef<{handle:HTMLButtonElement;href:string}|undefined>(undefined),byId=new Map(problems.map(problem=>[problem.id,problem]));
 const dirty=!sameStudyListOrder(draft.source,draft.ids),acknowledged=!!attempt&&(receipt===attempt.request||state.confirmation?.request_id===attempt.request),confirmed=acknowledged&&!!attempt&&sameStudyListOrder(current,attempt.ids),sourceChanged=!confirmed&&!sameStudyListOrder(draft.source,current),blocked=!!state.pending||state.loading||!!state.error,ownPending=!!attempt&&state.pending?.command.request_id===attempt.request;
 const title=(key:string)=>{const problem=byId.get(key);return problem?(locale==='en'?problem.translations?.en?.title||problem.title:problem.title):key;};
 function here(){return controlIsActive(root.current)&&state.snapshot?.owner_id===owner.current&&canAct();}
 useLayoutEffect(()=>{onDirty(dirty);},[dirty,onDirty]);
 useLayoutEffect(()=>{if(confirmed&&attempt){const focus=here()&&attempt.claim();onFinish(focus);}},[confirmed,attempt]);
 useLayoutEffect(()=>{
  const request=moved.current;moved.current=undefined;
  if(!request||!here()||location.href!==request.href||document.activeElement!==request.handle||!rows.current)return;
  const row=request.handle.closest('li');if(!row)return;
  const box=rows.current.getBoundingClientRect(),item=row.getBoundingClientRect(),delta=item.top<box.top?item.top-box.top:item.bottom>box.bottom?item.bottom-box.bottom:0;
  if(delta)rows.current.scrollTop=Math.max(0,Math.min(rows.current.scrollHeight-rows.current.clientHeight,rows.current.scrollTop+delta));
 },[draft.ids]);
 function move(key:string,to:number,trigger:HTMLButtonElement){
  if(!here()||blocked||sourceChanged||!root.current?.contains(trigger))return;
  const next=moveStudyListItem(draft.ids,key,to);if(!next)return;
  const row=trigger.closest('li'),handle=row?.querySelector<HTMLButtonElement>('.study-order-handle');
  if(handle&&row?.contains(document.activeElement)){handle.focus({preventScroll:true});if(document.activeElement===handle)moved.current={handle,href:location.href};}
  if(!here()){moved.current=undefined;return;}
  setDraft(value=>({...value,ids:next}));setMessage({id:key,position:to+1});
 }
 function save(){
  if(!here()||blocked||!store)return;
  const latest=store.getSnapshot(),live=latest.snapshot?.lists.find(row=>row.id===list.id),ids=live?.items.map(row=>row.problem_id);
  if(latest.pending||latest.loading||latest.error||latest.snapshot?.owner_id!==owner.current||!ids||!canSaveStudyListOrder(draft.source,draft.ids,ids))return;
  const claim=root.current?.contains(document.activeElement)?intent.create():()=>false,desired=[...draft.ids],result=store.mutate({action:'reorder',list_id:list.id,problem_ids:desired}),pending=store.getSnapshot().pending;
  if(pending&&pending.command.action==='reorder'&&pending.command.list_id===list.id){const request=pending.command.request_id;setAttempt({request,ids:desired,claim});void result.then(ok=>{if(ok)setReceipt(request);});}else intent.cancel();
 }
 function reload(){
  if(!here()||blocked||!store)return;if(root.current?.contains(document.activeElement))heading.current?.focus({preventScroll:true});if(!here())return;const latest=store.getSnapshot();if(latest.pending||latest.loading||latest.error)return;const live=latest.snapshot?.lists.find(row=>row.id===list.id);if(!live)return;
  const ids=live.items.map(row=>row.problem_id);setDraft({source:[...ids],ids:[...ids]});setAttempt(undefined);setReceipt('');setMessage(undefined);
 }
 return <section ref={root} className="study-list-order-editor" aria-labelledby={id+'-heading'}>
  <header><h3 ref={heading} tabIndex={-1} id={id+'-heading'}>{t('调整题单顺序','Arrange your study list')}</h3><span>{t(`${draft.ids.length} 道题`,`${draft.ids.length} problems`)}</span></header>
  <p id={id+'-help'}>{t('先预览，再一次保存到账号。聚焦左侧序号，用上下键移动，Home / End 置顶或移到底部。','Preview your order, then save it once to your account. Focus a position handle: arrows move one place; Home / End move to the first or last position.')}</p>
  {sourceChanged&&<div className="study-order-conflict" role="status"><p>{acknowledged?t('保存已确认，但题单随后有变化。请核对最新顺序。','Your save was confirmed, but the list changed afterward. Review the latest order.'):t('题单内容或顺序已在其他地方更新。草稿保留，重新载入顺序后再调整。','This list changed elsewhere. Your draft is kept. Load the latest order before continuing.')}</p><button type="button" className="text-button" disabled={blocked} onClick={reload}>{t('按最新题单重新调整','Use the latest list order')}</button></div>}
  <ol ref={rows} id={id+'-items'} className="study-order-items" role="list" aria-describedby={id+'-help'}>{draft.ids.map((key,index)=>{
   const problem=byId.get(key),name=title(key),disabled=blocked||sourceChanged;
   return <li key={key} data-order-id={key} aria-posinset={index+1} aria-setsize={draft.ids.length}>
    <button type="button" className="study-order-handle" aria-label={t(`调整 ${name}，当前位置 ${index+1} / ${draft.ids.length}`,`Arrange ${name}, position ${index+1} of ${draft.ids.length}`)} aria-controls={id+'-items'} aria-describedby={id+'-help'} aria-keyshortcuts="ArrowUp ArrowDown Home End" disabled={disabled} onKeyDown={event=>{
     if(event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey||document.activeElement!==event.currentTarget||!here())return;
     const to=event.key==='ArrowUp'?index-1:event.key==='ArrowDown'?index+1:event.key==='Home'?0:event.key==='End'?draft.ids.length-1:undefined;if(to===undefined)return;event.preventDefault();event.stopPropagation();move(key,to,event.currentTarget);
    }}><GripVertical size={15} aria-hidden="true"/><span>{index+1}</span></button>
    <div className="study-order-title"><strong>{problem?`${problem.number}. ${name}`:name}</strong>{!problem&&<small>{t('题目暂不可用，仍保留在题单中','Unavailable problem; kept in your list')}</small>}</div>
    <div className="study-order-moves">{moves.map(({kind,Icon,zh,en})=>{
     const to=kind==='first'?0:kind==='last'?draft.ids.length-1:index+(kind==='up'?-1:1);
     return <button type="button" className="icon-button" key={kind} aria-label={t(`${zh}：${name}`,`${en}: ${name}`)} disabled={disabled||to<0||to>=draft.ids.length||to===index} onClick={event=>move(key,to,event.currentTarget)}><Icon size={16} aria-hidden="true"/></button>;
    })}</div>
   </li>;
  })}</ol>
  <div className="study-order-feedback" role="status" aria-atomic="true">{message?t(`${title(message.id)} 已移至第 ${message.position} 位，尚未保存。`,`${title(message.id)} moved to position ${message.position}. Not saved yet.`):t('移动只改变预览，题单尚未修改。','Moves only change this preview. Your saved list is unchanged.')}</div>
  <footer><button type="button" className="button" disabled={blocked||sourceChanged||!dirty} aria-busy={ownPending&&!!state.pending?.busy} onClick={save}><Check size={16} aria-hidden="true"/>{ownPending?t('等待保存确认','Awaiting save confirmation'):t('保存顺序','Save order')}</button><button type="button" className="text-button" disabled={ownPending} onClick={()=>{if(here())onFinish(!!root.current?.contains(document.activeElement));}}><X size={16} aria-hidden="true"/>{t('取消调整','Cancel arrangement')}</button><small>{t('只保存题目顺序，不改动代码或学习记录。','Only saves problem order. Code and practice records stay intact.')}</small></footer>
 </section>;
}
