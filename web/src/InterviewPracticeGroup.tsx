import AppLink from './AppLink';
import {useId,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {ArrowDown,ArrowRight,ArrowUp,BookOpen,CheckCircle2,Layers3,Undo2,X} from 'lucide-react';
import {controlIsActive} from './activeControls';
import {useI18n} from './i18n';
import {interviewHref,interviewBatchView,type InterviewFilters} from './interviewNavigation';
import {withLibraryContext} from './libraryNavigation';
import {status,type Problem,type Saved} from './types';
import './interview-practice-group.css';
import InterviewBatchShare from './InterviewBatchShare';
import {PRACTICE_GROUP_LIMIT,interviewBatchShareHref} from './interviewBatchSharing';
import InterviewBudgetPlanner from './InterviewBudgetPlanner';
import {interviewBudgetPlan,type BudgetInput,type BudgetMode,type BudgetPlan} from './interviewBudgetPlan';

export {PRACTICE_GROUP_LIMIT} from './interviewBatchSharing';
type Source='reports'|'curated';
type Change={kind:'add'|'remove'|'move'|'clear'|'fill'|'budget'|'undo';number?:number;position?:number;count?:number;minutes?:number;budget?:number};
type UndoSnapshot={context:{scope:string;source:Source};numbers:number[]};
type Selection={scope:string;source:Source;numbers:number[];change?:Change;undo?:UndoSnapshot};
type Focus={scope:string;source:Source;href:string;opener:HTMLElement;number?:number;action?:'up'|'down'|'remove'|'catalog'};
function capturedGroup(queue:string){
 if(!queue)return [];
 const values=queue.split(',');
 if(values.length>PRACTICE_GROUP_LIMIT||values.some(value=>!/^[1-9]\d{0,4}$/.test(value)))return;
 const numbers=values.map(Number);
 return new Set(numbers).size===numbers.length?numbers:undefined;
}

/** Selection changes are local; only starting practice captures the ordered public IDs in its URL. */
export function useInterviewPracticeGroup({filters,source,problems,candidates,filterPending,saved,ready,selectionScope,pendingReason='progress',owner='guest'}:{filters:InterviewFilters;source:Source;problems:Problem[];candidates:Problem[];filterPending:boolean;saved:Record<string,Saved>;ready:boolean;selectionScope?:string;pendingReason?:'progress'|'catalog'|'sources';owner?:string}){
 const scope=JSON.stringify([owner,selectionScope??interviewHref(filters)]),initial=capturedGroup(filters.queue);
 const renderedHref=typeof location==='undefined'?'':location.href;
 const undoVisit=useRef({scope,source});
 // A later visit to the same scope must not revive an earlier undo snapshot.
 if(undoVisit.current.scope!==scope||undoVisit.current.source!==source)undoVisit.current={scope,source};
 const undoContext=undoVisit.current;
 const [selection,setSelection]=useState<Selection>(()=>({scope,source,numbers:initial||[]}));
 const current=selection.scope===scope&&selection.source===source?selection:{scope,source,numbers:initial||[]};
 const numbers=current.numbers;
 const byNumber=useMemo(()=>new Map(problems.map(problem=>[problem.number,problem])),[problems]);
 const modules=numbers.flatMap(number=>{const problem=byNumber.get(number);return problem?[problem]:[];});
 const missing=numbers.length-modules.length;
 const candidateNumbers=filterPending?[]:[...new Set(candidates.map(problem=>problem.number))].filter(number=>byNumber.has(number));
 const additions=(values:number[])=>candidateNumbers.filter(number=>!values.includes(number)).slice(0,Math.max(0,PRACTICE_GROUP_LIMIT-values.length));
 const fillCount=additions(numbers).length;
 const root=useRef<HTMLElement>(null),heading=useRef<HTMLHeadingElement>(null),focus=useRef<Focus|undefined>(undefined);
 const planningScope=JSON.stringify([scope,source,owner]),alive=useRef(true);
 useLayoutEffect(()=>{alive.current=true;return()=>{alive.current=false;};},[]);
 const latest=useRef({scope,source,owner,numbers,initial,problems,candidateNumbers,filterPending,saved,ready,undoContext});
 latest.current={scope,source,owner,numbers,initial,problems,candidateNumbers,filterPending,saved,ready,undoContext};
 function plan(budget:number|undefined,mode:BudgetMode,selected=latest.current.numbers){
  const value=latest.current;
  const input:BudgetInput={scope:JSON.stringify([value.scope,value.source]),owner:value.owner,selected,candidates:value.candidateNumbers,ready:value.ready,pending:value.filterPending,items:value.problems.map(problem=>({id:problem.id,number:problem.number,minutes:problem.estimated_minutes,state:value.ready?status(problem,value.saved[problem.id]):'unknown'}))};
  return interviewBudgetPlan(input,budget,mode);
 }
 function applyPlan(proposal:BudgetPlan){
  if(!alive.current)return;
  setSelection(previous=>{
   if(!alive.current)return previous;
   const value=latest.current,selected=previous.scope===value.scope&&previous.source===value.source?previous.numbers:value.initial||[];
   const current=plan(proposal.budget,proposal.mode,selected);
   // Recompute current metadata and progress, including queued selection edits.
   if(current.fingerprint!==proposal.fingerprint||current.reason||!current.additions.length)return previous;
   return {scope:value.scope,source:value.source,numbers:[...selected,...current.additions.map(item=>item.number)],undo:{context:value.undoContext,numbers:[...selected]},change:{kind:'budget',count:current.additions.length,minutes:current.minutes,budget:current.budget}};
  });
 }

 function update(transform:(numbers:number[])=>{numbers:number[];change:Change}|undefined){
  setSelection(previous=>{
   const values=previous.scope===scope&&previous.source===source?previous.numbers:initial||[],next=transform(values);
   if(!next||next.numbers.length===values.length&&next.numbers.every((number,index)=>number===values[index]))return previous;
   return {...next,scope,source,undo:{context:undoContext,numbers:[...values]}};
  });
 }
 function undo(opener:HTMLButtonElement){
  if(!alive.current||latest.current.undoContext!==undoContext||location.href!==renderedHref||!controlIsActive(root.current)||!controlIsActive(opener)||!root.current?.contains(opener))return;
  focus.current={scope,source,href:renderedHref,opener};
  setSelection(previous=>{
   if(!alive.current||latest.current.undoContext!==undoContext||previous.scope!==scope||previous.source!==source||previous.undo?.context!==undoContext)return previous;
   return {scope,source,numbers:[...previous.undo.numbers],change:{kind:'undo'}};
  });
 }
 function toggle(number:number){
  if(!byNumber.has(number))return;
  update(values=>values.includes(number)?{numbers:values.filter(value=>value!==number),change:{kind:'remove',number}}:values.length<PRACTICE_GROUP_LIMIT?{numbers:[...values,number],change:{kind:'add',number}}:undefined);
 }
 function fill(){
  update(values=>{const next=additions(values);return next.length?{numbers:[...values,...next],change:{kind:'fill',count:next.length}}:undefined;});
 }
 function remove(number:number,opener:HTMLElement){
  const index=numbers.indexOf(number);if(index<0)return;
  focus.current={scope,source,href:location.href,opener,number:numbers[index+1]??numbers[index-1],action:'remove'};
  update(values=>({numbers:values.filter(value=>value!==number),change:{kind:'remove',number}}));
 }
 function move(number:number,direction:-1|1,opener:HTMLElement){
  const index=numbers.indexOf(number),next=index+direction;
  if(index<0||next<0||next>=numbers.length)return;
  focus.current={scope,source,href:location.href,opener,number,action:direction<0?'up':'down'};
  update(values=>{const from=values.indexOf(number),to=from+direction;if(from<0||to<0||to>=values.length)return {numbers:values,change:{kind:'move',number,position:from+1}};const next=[...values];[next[from],next[to]]=[next[to],next[from]];return {numbers:next,change:{kind:'move',number,position:to+1}};});
 }
 function clear(opener:HTMLElement){
  focus.current={scope,source,href:location.href,opener};
  update(()=>({numbers:[],change:{kind:'clear'}}));
 }
 function retryCatalog(opener:HTMLElement,retry:()=>void){
  const number=numbers.find(value=>!byNumber.has(value));
  if(number!==undefined)focus.current={scope,source,href:location.href,opener,number,action:'catalog'};
  retry();
 }
 useLayoutEffect(()=>{
  const intent=focus.current;if(!intent)return;
  if(intent.scope!==scope||intent.source!==source||intent.href!==location.href){focus.current=undefined;return;}
  if(document.activeElement!==intent.opener&&!(document.activeElement===document.body&&(!intent.opener.isConnected||intent.opener.matches(':disabled')))){focus.current=undefined;return;}
  if(intent.action==='catalog'&&intent.number!==undefined&&!byNumber.has(intent.number))return;
  focus.current=undefined;
  const control=(action:string)=>root.current?.querySelector<HTMLButtonElement>(`[data-group-number="${intent.number}"][data-group-action="${action}"]`);
  const target=intent.action==='catalog'?root.current?.querySelector<HTMLAnchorElement>(`[data-selected-problem="${intent.number}"] .practice-group-title a`):intent.number?control(intent.action!)||undefined:undefined;
  const alternate=intent.action==='up'?control('down'):intent.action==='down'?control('up'):undefined;
  (target&&!('disabled' in target&&target.disabled)?target:alternate&&!alternate.disabled?alternate:heading.current)?.focus({preventScroll:true});
 },[selection,scope,source,missing,byNumber]);
 function href(problem:Problem){
  if(!numbers.length||missing||!numbers.includes(problem.number))return;
  return withLibraryContext(problem.number,interviewHref({...filters,view:interviewBatchView(filters)?filters.view:'questions',queue:numbers.join(',')}));
 }
 const first=(ready?modules.find(problem=>status(problem,saved[problem.id])!=='done'):undefined)||modules[0];
 const completed=ready?modules.filter(problem=>status(problem,saved[problem.id])==='done').length:undefined;
 const shareHref=numbers.length&&!missing?interviewBatchShareHref(filters,modules):undefined;
 return {shareHref,modules,root,heading,numbers,byNumber,missing,change:current.change,canUndo:current.undo?.context===undoContext,undo,uneditable:!!filters.queue&&!initial&&!current.change,toggle,fill,fillCount,filterPending,pendingReason,candidateCount:candidateNumbers.length,remove,move,clear,retryCatalog,href,first,startHref:first?href(first):undefined,completed,ready,state:(problem:Problem)=>ready?status(problem,saved[problem.id]):undefined,continuing:ready&&!!first&&status(first,saved[first.id])==='started',planningScope,planningDefault:filters.study==='done'?'all' as const:'remaining' as const,plan,applyPlan};
}
export type InterviewPracticeGroupState=ReturnType<typeof useInterviewPracticeGroup>;

export function InterviewGroupToggle({problem,name,group}:{problem:Problem;name:string;group:InterviewPracticeGroupState}){
 const {t}=useI18n(),selected=group.numbers.includes(problem.number),full=!selected&&group.numbers.length>=PRACTICE_GROUP_LIMIT;
 const label=selected?t(`从练习组移除${name}`,`Remove ${name} from this batch`):full?t(`练习组已满，请先移除一道再加入${name}`,`Batch full. Remove a concept before adding ${name}`):t(`将${name}加入练习组`,`Add ${name} to this batch`);
 return <button type="button" className="interview-group-toggle" aria-label={label} title={label} aria-pressed={selected} disabled={full} onClick={()=>group.toggle(problem.number)}>{selected?<CheckCircle2 size={12}/>:<BookOpen size={12}/>}<span>{selected?t('已加入练习组','Added to batch'):t('加入练习组','Add to batch')}</span></button>;
}

export default function InterviewPracticeGroup({group,onRetryCatalog}:{group:InterviewPracticeGroupState;onRetryCatalog?:()=>void}){
 const {t,locale}=useI18n(),headingId=useId(),fillHintId=useId();
 const title=(number:number)=>{const problem=group.byNumber.get(number);return problem?(locale==='en'?problem.translations?.en?.title||problem.title:problem.title):t(`第 ${number} 题`, `Problem ${number}`);};
 const change=group.change;
 const feedback=change?.kind==='budget'?t(`已加入 ${change.count} 道练习，整组约 ${change.minutes} / ${change.budget} 分钟。`,`Added ${change.count} exercises; about ${change.minutes} of ${change.budget} min for the batch.`):change?.kind==='fill'?t(`已按当前列表顺序加入 ${change.count} 道练习。`,`Added ${change.count} ${change.count===1?'concept':'concepts'} in the current list order.`):change?.kind==='move'?t(`已将${title(change.number!)}移到第 ${change.position} 位`,`Moved ${title(change.number!)} to position ${change.position}`):change?.kind==='clear'?t('已清空练习组。','Practice batch cleared.'):change?.kind==='add'?t(`已加入${title(change.number!)}。`,`Added ${title(change.number!)}.`):change?.kind==='remove'?t(`已移除${title(change.number!)}。`,`Removed ${title(change.number!)}.`):'';
 const allDone=group.completed!==undefined&&group.completed===group.numbers.length;
 const minutes=group.numbers.map(number=>group.byNumber.get(number)?.estimated_minutes);
 const estimate=minutes.length&&minutes.every(value=>typeof value==='number'&&Number.isFinite(value)&&value>0)?minutes.reduce<number>((total,value)=>total+value!,0):undefined;
 return <section ref={group.root} className="interview-practice-group" aria-labelledby={headingId}>
  <header><div><h3 ref={group.heading} id={headingId} tabIndex={-1}><Layers3 size={16}/>{t('自选练习组','Your practice batch')}</h3><p>{t('最多选 6 道，按自己的顺序连续练习。','Pick up to 6 concepts and practice in your own order.')}</p></div>{group.numbers.length>0&&<button type="button" className="text-button" onClick={event=>group.clear(event.currentTarget)}>{t('清空','Clear')}</button>}</header>
  <div className="practice-group-feedback"><p className="practice-group-status" role="status">{t(`已选 ${group.numbers.length} / ${PRACTICE_GROUP_LIMIT}`,`${group.numbers.length} of ${PRACTICE_GROUP_LIMIT} selected`)}{change?.kind==='undo'?<span> · {t('已撤销上次调整。','Last change undone.')}</span>:feedback&&<span> · {feedback}</span>}</p>{group.canUndo&&<button type="button" className="practice-group-undo" aria-label={t('撤销上一次练习组调整','Undo the last practice batch change')} onClick={event=>group.undo(event.currentTarget)}><Undo2 size={15} aria-hidden="true"/>{t('撤销','Undo')}</button>}</div>
  <div className="practice-group-fill"><button type="button" className="button subtle" aria-disabled={!group.fillCount} aria-describedby={fillHintId} onClick={group.fill}>{group.fillCount?group.numbers.length?t(`再加入 ${group.fillCount} 道`,`Add ${group.fillCount} more`):t(`加入前 ${group.fillCount} 道`,`Add the first ${group.fillCount}`):group.numbers.length>=PRACTICE_GROUP_LIMIT?t('练习组已满','Batch full'):group.filterPending?group.pendingReason==='catalog'?t('需先加载题库','Catalog required'):group.pendingReason==='sources'?t('需先读取来源','Sources required'):t('需先读取学习进度','Progress required'):group.candidateCount?t('当前列表已加入','Current list added'):t('当前筛选暂无练习','No concepts in this view')}</button><small id={fillHintId}>{group.filterPending?group.pendingReason==='catalog'?t('题库加载后可按完整对比列表组题；也可以先逐题加入已加载的练习。','Load the catalog to fill from the complete comparison. You can still add loaded concepts individually.'):group.pendingReason==='sources'?t('来源统计就绪后，可从当前对比范围组题。','Current source counts are needed to fill a comparison batch.'):t('学习进度就绪后，才能按所选状态组题；仍可逐题加入。','Learning progress is needed for this state filter. You can still add individual concepts.'):t('按当前筛选与排序补齐，保留已选题目和顺序。','Fill from the current filters and order, keeping your existing choices.')}</small></div>
  <InterviewBudgetPlanner key={group.planningScope} group={group}/>
  {group.uneditable&&<p className="practice-group-unavailable">{t('当前链接的练习组无法在此编辑，可以重新选择最多 6 道题。','This linked batch cannot be edited here. Choose up to 6 concepts to make a new batch.')}</p>}
  {group.numbers.length>0&&<>
   <ol>{group.numbers.map((number,index)=>{const problem=group.byNumber.get(number),name=title(number),href=problem?group.href(problem):undefined,state=problem?group.state(problem):undefined;
    const meta=<small className="practice-group-module-meta">{problem?<><span data-state={state||'unknown'}>{state==='done'?t('已通过','Passed'):state==='started'?t('进行中','In progress'):state==='todo'?t('未开始','Not started'):t('进度待同步','Progress pending')}</span>{typeof problem.estimated_minutes==='number'&&Number.isFinite(problem.estimated_minutes)&&problem.estimated_minutes>0&&<> · {t(`约 ${problem.estimated_minutes} 分钟`,`About ${problem.estimated_minutes} min`)}</>}</>:t('练习暂不可用','Practice unavailable')}</small>;
    return <li key={number} data-selected-problem={number}>
    <span className="practice-group-order" aria-hidden="true">{index+1}</span><span className="practice-group-title">{href?<AppLink href={href}><span>{name}</span>{meta}</AppLink>:<>{name}{meta}</>}</span>
    <div className="practice-group-controls"><button type="button" data-group-number={number} data-group-action="up" disabled={index===0} aria-label={t(`上移${name}`,`Move ${name} up`)} onClick={event=>group.move(number,-1,event.currentTarget)}><ArrowUp size={13}/></button><button type="button" data-group-number={number} data-group-action="down" disabled={index===group.numbers.length-1} aria-label={t(`下移${name}`,`Move ${name} down`)} onClick={event=>group.move(number,1,event.currentTarget)}><ArrowDown size={13}/></button><button type="button" data-group-number={number} data-group-action="remove" aria-label={t(`从练习组移除${name}`,`Remove ${name} from this batch`)} onClick={event=>group.remove(number,event.currentTarget)}><X size={13}/></button></div>
   </li>;})}</ol>
   {group.missing>0&&<p className="practice-group-unavailable" role="status">{t('部分练习尚不可用，请重试加载题库或先移除这些题目。','Some practice modules are unavailable. Retry the catalog or remove those concepts first.')}{onRetryCatalog&&<button type="button" className="text-button" onClick={event=>group.retryCatalog(event.currentTarget,onRetryCatalog)}>{t('重试题库','Retry catalog')}</button>}</p>}
   <InterviewBatchShare href={group.shareHref} modules={group.modules}/>
   <footer><small>{group.completed!==undefined?t(`${group.completed} 道已通过 · ${group.numbers.length-group.completed} 道待练习`,`${group.completed} passed · ${group.numbers.length-group.completed} remaining`):t('沿用当前工作区，进度就绪后显示完成情况。','Uses your workspace; completion appears when progress is ready.')}{estimate!==undefined&&<> · {t(`整组约 ${estimate} 分钟`,`About ${estimate} min for the whole batch`)}</>}</small>{group.startHref?<AppLink className="button" href={group.startHref}>{allDone?t('复习这组','Review this batch'):group.continuing?t('继续这组练习','Resume this batch'):group.ready?t('开始这组练习','Start this batch'):t('打开这组练习','Open this batch')}<ArrowRight size={13}/></AppLink>:<button type="button" className="button" disabled>{t('练习组暂不可用','Batch unavailable')}</button>}</footer>
  </>}
 </section>;
}
