import {appLocation} from './appLocation';
import AppLink from './AppLink';
import {useEffect,useId,useLayoutEffect,useMemo,useRef,useState,type KeyboardEvent} from 'react';
import {CheckCircle2,ChevronDown,ChevronLeft,ChevronRight,Circle,Search,X} from 'lucide-react';
import {readInterviewRoute} from './interviewNavigation';
import {preparationProgress} from './interviewPreparation';
import {interviewPracticeBatch} from './interviewPracticeBatch';
import {readLibraryRoute,withLibraryContext} from './libraryNavigation';
import {useI18n} from './i18n';
import {status,type Problem,type Saved} from './types';
import {controlIsActive} from './activeControls';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import './module-picker-search.css';

type Props={current:Problem;modules:Problem[];allModules?:Problem[];saved:Record<string,Saved>;ready:boolean;returnTo?:string;course?:boolean;scopeReady?:boolean};
export function ModuleNavigator(props:Props){return <ModuleNavigation key={JSON.stringify([props.current.id,props.returnTo])} {...props}/>;}
function ModuleNavigation({current,modules:paperModules,allModules,saved,ready,returnTo,course,scopeReady=true}:Props){
 const {t,locale}=useI18n();
 const track=returnTo?readInterviewRoute(returnTo)?.track:undefined;
 const progress=track?preparationProgress(track,allModules||paperModules,saved):undefined;
 const scoped=progress?.modules.some(module=>module.id===current.id)?progress:undefined;
 const batch=interviewPracticeBatch(returnTo,allModules||paperModules,saved,current);
 const modules=batch?.modules||scoped?.modules||paperModules;
 const library=readLibraryRoute(returnTo||''),company=!!library?.filters.company,direct=library?.filters.view==='modules'&&!course;
 const heading=company?t('当前公司练习','Current company practice'):direct?t('当前题目列表','Current exercise list'):batch?t('当前准备批次','This preparation batch'):scoped?(locale==='en'?scoped.path.name_en:scoped.path.name):course?t('本专题','This course'):t('本篇论文','This paper');
 const [open,setOpen]=useState(false),[draft,setDraft]=useState(''),[query,setQuery]=useState('');
 const root=useRef<HTMLDivElement>(null),trigger=useRef<HTMLButtonElement>(null),search=useRef<HTMLInputElement>(null),list=useRef<HTMLDivElement>(null),composing=useRef(false);
 const href=useRef(typeof window==='undefined'?'':window.location.href).current,popupId=useId(),listId=popupId+'-list',hint=popupId+'-hint';
 const scope=JSON.stringify([current.id,returnTo,scopeReady,modules.map(module=>module.id)]),intent=useWorkspaceIntent(scope);
 const opening=useRef<{keyboard:boolean;claim:()=>boolean}>(undefined);
 const index=modules.findIndex(module=>module.id===current.id),core=batch||scoped?modules:modules.filter(module=>!module.module.optional);
 const states=useMemo(()=>new Map(modules.map(module=>[module.id,ready?status(module,saved[module.id]):'todo'] as const)),[modules,saved,ready]);
 const solved=core.filter(module=>states.get(module.id)==='done').length,searchable=modules.length>8||!!draft||!!query;
 const folded=query.normalize('NFKC').toLowerCase().trim(),terms=folded.split(/\s+/u).filter(Boolean);
 const numeric=/^[0-9]+$/.test(folded)&&Number.isSafeInteger(Number(folded))?Number(folded):undefined;
 const visible=modules.filter(module=>{if(numeric!==undefined)return module.number===numeric;const text=[module.number,module.id,module.title,module.translations?.en?.title].join('\n').normalize('NFKC').toLowerCase();return terms.every(term=>text.includes(term));});
 const here=()=>controlIsActive(root.current)&&!root.current?.closest('details:not([open]),dialog:not([open])')&&location.href===href&&intent.isCurrent();
 const usable=()=>open&&here();
 const links=()=>Array.from(list.current?.querySelectorAll<HTMLAnchorElement>('a')||[]);
 function revealLink(target:HTMLElement){
  const pane=list.current;if(!pane?.clientHeight||!pane.contains(target))return;
  const edge=pane.getBoundingClientRect().top+pane.clientTop,box=target.getBoundingClientRect();
  const delta=box.top<edge?box.top-edge:box.bottom>edge+pane.clientHeight?box.bottom-edge-pane.clientHeight:0;
  if(delta)pane.scrollTop+=delta;
 }
 useLayoutEffect(()=>{
  const request=opening.current;if(!request||!open)return;opening.current=undefined;
  if(!here()||!request.claim())return;
  const rows=links(),target=rows.find(link=>link.getAttribute('aria-current')==='page')||rows[0];
  if(request.keyboard)(search.current||target)?.focus({preventScroll:true});
  if(target)revealLink(target);
 },[open,scope]);
 useEffect(()=>{
  if(!open)return;
  const outside=(event:PointerEvent)=>{if(!root.current?.contains(event.target as Node)){opening.current=undefined;intent.cancel();setOpen(false);}};
  document.addEventListener('pointerdown',outside);return()=>document.removeEventListener('pointerdown',outside);
 },[open,scope]);
 function toggle(keyboard=false){
  if(!here())return;
  if(open&&!keyboard){opening.current=undefined;intent.cancel();setOpen(false);return;}
  opening.current={keyboard,claim:intent.create()};setOpen(true);
  // A second keyboard opening can focus an already mounted search without a frame.
  if(open){const request=opening.current;opening.current=undefined;if(request.claim())(search.current||links().find(link=>link.getAttribute('aria-current')==='page')||links()[0])?.focus({preventScroll:true});}
 }
 function plain(event:KeyboardEvent<HTMLElement>){return !event.defaultPrevented&&!event.nativeEvent.isComposing&&event.nativeEvent.keyCode!==229&&!event.altKey&&!event.ctrlKey&&!event.metaKey&&!event.shiftKey&&!composing.current;}
 function close(){if(!usable())return;opening.current=undefined;intent.cancel();setOpen(false);trigger.current?.focus({preventScroll:true});}
 function apply(value:string,transient=false){
  if(!usable())return;
  const next=transient?value:Array.from(value).slice(0,200).join('');setDraft(next);if(!transient)setQuery(next.replace(/[\u0000-\u001f\u007f]/g,' '));
 }
 function clear(){if(!usable())return;composing.current=false;setDraft('');setQuery('');search.current?.focus({preventScroll:true});}
 function browse(event:KeyboardEvent<HTMLElement>){
  if(!usable()||!plain(event)||!['ArrowDown','ArrowUp','Home','End'].includes(event.key)||event.target!==document.activeElement)return;
  const rows=links(),active=document.activeElement,position=rows.findIndex(row=>row===active);
  if(position<0&&active!==search.current)return;
  if(active===search.current&&event.key!=='ArrowDown')return;
  const next=active===search.current?0:event.key==='Home'?0:event.key==='End'?rows.length-1:event.key==='ArrowUp'?position-1:Math.min(rows.length-1,position+1);
  const target=next<0?search.current:rows[next];if(!target)return;
  event.preventDefault();event.stopPropagation();target.focus({preventScroll:true});revealLink(target);
 }
 const guard=(event:import('react').MouseEvent<HTMLAnchorElement>)=>{if(!here())event.preventDefault();};
 if((company||direct)&&(!scopeReady||index<0))return <div ref={root} className="module-batch-unavailable" role="status">{!scopeReady?company?t('公司练习范围尚未确认，仍可编辑与运行当前题目。','Company scope is unconfirmed. You can edit and run this module.'):t('正在确认列表进度，仍可编辑与运行当前题目。','Checking list progress. You can edit and run this module.'):t('当前题目不在最新筛选范围，仍可继续编辑。','This module is outside the latest scope and remains editable.')}<AppLink onClick={guard} href={returnTo}>{company?t('返回公司练习','Back to company practice'):t('返回题目列表','Back to exercise list')}</AppLink></div>;
 if(batch&&!batch.currentIncluded)return <div ref={root} className="module-batch-unavailable" role="status">{t('这组练习已调整，仍可继续编辑当前题目。','This practice batch has changed. Your current module remains editable.')}<AppLink onClick={guard} href={returnTo}>{t('返回准备列表','Return to preparation')}</AppLink></div>;
 return <div className="problem-navigation module-navigator" ref={root} onBlur={event=>{if(event.relatedTarget&&!event.currentTarget.contains(event.relatedTarget as Node)){opening.current=undefined;intent.cancel();setOpen(false);}}} onKeyDown={event=>{
  if(!usable()||!plain(event)||event.key!=='Escape'||event.repeat||event.target!==document.activeElement)return;
  event.preventDefault();event.stopPropagation();close();
 }}>
  <button className="icon-button" type="button" aria-label={t('上一个模块','Previous module')} disabled={index<=0} onClick={()=>{if(here()&&index>0)appLocation.hash=withLibraryContext(modules[index-1].number,returnTo).slice(1);}}><ChevronLeft size={15}/></button>
  <button className="module-picker-trigger" type="button" ref={trigger} aria-expanded={open} aria-controls={open?popupId:undefined} aria-label={t(`选择模块，当前第 ${index+1} 个，共 ${modules.length} 个`,`Choose module, ${index+1} of ${modules.length}`)} onClick={()=>toggle()} onKeyDown={event=>{
   if(event.key!=='ArrowDown'||!plain(event)||event.repeat||!here()||event.currentTarget!==document.activeElement)return;
   event.preventDefault();event.stopPropagation();toggle(true);
  }}><span>{index+1} / {modules.length}</span><ChevronDown size={12}/></button>
  <button className="icon-button" type="button" aria-label={t('下一个模块','Next module')} disabled={index<0||index>=modules.length-1} onClick={()=>{if(here()&&index>=0&&index<modules.length-1)appLocation.hash=withLibraryContext(modules[index+1].number,returnTo).slice(1);}}><ChevronRight size={15}/></button>
  {open&&<nav className="module-picker" id={popupId} aria-label={heading}>
   <div className="module-picker-heading"><b>{heading}</b><span>{ready?`${solved} / ${batch?.total||scoped?.total||core.length} ${batch?t('本组练习已通过','batch modules solved'):scoped?t('路径模块已通过','path modules solved'):t('核心模块已通过','core modules solved')}`:t('正在加载进度…','Loading progress…')}</span></div>
   {!!batch?.missing&&<p className="module-batch-note" role="status">{t(`${batch.missing} 道练习暂不可用，当前顺序保留。`,`${batch.missing} modules are unavailable. The remaining order is preserved.`)}</p>}
   {searchable&&<div className="module-picker-tools">
    <div className="module-picker-search" role="search" aria-label={t('搜索当前题单','Search this list')}><Search size={16} aria-hidden="true"/><input ref={search} type="search" enterKeyHint="search" value={draft} maxLength={400} aria-label={t('搜索当前题单','Search this list')} aria-controls={listId} aria-describedby={hint+' '+popupId+'-count'} placeholder={t('编号、标题或题目 ID…','Number, title or module ID…')} onChange={event=>apply(event.target.value,composing.current||(event.nativeEvent as InputEvent).isComposing===true)} onCompositionStart={()=>{if(usable())composing.current=true;}} onCompositionEnd={event=>{composing.current=false;apply(event.currentTarget.value);}} onBlur={event=>{if(composing.current){composing.current=false;apply(event.currentTarget.value);}}} onKeyDown={event=>{
      if(!usable()||!plain(event))return;
      if(event.key==='Enter'){event.preventDefault();event.stopPropagation();return;}
      if(event.key==='Escape'&&!event.repeat&&draft){event.preventDefault();event.stopPropagation();clear();return;}
      browse(event);
     }}/>{!!draft&&<button type="button" className="icon-button" aria-label={t('清空题单搜索','Clear list search')} onClick={clear}><X size={16} aria-hidden="true"/></button>}</div>
    <p id={popupId+'-count'} role="status" aria-atomic="true">{t(`显示 ${visible.length} / ${modules.length} 道题目`,`${visible.length} of ${modules.length} modules`)}<span>{t('顺序保持不变','Original order kept')}</span></p>
   </div>}
   <span id={hint} className="sr-only">{t('最多200个字符，仅筛选当前题单。用上下方向键浏览结果，在题目链接按 Enter 打开；Esc先清空搜索，再关闭列表。','Up to 200 characters; filters this list only. Use arrow keys to browse results and Enter on a link to open it. Escape clears search, then closes the list.')}</span>
   <div className="module-picker-links" ref={list} id={listId} onKeyDown={browse}>{visible.map(module=>{
    const state=states.get(module.id)||'todo',active=module.id===current.id;
    const minutes=typeof module.estimated_minutes==='number'&&Number.isFinite(module.estimated_minutes)&&module.estimated_minutes>0?module.estimated_minutes:undefined;
    return <AppLink key={module.id} href={withLibraryContext(module.number,returnTo)} aria-current={active?'page':undefined} onClick={event=>{if(!usable()){event.preventDefault();return;}if(event.button===0&&!event.metaKey&&!event.ctrlKey&&!event.altKey&&!event.shiftKey)setOpen(false);}}>
     <span className={`module-picker-state ${state}`} role="img" aria-label={!ready?t('正在加载进度','Loading progress'):state==='done'?t('已通过','Solved'):state==='started'?t('进行中','In progress'):t('未开始','Not started')}>{state==='done'?<CheckCircle2 size={16}/>:<Circle size={16}/>}</span>
     <span><b>{module.number}. {locale==='en'?module.translations?.en?.title||module.title:module.title}</b><small>{minutes?t(`预计 ${minutes} 分钟`,`About ${minutes} min`):t('时长未标注','Time not listed')}{module.module.optional?` · ${t('可选拓展','Optional extension')}`:''}{active?` · ${t('当前模块','Current module')}`:''}</small></span>
     {active&&<span className="module-picker-current" aria-hidden="true"/>}
    </AppLink>;
   })}</div>
   {!visible.length&&<div className="module-picker-empty"><p>{t('当前题单没有匹配题目。','No matching modules in this list.')}</p><button type="button" className="text-button" onClick={clear}>{t('查看全部题目','Show all modules')}</button></div>}
  </nav>}
 </div>;
}
