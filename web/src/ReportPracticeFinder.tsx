import {useRef,useState} from 'react';
import {Search,X} from 'lucide-react';
import {useI18n} from './i18n';
import {activeSourcePreview} from './sourcePreviewKeyboard';
import {reportPracticeMatches,reportPracticeQuery} from './reportPracticeSearch';
import type {ProblemTopic} from './communityTopics';
import './report-practice-search.css';

/** This field filters linked public practice only; it cannot change a report or practice queue. */
export function useReportPracticeSearch({ids,topics,scope,expanded,listId,onCollapse}:{ids:readonly string[];topics:ReadonlyMap<string,ProblemTopic>;scope:string;expanded:boolean;listId:string;onCollapse:()=>void}){
 const {t}=useI18n(),root=useRef<HTMLDivElement>(null),input=useRef<HTMLInputElement>(null),composing=useRef(false),compositionScope=useRef<string>(undefined);
 const [state,setState]=useState({scope,draft:'',query:''});
 const lease=useRef({scope,href:location.href});
 if(lease.current.scope!==scope){lease.current={scope,href:location.href};composing.current=false;}
 if(state.scope!==scope)setState({scope,draft:'',query:''});
 const draft=state.scope===scope&&expanded?state.draft:'',query=state.scope===scope&&expanded?state.query.trim():'';
 const matches=reportPracticeMatches(ids,topics,query),missing=ids.filter(id=>!topics.has(id)).length;
 const canAct=()=>expanded&&activeSourcePreview(root.current)&&!root.current?.closest('details:not([open])')&&lease.current.scope===scope&&lease.current.href===location.href;
 function change(value:string,transient=false){
  if(!canAct()||transient&&compositionScope.current!==scope)return;
  const next=transient?value:reportPracticeQuery(value);
  setState(previous=>({scope,draft:next,query:transient&&previous.scope===scope?previous.query:transient?'':next}));
 }
 function reset(){composing.current=false;compositionScope.current=undefined;setState({scope,draft:'',query:''});}
 function clear(trigger:HTMLButtonElement){if(!canAct())return;reset();if(document.activeElement===trigger)input.current?.focus({preventScroll:true});}
 const controls=expanded&&ids.length>4?<div ref={root} className="report-practice-search" role="group" aria-label={t('查找本篇配套练习','Find practice in this experience')}>
  <div className="report-practice-search-field"><Search size={17} aria-hidden="true"/><input ref={input} type="search" value={draft} maxLength={400} aria-label={t('查找配套练习','Find linked practice')} aria-controls={listId} aria-describedby={listId+'-search-help '+listId+'-search-count'} placeholder={t('题号、中英文名称或模块…','Number, title or module…')} onChange={event=>change(event.target.value,composing.current||(event.nativeEvent as InputEvent).isComposing===true)} onCompositionStart={()=>{if(canAct()){composing.current=true;compositionScope.current=scope;}}} onCompositionEnd={event=>{const expected=compositionScope.current;composing.current=false;compositionScope.current=undefined;if(expected===scope)change(event.currentTarget.value);else if(canAct())event.currentTarget.value=draft;}} onBlur={event=>{if(composing.current){composing.current=false;change(event.currentTarget.value);}}} onKeyDown={event=>{
   if(event.defaultPrevented||event.repeat||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||composing.current||document.activeElement!==event.currentTarget||!canAct())return;
   if(event.key==='Enter'){event.preventDefault();event.stopPropagation();}
   if(event.key==='Escape'){event.preventDefault();event.stopPropagation();if(draft||query)reset();else{reset();onCollapse();}}
  }}/>{draft&&<button type="button" aria-label={t('清空配套练习搜索','Clear linked practice search')} onClick={event=>clear(event.currentTarget)}><X size={16} aria-hidden="true"/></button>}</div>
  <span id={listId+'-search-count'} className="report-practice-search-count" role="status" aria-atomic="true">{t(`显示 ${matches.length} / ${ids.length} 个配套知识点`,`${matches.length} of ${ids.length} linked concepts`)}</span>
  <p id={listId+'-search-help'}>{t('仅查找本篇已关联的练习，不改变题单与学习进度。Esc 清空搜索，再按一次收起。','Searches only this experience’s linked practice without changing your practice group or progress. Escape clears the search; press again to collapse.')}{missing>0&&<span>{t(` ${missing} 个知识点信息未齐，按名称或题号搜索可能不完整；仍可用模块 ID 查找。`,` ${missing} concepts lack details, so title or number matches may be incomplete. Module IDs remain searchable.`)}</span>}</p>
  {!!query.trim()&&!matches.length&&<div className="report-practice-search-empty" role="status"><span>{t('本篇配套练习没有匹配项，可换个关键词或清空搜索。','No linked practice matches in this experience. Try another keyword or clear the search.')}</span><button type="button" className="text-button" onClick={event=>clear(event.currentTarget)}>{t('显示全部配套练习','Show all linked practice')}</button></div>}
 </div>:null;
 return {controls,matches,query,input,reset};
}
