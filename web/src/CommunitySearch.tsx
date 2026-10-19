import {useEffect,useId,useRef,useState,type RefObject} from 'react';
import {Search,X} from 'lucide-react';
import {useI18n} from './i18n';
import './community-search.css';

const bounded=(value:string)=>Array.from(value).slice(0,200).join('');
const normalized=(value:string)=>bounded(value).replace(/[\u0000-\u001f\u007f]/g,' ').trim();

export default function CommunitySearch({applied,onApply,label,placeholder,disabled=false,className='community-search',describedBy,inputRef,resetScope}:{applied:string;onApply:(query:string)=>void;label?:string;placeholder?:string;disabled?:boolean;className?:string;describedBy?:string;inputRef?:RefObject<HTMLInputElement|null>;resetScope?:string}){
 const {t}=useI18n(),hint=useId(),fallback=useRef<HTMLInputElement>(null),composing=useRef(false),imeEnter=useRef(false);
 const input=inputRef||fallback;
 const [query,setQuery]=useState(applied);
 const pending=query!==applied&&normalized(query)!==applied;
 useEffect(()=>{setQuery(applied);composing.current=false;imeEnter.current=false;},[applied,resetScope]);
 useEffect(()=>{if(disabled){composing.current=false;imeEnter.current=false;}},[disabled]);
 function change(value:string,transient=composing.current){
  if(disabled)return;
  const next=transient?value:bounded(value);setQuery(next);
  if(!transient&&!next&&applied)onApply('');
 }
 function clear(){
  if(disabled)return;
  setQuery('');imeEnter.current=false;
  input.current?.focus({preventScroll:true});
  // Cancelling unsubmitted text does not reset the current feed's page.
  if(applied)onApply('');
 }
 function restore(){
  if(disabled||composing.current)return;
  setQuery(applied);imeEnter.current=false;
  input.current?.focus({preventScroll:true});
 }
 return <form className={`${className} community-query`} data-query={!!query||undefined} data-pending={pending||undefined} role="search" aria-label={label||t('搜索讨论','Search discussions')} onSubmit={event=>{
  event.preventDefault();if(disabled||composing.current||imeEnter.current)return;
  const next=normalized(query);setQuery(next);onApply(next);
 }}>
  <Search size={16} className="community-search-icon" aria-hidden="true"/>
  <span id={hint} className="sr-only">{t('最多200个字符。Enter搜索，Esc清空。','Up to 200 characters. Enter to search; Escape to clear.')}</span>
  <input ref={input} type="search" enterKeyHint="search" disabled={disabled} aria-label={label||t('搜索讨论','Search discussions')} aria-describedby={[hint,describedBy,pending?hint+'-pending':undefined].filter(Boolean).join(' ')} placeholder={placeholder||t('搜索讨论…','Search discussions…')} value={query} maxLength={400}
   onChange={event=>change(event.target.value,composing.current||(event.nativeEvent as InputEvent).isComposing===true)}
   onCompositionStart={()=>{composing.current=true;imeEnter.current=false;}}
   onCompositionEnd={event=>{composing.current=false;change(event.currentTarget.value);}}
   onKeyDown={event=>{
    const ime=composing.current||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229;
    if(event.key==='Enter'){imeEnter.current=ime;return;}
    if(event.key==='Escape'&&!ime&&query){event.preventDefault();event.stopPropagation();clear();}
   }} onKeyUp={event=>{if(event.key==='Enter')imeEnter.current=false;}} onBlur={()=>{imeEnter.current=false;}}/>
  {!!query&&<button type="button" disabled={disabled} className="community-search-clear" aria-label={t('清空搜索','Clear search')} title={t('清空搜索','Clear search')} onClick={clear}><X size={16} aria-hidden="true"/></button>}
  <button type="submit" className="community-search-submit" disabled={disabled} aria-label={t('搜索','Search')} onPointerDown={()=>{if(!composing.current)imeEnter.current=false;}}>{t('搜索','Search')}</button>
  {pending&&<div className="community-search-pending"><p id={hint+'-pending'} role="status" aria-atomic="true">{disabled?t('输入已保留 · 暂不可搜索','Input kept · Search unavailable'):applied?t('输入未应用','Not applied'):t('输入未应用 · 按 Enter 搜索','Not applied · Press Enter')}</p>{!!applied&&<button type="button" className="community-search-restore" disabled={disabled} aria-label={t('还原已应用的搜索','Restore applied search')} title={t('保留当前结果与页码，撤回未提交的输入','Keep the current results and page; discard unsubmitted edits')} onClick={restore}>{t('还原','Restore')}</button>}</div>}
 </form>;
}
