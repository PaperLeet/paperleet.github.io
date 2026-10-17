import {useEffect,useId,useRef,useState,type RefObject} from 'react';
import {ArrowRight,Search,X} from 'lucide-react';
import {useI18n} from './i18n';
import './community-search.css';

const bounded=(value:string)=>Array.from(value).slice(0,200).join('');

export default function CommunitySearch({applied,onApply,label,placeholder,disabled=false,className='community-search',describedBy,inputRef,resetScope}:{applied:string;onApply:(query:string)=>void;label?:string;placeholder?:string;disabled?:boolean;className?:string;describedBy?:string;inputRef?:RefObject<HTMLInputElement|null>;resetScope?:string}){
 const {t}=useI18n(),hint=useId(),fallback=useRef<HTMLInputElement>(null),composing=useRef(false),imeEnter=useRef(false);
 const input=inputRef||fallback;
 const [query,setQuery]=useState(applied);
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
 return <form className={className} role="search" aria-label={label||t('搜索讨论','Search discussions')} onSubmit={event=>{
  event.preventDefault();if(disabled||composing.current||imeEnter.current)return;
  const next=bounded(query).replace(/[\u0000-\u001f\u007f]/g,' ').trim();setQuery(next);onApply(next);
 }}>
  <Search size={16} aria-hidden="true"/>
  <span id={hint} className="sr-only">{t('最多200个字符。Enter搜索，Esc清空。','Up to 200 characters. Enter to search; Escape to clear.')}</span>
  <input ref={input} type="search" enterKeyHint="search" disabled={disabled} aria-label={label||t('搜索讨论','Search discussions')} aria-describedby={[hint,describedBy].filter(Boolean).join(' ')} placeholder={placeholder||t('搜索讨论…','Search discussions…')} value={query} maxLength={400}
   onChange={event=>change(event.target.value,composing.current||(event.nativeEvent as InputEvent).isComposing===true)}
   onCompositionStart={()=>{composing.current=true;imeEnter.current=false;}}
   onCompositionEnd={event=>{composing.current=false;change(event.currentTarget.value);}}
   onKeyDown={event=>{
    const ime=composing.current||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229;
    if(event.key==='Enter'){imeEnter.current=ime;return;}
    if(event.key==='Escape'&&!ime&&query){event.preventDefault();event.stopPropagation();clear();}
   }} onKeyUp={event=>{if(event.key==='Enter')imeEnter.current=false;}} onBlur={()=>{imeEnter.current=false;}}/>
  {!!query&&<button type="button" disabled={disabled} className="community-search-clear" aria-label={t('清空搜索','Clear search')} title={t('清空搜索','Clear search')} onClick={clear}><X size={16} aria-hidden="true"/></button>}
  <button type="submit" disabled={disabled} aria-label={t('搜索','Search')} onPointerDown={()=>{if(!composing.current)imeEnter.current=false;}}><ArrowRight size={16} aria-hidden="true"/></button>
 </form>;
}
