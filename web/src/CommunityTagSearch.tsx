import {useId,useRef,type RefObject} from 'react';
import {Search,X} from 'lucide-react';
import {useI18n} from './i18n';
import {communityTagQuery} from './communityTagBrowsing';

/** Enter belongs to this local finder, including inside a publication form. */
export default function CommunityTagSearch({value,onChange,inputRef,label,clearLabel,placeholder,className,onEmptyEscape,disabled=false,statusId}:{value:string;onChange:(value:string)=>void;inputRef:RefObject<HTMLInputElement|null>;label:string;clearLabel:string;placeholder:string;className:string;onEmptyEscape?:()=>void;disabled?:boolean;statusId:string}){
 const {t}=useI18n(),help=useId(),composing=useRef(false);
 function clear(){if(disabled)return;composing.current=false;onChange('');inputRef.current?.focus({preventScroll:true});}
 return <div className={className} role="search" aria-label={`${label} · ${t('搜索区域','Search area')}`}><Search size={16} aria-hidden="true"/>
  <input ref={inputRef} type="search" maxLength={200} value={value} disabled={disabled} aria-label={label} aria-describedby={`${help} ${statusId}`} placeholder={placeholder} onChange={event=>{if(!disabled)onChange(composing.current||(event.nativeEvent as InputEvent).isComposing?event.target.value:communityTagQuery(event.target.value));}} onCompositionStart={()=>{composing.current=true;}} onCompositionEnd={event=>{composing.current=false;if(!disabled)onChange(communityTagQuery(event.currentTarget.value));}} onKeyDown={event=>{
   const ime=composing.current||event.nativeEvent.isComposing||event.keyCode===229;
   if(event.key==='Enter'){event.stopPropagation();if(!ime)event.preventDefault();return;}
   if(event.key!=='Escape'||ime||event.defaultPrevented||event.repeat||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey)return;
   if(value||onEmptyEscape){event.preventDefault();event.stopPropagation();if(value)clear();else onEmptyEscape?.();}
  }}/>
  {!!value&&<button type="button" disabled={disabled} aria-label={clearLabel} onClick={clear}><X size={15} aria-hidden="true"/></button>}
  <span id={help} className="sr-only">{t('搜索只缩小标签选项，不会提交表单。最多100个字符。','Search only narrows tag choices and never submits a form. Up to 100 characters.')}</span>
 </div>;
}
