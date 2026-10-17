import {useEffect,useId,useRef,type RefObject} from 'react';
import {Search,X} from 'lucide-react';
import {useI18n} from './i18n';
import './related-module-search.css';

/** A search field inside a publication form; Enter never submits that outer form. */
export default function RelatedModuleSearch({value,onChange,label,disabled=false,inputRef}:{value:string;onChange:(value:string)=>void;label:string;disabled?:boolean;inputRef?:RefObject<HTMLInputElement|null>}){
 const {t}=useI18n(),hint=useId(),fallback=useRef<HTMLInputElement>(null),composing=useRef(false);
 const input=inputRef||fallback;
 useEffect(()=>{if(disabled)composing.current=false;},[disabled]);
 function clear(){if(disabled)return;composing.current=false;onChange('');input.current?.focus({preventScroll:true});}
 return <div className="related-module-search" role="search" aria-label={label}>
  <Search size={16} aria-hidden="true"/>
  <span id={hint} className="sr-only">{t('按题号、中英文名称或模块 ID 筛选，最多100个字符。Esc清空搜索，不会更换已选题目。','Filter by number, Chinese or English title, or module ID. Up to 100 characters. Escape clears the search without changing your selection.')}</span>
  <input ref={input} type="search" value={value} maxLength={200} disabled={disabled} aria-label={label} aria-describedby={hint} placeholder={t('搜索题号、名称或模块 ID…','Search number, title or module ID…')}
   onChange={event=>{if(!disabled)onChange(composing.current||(event.nativeEvent as InputEvent).isComposing?event.target.value:Array.from(event.target.value).slice(0,100).join(''));}}
   onCompositionStart={()=>{composing.current=true;}}
   onCompositionEnd={event=>{composing.current=false;if(!disabled)onChange(Array.from(event.currentTarget.value).slice(0,100).join(''));}}
   onKeyDown={event=>{
    const ime=composing.current||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229;
    if(event.key==='Enter'){event.stopPropagation();if(!ime)event.preventDefault();return;}
    if(event.key==='Escape'&&!ime&&value){event.preventDefault();event.stopPropagation();clear();}
   }}/>
  {!!value&&<button type="button" disabled={disabled} aria-label={t('清空题目搜索','Clear module search')} onClick={clear}><X size={16} aria-hidden="true"/></button>}
 </div>;
}
