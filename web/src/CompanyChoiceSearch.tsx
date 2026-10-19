import {useId,useRef} from 'react';
import {Search,X} from 'lucide-react';
import {useI18n} from './i18n';

/** A local catalog finder; it never changes the applied comparison or submits a form. */
export default function CompanyChoiceSearch({value,onChange,count,statusId}:{value:string;onChange:(query:string)=>void;count:number;statusId:string}){
 const {t}=useI18n(),help=useId(),input=useRef<HTMLInputElement>(null),composing=useRef(false);
 function clear(){composing.current=false;onChange('');input.current?.focus({preventScroll:true});}
 return <div className="company-choice-finder">
  <div className="company-choice-search" role="search" aria-label={t('查找对比公司','Find companies to compare')}><Search size={16} aria-hidden="true"/>
   <input ref={input} type="search" maxLength={200} value={value} aria-label={t('查找对比公司','Find companies to compare')} aria-describedby={`${help} ${statusId}`} placeholder={t('搜索公司中英文名称或标识…','Search company name or identifier…')} onChange={event=>onChange(composing.current||(event.nativeEvent as InputEvent).isComposing?event.target.value:Array.from(event.target.value).slice(0,100).join(''))} onCompositionStart={()=>{composing.current=true;}} onCompositionEnd={event=>{composing.current=false;onChange(Array.from(event.currentTarget.value).slice(0,100).join(''));}} onKeyDown={event=>{
    const ime=composing.current||event.nativeEvent.isComposing||event.keyCode===229;
    if(event.key==='Enter'){event.stopPropagation();if(!ime)event.preventDefault();return;}
    if(event.key==='Escape'&&!ime&&value){event.preventDefault();event.stopPropagation();clear();}
   }}/>
   {!!value&&<button type="button" aria-label={t('清空对比公司搜索','Clear comparison company search')} onClick={clear}><X size={16} aria-hidden="true"/></button>}
  </div>
  <p id={help}>{t('搜索只缩小两个下拉列表的选项；已选公司会保留，选择公司后才更新对比。最多100个字符，Esc清空搜索。','Search only narrows the two company lists. Selected companies are kept; choosing a company updates the comparison. Up to 100 characters; Escape clears search.')}</p>
  <p id={statusId} role="status" aria-live="polite" aria-atomic="true">{count?t(`匹配 ${count} 家公司`,`${count} ${count===1?'company matches':'companies match'}`):t('没有匹配的公司，已选项保留。可以清空搜索或尝试其他名称。','No companies match. Your selections are kept. Clear search or try another name.')}</p>
 </div>;
}
