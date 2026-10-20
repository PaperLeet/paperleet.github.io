import {useRef,useState} from 'react';
import {Search,X} from 'lucide-react';
import {useI18n} from './i18n';
import {activeSourcePreview} from './sourcePreviewKeyboard';
import {interviewCatalog,interviewLabel,type InterviewSummary} from './interviewModels';
import {matchesInterviewCompany} from './interviewSearch';
import {sourcePublicationTime} from './interviewEvidenceOrder';
import './source-company-directory.css';

export type SourceCompany=InterviewSummary['questions'][number]['companies'][number];
export type SourceCompanyOrder='default'|'contributors'|'recent'|'name';
const orders:SourceCompanyOrder[]=['default','contributors','recent','name'];

/** Local presentation of confirmed evidence; never rewrite counts or the source array. */
export function sourceCompanyRows(companies:readonly SourceCompany[],query:string,order:SourceCompanyOrder,locale:string,problemId:string){
 return companies.map((row,index)=>({row,index,stamp:row.problem_id===problemId?sourcePublicationTime(row.latest):undefined})).filter(({row})=>{
  const company=interviewCatalog.companies.find(company=>company.id===row.company);
  return company&&matchesInterviewCompany(company,query);
 }).sort((a,b)=>{
  if(order==='contributors')return b.row.contributors-a.row.contributors||a.index-b.index;
  if(order==='name')return interviewLabel('companies',a.row.company,locale==='en').localeCompare(interviewLabel('companies',b.row.company,locale==='en'),locale==='en'?'en':'zh-CN')||a.index-b.index;
  if(order==='recent')return a.stamp===undefined?b.stamp===undefined?a.index-b.index:1:b.stamp===undefined?-1:b.stamp-a.stamp||a.index-b.index;
  return a.index-b.index;
 }).map(({row})=>row);
}

/** Keep the field mounted across publication-period refreshes after it first appears. */
export function useSourceCompanyDirectory({companies,problemId,listId,ready,canAct,onQuery,onClose}:{companies:readonly SourceCompany[];problemId:string;listId:string;ready:boolean;canAct:()=>boolean;onQuery:()=>void;onClose:()=>void}){
 const {t,locale}=useI18n(),root=useRef<HTMLDivElement>(null),input=useRef<HTMLInputElement>(null),composing=useRef(false);
 const [query,setQuery]=useState(''),[order,setOrder]=useState<SourceCompanyOrder>('default'),[shown,setShown]=useState(false);
 if(!shown&&companies.length>4)setShown(true);
 const rows=sourceCompanyRows(companies,query,order,locale,problemId);
 function active(){return activeSourcePreview(root.current)&&canAct();}
 function change(value:string,transient=false){
  if(!active())return;
  const next=transient?value:Array.from(value).slice(0,200).join('');
  if(next!==query){setQuery(next);onQuery();}
 }
 function clear(trigger:HTMLButtonElement){
  if(!active())return;change('');
  if(trigger.ownerDocument.activeElement===trigger)input.current?.focus({preventScroll:true});
 }
 const controls=shown||companies.length>4?<div ref={root} className="source-company-directory" role="group" aria-label={t('查找来源公司','Find source companies')}>
  <div className="source-company-search"><Search size={17} aria-hidden="true"/><input ref={input} type="search" value={query} maxLength={400} aria-label={t('筛选来源公司','Filter source companies')} aria-controls={listId} aria-describedby={listId+'-help '+listId+'-status'} placeholder={t('公司中文名或英文名…','Company name in Chinese or English…')} onChange={event=>change(event.target.value,composing.current||(event.nativeEvent as InputEvent).isComposing===true)}
   onCompositionStart={()=>{if(active())composing.current=true;}} onCompositionEnd={event=>{composing.current=false;change(event.currentTarget.value);}} onBlur={()=>{composing.current=false;}}
   onKeyDown={event=>{
    if(event.key!=='Escape'||event.defaultPrevented||event.repeat||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||composing.current||event.currentTarget.ownerDocument.activeElement!==event.currentTarget||!active())return;
    event.preventDefault();event.stopPropagation();if(query)change('');else onClose();
   }}/>{query&&<button type="button" aria-label={t('清空公司搜索','Clear company search')} title={t('清空公司搜索','Clear company search')} onClick={event=>clear(event.currentTarget)}><X size={16} aria-hidden="true"/></button>}</div>
  <label className="source-company-order"><span>{t('排序','Order')}</span><select aria-label={t('来源公司排序','Source company order')} aria-controls={listId} aria-describedby={listId+'-help'} value={order} onChange={event=>{const value=event.target.value;if(active()&&orders.includes(value as SourceCompanyOrder))setOrder(value as SourceCompanyOrder);}}><option value="default">{t('默认顺序','Default order')}</option><option value="contributors">{t('分享者最多','Most contributors')}</option><option value="recent">{t('最新分享','Latest report')}</option><option value="name">{t('公司名称','Company name')}</option></select></label>
  <span id={listId+'-status'} className="source-company-count" role="status" aria-atomic="true">{ready?t(`显示 ${rows.length} / ${companies.length} 家来源公司`,`${rows.length} of ${companies.length} source companies`):t('公司筛选已保留，待来源确认后显示结果。','Company filters kept until source data is confirmed.')}</span>
  <span id={listId+'-help'} className="sr-only">{t('输入即时筛选，不改变发布时间。Esc 清空公司搜索，再按一次收起来源。最新分享按公开面经发布时间排序；未知日期排在末尾。','Filters as you type without changing the publication period. Escape clears the company search; press it again to collapse sources. Latest report uses public report publication dates, with unknown dates last.')}</span>
 </div>:null;
 return {controls,rows,query,listId,clear};
}
