import {useId,useRef,useState} from 'react';
import {ChevronLeft,ChevronRight} from 'lucide-react';
import {useI18n} from './i18n';
import {MAX_INTERVIEW_PAGE} from './interviewNavigation';
import './interview-pagination.css';

export default function InterviewPagination({page,total,onChange}:{page:number;total:number;onChange:(page:number)=>void}){
 const {t}=useI18n(),id=useId(),input=useRef<HTMLInputElement>(null),composing=useRef(false),imeEnter=useRef(false);
 const [value,setValue]=useState(String(page)),[invalid,setInvalid]=useState(false);
 const pages=Math.ceil(total/20),available=Math.min(pages,MAX_INTERVIEW_PAGE);
 const error=t(`请输入 1 至 ${available} 之间的整数页码。`,`Enter a whole page number from 1 to ${available}.`);
 function jump(){
  if(composing.current||imeEnter.current)return;
  const normalized=value.normalize('NFKC').trim(),next=Number(normalized);
  if(!/^\d+$/.test(normalized)||!Number.isSafeInteger(next)||next<1||next>available){setInvalid(true);input.current?.focus({preventScroll:true});return;}
  setInvalid(false);setValue(String(next));
  if(next!==page)onChange(next);
 }
 return <nav className="interview-pagination" aria-label={t('面经分页','Experience pages')}>
  <button type="button" disabled={page===1} onClick={()=>onChange(page-1)}><ChevronLeft size={15}/>{t('上一页','Previous')}</button>
  <span aria-current="page">{page>pages?t(`请求第 ${page} 页 · 当前共 ${pages} 页`,`Requested page ${page} · ${pages} pages available`):t(`第 ${page} 页 · 共 ${pages} 页`,`Page ${page} of ${pages}`)}</span>
  <button type="button" disabled={page>=available} onClick={()=>onChange(page+1)}>{t('下一页','Next')}<ChevronRight size={15}/></button>
  {pages>=4&&<form className="interview-page-jump" aria-label={t('跳转面经页码','Jump to experience page')} onSubmit={event=>{event.preventDefault();jump();}}>
   <span id={id+'-hint'} className="sr-only">{t(`页码范围 1 至 ${available}，Enter 跳转，Esc 恢复当前页。`,`Pages 1 to ${available}. Enter to go; Escape to restore the current page.`)}</span>
   <label><span>{t('前往','Go to')}</span><input ref={input} type="text" inputMode="numeric" enterKeyHint="go" autoComplete="off" maxLength={20} value={value} aria-label={t('跳转页码','Jump to page')} aria-invalid={invalid||undefined} aria-describedby={id+'-hint'+(invalid?' '+id+'-error':'')}
    onChange={event=>{setValue(event.target.value);setInvalid(false);}}
    onCompositionStart={()=>{composing.current=true;imeEnter.current=false;}}
    onCompositionEnd={()=>{composing.current=false;}}
    onKeyDown={event=>{
     const ime=composing.current||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229;
     if(event.key==='Enter'){imeEnter.current=ime;return;}
     if(event.key==='Escape'&&!ime){event.preventDefault();event.stopPropagation();setValue(String(page));setInvalid(false);imeEnter.current=false;}
    }} onKeyUp={event=>{if(event.key==='Enter')imeEnter.current=false;}} onBlur={()=>{imeEnter.current=false;}}/>
   </label><span aria-hidden="true">/ {available}</span>
   <button type="submit" onPointerDown={()=>{if(!composing.current)imeEnter.current=false;}}>{t('跳转','Go')}</button>
  </form>}
  {invalid&&<p id={id+'-error'} className="interview-page-error" role="alert">{error}</p>}
  {pages>MAX_INTERVIEW_PAGE&&<p className="interview-page-limit">{t(`最多可浏览前 ${MAX_INTERVIEW_PAGE} 页，请按公司、岗位或时间缩小范围。`,`The first ${MAX_INTERVIEW_PAGE} pages are available. Narrow the company, role or time range to find more relevant experiences.`)}</p>}
 </nav>;
}
