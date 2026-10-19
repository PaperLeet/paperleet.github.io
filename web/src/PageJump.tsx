import {useId,useRef,useState} from 'react';
import {useI18n} from './i18n';
import './page-jump.css';

export default function PageJump({page,maxPage,onChange,label,className=''}:{page:number;maxPage:number;onChange:(page:number)=>void;label:string;className?:string}){
 const {t}=useI18n(),id=useId(),input=useRef<HTMLInputElement>(null),composing=useRef(false),imeEnter=useRef(false);
 const [value,setValue]=useState(String(page)),[invalid,setInvalid]=useState(false);
 function jump(){
  if(composing.current||imeEnter.current)return;
  const normalized=value.normalize('NFKC').trim(),next=Number(normalized);
  if(!/^\d+$/.test(normalized)||!Number.isSafeInteger(next)||next<1||next>maxPage){setInvalid(true);input.current?.focus({preventScroll:true});return;}
  setInvalid(false);setValue(String(next));
  if(next!==page)onChange(next);
 }
 return <form className={`page-jump ${className}`} aria-label={label} onSubmit={event=>{event.preventDefault();jump();}}>
  <span id={id+'-hint'} className="sr-only">{t(`页码范围 1 至 ${maxPage}，Enter 跳转，Esc 恢复当前页。`,`Pages 1 to ${maxPage}. Enter to go; Escape to restore the current page.`)}</span>
  <label><span>{t('前往','Go to')}</span><input ref={input} type="text" inputMode="numeric" enterKeyHint="go" autoComplete="off" maxLength={20} value={value} aria-label={t('跳转页码','Jump to page')} aria-invalid={invalid||undefined} aria-describedby={id+'-hint'+(invalid?' '+id+'-error':'')}
   onChange={event=>{setValue(event.target.value);setInvalid(false);}}
   onCompositionStart={()=>{composing.current=true;imeEnter.current=false;}}
   onCompositionEnd={()=>{composing.current=false;}}
   onKeyDown={event=>{
    const ime=composing.current||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229;
    if(event.key==='Enter'){imeEnter.current=ime;return;}
    if(event.key==='Escape'&&!ime&&!event.repeat&&!event.ctrlKey&&!event.metaKey&&!event.altKey&&!event.shiftKey){event.preventDefault();event.stopPropagation();setValue(String(page));setInvalid(false);imeEnter.current=false;}
   }} onKeyUp={event=>{if(event.key==='Enter')imeEnter.current=false;}} onBlur={()=>{imeEnter.current=false;}}/>
  </label><span aria-hidden="true">/ {maxPage}</span>
  <button type="submit" onPointerDown={()=>{if(!composing.current)imeEnter.current=false;}}>{t('跳转','Go')}</button>
  {invalid&&<p id={id+'-error'} className="page-jump-error" role="alert">{t(`请输入 1 至 ${maxPage} 之间的整数页码。`,`Enter a whole page number from 1 to ${maxPage}.`)}</p>}
 </form>;
}
