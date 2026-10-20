import {useId,useLayoutEffect,useRef,useState} from 'react';
import {useI18n} from './i18n';
import {controlIsActive} from './activeControls';
import './page-jump.css';

export default function PageJump({page,maxPage,onChange,label,className='',controls,canAct}:{page:number;maxPage:number;onChange:(page:number)=>void;label:string;className?:string;controls?:string;canAct?:()=>boolean}){
 const {t}=useI18n(),id=useId(),input=useRef<HTMLInputElement>(null),composing=useRef(false),imeEnter=useRef(false);
 const recoveryFocus=useRef(false),form=useRef<HTMLFormElement>(null),href=typeof window==='undefined'?'':window.location.href;
 function active(){return controlIsActive(form.current)&&(canAct?canAct():form.current?.ownerDocument.location.href===href);}
 const [value,setValue]=useState(String(page)),[invalid,setInvalid]=useState(false);
 useLayoutEffect(()=>{const requested=recoveryFocus.current;recoveryFocus.current=false;if(requested&&active()&&input.current===document.activeElement)input.current?.select();});
 const normalized=value.normalize('NFKC').trim(),numeric=Number(normalized);
 const pending=!invalid&&/^\d+$/.test(normalized)&&Number.isSafeInteger(numeric)&&numeric>=1&&numeric<=maxPage&&numeric!==page;
 const outsideRange=/^-?\d+$/.test(normalized)&&Number.isSafeInteger(numeric)&&(numeric<1||numeric>maxPage);
 const currentValid=Number.isSafeInteger(page)&&page>=1&&page<=maxPage;
 const recovery=outsideRange?(numeric<1?1:maxPage):currentValid?page:Number.isSafeInteger(page)?Math.max(1,Math.min(page,maxPage)):1;
 const recoveryLabel=outsideRange||!currentValid?(recovery===1?t('使用首页（1）','Use first page (1)'):t(`使用末页（${maxPage}）`,`Use last page (${maxPage})`)):t(`恢复当前页（${page}）`,`Restore current page (${page})`);
 function recover(){
  if(!active()||composing.current||imeEnter.current)return;
  recoveryFocus.current=true;setValue(String(recovery));setInvalid(false);
  input.current?.focus({preventScroll:true});
 }
 function jump(){
  if(!active()||composing.current||imeEnter.current)return;
  const next=numeric;
  if(!/^\d+$/.test(normalized)||!Number.isSafeInteger(next)||next<1||next>maxPage){setInvalid(true);input.current?.focus({preventScroll:true});return;}
  setInvalid(false);setValue(String(next));
  if(next!==page)onChange(next);
 }
 return <form ref={form} className={`page-jump ${className}`} data-pending={pending||undefined} style={{'--page-jump-digits':Math.min(12,Math.max(2,String(maxPage).length,value.length))} as import('react').CSSProperties} aria-label={label} onSubmit={event=>{event.preventDefault();jump();}}>
  <span id={id+'-hint'} className="sr-only">{t(`页码范围 1 至 ${maxPage}，Enter 跳转，Esc 恢复当前页。`,`Pages 1 to ${maxPage}. Enter to go; Escape to restore the current page.`)}</span>
  <label><span>{t('前往','Go to')}</span><input ref={input} type="text" inputMode="numeric" enterKeyHint="go" autoComplete="off" maxLength={20} value={value} aria-label={t('跳转页码','Jump to page')} aria-controls={controls} aria-invalid={invalid||undefined} aria-describedby={id+'-hint'+(invalid?' '+id+'-error':'')}
   onFocus={event=>{if(active())event.currentTarget.select();}}
   onChange={event=>{setValue(event.target.value);setInvalid(false);}}
   onCompositionStart={()=>{composing.current=true;imeEnter.current=false;}}
   onCompositionEnd={()=>{composing.current=false;}}
   onKeyDown={event=>{
    if(!active())return;
    const ime=composing.current||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229;
    if(event.key==='Enter'){imeEnter.current=ime;return;}
    if(event.key==='Escape'&&!ime&&!event.repeat&&!event.ctrlKey&&!event.metaKey&&!event.altKey&&!event.shiftKey){event.preventDefault();event.stopPropagation();setValue(String(page));setInvalid(false);imeEnter.current=false;}
   }} onKeyUp={event=>{if(event.key==='Enter')imeEnter.current=false;}} onBlur={()=>{imeEnter.current=false;}}/>
  </label><span aria-hidden="true">/ {maxPage}</span>
  <button type="submit" onPointerDown={()=>{if(!composing.current)imeEnter.current=false;}}>{t('跳转','Go')}</button>
  {invalid&&<div className="page-jump-feedback"><p id={id+'-error'} className="page-jump-error" role="alert">{t(`请输入 1 至 ${maxPage} 之间的整数页码。`,`Enter a whole page number from 1 to ${maxPage}.`)}</p><button type="button" className="page-jump-recovery" onClick={recover}>{recoveryLabel}</button></div>}
 </form>;
}
