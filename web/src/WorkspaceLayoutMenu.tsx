import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { Columns2, RotateCcw, ChevronDown } from 'lucide-react';
import { useI18n } from './i18n';

export default function WorkspaceLayoutMenu({onLayout}:{onLayout:(ratio:number,reset?:boolean)=>void}) {
  const {t}=useI18n();
  const [open,setOpen]=useState(false);
  const root=useRef<HTMLDivElement>(null);
  const trigger=useRef<HTMLButtonElement>(null);
  const focusIntent=useRef<{opening:boolean;opener:HTMLElement;href:string}|undefined>(undefined);
  const id=useId();
  useLayoutEffect(()=>{
    const intent=focusIntent.current;if(!intent)return;focusIntent.current=undefined;
    if(intent.href!==location.href)return;
    const active=document.activeElement;
    if(active!==intent.opener&&!(active===document.body&&!intent.opener.isConnected))return;
    const control=intent.opening?root.current?.querySelector<HTMLButtonElement>('.workspace-layout-popover button'):trigger.current;
    control?.focus({preventScroll:true});
  },[open]);
  useEffect(()=>{
    if(!open)return;
    const outside=(event:PointerEvent)=>{if(!root.current?.contains(event.target as Node)){focusIntent.current=undefined;setOpen(false);}};
    document.addEventListener('pointerdown',outside);
    return()=>document.removeEventListener('pointerdown',outside);
  },[open]);
  const choose=(ratio:number,opener:HTMLElement,reset=false)=>{
    focusIntent.current={opening:false,opener,href:location.href};
    onLayout(ratio,reset);setOpen(false);
  };
  return <div className="workspace-layout-menu" ref={root} onBlur={event=>{
    if(event.relatedTarget&&!event.currentTarget.contains(event.relatedTarget as Node)){focusIntent.current=undefined;setOpen(false);}
  }} onKeyDown={event=>{
    if(!open||event.key!=='Escape'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
    const active=document.activeElement;
    if(!(active instanceof HTMLElement)||!event.currentTarget.contains(active))return;
    event.preventDefault();event.stopPropagation();focusIntent.current={opening:false,opener:active,href:location.href};setOpen(false);
  }}>
    <button type="button" className="workspace-layout-trigger" ref={trigger} aria-expanded={open} aria-controls={id} onClick={()=>{focusIntent.current=undefined;setOpen(value=>!value);}} onKeyDown={event=>{
      if(event.key!=='ArrowDown'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
      event.preventDefault();
      if(open){root.current?.querySelector<HTMLButtonElement>('.workspace-layout-popover button')?.focus({preventScroll:true});return;}
      focusIntent.current={opening:true,opener:event.currentTarget,href:location.href};setOpen(true);
    }}><Columns2 size={16} aria-hidden="true"/><span>{t('布局','Layout')}</span><ChevronDown size={14} aria-hidden="true"/></button>
    {open&&<section id={id} className="workspace-layout-popover" aria-label={t('工作区布局','Workspace layout')}>
      <div className="workspace-layout-heading"><b>{t('适合你的工作区','Make room for your work')}</b><small>{t('随时拖动分隔线微调，布局自动记忆','Drag the dividers to fine-tune. Your layout is saved.')}</small></div>
      <div className="workspace-layout-options">
        {([{ratio:.62,label:t('阅读优先','Reading')},{ratio:.5,label:t('均衡分栏','Balanced')},{ratio:.36,label:t('代码优先','Coding')}]).map(preset=><button type="button" key={preset.ratio} onClick={event=>choose(preset.ratio,event.currentTarget)}>
          <span className="workspace-layout-preview" aria-hidden="true" style={{gridTemplateColumns:`${preset.ratio}fr ${1-preset.ratio}fr`}}><i/><i/></span><span>{preset.label}</span>
        </button>)}
      </div>
      <button type="button" className="workspace-layout-reset" onClick={event=>choose(.5,event.currentTarget,true)}><RotateCcw size={14} aria-hidden="true"/>{t('恢复默认布局','Reset layout')}<small>{t('均分题面与代码','Equal pane widths')}</small></button>
      <div className="workspace-layout-hints"><span>{t('运行测试','Run tests')}<kbd>⌘ / Ctrl ↵</kbd></span><span>{t('收起此面板','Close this panel')}<kbd>Esc</kbd></span></div>
    </section>}
  </div>;
}
