import { useEffect, useId, useRef, useState } from 'react';
import { Columns2, RotateCcw, ChevronDown } from 'lucide-react';
import { useI18n } from './i18n';

export default function WorkspaceLayoutMenu({onLayout}:{onLayout:(ratio:number,reset?:boolean)=>void}) {
  const {t}=useI18n();
  const [open,setOpen]=useState(false);
  const root=useRef<HTMLDivElement>(null);
  const trigger=useRef<HTMLButtonElement>(null);
  const id=useId();
  useEffect(()=>{
    if(!open)return;
    const outside=(event:PointerEvent)=>{if(!root.current?.contains(event.target as Node))setOpen(false);};
    const escape=(event:KeyboardEvent)=>{
      if(event.key!=='Escape')return;
      event.preventDefault();setOpen(false);trigger.current?.focus({preventScroll:true});
    };
    document.addEventListener('pointerdown',outside);
    document.addEventListener('keydown',escape);
    return()=>{document.removeEventListener('pointerdown',outside);document.removeEventListener('keydown',escape);};
  },[open]);
  const choose=(ratio:number,reset=false)=>{onLayout(ratio,reset);setOpen(false);trigger.current?.focus({preventScroll:true});};
  return <div className="workspace-layout-menu" ref={root} onBlur={event=>{
    if(event.relatedTarget&&!event.currentTarget.contains(event.relatedTarget as Node))setOpen(false);
  }}>
    <button type="button" className="workspace-layout-trigger" ref={trigger} aria-expanded={open} aria-controls={id} onClick={()=>setOpen(value=>!value)} onKeyDown={event=>{
      if(event.key!=='ArrowDown')return;
      event.preventDefault();setOpen(true);
      requestAnimationFrame(()=>root.current?.querySelector<HTMLButtonElement>('.workspace-layout-popover button')?.focus());
    }}><Columns2 size={14}/><span>{t('布局','Layout')}</span><ChevronDown size={12}/></button>
    {open&&<section id={id} className="workspace-layout-popover" aria-label={t('工作区布局','Workspace layout')}>
      <div className="workspace-layout-heading"><b>{t('适合你的工作区','Make room for your work')}</b><small>{t('随时拖动分隔线微调，布局自动记忆','Drag the dividers to fine-tune. Your layout is saved.')}</small></div>
      <div className="workspace-layout-options">
        {([{ratio:.62,label:t('阅读优先','Reading')},{ratio:.5,label:t('均衡分栏','Balanced')},{ratio:.36,label:t('代码优先','Coding')}]).map(preset=><button type="button" key={preset.ratio} onClick={()=>choose(preset.ratio)}>
          <span className="workspace-layout-preview" aria-hidden="true" style={{gridTemplateColumns:`${preset.ratio}fr ${1-preset.ratio}fr`}}><i/><i/></span><span>{preset.label}</span>
        </button>)}
      </div>
      <button type="button" className="workspace-layout-reset" onClick={()=>choose(.5,true)}><RotateCcw size={13}/>{t('恢复默认布局','Reset layout')}<small>{t('包含结果区高度','Includes results height')}</small></button>
      <div className="workspace-layout-hints"><span>{t('运行测试','Run tests')}<kbd>⌘ / Ctrl ↵</kbd></span><span>{t('退出专注','Exit focus')}<kbd>Esc</kbd></span></div>
    </section>}
  </div>;
}
