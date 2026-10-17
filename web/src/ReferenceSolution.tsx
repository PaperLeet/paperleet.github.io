import {useId,useLayoutEffect,useRef,useState} from 'react';
import {ArrowUp,BookOpen,ChevronDown,Code2} from 'lucide-react';
import {useI18n} from './i18n';
import CommunityCodeBlock from './CommunityCodeBlock';
import {keepReadingTriggerVisible} from './readingViewport';

/** Spoiler consent and copy state belong to this exact reference, including A → B → A. */
export default function ReferenceSolution({solution,onCode}:{solution?:string|null;onCode:()=>void}) {
  return <ReferenceReader key={solution??''} solution={solution} onCode={onCode}/>;
}

function ReferenceReader({solution,onCode}:{solution?:string|null;onCode:()=>void}) {
  const {t}=useI18n(),regionId=useId();
  const [open,setOpen]=useState(false);
  const summary=useRef<HTMLElement>(null);
  const intent=useRef<{opener:HTMLElement;href:string}|undefined>(undefined);
  function collapse(opener:HTMLElement){intent.current={opener,href:location.href};setOpen(false);}
  useLayoutEffect(()=>{
    const pending=intent.current;if(open||!pending)return;intent.current=undefined;
    if(pending.href!==location.href)return;
    const active=document.activeElement;
    if(active!==pending.opener&&!(active===document.body&&!pending.opener.isConnected))return;
    const target=summary.current;if(!target)return;
    target.focus({preventScroll:true});keepReadingTriggerVisible(target);
  },[open]);
  return <details className="reference-reader" open={open} onToggle={event=>setOpen(event.currentTarget.open)} onKeyDown={event=>{
    if(!open||event.key!=='Escape'||event.defaultPrevented||event.repeat||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229)return;
    const active=document.activeElement;
    if(!(active instanceof HTMLElement)||!event.currentTarget.contains(active)||active.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"])'))return;
    event.preventDefault();event.stopPropagation();collapse(active);
  }}>
    <summary ref={summary} aria-expanded={open} aria-controls={regionId} aria-keyshortcuts={open?'Escape':undefined}>
      <BookOpen size={16} aria-hidden="true"/><span>{open?t('收起参考解','Collapse reference'):t('展开参考解（包含答案）','Reveal reference solution (spoilers)')}</span><ChevronDown size={16} className="reference-reader-chevron" aria-hidden="true"/>
    </summary>
    <div id={regionId} role="region" aria-label={t('参考实现代码','Reference implementation code')}>
      {open&&<>
        {solution?<CommunityCodeBlock code={solution} language="python" initiallyExpanded/>:<div className="notice">{t('参考实现尚未提供。','No reference implementation is available yet.')}</div>}
        <div className="reference-reading-actions" role="group" aria-label={t('参考实现阅读操作','Reference reading actions')}>
          <button type="button" onClick={event=>collapse(event.currentTarget)}><ArrowUp size={15} aria-hidden="true"/>{t('收起参考解','Collapse reference')}</button>
          <button type="button" onClick={onCode}><Code2 size={15} aria-hidden="true"/>{t('返回代码','Back to code')}</button>
        </div>
      </>}
    </div>
  </details>;
}
