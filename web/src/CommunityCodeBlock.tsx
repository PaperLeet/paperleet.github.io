import {useId,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {Check,ChevronDown,ChevronUp,Copy,WrapText} from 'lucide-react';
import {Highlight} from './SyntaxHighlight';
import {useI18n} from './i18n';
import {useClipboardCopy} from './useClipboardCopy';
import {codeReadingPreview} from './codeReadingPreview';
import {keepReadingTriggerVisible} from './readingViewport';
import './community-code-block.css';

type Props={code:string;language?:string;initiallyExpanded?:boolean};

export default function CommunityCodeBlock({code,language='text',initiallyExpanded=false}:Props) {
  // A replaced snippet starts a new lifetime, including A → B → A replacements.
  return <CodeBlock key={JSON.stringify([code,language,initiallyExpanded])} code={code} language={language} initiallyExpanded={initiallyExpanded}/>;
}

function CodeBlock({code,language,initiallyExpanded}:Required<Props>) {
  const {t}=useI18n(),regionId=useId(),toggle=useRef<HTMLButtonElement>(null);
  const reading=useMemo(()=>codeReadingPreview(code),[code]);
  const [expanded,setExpanded]=useState(initiallyExpanded);
  const closing=useRef<{opener:HTMLElement;href:string}|undefined>(undefined);
  function collapse(opener:HTMLElement){closing.current={opener,href:location.href};setExpanded(false);}
  useLayoutEffect(()=>{
    const pending=closing.current;if(expanded||!pending)return;closing.current=undefined;
    if(pending.href!==location.href)return;
    const active=document.activeElement;
    if(active!==pending.opener&&!(active===document.body&&!pending.opener.isConnected))return;
    if(toggle.current){toggle.current.focus({preventScroll:true});keepReadingTriggerVisible(toggle.current);}
  },[expanded]);
  const expandLabel=t(`展开完整代码（${reading.lines} 行）`,`Show all ${reading.lines} lines`);
  const collapseLabel=t('收起完整代码','Collapse full code');
  const [wrapped,setWrapped]=useState(false),{status,copy}=useClipboardCopy();
  return <div className={`community-code-block${wrapped?' is-wrapped':''}${reading.collapsible?' has-preview':''}`} onKeyDown={event=>{
    if(!reading.collapsible||!expanded||event.key!=='Escape'||event.defaultPrevented||event.repeat||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229)return;
    const active=document.activeElement;
    if(!(active instanceof HTMLElement)||!event.currentTarget.contains(active)||active.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"])'))return;
    event.preventDefault();event.stopPropagation();collapse(active);
  }}>
    <div className="community-code-header">
      <span>{language}{reading.collapsible&&<small>{t(`${reading.lines} 行`,`${reading.lines} lines`)}</small>}</span>
      <div className="community-code-actions">
        {reading.collapsible&&<button ref={toggle} type="button" aria-label={expanded?collapseLabel:expandLabel} aria-expanded={expanded} aria-controls={regionId} onClick={event=>expanded?collapse(event.currentTarget):setExpanded(true)}>
          {expanded?<ChevronUp size={15} aria-hidden="true"/>:<ChevronDown size={15} aria-hidden="true"/>}{expanded?t('收起','Collapse'):t('展开','Expand')}
        </button>}
        <button type="button" aria-pressed={wrapped} aria-label={t(`自动换行 ${language} 代码`,`Wrap ${language} code`)} onClick={()=>setWrapped(value=>!value)}>
          <WrapText size={15} aria-hidden="true"/>{t('自动换行','Wrap lines')}
        </button>
        <button type="button" aria-disabled={status==='copying'} aria-busy={status==='copying'} onClick={()=>copy(()=>code)} aria-label={t(`复制 ${language} 代码`,`Copy ${language} code`)}>
          {status==='copied'?<Check size={15} aria-hidden="true"/>:<Copy size={15} aria-hidden="true"/>}
          {status==='copied'?t('已复制','Copied'):status==='copying'?t('正在复制…','Copying…'):t('复制代码','Copy code')}
        </button>
      </div>
    </div>
    <span id={regionId+'-help'} className="sr-only">{t('代码可使用方向键滚动浏览，按 Tab 离开。可在上方切换自动换行。','Use arrow keys to scroll the code and Tab to leave. Toggle line wrapping above.')}</span>
    <div id={regionId} className="community-code-content" tabIndex={0} role="region" aria-label={t(`${language} 代码`,`${language} code`)} aria-describedby={regionId+'-help'}>
      <Highlight code={!reading.collapsible||expanded?code:reading.preview} lang={language}/>
    </div>
    {reading.collapsible&&<div className="community-code-disclosure">
      <span>{expanded?t(`完整代码 · ${reading.lines} 行`,`Full code · ${reading.lines} lines`):t(`预览前 16 / ${reading.lines} 行；复制将包含完整代码。`,`Previewing 16 of ${reading.lines} lines; copying includes the full code.`)}</span>
      <button type="button" aria-expanded={expanded} aria-controls={regionId} onClick={event=>expanded?collapse(event.currentTarget):setExpanded(true)}>{expanded?<ChevronUp size={15} aria-hidden="true"/>:<ChevronDown size={15} aria-hidden="true"/>}{expanded?collapseLabel:expandLabel}</button>
    </div>}
    <span className="sr-only" role="status">{status==='copying'?t('正在复制代码…','Copying code…'):status==='copied'?t('代码已复制。','Code copied.'):status==='failed'?t('自动复制不可用，请使用下方的手动复制区域。','Automatic copy is unavailable. Use the manual copy field below.'):null}</span>
    {status==='failed'&&<label className="community-code-manual">
      <span>{t('请选择下方代码，使用系统复制快捷键或菜单复制。','Select the code below and use your system’s copy shortcut or menu.')}</span>
      <textarea aria-label={t(`手动复制 ${language} 代码`,`Copy ${language} code manually`)} readOnly spellCheck={false} wrap="off" rows={Math.min(8,Math.max(2,code.split(/\r\n|\r|\n/).length))} value={code} onFocus={event=>event.currentTarget.select()}/>
    </label>}
  </div>;
}
