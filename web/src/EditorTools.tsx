import {useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import {Check,Copy,X} from 'lucide-react';
import {useI18n} from './i18n';
import {ResetCodeButton} from './deferredPages';
import {editorFontSizes} from './editorPreferences';
import {useClipboardCopy} from './useClipboardCopy';
import type {KeyboardEvent} from 'react';
import './editor-tools.css';

type ResetProps=Parameters<typeof import('./ResetCodeButton').ResetCodeButton>[0];
export type EditorToolsProps=ResetProps&{font:number;onFont:(font:number)=>void;canUseCode:()=>boolean};
export default function EditorTools(props:EditorToolsProps){
  const {t}=useI18n(),{font,onFont,canUseCode,...reset}=props;
  const source=props.saved.code??props.problem.starter;
  return <>
    <label className="editor-font-size"><span aria-hidden="true">Aa</span><select className="editor-font-size-select" value={font} title={t('代码字号','Code font size')} aria-label={t('代码字号','Code font size')} onChange={event=>{
      const size=Number(event.target.value);if(!editorFontSizes.includes(size))return;
      onFont(size);try{localStorage.setItem('paperleet.editor.fontSize',String(size));}catch{/* This session keeps the selected size. */}
    }}>{editorFontSizes.map(size=><option key={size} value={size}>{size} px</option>)}</select></label>
    <SourceCopy key={JSON.stringify([props.owner,props.problem.id,source,props.ready])} source={source} ready={props.ready} canUseCode={canUseCode}/>
    <ResetCodeButton key={`${props.owner}:${props.problem.id}`} {...reset}/>
  </>;
}
function SourceCopy({source,ready,canUseCode}:{source:string;ready:boolean;canUseCode:()=>boolean}){
  const {t}=useI18n(),{status,value,copy}=useClipboardCopy(),[dismissed,setDismissed]=useState(false);
  const root=useRef<HTMLSpanElement>(null),button=useRef<HTMLButtonElement>(null);
  const manual=status==='failed'&&value!==undefined&&!dismissed;
  function close(){setDismissed(true);if(ready&&canUseCode())button.current?.focus({preventScroll:true});}
  function escape(event:KeyboardEvent){
    if(!manual||event.key!=='Escape'||event.defaultPrevented||event.repeat||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229)return;
    event.preventDefault();event.stopPropagation();close();
  }
  const target=manual?root.current?.closest('.editor-code-panel')?.querySelector('.editor-copy-recovery'):null;
  return <span className="editor-source-copy" ref={root} onKeyDown={escape}>
    <button ref={button} type="button" className="icon-button" title={status==='copied'?t('代码已复制','Code copied'):t('复制代码','Copy code')} aria-label={t('复制代码','Copy code')} aria-disabled={!ready||status==='copying'} aria-busy={status==='copying'} onClick={()=>{
      if(!ready||!canUseCode())return;
      copy(()=>{setDismissed(false);return source;});
    }}>{status==='copied'?<Check size={14} aria-hidden="true"/>:<Copy size={14} aria-hidden="true"/>}</button>
    <span className="sr-only" role="status">{status==='copying'?t('正在复制代码…','Copying code…'):status==='copied'?t('代码已复制。','Code copied.'):manual?t('自动复制不可用，已显示完整代码供手动复制。','Automatic copy is unavailable. The complete source is available for manual copying.'):null}</span>
    {target&&createPortal(<div className="editor-copy-manual" onKeyDown={escape}>
      <div><label htmlFor="editor-manual-copy-source">{t('选择以下完整代码，使用系统快捷键或菜单复制。','Select the complete code below and use your system copy shortcut or menu.')}</label><button type="button" className="icon-button" aria-label={t('关闭手动复制','Close manual copy')} onClick={close}><X size={16} aria-hidden="true"/></button></div>
      {source===''&&<small>{t('当前文件为空（0 字节）。','The current file is empty (0 bytes).')}</small>}
      <textarea id="editor-manual-copy-source" aria-label={t('手动复制当前代码','Copy current code manually')} readOnly spellCheck={false} wrap="off" rows={6} value={value} onFocus={event=>event.currentTarget.select()}/>
    </div>,target)}
  </span>;
}
