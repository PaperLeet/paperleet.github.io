import {useEffect,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import {Check,Copy,X,Search,Replace,Code2,CornerDownRight,Keyboard} from 'lucide-react';
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
  const actions=useRef<HTMLDivElement>(null);
  useEffect(()=>{
    const menu=actions.current?.closest<HTMLDetailsElement>('details.editor-actions');if(!menu)return;
    const outside=(event:PointerEvent)=>{if(menu.open&&!menu.contains(event.target as Node)&&!menu.querySelector('dialog[open]'))menu.open=false;};
    const escape=(event:globalThis.KeyboardEvent)=>{if(!menu.open||event.key!=='Escape'||event.defaultPrevented||event.repeat||event.isComposing||event.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||menu.querySelector('dialog[open]'))return;event.preventDefault();event.stopPropagation();menu.open=false;menu.querySelector('summary')?.focus();};
    document.addEventListener('pointerdown',outside);menu.addEventListener('keydown',escape);
    return()=>{document.removeEventListener('pointerdown',outside);menu.removeEventListener('keydown',escape);};
  },[]);
  function command(name:string){
    const menu=actions.current?.closest<HTMLDetailsElement>('details.editor-actions'),editor=actions.current?.closest('.editor-pane,.editor-code-panel')?.querySelector('.python-editor');
    if(!editor)return;
    if(menu)menu.open=false;
    editor.dispatchEvent(new CustomEvent('paperleet:editor-command',{detail:name}));
  }
  return <>
    <div ref={actions} className="editor-tool-actions">
      {[
        {name:'find',label:t('查找','Find'),keys:'Ctrl / ⌘ F',icon:Search},
        {name:'replace',label:t('查找和替换','Find and replace'),keys:'Ctrl H / ⌘ ⌥ F',icon:Replace},
        {name:'symbols',label:t('跳转到函数或类','Go to function or class'),keys:'Ctrl / ⌘ Shift O',icon:Code2},
        {name:'line',label:t('跳转到行','Go to line'),keys:'Ctrl G',icon:CornerDownRight},
        {name:'shortcuts',label:t('键盘快捷键','Keyboard shortcuts'),keys:'F1',icon:Keyboard},
      ].map(action=><button type="button" disabled={!props.ready} key={action.name} onClick={()=>command(action.name)}><action.icon size={14}/><span>{action.label}</span><kbd>{action.keys}</kbd></button>)}
    </div>
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
  function close(){setDismissed(true);if(ready&&canUseCode()){const menu=button.current?.closest<HTMLDetailsElement>('details.editor-actions');(menu&&!menu.open?menu.querySelector<HTMLElement>('summary'):button.current)?.focus({preventScroll:true});}}
  function escape(event:KeyboardEvent){
    if(!manual||event.key!=='Escape'||event.defaultPrevented||event.repeat||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229)return;
    event.preventDefault();event.stopPropagation();close();
  }
  const target=manual?root.current?.closest('.editor-pane,.editor-code-panel')?.querySelector('.editor-copy-recovery'):null;
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
