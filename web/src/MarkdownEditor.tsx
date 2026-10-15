import {useId,useLayoutEffect,useRef,useState} from 'react';
import type {RefObject} from 'react';
import {Bold,Italic,Code,SquareCode,List,Link,SquareFunction,Undo2} from 'lucide-react';
import CommunityMarkdown,{CommunityFormatHelp} from './CommunityMarkdown';
import {useI18n} from './i18n';
import {formatMarkdown,type MarkdownEdit,type MarkdownFormat} from './markdownFormatting';
import './markdown-editor.css';

const tools=[
  ['bold',Bold,'加粗','Bold','B'],['italic',Italic,'斜体','Italic','I'],
  ['code',Code,'行内代码','Inline code',''],['codeblock',SquareCode,'代码块','Code block',''],
  ['list',List,'无序列表','Bullet list',''],['link',Link,'插入链接','Insert link','K'],
  ['math',SquareFunction,'独立公式','Display math',''],
] as const;
type Selection={start:number;end:number;scroll:number};
export default function MarkdownEditor({value,onChange,preview,onPreviewChange,label,modeLabel,rows=10,maxLength,disabled=false,placeholder,textareaRef,id,describedBy,autoFocus=false}:{
  value:string;onChange:(value:string)=>void;preview:boolean;onPreviewChange:(preview:boolean)=>void;
  label:string;modeLabel:string;rows?:number;maxLength:number;disabled?:boolean;placeholder?:string;
  textareaRef?:RefObject<HTMLTextAreaElement|null>;id?:string;describedBy?:string;autoFocus?:boolean;
}){
  const {t,locale}=useI18n(),generatedId=useId(),ownField=useRef<HTMLTextAreaElement>(null);
  const field=textareaRef||ownField;
  const [operation,setOperation]=useState<(MarkdownEdit&{selection:Selection})>();
  const [error,setError]=useState('');
  const [tick,setTick]=useState(0);
  const pending=useRef<{opener:Element|null;selection:Selection;value:string}>(undefined);
  const lastValue=useRef(value);lastValue.current=value;
  const live=useRef({disabled,preview});live.current={disabled,preview};
  const savedSelection=useRef<Selection>(undefined);
  useLayoutEffect(()=>{
    const request=pending.current;if(!request||preview)return;pending.current=undefined;
    const undoLostFocus=document.activeElement===document.body&&request.opener instanceof HTMLElement&&request.opener.matches('button:disabled');
    if(disabled||value!==request.value||(document.activeElement!==request.opener&&!undoLostFocus))return;
    const textarea=field.current;if(!textarea)return;
    textarea.focus({preventScroll:true});textarea.setSelectionRange(request.selection.start,request.selection.end);textarea.scrollTop=request.selection.scroll;
  },[value,preview,disabled,tick,field]);
  const selection=():Selection=>({start:field.current?.selectionStart??value.length,end:field.current?.selectionEnd??value.length,scroll:field.current?.scrollTop||0});
  function restore(next:Selection,nextValue:string){pending.current={opener:document.activeElement,selection:next,value:nextValue};setTick(current=>current+1);}
  function format(kind:MarkdownFormat){
    if(live.current.disabled||live.current.preview)return;
    const original=selection(),edit=formatMarkdown(lastValue.current,kind,original.start,original.end,maxLength,locale==='en');
    if(!edit){setError(t('加入此格式会超过长度上限。请精简内容后重试。','This formatting would exceed the length limit. Shorten the text and try again.'));return;}
    setError('');setOperation({...edit,selection:original});onChange(edit.after);
    restore({start:edit.start,end:edit.end,scroll:original.scroll},edit.after);
  }
  function changePreview(next:boolean){
    if(disabled)return;
    if(next)savedSelection.current=selection();
    else if(savedSelection.current)restore(savedSelection.current,value);
    onPreviewChange(next);
  }
  const canUndo=!!operation&&operation.after===value;
  const errorId=generatedId+'-error';
  return <div className="markdown-editor">
    <div className="editor-toggle" role="group" aria-label={modeLabel}>
      <button type="button" disabled={disabled} className={!preview?'selected':''} aria-pressed={!preview} onClick={()=>changePreview(false)}>{t('编写','Write')}</button>
      <button type="button" disabled={disabled} className={preview?'selected':''} aria-pressed={preview} onClick={()=>changePreview(true)}>{t('预览','Preview')}</button>
      <small>{t('Markdown · 公式与代码','Markdown · math & code')}</small>
    </div>
    <div className="markdown-format-tools" role="group" aria-label={t('正文格式工具','Text formatting')} hidden={preview}>
      {tools.map(([kind,Icon,zh,en,shortcut])=><button key={kind} type="button" disabled={disabled} title={t(zh,en)+(shortcut?` (⌘ / Ctrl + ${shortcut})`:'')} aria-label={t(zh,en)} aria-keyshortcuts={shortcut?`Control+${shortcut.toLowerCase()} Meta+${shortcut.toLowerCase()}`:undefined} onMouseDown={event=>event.preventDefault()} onClick={()=>format(kind)}><Icon size={16} aria-hidden="true"/></button>)}
      <button type="button" className="markdown-format-undo" disabled={disabled||!canUndo} onMouseDown={event=>event.preventDefault()} onClick={()=>{
        if(!operation||live.current.disabled||lastValue.current!==operation.after)return;
        onChange(operation.before);restore(operation.selection,operation.before);setOperation(undefined);setError('');
      }}><Undo2 size={15} aria-hidden="true"/>{t('撤销格式','Undo formatting')}</button>
    </div>
    <label hidden={preview} className="markdown-body-label">{label}
      <textarea ref={field} id={id} required autoFocus={autoFocus} rows={rows} maxLength={maxLength} value={value} disabled={disabled} placeholder={placeholder} aria-describedby={[describedBy,error?errorId:undefined].filter(Boolean).join(' ')||undefined} onChange={event=>{setError('');onChange(event.target.value);}} onKeyDown={event=>{
        if(event.nativeEvent.isComposing||event.altKey||event.shiftKey||!(event.metaKey||event.ctrlKey))return;
        const kind=({b:'bold',i:'italic',k:'link'} as const)[event.key.toLowerCase() as 'b'|'i'|'k'];
        if(kind){event.preventDefault();format(kind);}
      }}/>
    </label>
    {preview&&<div className="compose-preview" role="region" aria-label={t('正文预览','Content preview')}><CommunityMarkdown body={value||t('写下内容后在这里预览。','Your preview will appear here.')}/></div>}
    <div className="markdown-editor-footer"><CommunityFormatHelp/><small className={value.length>maxLength?'over-limit':''}>{value.length.toLocaleString(locale)} / {maxLength.toLocaleString(locale)}</small></div>
    {error&&<p id={errorId} className="markdown-format-error" role="status">{error}</p>}
  </div>;
}
