import {useDeferredValue,useId,useLayoutEffect,useRef,useState} from 'react';
import type {RefObject} from 'react';
import {Bold,Italic,Code,SquareCode,List,ListOrdered,Heading2,Quote,Link,SquareFunction,Undo2,LoaderCircle} from 'lucide-react';
import CommunityMarkdown,{CommunityFormatHelp} from './CommunityMarkdown';
import {useI18n} from './i18n';
import {formatMarkdown,type MarkdownEdit,type MarkdownFormat} from './markdownFormatting';
import './markdown-editor.css';

const tools=[
  ['heading',Heading2,'小标题','Section heading',''],
  ['bold',Bold,'加粗','Bold','B'],['italic',Italic,'斜体','Italic','I'],
  ['quote',Quote,'引用','Block quote',''],
  ['ordered',ListOrdered,'编号步骤','Numbered steps',''],
  ['code',Code,'行内代码','Inline code',''],['codeblock',SquareCode,'代码块','Code block',''],
  ['list',List,'无序列表','Bullet list',''],['link',Link,'插入链接','Insert link','K'],
  ['math',SquareFunction,'独立公式','Display math',''],
] as const;
type Selection={start:number;end:number;scroll:number;left?:number;direction?:HTMLTextAreaElement['selectionDirection']};
export type MarkdownEditorView={value:string;selection:Selection;instance:object;scopeKey?:string;livePreview?:true};
type Props={
  value:string;onChange:(value:string)=>void;preview:boolean;onPreviewChange:(preview:boolean)=>void;
  label:string;modeLabel:string;rows?:number;maxLength:number;disabled?:boolean;placeholder?:string;
  textareaRef?:RefObject<HTMLTextAreaElement|null>;viewStateRef?:RefObject<MarkdownEditorView|undefined>;id?:string;describedBy?:string;autoFocus?:boolean;submitShortcut?:boolean;busy?:boolean;scopeKey?:string;
};
export default function MarkdownEditor(props:Props){
  return <ScopedMarkdownEditor key={props.scopeKey} {...props}/>;
}
function ScopedMarkdownEditor({value,onChange,preview,onPreviewChange,label,modeLabel,rows=10,maxLength,disabled=false,placeholder,textareaRef,viewStateRef,id,describedBy,autoFocus=false,submitShortcut=false,busy=false,scopeKey}:Props){
  const {t,locale}=useI18n(),generatedId=useId(),ownField=useRef<HTMLTextAreaElement>(null);
  const field=textareaRef||ownField,locked=disabled||busy;
  const [operation,setOperation]=useState<(MarkdownEdit&{selection:Selection})>();
  const [error,setError]=useState('');
  const [tick,setTick]=useState(0);
  const [livePreview,setLivePreview]=useState(()=>viewStateRef?.current?.scopeKey===scopeKey&&viewStateRef?.current?.value===value&&viewStateRef.current.livePreview===true);
  const deferredValue=useDeferredValue(value),paired=!preview&&livePreview;
  const pending=useRef<{opener:Element|null;selection:Selection;value:string}>(undefined);
  const lastValue=useRef(value);lastValue.current=value;
  const live=useRef({disabled:locked,preview,submitShortcut,paired});live.current={disabled:locked,preview,submitShortcut,paired};
  const savedSelection=useRef<Selection>(undefined);
  const previewValue=useRef<string>(undefined);
  const viewInstance=useRef({});
  useLayoutEffect(()=>{
    const request=pending.current;if(!request||preview)return;pending.current=undefined;
    const undoLostFocus=document.activeElement===document.body&&request.opener instanceof HTMLElement&&request.opener.matches('button:disabled');
    if(locked||value!==request.value||(document.activeElement!==request.opener&&!undoLostFocus))return;
    const textarea=field.current;if(!textarea)return;
    textarea.focus({preventScroll:true});textarea.setSelectionRange(request.selection.start,request.selection.end,request.selection.direction);textarea.scrollTop=request.selection.scroll;textarea.scrollLeft=request.selection.left||0;
  },[value,preview,locked,tick,field]);
  useLayoutEffect(()=>{
    if(!viewStateRef)return;
    // Effect replay keeps this editor; only a new editor consumes its snapshot.
    if(viewStateRef.current&&(viewStateRef.current.instance===viewInstance.current||viewStateRef.current.scopeKey!==scopeKey))viewStateRef.current=undefined;
    const node=field.current;
    return()=>{
      const current=lastValue.current;
      const position=node&&node.value===current?(live.current.preview
        ?previewValue.current===current?savedSelection.current:undefined
        :{start:node.selectionStart,end:node.selectionEnd,direction:node.selectionDirection,scroll:node.scrollTop,left:node.scrollLeft}):undefined;
      viewStateRef.current=position?{value:current,selection:position,instance:viewInstance.current,scopeKey,...(live.current.paired?{livePreview:true as const}:{})}:undefined;
    };
  },[field,viewStateRef,scopeKey]);
  const selection=():Selection=>({start:field.current?.selectionStart??value.length,end:field.current?.selectionEnd??value.length,direction:field.current?.selectionDirection,scroll:field.current?.scrollTop||0,left:field.current?.scrollLeft||0});
  function restore(next:Selection,nextValue:string){pending.current={opener:document.activeElement,selection:next,value:nextValue};setTick(current=>current+1);}
  function format(kind:MarkdownFormat){
    if(live.current.disabled||live.current.preview)return;
    const original=selection(),edit=formatMarkdown(lastValue.current,kind,original.start,original.end,maxLength,locale==='en');
    if(!edit){setError(t('加入此格式会超过长度上限。请精简内容后重试。','This formatting would exceed the length limit. Shorten the text and try again.'));return;}
    setError('');setOperation({...edit,selection:original});onChange(edit.after);
    restore({start:edit.start,end:edit.end,scroll:original.scroll,left:original.left},edit.after);
  }
  function changeMode(next:'write'|'preview'|'live'){
    if(locked)return;
    if(next==='preview'){savedSelection.current=selection();previewValue.current=value;}
    else if(!preview)restore(selection(),value);
    else if(savedSelection.current&&previewValue.current===value)restore(savedSelection.current,value);
    setLivePreview(next==='live');onPreviewChange(next==='preview');
  }
  const canUndo=!!operation&&operation.after===value;
  const errorId=generatedId+'-error',shortcutId=generatedId+'-submit-shortcut',busyId=generatedId+'-busy',showShortcut=submitShortcut&&!preview&&!busy;
  return <div className={`markdown-editor${paired?' has-live-preview':''}`}>
    <div className="editor-toggle" role="group" aria-label={modeLabel}>
      <button type="button" disabled={locked} className={!preview&&!paired?'selected':''} aria-pressed={!preview&&!paired} onClick={()=>changeMode('write')}>{t('编写','Write')}</button>
      <button type="button" disabled={locked} className={preview?'selected':''} aria-pressed={preview} onClick={()=>changeMode('preview')}>{t('预览','Preview')}</button>
      <button type="button" disabled={locked} className={paired?'selected':''} aria-pressed={paired} onClick={()=>changeMode('live')}>{t('边写边预览','Live preview')}</button>
      <small>{t('Markdown · 公式与代码','Markdown · math & code')}</small>
    </div>
    <div className="markdown-format-tools" role="group" aria-label={t('正文格式工具','Text formatting')} hidden={preview}>
      {tools.map(([kind,Icon,zh,en,shortcut])=><button key={kind} type="button" disabled={locked} title={t(zh,en)+(shortcut?` (⌘ / Ctrl + ${shortcut})`:'')} aria-label={t(zh,en)} aria-keyshortcuts={shortcut?`Control+${shortcut.toLowerCase()} Meta+${shortcut.toLowerCase()}`:undefined} onMouseDown={event=>event.preventDefault()} onClick={()=>format(kind)}><Icon size={16} aria-hidden="true"/></button>)}
      <button type="button" className="markdown-format-undo" disabled={locked||!canUndo} onMouseDown={event=>event.preventDefault()} onClick={()=>{
        if(!operation||live.current.disabled||lastValue.current!==operation.after)return;
        onChange(operation.before);restore(operation.selection,operation.before);setOperation(undefined);setError('');
      }}><Undo2 size={15} aria-hidden="true"/>{t('撤销格式','Undo formatting')}</button>
    </div>
    <div className="markdown-editor-body"><label hidden={preview} className="markdown-body-label">{label}
      <textarea ref={field} id={id} required autoFocus={autoFocus} rows={rows} maxLength={maxLength} value={value} disabled={disabled} readOnly={busy} aria-busy={busy||undefined} placeholder={placeholder} aria-keyshortcuts={submitShortcut&&!locked&&!preview?'Control+Enter Meta+Enter':undefined} aria-describedby={[describedBy,error?errorId:undefined,showShortcut?shortcutId:undefined,busy?busyId:undefined].filter(Boolean).join(' ')||undefined} onChange={event=>{setError('');onChange(event.target.value);}} onKeyDown={event=>{
        if(event.defaultPrevented||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.altKey||event.shiftKey||!(event.metaKey||event.ctrlKey))return;
        if(live.current.submitShortcut&&event.key==='Enter'){
          event.preventDefault();
          const textarea=event.currentTarget;
          if(live.current.disabled||live.current.preview||event.repeat||!textarea.isConnected||textarea.matches(':disabled')||textarea.closest('[hidden],[aria-hidden="true"]')||document.activeElement!==textarea)return;
          const form=textarea.form;
          const submitter=form?Array.from(form.elements).find(element=>element.matches('button,input')&&(element as HTMLButtonElement).type==='submit') as HTMLButtonElement|HTMLInputElement|undefined:undefined;
          if(form&&submitter&&!submitter.matches(':disabled')&&submitter.getAttribute('aria-disabled')!=='true'&&!submitter.closest('[hidden],[aria-hidden="true"]'))form.requestSubmit(submitter);
          return;
        }
        const kind=({b:'bold',i:'italic',k:'link'} as const)[event.key.toLowerCase() as 'b'|'i'|'k'];
        if(kind){event.preventDefault();format(kind);}
      }}/>
    </label>
    {(preview||paired)&&<div className="compose-preview" role="region" aria-label={t('正文预览','Content preview')} aria-busy={paired&&deferredValue!==value}>
      {paired&&<span className="markdown-preview-heading">{t('正文预览','Content preview')}</span>}
      {paired&&deferredValue!==value&&<span className="sr-only" role="status">{t('预览更新中…','Updating preview…')}</span>}
      <CommunityMarkdown body={(preview?value:value?deferredValue:'')||t('写下内容后在这里预览。','Your preview will appear here.')}/>
    </div>}</div>
    <div className={`markdown-editor-footer${showShortcut?' has-submit-shortcut':busy?' has-editor-status':''}`}><CommunityFormatHelp/>{busy?<span id={busyId} className="markdown-busy-status" role="status"><span className="markdown-busy-indicator" aria-hidden="true"><LoaderCircle size={12}/></span><span>{t('处理中…','Working…')}</span></span>:showShortcut&&<span id={shortcutId} className="markdown-submit-shortcut"><kbd>⌘ / Ctrl</kbd><span>+</span><kbd>Enter</kbd><span>{t('提交','to submit')}</span></span>}<small className={value.length>maxLength?'over-limit':''}>{value.length.toLocaleString(locale)} / {maxLength.toLocaleString(locale)}</small></div>
    {error&&<p id={errorId} className="markdown-format-error" role="status">{error}</p>}
  </div>;
}
