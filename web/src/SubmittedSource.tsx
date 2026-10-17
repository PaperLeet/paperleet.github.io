import {useState,type ReactNode} from 'react';
import {Check,ChevronDown,Code2,Copy,WrapText} from 'lucide-react';
import {useI18n} from './i18n';
import {useClipboardCopy} from './useClipboardCopy';
import './submitted-source.css';

type Props={scope:string;code:string;title?:string;children?:ReactNode};

/** A different owner, record or source starts a fresh copy/read lifetime. */
export default function SubmittedSource(props:Props){
 return <Source key={JSON.stringify([props.scope,props.code])} {...props}/>;
}
function Source({code,title,children}:Props){
 const {t}=useI18n(),[wrapped,setWrapped]=useState(false),{status,copy}=useClipboardCopy();
 return <div className={`submitted-source${wrapped?' is-wrapped':''}`}>
  <div className="submitted-source-heading"><strong>{title||t('提交时的代码','Submitted code')}</strong><div className="submitted-source-actions">
   {children}
   <button type="button" aria-pressed={wrapped} onClick={()=>setWrapped(value=>!value)}><WrapText size={15} aria-hidden="true"/>{t('自动换行','Wrap lines')}</button>
   <button type="button" aria-disabled={status==='copying'} aria-busy={status==='copying'} onClick={()=>copy(()=>code)}>{status==='copied'?<Check size={15} aria-hidden="true"/>:<Copy size={15} aria-hidden="true"/>}{status==='copying'?t('正在复制…','Copying…'):status==='copied'?t('已复制','Copied'):t('复制代码','Copy code')}</button>
  </div></div>
  <span className="sr-only" role="status">{status==='copying'?t('正在复制提交时的代码…','Copying submitted code…'):status==='copied'?t('提交代码已复制。','Submitted code copied.'):status==='failed'?t('自动复制不可用，请使用下方的手动复制区域。','Automatic copy is unavailable. Use the manual copy field below.'):null}</span>
  {status==='failed'&&<label className="submitted-source-manual"><span>{t('请选择下方代码，使用系统复制快捷键或菜单复制。','Select the code below and use your system’s copy shortcut or menu.')}</span><textarea aria-label={t('手动复制提交代码','Copy submitted code manually')} readOnly spellCheck={false} wrap="off" rows={Math.min(8,Math.max(2,code.split(/\r\n|\r|\n/).length))} value={code} onFocus={event=>event.currentTarget.select()}/></label>}
  <pre tabIndex={0} aria-label={t('提交时的 Python 代码（只读）','Submitted Python code (read only)')}><code>{code}</code></pre>
 </div>;
}

export function SubmittedSourceDisclosure(props:Props){
 return <SourceDisclosure key={JSON.stringify([props.scope,props.code])} {...props}/>;
}
function SourceDisclosure(props:Props){
 const {t}=useI18n(),[opened,setOpened]=useState(false);
 return <details className="submitted-source-disclosure" onToggle={event=>{if(event.currentTarget.open)setOpened(true);}}>
  <summary><Code2 size={18} aria-hidden="true"/><span>{t('提交时的代码','Submitted code')}</span><ChevronDown size={18} aria-hidden="true"/></summary>
  {opened&&<SubmittedSource {...props} title={t('Python · 只读','Python · Read only')}/>}
 </details>;
}
