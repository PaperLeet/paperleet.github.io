import { useMemo, useState } from 'react';
import type {ReactNode} from 'react';
import { Check, Copy, WrapText } from 'lucide-react';
import { useI18n } from './i18n';
import { failureExcerpt } from './reportFailureModel';
import { SourceLocationButton } from './SourceLocationButton';
import type { LearnerTraceFrame, SourceLocation } from './sourceLocation';
import {useClipboardCopy} from './useClipboardCopy';
import './failure-diagnostic.css';

type Props={message?:string;location?:SourceLocation;frames?:LearnerTraceFrame[];onLocate?:(location:SourceLocation)=>void};
export default function FailureDiagnostic(props:Props) {
  return <Diagnostic key={props.message} {...props}/>;
}

function Diagnostic({message,location,frames,onLocate}:Props) {
  const {t} = useI18n();
  const [open,setOpen]=useState(false),[wrapped,setWrapped]=useState(true),{status,copy}=useClipboardCopy();
  const frameRows=useMemo(()=>new Map(frames?.map(frame=>[frame.row,frame.location])),[frames]);
  const trace=useMemo(()=>{
    if(!open||!onLocate||!frameRows.size)return message;
    const parts:ReactNode[]=[];let plain='';
    message?.split(/(\r\n|\r|\n)/).forEach((text,index)=>{
      const target=index%2?undefined:frameRows.get(index/2);
      if(!target){plain+=text;return;}
      if(plain){parts.push(plain);plain='';}
      parts.push(<button key={index} type="button" className="failure-frame-link" aria-label={t(`定位代码第 ${target.line} 行`,`Go to code line ${target.line}`)} onClick={()=>onLocate(target)}>{text}</button>);
    });
    if(plain)parts.push(plain);
    return parts;
  },[open,onLocate,message,frameRows,t]);
  return <div className="failure-diagnostic">
    <div className="failure-insight"><span>{t('错误摘要','ERROR SUMMARY')}</span><p>{failureExcerpt(message) || t('测试未通过，请结合数值对比和性质说明检查实现。','This test failed. Review the comparison and property specification below.')}</p><SourceLocationButton location={location} onLocate={onLocate}/></div>
    {message && <details className={`failure-trace${wrapped?'':' is-unwrapped'}`} open={open} onToggle={event=>setOpen(event.currentTarget.open)}>
      <summary>{t('完整堆栈与诊断','Full traceback and diagnostic')}</summary>
      <div className="failure-trace-toolbar"><span>{t('原始错误输出','Original diagnostic output')}</span>
        <button type="button" className="text-button diagnostic-wrap" aria-pressed={wrapped} onClick={()=>setWrapped(value=>!value)}><WrapText size={15} aria-hidden="true"/>{t('自动换行','Wrap lines')}</button>
        <button type="button" className="text-button diagnostic-copy" aria-disabled={status==='copying'} aria-busy={status==='copying'} onClick={()=>copy(()=>message)}>{status==='copied'?<Check size={15} aria-hidden="true"/>:<Copy size={15} aria-hidden="true"/>}{status==='copied'?t('已复制','Copied'):status==='copying'?t('正在复制…','Copying…'):t('复制诊断','Copy diagnostic')}</button>
      </div>
      {!!frames?.length&&onLocate&&<p className="failure-frame-hint">{t('点击堆栈中标出的实现位置，可直接返回对应代码。','Select a highlighted implementation frame to return to that code.')}</p>}
      {status!=='idle'&&<p className="diagnostic-copy-status" role={status==='failed'?'alert':'status'}>{status==='failed'?t('自动复制不可用，请选择下方文本并使用系统复制快捷键或菜单。','Automatic copy is unavailable. Select the text below and use your system’s copy shortcut or menu.'):status==='copying'?t('正在复制完整诊断…','Copying full diagnostic…'):t('完整诊断已复制。','Full diagnostic copied.')}</p>}
      {status==='failed'&&<label className="failure-diagnostic-manual">{t('手动复制完整诊断','Copy full diagnostic manually')}<textarea aria-label={t('可手动复制的完整诊断','Full diagnostic to copy manually')} readOnly spellCheck={false} wrap="off" rows={6} value={message} onFocus={event=>event.currentTarget.select()}/></label>}
      <pre className="failure-message">{trace}</pre>
    </details>}
  </div>;
}
