import { useEffect, useRef, useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { useI18n } from './i18n';
import { failureExcerpt } from './reportFailureModel';
import { SourceLocationButton } from './SourceLocationButton';
import type { SourceLocation } from './sourceLocation';

export default function FailureDiagnostic({message,location,onLocate}:{message?:string;location?:SourceLocation;onLocate?:(location:SourceLocation)=>void}) {
  const {t} = useI18n();
  const [copyState,setCopyState] = useState<'idle'|'copied'|'failed'>('idle');
  const live = useRef(true);
  const currentMessage = useRef(message);currentMessage.current=message;
  useEffect(() => {live.current=true;return()=>{live.current=false;};},[]);
  useEffect(() => {setCopyState('idle');},[message]);
  async function copy() {
    const value=message;
    try {await navigator.clipboard.writeText(value || '');if(live.current && currentMessage.current===value)setCopyState('copied');}
    catch {if(live.current && currentMessage.current===value)setCopyState('failed');}
  }
  return <div className="failure-diagnostic">
    <div className="failure-insight"><span>{t('错误摘要','ERROR SUMMARY')}</span><p>{failureExcerpt(message) || t('测试未通过，请结合数值对比和性质说明检查实现。','This test failed. Review the comparison and property specification below.')}</p><SourceLocationButton location={location} onLocate={onLocate}/></div>
    {message && <details className="failure-trace"><summary>{t('完整堆栈与诊断','Full traceback and diagnostic')}</summary><div className="failure-trace-toolbar"><span>{t('原始错误输出','Original diagnostic output')}</span><button type="button" className="text-button" onClick={() => void copy()}>{copyState === 'copied' ? <Check size={13} aria-hidden="true"/> : <Copy size={13} aria-hidden="true"/>}{copyState === 'copied' ? t('已复制','Copied') : t('复制诊断','Copy diagnostic')}</button></div>{copyState !== 'idle' && <p className="diagnostic-copy-status" role={copyState === 'failed' ? 'alert' : 'status'}>{copyState === 'failed' ? t('未能复制，请选择下方文本手动复制。','Could not copy. Select the text below to copy it manually.') : t('完整诊断已复制。','Full diagnostic copied.')}</p>}<pre className="failure-message">{message}</pre></details>}
  </div>;
}
