import AppLink from './AppLink';
import {useRef} from 'react';
import {FileCode2,X} from 'lucide-react';
import {useI18n} from './i18n';
import type {SubmissionCompletionProps} from './submissionCompletion';
import './submission-completion-notice.css';

/** Arrival never takes focus; dismissal may return to the last outside control. */
export default function SubmissionCompletionNotice({notice,inline,onView,onDismiss}:SubmissionCompletionProps) {
  const {t}=useI18n(),root=useRef<HTMLDivElement>(null),previous=useRef<HTMLElement|undefined>(undefined);
  function remember(target:EventTarget|null) {
    if(target instanceof HTMLElement&&!root.current?.contains(target))previous.current=target===document.body?undefined:target;
  }
  return <div ref={root} className="toast submission-completion-notice" role="status"
    onPointerDownCapture={()=>remember(document.activeElement)} onFocusCapture={event=>remember(event.relatedTarget)}
    onKeyDown={event=>{
      if(event.key!=='Escape'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
      const active=document.activeElement;if(!(active instanceof HTMLElement)||!event.currentTarget.contains(active))return;
      event.preventDefault();event.stopPropagation();onDismiss(previous.current);
    }}>
    <FileCode2 size={20} aria-hidden="true"/>
    <div className="submission-completion-copy"><strong>{t('提交已完成','Submission complete')} <span>#{notice.number}</span></strong>
      <p>{inline?t('本题的测试反馈已更新。','This problem’s test feedback is ready.'):t('可以查看这次提交的完整结果。','This submission is ready to review.')}</p></div>
    <button type="button" className="icon-button" aria-label={t('关闭提交完成提示','Dismiss submission notification')} onClick={()=>onDismiss(previous.current)}><X size={16} aria-hidden="true"/></button>
    <AppLink className="button subtle submission-completion-view" href={notice.href} onClick={event=>{
      if(event.defaultPrevented)return;
      const plain=event.button===0&&!event.altKey&&!event.ctrlKey&&!event.metaKey&&!event.shiftKey;
      if(onView(plain&&inline,!plain)!=='saved')event.preventDefault();
    }}>{inline?t('展开本题结果','Open Result'):t('查看提交结果','View saved result')}</AppLink>
  </div>;
}
