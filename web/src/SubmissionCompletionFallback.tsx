import AppLink from './AppLink';
import {useI18n} from './i18n';
import type {SubmissionCompletionProps} from './submissionCompletion';

export function SubmissionCompletionFallback({notice,inline,onView,onDismiss}:SubmissionCompletionProps) {
  const {t}=useI18n();
  return <div className="toast submission-completion-fallback" role="status"><span>{t('提交已完成','Submission complete')} · #{notice.number}</span>
    <AppLink className="text-button" href={notice.href} onClick={event=>{
      if(event.defaultPrevented)return;
      const plain=event.button===0&&!event.altKey&&!event.ctrlKey&&!event.metaKey&&!event.shiftKey;
      if(onView(plain&&inline,!plain)!=='saved')event.preventDefault();
    }}>{t('查看结果','View result')}</AppLink>
    <button type="button" className="icon-button" aria-label={t('关闭提交完成提示','Dismiss submission notification')} onClick={()=>onDismiss()}>×</button>
  </div>;
}
