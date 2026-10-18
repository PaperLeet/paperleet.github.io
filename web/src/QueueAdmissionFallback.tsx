import AppLink from './AppLink';
import { useI18n } from './i18n';
import type { AdmissionWait } from './queueAdmission';
import type { ReactNode } from 'react';

export type QueueAdmissionProps = { waiting: AdmissionWait; changed: boolean; href?: string; onStop: () => void; onDownload: () => void };
export function QueueAdmissionFallback({waiting,changed,href,onStop,onDownload,children}: QueueAdmissionProps & {children?:ReactNode}) {
  const {t}=useI18n();
  return <section className="practice-notice queue-admission" aria-label={t('等待判题名额','Waiting for a judge slot')}>
    <strong role="status">{waiting.attempting?t('正在确认能否入队，请等待本次提交的回应。','Checking admission. Waiting for this submission to be acknowledged.'):t('队列已满，尚未入队；正在自动等待名额。','The queue is full. Your submission has not joined it; waiting for a slot.')}</strong>
    <small>{t('自动等待最多两分钟，固定使用点击提交时的代码。刷新会停止等待名额；任务入队后可刷新恢复。','Automatic waiting lasts up to two minutes and uses the code captured at your click. Refresh stops waiting for a slot; admitted jobs reconnect after refresh.')}</small>
    {changed&&<p>{t('编辑器已有新修改；等待提交的仍是此前版本。','The editor has new changes; the waiting submission still uses the earlier version.')}</p>}
    {children}
    <div className="queue-admission-actions">
      <button type="button" disabled={waiting.attempting} onClick={onStop}>{t('停止等待','Stop waiting')}</button>
      <button type="button" onClick={onDownload}>{t('下载本次提交代码','Download captured code')}</button>
      {href&&<AppLink href={href}>{t('查看等待中的题目','View waiting module')}</AppLink>}
    </div>
  </section>;
}
