import { useEffect, useState } from 'react';
import { useI18n } from './i18n';
import { QueueAdmissionFallback } from './QueueAdmissionFallback';
import type { QueueAdmissionProps } from './QueueAdmissionFallback';

export default function QueueAdmissionNotice(props:QueueAdmissionProps) {
  const {t}=useI18n();
  const [now,setNow]=useState(Date.now),[open,setOpen]=useState(false);
  useEffect(()=>{
    setNow(Date.now());
    const interval=setInterval(()=>setNow(Date.now()),1000);
    return()=>clearInterval(interval);
  },[props.waiting.retryAt]);
  const seconds=Math.max(0,Math.ceil((props.waiting.retryAt-now)/1000));
  return <QueueAdmissionFallback {...props}>
    {!props.waiting.attempting&&<span aria-live="off">{seconds>0?t(`约 ${seconds} 秒后再次尝试入队；这不是判题开始时间。`,`Trying to join again in about ${seconds}s; this is not a judge start estimate.`):t('正在等待确认下一次入队尝试…','Waiting to confirm the next admission attempt…')}</span>}
    <details open={open} onToggle={event=>setOpen(event.currentTarget.open)}>
      <summary>{t('查看本次提交的代码','Review captured code')}</summary>
      {open&&<pre tabIndex={0} aria-label={t('本次等待提交的代码','Captured source waiting for admission')}>{props.waiting.source}</pre>}
    </details>
  </QueueAdmissionFallback>;
}
