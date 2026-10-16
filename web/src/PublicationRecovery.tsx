import { useI18n } from './i18n';
import { samePublication, type PendingPublication, type PublicationPayload } from './publicationRequests';
import {readInterviewDetails} from './interviewModels';
import InterviewMeta from './InterviewMeta';
import PaperPreviewCard from './PaperPreviewCard';

export default function PublicationRecovery({pending,payload,busy,disabled=false,onRetry}:{pending?:PendingPublication;payload:PublicationPayload;busy:boolean;disabled?:boolean;onRetry:()=>void}) {
  const {t}=useI18n();
  if(!pending||busy)return null;
  const changed=!samePublication(pending.receipt,payload);
  let interview;
  try{if(pending.receipt.kind==='interview')interview=readInterviewDetails(JSON.parse(pending.receipt.payload.interview as string));}catch{/* A malformed recovery record cannot add metadata to the review. */}
  return <div className="publication-recovery" role="status">
    <p>{t('此前发布的结果尚未确认。重试相同内容会继续确认这一次发布，不会重复创建；网络错误不代表发布失败。',
      'The previous publication is not confirmed. Retrying the same content confirms this publication without creating a duplicate; a network error does not prove it failed.')}</p>
    {changed&&<p>{t('当前草稿已经改动。直接发布会创建新内容，不会修改此前的发布；也可先重试此前发布，当前草稿会保留。',
      'This draft has changed. Publishing it creates new content rather than editing the earlier publication. You can retry the earlier publication while keeping this draft.')}</p>}
    {!pending.durable&&<p role="alert">{t('此标签页无法保存待确认记录。当前页面内可以安全重试；刷新或关闭后无法保证找回，请先复制草稿。',
      'This tab cannot save the pending publication. Retries in this page keep it; recovery after reload or closing is not guaranteed. Copy your draft first.')}</p>}
    {pending.receipt.kind!=='reply'&&<p>{pending.receipt.payload.anonymous===true?t('此前发布身份：匿名用户，所属机构公开。','Previous posting identity: anonymous, with public affiliation.'):t('此前发布身份：公开昵称。','Previous posting identity: public name.')}</p>}
    <details><summary>{t('查看待确认内容','Review pending publication')}</summary>
      {pending.receipt.payload.title&&<strong>{pending.receipt.payload.title}</strong>}
      {interview&&<InterviewMeta details={interview} compact/>}
      <pre tabIndex={0}>{pending.receipt.payload.body}</pre>
      <PaperPreviewCard url={typeof pending.receipt.payload.paper_url==='string'?pending.receipt.payload.paper_url:undefined} compact/>
      {pending.receipt.payload.parent_id&&<small>{t('这份发布包含指定回复对象。','This publication targets a specific reply.')}</small>}
    </details>
    {changed&&<button className="button subtle" type="button" disabled={disabled} onClick={onRetry}>{t('重试此前发布','Retry previous publication')}</button>}
  </div>;
}
