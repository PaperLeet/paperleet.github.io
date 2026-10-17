import {useState} from 'react';
import {Check,Copy,Link} from 'lucide-react';
import {useI18n} from './i18n';
import {useAccount} from './social';
import {useClipboardCopy} from './useClipboardCopy';
import {interviewBatchChecklist,interviewBatchShareUrl} from './interviewBatchSharing';
import type {Problem} from './types';
import './interview-batch-share.css';

type Props={href?:string;modules:Problem[]};
export default function InterviewBatchShare({href,modules}:Props){
  const {user,loading}=useAccount();
  if(loading||!href||!modules.length)return null;
  const source=modules.map(module=>[module.id,module.number,module.title,module.translations?.en?.title,module.estimated_minutes]);
  return <BatchShare key={JSON.stringify([user?.id||'guest',href,source])} href={href} modules={modules}/>;
}
function BatchShare({href,modules}:{href:string;modules:Problem[]}){
  const {t,locale}=useI18n(),[kind,setKind]=useState<'link'|'checklist'>('link');
  const {status,value,copy}=useClipboardCopy();
  function prepare(next:'link'|'checklist'){
    copy(()=>{setKind(next);return next==='link'?interviewBatchShareUrl(location.href,href,modules):interviewBatchChecklist(location.href,href,modules,locale==='en');});
  }
  return <div className="interview-batch-share" role="group" aria-label={t('保存与分享练习组','Save and share this practice batch')}>
    <div className="interview-batch-share-actions">
      <button type="button" data-batch-copy="link" aria-disabled={status==='copying'} aria-busy={status==='copying'&&kind==='link'} onClick={()=>prepare('link')}>{status==='copied'&&kind==='link'?<Check size={15} aria-hidden="true"/>:<Link size={15} aria-hidden="true"/>}{t('复制练习组链接','Copy batch link')}</button>
      <button type="button" data-batch-copy="checklist" aria-disabled={status==='copying'} aria-busy={status==='copying'&&kind==='checklist'} onClick={()=>prepare('checklist')}>{status==='copied'&&kind==='checklist'?<Check size={15} aria-hidden="true"/>:<Copy size={15} aria-hidden="true"/>}{t('复制准备清单','Copy study checklist')}</button>
    </div>
    <small>{t('链接保留题目顺序，清单可保存到笔记；打开者使用自己的学习进度。','Keep the selected order in a link or save the checklist to your notes. Viewers use their own progress.')}</small>
    <p className={status==='copied'?'interview-batch-share-feedback':'sr-only'} role="status">{status==='copying'?kind==='link'?t('正在复制练习组链接…','Copying the batch link…'):t('正在复制准备清单…','Copying the study checklist…'):status==='copied'?kind==='link'?t('练习组链接已复制。','Batch link copied.'):t('准备清单已复制。','Study checklist copied.'):status==='failed'&&value?t('自动复制不可用，请手动复制下方内容。','Automatic copy is unavailable. Copy the content below manually.'):null}</p>
    {status==='failed'&&(value?<label className="interview-batch-share-manual"><span>{t('请选择并复制以下内容。','Select and copy the content below.')}</span>{kind==='link'?<input aria-label={t('手动复制练习组链接','Copy batch link manually')} readOnly value={value} onFocus={event=>event.currentTarget.select()}/>:<textarea aria-label={t('手动复制准备清单','Copy study checklist manually')} readOnly spellCheck={false} rows={8} wrap="off" value={value} onFocus={event=>event.currentTarget.select()}/>}</label>:<p className="interview-batch-share-feedback" role="alert">{t('暂时无法生成分享内容，请检查地址和练习组是否完整。','Could not prepare sharing content. Check the address and whether the batch is complete.')}</p>)}
  </div>;
}
