import { useI18n } from './i18n';

export function BrowserLearningNotice({count,failed,saving,onRetry}:{count:number;failed:boolean;saving:boolean;onRetry:()=>void}) {
  const {t}=useI18n();
  if(!count&&!failed)return null;
  return <aside className="practice-notice browser-learning-notice" aria-label={t('学习进度','Learning progress')}>
    {count>0&&<span>{t(`已在当前工作区完成 ${count} 道题，可复盘坑点并继续练习。提交后才能计入账号成绩与统计。`,
      `You have passed ${count} ${count === 1 ? 'module' : 'modules'} in this workspace. Review Notes and keep practicing. Submit your solutions to record account results and statistics.`)}</span>}
    {failed&&<div role="alert"><span>{t('本机学习记录暂未保存，当前结果仍可查看。','Learning records could not be saved; current results remain available.')}</span>
      <button className="button subtle" type="button" disabled={saving} onClick={onRetry}>{t('重试保存学习记录','Retry saving progress')}</button></div>}
  </aside>;
}
