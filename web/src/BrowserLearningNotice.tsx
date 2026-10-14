import { useI18n } from './i18n';

export function BrowserLearningNotice({count,failed,saving,onRetry}:{count:number;failed:boolean;saving:boolean;onRetry:()=>void}) {
  const {t}=useI18n();
  if(!count&&!failed)return null;
  return <aside className="practice-notice browser-learning-notice" aria-label={t('本机学习进度','Browser learning progress')}>
    {count>0&&<span>{t(`本机学习进度包含 ${count} 道浏览器通过的题目，可复盘坑点并继续练习。账号成绩与统计仍需服务端确认。`,
      `Learning progress includes ${count} ${count === 1 ? 'module' : 'modules'} passed in this browser. Review Notes and continue practicing; account results and statistics still require server verification.`)}</span>}
    {failed&&<div role="alert"><span>{t('本机学习记录暂未保存，当前结果仍可查看。','Learning records could not be saved; current results remain available.')}</span>
      <button className="button subtle" type="button" disabled={saving} onClick={onRetry}>{t('重试保存学习记录','Retry saving progress')}</button></div>}
  </aside>;
}
