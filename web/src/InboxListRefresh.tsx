import {RefreshCw} from 'lucide-react';
import {useI18n} from './i18n';

export default function InboxListRefresh({changed,busy,onRefresh}:{changed:boolean;busy:boolean;onRefresh:()=>void}) {
  const {t}=useI18n();
  return <div className={`inbox-list-refresh${changed?' has-updates':''}`}>
    <span role="status">{changed?<>{t('有更新','Updates available')}<span className="sr-only">{t('，当前列表已保留。', '. Your current list is kept.')}</span></>:null}</span>
    <button type="button" aria-disabled={busy} aria-busy={busy} onClick={()=>{if(!busy)onRefresh();}}><RefreshCw size={14} aria-hidden="true"/>{busy?t('正在刷新…','Refreshing…'):changed?t('查看最新','Show latest'):t('刷新','Refresh')}</button>
  </div>;
}
