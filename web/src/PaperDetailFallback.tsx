import type { EditorRecovery } from './EditorFallback';
import { useI18n } from './i18n';

export function PaperDetailFailure({returnTo,recovery}:{returnTo?:string;recovery:EditorRecovery}) {
  const {t}=useI18n();
  return <main className="home">
    <section className="feature-failure">
      <p role="alert">{t('论文详情暂时无法加载。','Paper details could not load.')}</p>
      <p>{t('重新加载会保留当前论文和题库筛选。','Reloading keeps this paper and your library filters.')}</p>
      <div className="feature-recovery-actions">
        <button className="button subtle" type="button" disabled={recovery.blocked} onClick={()=>{if(!recovery.blocked)recovery.reload();}}>{t('重新加载论文','Reload paper')}</button>
        <a className="text-button" href={returnTo||'#'}>{t('返回题库结果','Back to library results')}</a>
      </div>
      {recovery.reason&&<p role="status">{recovery.reason}</p>}
    </section>
  </main>;
}
