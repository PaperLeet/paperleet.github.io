import { useState } from 'react';
import { useI18n } from './i18n';
import type { BrowserBackup } from './browserWorkspaceStore';
import type { Saved } from './types';

export function BrowserDraftBackups({backups,source,conflict,busy,onRestore,onDiscard}: {
  backups: BrowserBackup[]; source: string; conflict?: Saved; busy: boolean;
  onRestore: (backup: BrowserBackup) => void; onDiscard: (backup: BrowserBackup) => Promise<void>;
}) {
  const {t,locale} = useI18n();
  const [discard,setDiscard] = useState<string>();
  const groups = new Map<string,BrowserBackup[]>();
  for (const backup of backups) if (backup.code !== source) groups.set(backup.code,[...(groups.get(backup.code) || []),backup]);
  const alternatives = [...groups.values()];
  if (!conflict && !alternatives.length) return null;
  function download(backup: BrowserBackup) {
    const url = URL.createObjectURL(new Blob([backup.code],{type:'text/x-python;charset=utf-8'}));
    const anchor = document.createElement('a'); anchor.href = url;
    anchor.download = `paperleet-${backup.id.replaceAll('/','-')}-recovered.py`; anchor.click(); URL.revokeObjectURL(url);
  }
  return <section className="browser-draft-backups" aria-label={t('本机草稿版本','Saved draft versions')}>
    {conflict && <details open><summary>{t('另一标签页保存的版本','Version saved in another tab')}</summary>
      <p>{t('编辑器保留你的代码。核对下面的版本，再选择保留哪份；修改不会自动合并。','Your code stays in the editor. Review this version before choosing which to keep; edits are not merged automatically.')}</p>
      <pre tabIndex={0} aria-label={t('另一份代码','Other source')}>{conflict.code || t('尚无代码','No source saved')}</pre>
    </details>}
    {!!alternatives.length && <details><summary>{t(`可恢复的其他版本（${alternatives.length}）`,`Other recoverable versions (${alternatives.length})`)}</summary>
      <p>{t('这些版本只保存在此浏览器，不会写入账号。载入后仍会检查是否与最新草稿冲突。','These versions stay in this browser and do not update an account. Loading one still checks it against the latest draft.')}</p>
      {alternatives.map(copies => {
        const backup = copies.reduce((a,b)=>b.updated > a.updated ? b : a);
        return <article key={backup.version}>
        <small>{Number.isFinite(Date.parse(backup.updated)) ? new Date(backup.updated).toLocaleString(locale === 'zh' ? 'zh-CN' : 'en-US') : t('时间未知','Unknown time')}</small>
        {copies.length > 1 && <small> · {t(`${copies.length} 份相同代码的备份`,`${copies.length} backups with identical source`)}</small>}
        <pre tabIndex={0}>{backup.code}</pre>
        <div><button type="button" disabled={busy} onClick={()=>onRestore(backup)}>{t('载入此版本','Load this version')}</button>
          <button type="button" onClick={()=>download(backup)}>{t('下载备份','Download backup')}</button>
          <button type="button" disabled={busy} onClick={()=>setDiscard(backup.version)}>{t('放弃备份','Discard backup')}</button></div>
        {discard === backup.version && <div role="group" aria-label={t('确认放弃备份','Confirm discarding backup')}>
          <p>{t('放弃这份代码的已显示本机备份？编辑器中的代码不会改变。','Discard the displayed local backups of this source? The editor stays as it is.')}</p>
          <button type="button" disabled={busy} onClick={()=>{void Promise.all(copies.map(onDiscard));setDiscard(undefined);}}>{t('确认放弃','Discard this backup')}</button>
          <button type="button" onClick={()=>setDiscard(undefined)}>{t('取消','Cancel')}</button>
        </div>}
      </article>;})}
    </details>}
  </section>;
}
