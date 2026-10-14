import type { EditorRecovery } from './EditorFallback';
import { useI18n } from './i18n';

export function WorkspaceBackupFailure({ onClose, recovery }: { onClose: () => void; recovery: EditorRecovery }) {
  const { t } = useI18n();
  return <section className="feature-failure" aria-label={t('代码备份加载失败', 'Code backup loading failed')}>
    <p role="alert">{t('代码备份面板暂时无法加载。可以先通过各题编辑器的下载按钮保留代码。', 'The code backup panel could not load. You can still download code from each module’s editor.')}</p>
    <div className="feature-recovery-actions">
      <button className="button subtle" type="button" onClick={onClose}>{t('关闭', 'Close')}</button>
      <button className="button subtle" type="button" disabled={recovery.blocked} onClick={recovery.reload}>{t('重新加载备份面板', 'Reload backup panel')}</button>
    </div>
    {recovery.reason && <p role="status">{recovery.reason}</p>}
  </section>;
}
